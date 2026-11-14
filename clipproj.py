"""A small Qwen3-VL text encoder in place of H3's 32B one, through a learned projection.

Ported from ComfyUI-ClipProj by nicolab28 (MIT, see NOTICE and LICENSES/ClipProj-MIT.txt):
the projection file format, the wrapper that makes a 4B or 8B encoder answer like H3's 32B,
and its fixes for video blocks, int8 vision weights and the attention sink. The loaders,
GPU pinning and text generation of the original are left out: the encoder is loaded by
ComfyUI's own loader, which moves it off the card before sampling.

A projection file holds W [d_in, d_out] (optional when it carries a residual network), the
mean and standard deviation of each side, the layer to read (tap), optionally the residual
network as mlp.N.weight / mlp.N.bias, and the attention-sink vector sink_out:

    cond = ((h - mean_in) / std_in) @ W * std_out + mean_out   (+ the residual, in that space)
"""

import json
import logging
import os
import struct

import torch

FOLDER = "clip_projections"

PAD_TOKEN = 151643
VISION_START = 151652
VISION_END = 151653

# The width of the vision merger's output names the Qwen3-VL size, and the ComfyUI encoder
# type that builds it. Read from the file header, so quantized files are recognised too.
ARCH_BY_DIM = {2560: ("krea2", "4B"), 4096: ("boogu", "8B"), 5120: ("minimax", "32B")}
MERGER_KEYS = ("model.visual.merger.linear_fc2.weight", "visual.merger.linear_fc2.weight")

META_TYPES = {"tap": int, "d_in": int, "d_out": int, "n_train_prompts": int, "n_train_tokens": int,
              "lambda": float, "cos_test": float, "r2_test": float}

_CACHE = {}


def register_folder():
    """Declare models/clip_projections to ComfyUI, so projections are listed by name."""
    import folder_paths

    path = os.path.join(folder_paths.models_dir, FOLDER)
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        pass
    known = folder_paths.folder_names_and_paths.get(FOLDER)
    if known is None:
        folder_paths.folder_names_and_paths[FOLDER] = ([path], {".safetensors"})
    elif path not in known[0]:
        known[0].append(path)


def list_projections():
    import folder_paths

    try:
        return sorted((f for f in folder_paths.get_filename_list(FOLDER) if f.lower().endswith(".safetensors")), key=str.lower)
    except Exception:
        return []


def read_header(path):
    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(size))


def encoder_arch(path):
    """(ComfyUI encoder type, size label) of a Qwen3-VL text encoder file, or None."""
    try:
        header = read_header(path)
    except Exception:
        return None
    for key in MERGER_KEYS:
        entry = header.get(key)
        if entry and entry.get("shape"):
            return ARCH_BY_DIM.get(entry["shape"][0])
    return None


def projection_width(path):
    """The encoder width a projection file expects, from its header, or None."""
    try:
        entry = read_header(path).get("mean_in")
        return entry["shape"][0] if entry else None
    except Exception:
        return None


def load_projection(path):
    """A projection's tensors and scalars, cached by path and modification time."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    key = (path, os.path.getmtime(path))
    if key in _CACHE:
        return _CACHE[key]
    data = load_file(path)
    with safe_open(path, framework="pt") as f:
        meta = f.metadata() or {}
    for k, v in meta.items():
        cast = META_TYPES.get(k)
        try:
            data[k] = cast(v) if cast else v
        except (TypeError, ValueError):
            data[k] = v
    for k in ("mean_in", "std_in", "mean_out", "std_out", "tap"):
        if k not in data:
            raise KeyError("%s has no '%s': it is not a ClipProj projection, or the download is incomplete" % (os.path.basename(path), k))
    for k in ("W", "mean_in", "std_in", "mean_out", "std_out"):
        if k in data:
            data[k] = data[k].float()
    if "W" not in data and not any(k.startswith("mlp.") for k in data):
        raise ValueError("%s has neither a matrix nor a residual network: download it again" % os.path.basename(path))
    _CACHE.clear()  # one projection at a time
    _CACHE[key] = data
    return data


def build_residual(proj, device):
    """The residual network stored as mlp.N.weight / mlp.N.bias, in the file's dtype, or None."""
    layers = sorted({int(k.split(".")[1]) for k in proj if k.startswith("mlp.")})
    if not layers:
        return None
    dtype = proj["mlp.%d.weight" % layers[0]].dtype
    modules = []
    for n, i in enumerate(layers):
        w = proj["mlp.%d.weight" % i]
        lin = torch.nn.Linear(w.shape[1], w.shape[0], bias=("mlp.%d.bias" % i) in proj)
        lin.weight.data = w.to(device=device, dtype=dtype)
        if lin.bias is not None:
            lin.bias.data = proj["mlp.%d.bias" % i].to(device=device, dtype=dtype)
        modules.append(lin)
        if n < len(layers) - 1:
            modules.append(torch.nn.GELU())
    net = torch.nn.Sequential(*modules).to(device=device, dtype=dtype)
    net.train(False)
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def project(h, proj, cache):
    """Map a small encoder's hidden states [B, seq, d_in] into H3's conditioning [B, seq, d_out]."""
    dev = h.device
    if cache.get("device") != dev:
        cache.clear()
        cache["device"] = dev
    if "p" not in cache:
        width = proj["mean_in"].shape[0]
        if h.shape[-1] != width:
            sizes = {2560: "4B", 4096: "8B", 5120: "32B"}
            raise ValueError("This projection is for a %s text encoder (%d wide) and the encoder is a %s (%d wide). "
                             "Pick the mmh3-%s projection, or a %s encoder."
                             % (sizes.get(width, "?"), width, sizes.get(h.shape[-1], "?"), h.shape[-1],
                                sizes.get(h.shape[-1], "?").lower(), sizes.get(width, "?")))
        cache["p"] = {k: proj[k].to(dev) for k in ("W", "mean_in", "std_in", "mean_out", "std_out") if k in proj}
        cache["mlp"] = build_residual(proj, dev)
    p = cache["p"]
    # The ridge and the residual both work in the standardised space they were fitted in.
    xn = (h.float() - p["mean_in"]) / p["std_in"]
    yn = xn @ p["W"] if "W" in p else None
    net = cache.get("mlp")
    if net is not None:
        out = net(xn.to(net[0].weight.dtype)).float()
        yn = out if yn is None else yn + out
    cond = yn * p["std_out"] + p["mean_out"]
    # Token 0 is an attention sink: constant in direction, huge in norm, and never seen in
    # calibration. The file carries its measured value, which is exact since it is constant.
    sink = proj.get("sink_out")
    if sink is not None and cond.shape[1] > 0:
        cond[:, 0] = sink.to(device=cond.device, dtype=cond.dtype)
    return cond


def submodel(clip):
    """The inner SDClipModel, past encoder wrappers that strip templates or stack layers."""
    csm = clip.cond_stage_model
    name = getattr(csm, "clip", None)
    if name is not None and hasattr(csm, name):
        return getattr(csm, name)
    for attr in dir(csm):
        if attr.startswith("_"):
            continue
        sub = getattr(csm, attr, None)
        if hasattr(sub, "encode_token_weights") and hasattr(sub, "transformer"):
            return sub
    raise RuntimeError("no text model found in this encoder")


def raw_tokenizer(clip):
    tk = clip.tokenizer
    name = getattr(tk, "clip_name", None)
    if name is not None and hasattr(getattr(tk, name, None), "tokenizer"):
        return getattr(tk, name).tokenizer
    for attr in dir(tk):
        if attr.startswith("_"):
            continue
        sub = getattr(tk, attr, None)
        if hasattr(sub, "tokenizer"):
            return sub.tokenizer
    raise RuntimeError("no tokenizer found in this encoder")


def token_tags(seq_len, embeds_info):
    """H3's tags: 0 on vision blocks, with their start and end tokens, and 1 on text."""
    tags = torch.ones(seq_len, dtype=torch.long)
    for e in embeds_info:
        if e.get("type") == "image":
            tags[max(0, e["index"] - 1):e["index"] + e["size"] + 1] = 0
    return tags


def install_video_blocks(sm):
    """Let a 4B or 8B read H3's two-frame video blocks, which ComfyUI implements on the 32B only."""
    tr = getattr(sm, "transformer", None)
    if tr is None or getattr(tr, "_h3_video_blocks", False):
        return
    try:
        from comfy.text_encoders.minimax import process_video_block
    except Exception as e:
        logging.warning("H3 Editor: video references unavailable on a small text encoder: %s", e)
        return
    original = tr.preprocess_embed

    def preprocess_embed(embed, device):
        if embed.get("type") == "image" and embed.get("minimax_video_block", False):
            flat, grid = process_video_block(embed["data"])
            merged, deepstack = tr.visual(flat.to(device, dtype=torch.float32), grid)
            return merged, {"grid": grid, "deepstack": deepstack}
        return original(embed, device)

    tr.preprocess_embed = preprocess_embed
    tr._h3_video_blocks = True


def install_int8_vision_fix(sm):
    """Keep an int8 encoder working with reference pictures.

    ComfyUI's int8 embedding path can be handed a weight its cast already dequantized, and then
    fails in the vision tower's position embedding. Retrying with the format cleared takes the
    plain lookup, which is right for a dequantized weight.
    """
    tr = getattr(sm, "transformer", None)
    pe = getattr(getattr(tr, "visual", None), "pos_embed", None)
    if pe is None or getattr(pe, "_h3_int8_fix", False) or getattr(pe, "quant_format", None) != "int8_tensorwise":
        return
    original = pe.forward_comfy_cast_weights
    state = {"plain": False}

    def forward_comfy_cast_weights(input, out_dtype=None):
        if not state["plain"]:
            try:
                return original(input, out_dtype=out_dtype)
            except Exception as e:
                state["plain"] = True
                logging.info("H3 Editor: int8 position embedding fell back to a plain lookup (%s)", e)
        fmt = pe.quant_format
        pe.quant_format = None
        try:
            return original(input, out_dtype=out_dtype)
        finally:
            pe.quant_format = fmt

    pe.forward_comfy_cast_weights = forward_comfy_cast_weights
    pe._h3_int8_fix = True


class ProjectedCLIP:
    """A small encoder that tokenizes and encodes the way ComfyUI's MiniMax H3 nodes expect of the 32B."""

    def __init__(self, base, projection_path):
        self.__dict__["_base"] = base
        self.__dict__["_path"] = projection_path
        self.__dict__["_proj"] = load_projection(projection_path)
        self.__dict__["_key"] = getattr(base.cond_stage_model, "clip", "qwen3vl_4b")
        self.__dict__["_cache"] = {}

    def __getattr__(self, name):
        return getattr(self.__dict__["_base"], name)

    def __setattr__(self, name, value):
        if name in self.__dict__:
            self.__dict__[name] = value
        else:
            setattr(self.__dict__["_base"], name, value)

    def clone(self):
        return ProjectedCLIP(self._base.clone(), self._path)

    def tokenize(self, text, return_word_ids=False, images=[], minimax_ref_items=None, **kwargs):
        """H3's tokenization: raw text, with "<Picture i>: " and a vision block per reference."""
        tok = raw_tokenizer(self._base)
        entries = []

        def add_text(s):
            entries.extend((t, 1.0) for t in tok(s, add_special_tokens=False)["input_ids"])

        def add_vision(data, video_block=False):
            entries.append((VISION_START, 1.0))
            embed = {"type": "image", "data": data, "original_type": "image"}
            if video_block:
                embed["minimax_video_block"] = True
            entries.append((embed, 1.0))
            entries.append((VISION_END, 1.0))

        if minimax_ref_items:
            counters = {"image": 0, "audio": 0, "video": 0}
            for item in minimax_ref_items:
                kind = item["type"]
                counters[kind] = counters.get(kind, 0) + 1
                if kind == "image":
                    add_text("<Picture %d>: " % counters["image"])
                    add_vision(item["data"])
                elif kind == "audio":
                    add_text("<Audio %d>: " % counters["audio"])  # sound never enters the text encoder
                else:
                    # A video goes in as frame pairs at 2 fps, each after the time of its middle.
                    frames = item["data"]
                    stamps = item.get("timestamps")
                    stamps = list(stamps) if stamps is not None else [i / 2.0 for i in range(frames.shape[0])]
                    if frames.shape[0] % 2 == 1:
                        frames = torch.cat([frames, frames[-1:]], dim=0)
                        stamps.append(stamps[-1])
                    add_text("<Video %d>: " % counters["video"])
                    for k in range(0, frames.shape[0], 2):
                        add_text("<%.1f seconds>" % ((stamps[k] + stamps[k + 1]) / 2.0))
                        add_vision(frames[k:k + 2], video_block=True)
        else:
            for i, img in enumerate(images):
                add_text("<Picture %d>: " % (i + 1))
                add_vision(img)
        add_text(text)
        if not entries:
            entries.append((PAD_TOKEN, 1.0))
        if return_word_ids:
            entries = [t + (0,) for t in entries]
        return {self._key: [entries]}

    def _encode(self, tokens):
        import comfy.model_management as mm

        base = self._base
        sm = submodel(base)
        install_video_blocks(sm)
        install_int8_vision_fix(sm)
        mm.load_models_gpu([base.patcher])
        device = base.patcher.load_device
        pairs = tokens[self._key] if isinstance(tokens, dict) else tokens
        pairs = [[(t[0], t[1]) for t in seq] for seq in pairs]
        tap = int(self._proj["tap"])

        # Where the vision blocks landed is known only inside the forward call: catch it there.
        captured = {}
        forward = sm.transformer.forward

        def catching(*args, **kwargs):
            captured["embeds_info"] = kwargs.get("embeds_info", [])
            return forward(*args, **kwargs)

        saved = (sm.layer, sm.layer_idx, sm.layer_norm_hidden_state, sm.execution_device)
        try:
            sm.transformer.forward = catching
            sm.layer = [tap] if tap >= 0 else "last"
            sm.layer_idx = None
            sm.layer_norm_hidden_state = False
            sm.execution_device = device
            with mm.cuda_device_context(device):
                with torch.no_grad():
                    out = sm.encode_token_weights(pairs)
        finally:
            sm.transformer.forward = forward
            (sm.layer, sm.layer_idx, sm.layer_norm_hidden_state, sm.execution_device) = saved

        h = out[0]
        if h.dim() == 4:  # [B, taps, seq, d]: one tap was asked for
            h = h[:, 0]
        with torch.no_grad():
            cond = project(h, self._proj, self.__dict__["_cache"])
        cond = cond.to(mm.intermediate_device())
        return cond, token_tags(cond.shape[1], captured.get("embeds_info", []))

    def encode_from_tokens(self, tokens, return_pooled=False, return_dict=False):
        cond, tags = self._encode(tokens)
        if return_dict:
            out = {"cond": cond, "pooled_output": None, "minimax_token_tags": tags}
            self._base.add_hooks_to_dict(out)
            return out
        return (cond, None) if return_pooled else cond

    def encode_from_tokens_scheduled(self, tokens, unprojected=False, add_dict={}, show_pbar=True):
        cond, tags = self._encode(tokens)
        extra = {"pooled_output": None, "minimax_token_tags": tags}
        extra.update(add_dict)
        self._base.add_hooks_to_dict(extra)
        return [[cond, extra]]

    def encode(self, text):
        return self.encode_from_tokens(self.tokenize(text))


def load(clip_name, projection_name):
    """A small text encoder from models/text_encoders, wrapped with its projection."""
    import comfy.sd
    import folder_paths

    path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
    proj_path = folder_paths.get_full_path_or_raise(FOLDER, projection_name)
    arch = encoder_arch(path)
    if arch is None or arch[0] == "minimax":
        raise ValueError("%s is not a Qwen3-VL 4B or 8B text encoder. A projection needs one: a text-only Qwen3 of "
                         "the same size loads but ignores the prompt." % clip_name)
    clip = comfy.sd.load_clip(ckpt_paths=[path], embedding_directory=folder_paths.get_folder_paths("embeddings"),
                              clip_type=getattr(comfy.sd.CLIPType, arch[0].upper()))
    wrapped = ProjectedCLIP(clip, proj_path)
    p = wrapped._proj
    logging.info("H3 Editor: %s (%s) through %s, tap %d, %d -> %d%s", clip_name, arch[1], projection_name, int(p["tap"]),
                 p["mean_in"].shape[0], p["mean_out"].shape[0], "" if "W" in p else ", residual only")
    return wrapped
