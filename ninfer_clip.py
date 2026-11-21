"""Remote NInfer text/vision encoder for ComfyUI.

Holds zero encoder weights in ComfyUI VRAM: forwards prompt text and optional reference
images to a resident NInfer server's POST /v1/hidden_states endpoint, unpacks the BF16
hidden states, and optionally passes them through a ClipProj projection.
"""

import base64
import io
import json
import logging
import struct
import time
import urllib.error
import urllib.request

import numpy as np
import torch
from PIL import Image

try:
    from . import clipproj
except ImportError:
    import clipproj

DEFAULT_URL = "http://host.docker.internal:8010"


def encode_image_data_url(image, quality=92):
    """Encode a ComfyUI image tensor [H, W, C] or [1, H, W, C] in [0, 1] as a JPEG data URL."""
    t = image[0] if image.dim() == 4 else image
    arr = (t.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).round().astype(np.uint8)
    if arr.ndim == 2:
        pil = Image.fromarray(arr, mode="L").convert("RGB")
    elif arr.shape[-1] == 4:
        pil = Image.fromarray(arr, mode="RGBA").convert("RGB")
    else:
        pil = Image.fromarray(arr[:, :, :3], mode="RGB")
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return "data:image/jpeg;base64," + encoded


def build_parts(text, images=(), minimax_ref_items=None, picture_label="<Picture %d>: "):
    """Build OpenAI-style multimodal content parts matching H3's prompt layout."""
    parts = []

    def add_text(s):
        if not s:
            return
        if parts and parts[-1]["type"] == "text":
            parts[-1]["text"] += s
        else:
            parts.append({"type": "text", "text": s})

    def add_image(tensor):
        parts.append({"type": "image_url", "image_url": {"url": encode_image_data_url(tensor)}})

    if minimax_ref_items:
        counters = {"image": 0, "audio": 0, "video": 0}
        for item in minimax_ref_items:
            kind = item["type"]
            counters[kind] = counters.get(kind, 0) + 1
            if kind == "image":
                add_text(picture_label % counters["image"])
                add_image(item["data"])
            elif kind == "audio":
                add_text("<Audio %d>: " % counters["audio"])
            else:
                frames = item["data"]
                stamps = item.get("timestamps")
                stamps = list(stamps) if stamps is not None else [i / 2.0 for i in range(frames.shape[0])]
                add_text("<Video %d>: " % counters["video"])
                for k in range(frames.shape[0]):
                    add_text("<%.1f seconds>" % stamps[k])
                    add_image(frames[k])
    else:
        for i, img in enumerate(images or ()):
            add_text(picture_label % (i + 1))
            add_image(img)

    add_text(text or "")
    if not parts:
        parts.append({"type": "text", "text": " "})
    return parts


def decode_hidden_response(raw):
    """Unpack NInfer's [u64 hdr_len][JSON header][LE BF16 payload] response."""
    if len(raw) < 8:
        raise ValueError("NInfer hidden_states response truncated before header length (%d bytes)" % len(raw))
    header_len = struct.unpack("<Q", raw[:8])[0]
    if len(raw) < 8 + header_len:
        raise ValueError("NInfer hidden_states response truncated inside header (%d < %d)" % (len(raw), 8 + header_len))
    meta = json.loads(raw[8:8 + header_len].decode("utf-8"))
    seq_len = int(meta["seq_len"])
    hidden_size = int(meta["hidden_size"])
    expected_bytes = seq_len * hidden_size * 2
    payload = raw[8 + header_len:]
    if len(payload) != expected_bytes:
        raise ValueError("NInfer hidden_states payload size mismatch: got %d bytes, expected %d (%dx%d bf16)"
                         % (len(payload), expected_bytes, seq_len, hidden_size))
    # Copy into writable bytearray so torch.frombuffer avoids non-writable warnings.
    tensor = torch.frombuffer(bytearray(payload), dtype=torch.bfloat16).view(1, seq_len, hidden_size).float()
    return meta, tensor


def find_user_span(token_texts, seq_len):
    """Locate (start, end) of the user turn inside Qwen's chat-templated prompt.

    Returns (start, end) where tokens[start:end] are the user content tokens (excluding
    `<|im_start|>user\\n` and `<|im_end|>...`). Falls back to (0, seq_len) if not templated.
    """
    if not token_texts or len(token_texts) != seq_len:
        return 0, seq_len
    start = 0
    if token_texts[0] == "<|im_start|>":
        acc = ""
        for idx in range(1, min(seq_len, 8)):
            acc += token_texts[idx]
            if acc.endswith("user\n"):
                start = idx + 1
                break
    end = seq_len
    for idx in range(start, seq_len):
        if token_texts[idx] == "<|im_end|>":
            end = idx
            break
    if end <= start:
        return 0, seq_len
    return start, end


def token_tags_from_types(seq_len, token_types, token_texts=None):
    """Convert NInfer token_types (0=text, non-zero=vision) to H3 tags (0=vision, 1=text).

    MiniMax H3 tags the entire vision block — including flanking <|vision_start|> and
    <|vision_end|> tokens — as 0 (video modality) and all other text tokens as 1.
    """
    if not token_types or len(token_types) != seq_len:
        return torch.ones(seq_len, dtype=torch.long)
    tags = torch.tensor([0 if int(t) != 0 else 1 for t in token_types], dtype=torch.long)
    # Widen each vision span by 1 on each side when flanked by <|vision_start|> / <|vision_end|>,
    # matching comfy/text_encoders/minimax.py:token_tags_from_embeds_info.
    if token_texts and len(token_texts) == seq_len:
        for i, tok in enumerate(token_texts):
            if tok in ("<|vision_start|>", "<|vision_end|>"):
                tags[i] = 0
    else:
        vis = (tags == 0).nonzero(as_tuple=False).flatten().tolist()
        for i in vis:
            if i > 0:
                tags[i - 1] = 0
            if i + 1 < seq_len:
                tags[i + 1] = 0
    return tags


def project_with_modality(h, proj, tags, cache):
    """Apply ClipProj projection, optionally routing vision tokens (tags == 0) through a dedicated branch."""
    if "W_vis" not in proj and not any(k.startswith("mlp_vis.") for k in proj):
        return clipproj.project(h, proj, cache)

    cond = clipproj.project(h, proj, cache)
    vis_mask = tags.to(device=h.device) == 0
    if not vis_mask.any():
        return cond

    vis_cache = cache.setdefault("_vis_cache", {})
    if "_vis_proj" not in cache:
        vp = {
            "tap": proj["tap"],
            "mean_in": proj.get("mean_in_vis", proj["mean_in"]),
            "std_in": proj.get("std_in_vis", proj["std_in"]),
            "mean_out": proj.get("mean_out_vis", proj["mean_out"]),
            "std_out": proj.get("std_out_vis", proj["std_out"]),
        }
        if "W_vis" in proj:
            vp["W"] = proj["W_vis"].float()
        for k, v in proj.items():
            if k.startswith("mlp_vis."):
                vp["mlp." + k[len("mlp_vis."):]] = v
        # Ensure float32 stats just like clipproj.load_projection
        for k in ("mean_in", "std_in", "mean_out", "std_out"):
            vp[k] = vp[k].float()
        cache["_vis_proj"] = vp

    vis_h = h[:, vis_mask, :]
    vis_cond = clipproj.project(vis_h, cache["_vis_proj"], vis_cache)
    cond = cond.clone()
    cond[:, vis_mask, :] = vis_cond.to(dtype=cond.dtype)
    # Re-pin token 0 attention sink if present
    sink = proj.get("sink_out")
    if sink is not None and cond.shape[1] > 0:
        cond[:, 0] = sink.to(device=cond.device, dtype=cond.dtype)
    return cond


def fetch_hidden_states(url, parts, model="", layer=-1, api_key="", timeout=120):
    """POST /v1/hidden_states to NInfer and return (meta, hidden_fp32[1, seq, H])."""
    endpoint = url.rstrip("/") + "/v1/hidden_states"
    payload = {
        "messages": [{"role": "user", "content": parts}],
        "enable_thinking": False,
    }
    if model:
        payload["model"] = model
    if layer is not None and int(layer) >= 0:
        payload["layer"] = int(layer)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(endpoint, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError("NInfer /v1/hidden_states HTTP %d from %s: %s" % (err.code, endpoint, detail)) from err
    except urllib.error.URLError as err:
        raise RuntimeError("Cannot reach NInfer at %s: %s" % (endpoint, err.reason)) from err
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    meta, hidden = decode_hidden_response(raw)
    meta["http_ms"] = round(elapsed_ms, 2)
    return meta, hidden


class NInferRemoteCLIP:
    """Duck-typed ComfyUI CLIP backed by a remote NInfer server (0 bytes encoder VRAM in ComfyUI)."""

    def __init__(self, url=DEFAULT_URL, model="", layer=-1, projection_path=None, api_key="", timeout=120):
        self.url = url or DEFAULT_URL
        self.model = model or ""
        self.layer = int(layer) if layer is not None else -1
        self.projection_path = projection_path
        self.api_key = api_key or ""
        self.timeout = timeout
        self._proj = clipproj.load_projection(projection_path) if projection_path else None
        self._proj_cache = {}
        self.cond_stage_model = None
        self.patcher = None

    def clone(self):
        return NInferRemoteCLIP(
            url=self.url,
            model=self.model,
            layer=self.layer,
            projection_path=self.projection_path,
            api_key=self.api_key,
            timeout=self.timeout,
        )

    def add_hooks_to_dict(self, pooled_dict):
        return pooled_dict

    def set_clip_options(self, options):
        pass

    def clip_layer(self, layer_idx):
        pass

    def tokenize(self, text, return_word_ids=False, images=(), minimax_ref_items=None,
                 picture_label="<Picture %d>: ", **kwargs):
        parts = build_parts(text, images=images, minimax_ref_items=minimax_ref_items, picture_label=picture_label)
        return {"ninfer": {"parts": parts}}

    def _encode(self, tokens):
        try:
            import comfy.model_management as mm
            out_device = mm.intermediate_device()
        except ImportError:
            out_device = torch.device("cpu")

        if isinstance(tokens, dict) and "ninfer" in tokens:
            parts = tokens["ninfer"]["parts"]
        elif isinstance(tokens, str):
            parts = build_parts(tokens)
        else:
            raise ValueError("NInferRemoteCLIP received unsupported token payload: %r" % (type(tokens),))

        tap = self.layer
        if tap < 0 and self._proj is not None and "tap" in self._proj:
            tap = int(self._proj["tap"])

        meta, hidden = fetch_hidden_states(
            self.url, parts, model=self.model, layer=tap, api_key=self.api_key, timeout=self.timeout
        )
        raw_seq = hidden.shape[1]
        token_types = meta.get("token_types") or [0] * raw_seq
        token_texts = meta.get("token_texts")
        start, end = find_user_span(token_texts, raw_seq)
        # When a projection supplies an explicit attention-sink vector (`sink_out`), keep one
        # leading slot (`start - 1`) for `clipproj.project()` to overwrite with `sink_out`, so
        # every single user prompt token survives intact at indices 1..N.
        has_sink = self._proj is not None and self._proj.get("sink_out") is not None
        if has_sink and start > 0:
            slice_start = start - 1
        else:
            slice_start = start
        hidden = hidden[:, slice_start:end].contiguous()
        token_types = token_types[slice_start:end]
        sliced_texts = token_texts[slice_start:end] if token_texts and len(token_texts) == raw_seq else None
        tags = token_tags_from_types(hidden.shape[1], token_types, sliced_texts)

        if self._proj is not None:
            with torch.no_grad():
                cond = project_with_modality(hidden, self._proj, tags, self._proj_cache)
        else:
            cond = hidden

        cond = cond.to(out_device)
        logging.info(
            "H3 NInfer CLIP: model=%s seq=%d/%d dim=%d layer=%s prefill=%.1fms vision=%.1fms http=%.1fms%s",
            meta.get("model", "?"),
            cond.shape[1],
            raw_seq,
            cond.shape[2],
            meta.get("layer", "final"),
            float(meta.get("prefill_ms", 0.0)),
            float(meta.get("vision_ms", 0.0)),
            float(meta.get("http_ms", 0.0)),
            " (projected)" if self._proj is not None else " (raw)",
        )
        return cond, tags

    def encode_from_tokens(self, tokens, return_pooled=False, return_dict=False):
        cond, tags = self._encode(tokens)
        if return_dict:
            return {"cond": cond, "pooled_output": None, "minimax_token_tags": tags}
        return (cond, None) if return_pooled else cond

    def encode_from_tokens_scheduled(self, tokens, unprojected=False, add_dict=None, show_pbar=True):
        cond, tags = self._encode(tokens)
        extra = {"pooled_output": None, "minimax_token_tags": tags}
        if add_dict:
            extra.update(add_dict)
        return [[cond, extra]]

    def encode(self, text):
        return self.encode_from_tokens(self.tokenize(text))


def load(url=DEFAULT_URL, model="", layer=-1, projection="none", api_key=""):
    proj_path = None
    if projection and projection != "none":
        import folder_paths
        proj_path = folder_paths.get_full_path_or_raise(clipproj.FOLDER, projection)
    return NInferRemoteCLIP(url=url, model=model, layer=layer, projection_path=proj_path, api_key=api_key)
