"""Calibrate a ClipProj projection from NInfer's resident Qwen3.8-27B to MiniMax H3's Qwen3-VL-32B.

Run inside the ComfyUI container so ComfyUI's native 32B loader is available:
    docker exec -w /opt/ComfyUI/custom_nodes/comfyui-h3-take -e PYTHONPATH=/opt/ComfyUI \
        comfyui /opt/venv/bin/python tools/calibrate_ninfer.py --auto

Steps:
1. Build a diverse corpus of cinematic video/audio/dialogue/multimodal prompts.
2. Run ComfyUI's native MiniMax H3 Qwen3-VL-32B encoder on every prompt (once), recording
   unnormalized layer-50 hidden states, 32B token spans, vision spans, and token-0 attention sink.
3. Unload the 32B encoder to free VRAM.
4. Sweep candidate Qwen3.8-27B layers via NInfer's POST /v1/hidden_states, using character-span
   alignment (exact 1:1 + subword span-mean) for text and exact 1:1 spatial alignment for vision,
   solving closed-form standardized Ridge regression on GPU.
5. Fit the full Ridge matrix W and an optional 2-layer GELU residual MLP on the best layer,
   evaluate held-out test cosine similarity and R^2, and save ClipProj-compatible `.safetensors`
   files into /opt/ComfyUI/models/clip_projections/.
"""

import argparse
import gc
import glob
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ninfer_clip  # noqa: E402


# --- Diverse prompt bank tailored to MiniMax H3 (cinematic AV + dialogue + styles + refs) ---

SUBJECTS = [
    "a red sports car", "an elderly Japanese ceramicist", "a golden retriever puppy",
    "a tired jazz saxophonist", "two teenagers on bicycles", "a sleek silver freight train",
    "a street vendor flipping dumplings", "a ballerina tying pointe shoes",
    "a weathered fisherman in yellow oilskins", "a young woman in a crimson trench coat",
    "a tabby cat perched on a windowsill", "a blacksmith hammering glowing iron",
    "a lone astronaut adjusting a helmet visor", "a toddler splashing in rain puddles",
    "a cello player on a misty pier", "a chef searing scallops in a copper pan",
    "a skateboarder carving down an empty concrete ditch", "a fox trotting across fresh snow",
    "a vintage biplane banking through clouds", "a clockmaker inspecting brass gears",
    "a street dancer spinning under sodium streetlights", "a botanist sketching orchids",
    "a motorcycle courier weaving through gridlock", "an owl gliding silently between pines",
    "a glassblower shaping molten amber glass", "a detective unfolding a damp paper map",
    "a choir singing in a stone chapel", "a surfer paddling into a glassy green swell",
    "a grandmother kneading sourdough bread", "a violinist busking on a subway platform",
    "a robotics engineer soldering a circuit board", "a flock of starlings murmuring at dusk",
    "a barista pouring rosetta latte art", "a mountain climber chalking both hands",
    "a tailor chalking dark wool suiting", "a lighthouse keeper wiping salt spray from glass",
]

ACTIONS = [
    "moving slowly through the frame while glancing back over a shoulder",
    "pausing mid-motion as wind catches loose fabric and hair",
    "leaning forward with quiet focus as tiny sparks drift upward",
    "turning toward the lens with a sudden, unguarded laugh",
    "crossing a narrow threshold from deep shadow into warm sunlight",
    "accelerating past parked storefronts while water sheets off the ground",
    "settling into stillness while breathing softly in the cold air",
    "reaching out carefully to steady a trembling glass of water",
    "arguing in hushed, urgent tones before walking away abruptly",
    "lifting a heavy wooden crate onto a rusted metal counter",
    "wiping rain from narrowed eyes and checking a wristwatch",
    "drifting effortlessly along a curving track surrounded by mist",
    "tapping a steady rhythm with fingertips against a scratched tabletop",
    "stopping abruptly at the curb as distant sirens echo off wet brick",
    "holding up a small glowing lantern to inspect peeling paint",
    "spinning in a tight arc that kicks up dust and autumn leaves",
]

SETTINGS = [
    "in a rain-slicked Tokyo alleyway packed with glowing kanji signs",
    "inside a sunlit Tuscan workshop cluttered with terracotta and dust motes",
    "on a fog-bound Scottish headland overlooking dark churning surf",
    "in a cramped retro diner with red vinyl booths and flickering neon",
    "along a cobblestone Lisbon street at blue hour as trams clatter past",
    "inside an echoing brutalist train station bathed in shafts of afternoon sun",
    "in a lush greenhouse dripping with condensation and hanging ferns",
    "on a desert highway shimmering with heat haze under a cobalt sky",
    "inside a wood-paneled recording studio lit only by warm VU meters",
    "at a bustling night market thick with charcoal smoke and lantern light",
    "on a frozen alpine lake cracking softly beneath pink alpenglow",
    "in a cluttered second-hand bookstore with towering spiral staircases",
    "on the wet deck of a ferry pulling out of a harbor at dawn",
    "inside an abandoned greenhouse reclaimed by ivy and wildflowers",
    "along a quiet suburban cul-de-sac during a golden late-summer evening",
    "in a neon-lit underground parking garage echoing with footsteps",
]

CAMERAS = [
    "35mm anamorphic tracking shot with shallow depth of field",
    "handheld close-up with natural micro-jitter and soft lens flare",
    "slow dolly push-in on a 50mm prime lens",
    "wide static establishing frame locked on a tripod",
    "low-angle steadicam orbit smoothly circling the subject",
    "overhead drone shot descending gently toward the scene",
    "intimate over-the-shoulder medium shot with soft bokeh",
    "crisp macro detail shot racking focus from foreground to background",
    "cinematic 24fps documentary coverage with naturalistic framing",
    "sweeping crane shot rising above rooftops into the evening sky",
]

LIGHTING_AND_STYLES = [
    "warm golden-hour rim lighting, natural Kodak film grain, rich shadow detail",
    "cool cyan and magenta neon reflections, wet specular highlights, moody contrast",
    "soft overcast daylight, muted pastel palette, gentle skin tones",
    "dramatic chiaroscuro candlelight, deep velvet blacks, warm amber falloff",
    "harsh fluorescent overhead flicker, desaturated teal shadows, gritty realism",
    "painterly Studio Ghibli watercolor aesthetic, hand-inked outlines, soft cel shading",
    "tactile stop-motion clay animation with visible fingerprints and miniature set lighting",
    "high-contrast black-and-white 16mm silver gelatin film with crisp halation",
    "vivid stylized 3D animation with subsurface scattering and expressive lighting",
    "crisp commercial photography look, clean key light, balanced neutral color",
]

AUDIO_CUES = [
    "Audio: steady rainfall pattering on metal awnings, distant tires hissing on wet asphalt, low rumble of thunder.",
    "Audio: muffled crowd chatter, sizzling oil on a hot griddle, clinking ceramic bowls.",
    "Audio: crisp footsteps crunching on gravel, wind rustling dry leaves, a lone bird calling far off.",
    "Audio: warm upright bass and brushed snare playing softly from another room, faint vinyl crackle.",
    "Audio: mechanical hum of a fluorescent tube, steady ticking of a wall clock, quiet breathing.",
    "Audio: seagulls crying overhead, creaking timber dock, rhythmic slap of harbor waves.",
    'Dialogue: <d>Wait—did you hear that? Look down the street.</d> Followed by hurried footsteps and heavy breathing.',
    'Dialogue: <d>We only get one shot at this before sunrise, so keep your eyes on the mirror.</d> Soft engine idle underneath.',
    'Dialogue: <d>I told you the storm would clear by noon.</d> Cheerful laugh over gentle acoustic guitar strumming.',
    'Dialogue: <d>Hold still for just two seconds, the light is perfect right there.</d> Camera shutter clicks crisply.',
    "Audio: silence broken only by the hiss of steam escaping a radiator and distant subway rumble.",
    "Audio: triumphant orchestral strings swelling underneath cheering voices and echoing applause.",
]

EXTRA_NATURAL_PROMPTS = [
    "A slow cinematic zoom into a steaming porcelain teacup on a dark oak desk as morning fog drifts past a tall window.",
    "Close-up portrait of a woman with freckles smiling softly in warm sunset light, wind lifting strands of auburn hair.",
    "A vintage steam locomotive crossing a stone viaduct in the Scottish Highlands, thick white plumes billowing into grey clouds.",
    "Time-lapse of storm clouds boiling over Manhattan skyscrapers as office windows flicker on at dusk.",
    "A chef slices fresh yellowfin tuna with a gleaming yanagiba knife on a hinoki cutting board, subtle kitchen ambiance.",
    "Two stray cats chase a tumbling paper bag across a quiet moonlit courtyard in Rome.",
    "An underwater tracking shot following a sea turtle gliding effortlessly over a sunlit coral reef with drifting bubbles.",
    "Macro shot of dewdrops trembling on a spiderweb at dawn while a ladybug crawls slowly along a silk strand.",
    "A wooden sailboat heels gracefully across choppy turquoise water, white spray arcing over the bow.",
    "A cozy cabin interior during a blizzard, orange firelight dancing across knit blankets and fogged windowpanes.",
    "A futuristic bullet train glides silently through a cherry-blossom canyon at sunset, petals swirling in its wake.",
    "Stop-motion clay figure of a tiny fox baking a blueberry pie in a miniature woodland kitchen.",
    "Anime style wide shot of a teenage girl standing at a coastal railway crossing as a commuter train rushes past.",
    "Oil painting come to life: swirling golden wheat fields bending under a turbulent indigo twilight sky.",
    "A firefighter emerges from drifting smoke carrying a rescued terrier, flashing red emergency lights cutting through mist.",
    "Overhead shot of colorful kayaks drifting down a crystal-clear emerald canyon river surrounded by sheer cliffs.",
    "A drummer locks into a tight funk groove in a dimly lit rehearsal room, cymbals shimmering in slow motion.",
    "A snowy owl turns its head smoothly 180 degrees while snow falls silently in a boreal forest.",
    "A neon-lit arcade interior in 1988, CRT screens reflecting across excited faces, synth music pulsing in the background.",
    "A street artist sprays vibrant turquoise arcs onto a brick wall while passing trains rattle an elevated track.",
]


def build_prompt_corpus(n_prompts, seed=42):
    rng = random.Random(seed)
    prompts = list(EXTRA_NATURAL_PROMPTS)
    # Add short direct prompts + rich multi-sentence H3 prompts + tag-prefixed prompts
    while len(prompts) < n_prompts:
        mode = rng.random()
        subj = rng.choice(SUBJECTS)
        act = rng.choice(ACTIONS)
        setting = rng.choice(SETTINGS)
        cam = rng.choice(CAMERAS)
        style = rng.choice(LIGHTING_AND_STYLES)
        audio = rng.choice(AUDIO_CUES)
        if mode < 0.25:
            p = f"{cam.capitalize()} of {subj} {setting}. {style.capitalize()}."
        elif mode < 0.65:
            p = f"{cam.capitalize()} of {subj} {act} {setting}. {style.capitalize()}. {audio}"
        elif mode < 0.85:
            subj2 = rng.choice(SUBJECTS)
            p = (
                f"Scene opens on {subj} {setting}, then cuts smoothly to {subj2} {act}. "
                f"Shot as a {cam}, {style}. {audio}"
            )
        else:
            p = (
                f"<Audio 1>: {cam.capitalize()} featuring {subj} {act} {setting}. "
                f"{style.capitalize()}. {audio}"
            )
        prompts.append(p)
    rng.shuffle(prompts)
    return prompts[:n_prompts]


def collect_sample_images(n_images, search_dir="/opt/ComfyUI/output", seed=42):
    """Collect diverse real images from output folder (all >= 256x256, multiples of 32) for 1:1 vision alignment."""
    rng = random.Random(seed)
    candidates = []
    if os.path.isdir(search_dir):
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            candidates.extend(glob.glob(os.path.join(search_dir, "**", ext), recursive=True))
    # Avoid sampling our own tiny debug frames repeatedly if plenty of other images exist
    filtered = [p for p in candidates if "ninfer_wire_test" not in p and "ninfer_ab" not in p]
    if len(filtered) >= n_images:
        candidates = filtered
    rng.shuffle(candidates)
    # Multiples of 32 with area >= 65536 so both Qwen3-VL-32B and NInfer Qwen3.8-27B produce identical grids!
    landscape_sizes = [(320, 256), (384, 256), (512, 320)]
    portrait_sizes = [(256, 320), (256, 384), (320, 512)]
    square_sizes = [(256, 256), (320, 320), (384, 384)]
    tensors = []
    for path in candidates:
        if len(tensors) >= n_images:
            break
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                w, h = im.size
                if w > h * 1.15:
                    tw, th = rng.choice(landscape_sizes)
                elif h > w * 1.15:
                    tw, th = rng.choice(portrait_sizes)
                else:
                    tw, th = rng.choice(square_sizes)
                im = im.resize((tw, th), Image.Resampling.BICUBIC)
                arr = np.asarray(im, dtype=np.float32) / 255.0
                tensors.append((os.path.basename(path), torch.from_numpy(arr).unsqueeze(0)))
        except Exception:
            continue
    while len(tensors) < n_images:
        idx = len(tensors)
        g = torch.Generator().manual_seed(1000 + idx)
        ys = torch.linspace(0, 1, 256).view(1, 256, 1, 1)
        xs = torch.linspace(0, 1, 256).view(1, 1, 256, 1)
        c1 = torch.rand(1, 1, 1, 3, generator=g)
        c2 = torch.rand(1, 1, 1, 3, generator=g)
        img = (c1 * xs + c2 * ys).clamp(0, 1)
        tensors.append((f"synth_{idx}.jpg", img))
    return tensors


# --- Character-span alignment between 248k (Qwen3.8-27B) and 152k (Qwen3-VL-32B) ---

def token_char_spans(token_texts):
    """Cumulative character spans [(c0, c1), ...] from decoded token strings."""
    spans = []
    pos = 0
    for s in token_texts:
        n = len(s)
        spans.append((pos, pos + n))
        pos += n
    return spans


def align_text_spans(spans_src, spans_tgt, include_span_mean=True, skip_tgt_zero=True):
    """Map source token indices to target token index lists via exact character offsets.

    Supports:
    - 1:1 exact matches (same character span)
    - 1:N forward span matches (1 source token covering N target subwords -> mean target)
    - N:1 reverse span matches (N source tokens covering 1 target token, e.g. '<', 'd', '>' -> '<d>')
    Returns list of (src_idx, [tgt_indices], is_exact_1to1).
    """
    tgt_by_start = {}
    for j, (d0, d1) in enumerate(spans_tgt):
        if skip_tgt_zero and j == 0:
            continue
        if d1 > d0:
            tgt_by_start.setdefault(d0, []).append((j, d1))

    pairs = []
    matched_src = set()
    for i, (c0, c1) in enumerate(spans_src):
        if c1 <= c0:
            continue
        cur = c0
        matched = []
        ok = True
        while cur < c1:
            opts = tgt_by_start.get(cur)
            if not opts:
                ok = False
                break
            picked = None
            for j, d1 in opts:
                if d1 <= c1 and (picked is None or d1 > picked[1]):
                    picked = (j, d1)
            if picked is None:
                ok = False
                break
            matched.append(picked[0])
            cur = picked[1]
        if ok and matched:
            is_exact = len(matched) == 1
            if is_exact or include_span_mean:
                pairs.append((i, matched, is_exact))
                matched_src.add(i)

    # N:1 reverse-span matching: e.g. Qwen3.8 splits MiniMax special tokens '<d>', '</d>',
    # '<|lyrics_start|>', or accented subwords into multiple source tokens covering 1 target token.
    if include_span_mean:
        for j, (d0, d1) in enumerate(spans_tgt):
            if (skip_tgt_zero and j == 0) or d1 <= d0:
                continue
            covered = [i for i, (c0, c1) in enumerate(spans_src) if c1 > c0 and c0 >= d0 and c1 <= d1 and i not in matched_src]
            if covered and spans_src[covered[0]][0] == d0 and spans_src[covered[-1]][1] == d1:
                for i in covered:
                    pairs.append((i, [j], False))
                    matched_src.add(i)
    return pairs


# --- Native 32B extraction ---

def extract_32b_targets(clip_name, prompts, vision_items):
    """Load ComfyUI's MiniMax H3 32B encoder once and extract targets for all prompts + vision items."""
    import comfy.model_management as mm
    import comfy.sd
    import folder_paths

    path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
    print(f"[32B] Loading {path} ...", flush=True)
    t0 = time.perf_counter()
    clip = comfy.sd.load_clip(
        ckpt_paths=[path],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=comfy.sd.CLIPType.MINIMAX,
    )
    sm = clip.cond_stage_model.qwen3vl_32b
    hf_tok = clip.tokenizer.qwen3vl_32b.tokenizer
    mm.load_models_gpu([clip.patcher])
    print(f"[32B] Loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    # Capture embeds_info during forward so we know exact image_pad spans in 32B
    captured = {}
    orig_forward = sm.transformer.forward

    def catching(*args, **kwargs):
        captured["embeds_info"] = kwargs.get("embeds_info", [])
        return orig_forward(*args, **kwargs)

    sm.transformer.forward = catching
    text_targets = []
    vision_targets = []
    sinks = []

    try:
        t1 = time.perf_counter()
        for idx, text in enumerate(prompts):
            tokens = clip.tokenize(text)
            raw_ids = [t[0] for t in tokens["qwen3vl_32b"][0] if isinstance(t[0], int)]
            tok_texts = [hf_tok.decode([tid], skip_special_tokens=False) for tid in raw_ids]
            with torch.no_grad():
                out = clip.encode_from_tokens(tokens, return_dict=True)
            h = out["cond"][0].to(dtype=torch.bfloat16, device="cpu")  # [seq, 5120]
            if h.shape[0] != len(raw_ids):
                continue
            sinks.append(h[0].float())
            text_targets.append({
                "prompt": text,
                "token_texts": tok_texts,
                "spans": token_char_spans(tok_texts),
                "h": h,
            })
            if (idx + 1) % 100 == 0 or idx + 1 == len(prompts):
                print(f"[32B] Text {idx + 1}/{len(prompts)} ({time.perf_counter() - t1:.1f}s)", flush=True)

        t2 = time.perf_counter()
        for idx, (img_name, img_tensor, prompt_text) in enumerate(vision_items):
            captured.clear()
            tokens = clip.tokenize(prompt_text, images=[img_tensor])
            with torch.no_grad():
                out = clip.encode_from_tokens(tokens, return_dict=True)
            h = out["cond"][0].to(dtype=torch.bfloat16, device="cpu")
            embeds_info = captured.get("embeds_info", [])
            img_span = None
            for e in embeds_info:
                if e.get("type") == "image":
                    img_span = (int(e["index"]), int(e["size"]))
                    break
            if img_span is not None:
                v0 = max(0, img_span[0] - 1)
                v1 = min(h.shape[0], img_span[0] + img_span[1] + 1)
                vision_targets.append({
                    "name": img_name,
                    "prompt": prompt_text,
                    "img_tensor": img_tensor,
                    "vis_index": img_span[0],
                    "vis_size": img_span[1],
                    "vis_block_len": v1 - v0,
                    "h_vis": h[v0:v1],
                })
        print(f"[32B] Vision {len(vision_targets)}/{len(vision_items)} ({time.perf_counter() - t2:.1f}s)", flush=True)
    finally:
        sm.transformer.forward = orig_forward
        del clip
        mm.unload_all_models()
        mm.soft_empty_cache()
        gc.collect()
        torch.cuda.empty_cache()

    sink_out = torch.stack(sinks, dim=0).mean(dim=0)
    print(
        f"[32B] Unloaded. sink_out norm={sink_out.norm().item():.2f} "
        f"(mean cos across prompts={F.cosine_similarity(torch.stack(sinks), sink_out.unsqueeze(0), dim=-1).mean().item():.6f})",
        flush=True,
    )
    return text_targets, vision_targets, sink_out


# --- NInfer 27B extraction & alignment ---

def collect_27b_aligned(url, layer, text_targets, vision_targets, include_span_mean=True, max_text=None):
    """Query NInfer /v1/hidden_states for `layer` and return aligned (X_txt, Y_txt, X_vis, Y_vis, stats)."""
    x_txt_list, y_txt_list = [], []
    exact_flags = []
    prompt_ids = []
    n_exact = n_span = n_unmatched = 0

    subset = text_targets[:max_text] if max_text else text_targets
    t0 = time.perf_counter()
    for p_idx, item in enumerate(subset):
        parts = ninfer_clip.build_parts(item["prompt"])
        meta, h_full = ninfer_clip.fetch_hidden_states(url, parts, layer=layer)
        raw_seq = h_full.shape[1]
        tok_texts = meta.get("token_texts") or []
        start, end = ninfer_clip.find_user_span(tok_texts, raw_seq)
        u_texts = tok_texts[start:end]
        u_h = h_full[0, start:end]  # [u_seq, 5120] fp32
        src_spans = token_char_spans(u_texts)
        pairs = align_text_spans(src_spans, item["spans"], include_span_mean=include_span_mean, skip_tgt_zero=True)
        n_unmatched += max(0, len(src_spans) - len(pairs))
        y_h = item["h"].float()
        for src_i, tgt_js, is_exact in pairs:
            x_txt_list.append(u_h[src_i].to(torch.bfloat16))
            if len(tgt_js) == 1:
                y_txt_list.append(y_h[tgt_js[0]].to(torch.bfloat16))
            else:
                y_txt_list.append(y_h[tgt_js].mean(dim=0).to(torch.bfloat16))
            if is_exact:
                n_exact += 1
            else:
                n_span += 1
            exact_flags.append(is_exact)
            prompt_ids.append(p_idx)

    x_vis_list, y_vis_list = [], []
    vis_img_ids = []
    vis_matched_imgs = 0
    vis_mismatched_imgs = 0
    for v_idx, item in enumerate(vision_targets):
        parts = ninfer_clip.build_parts(item["prompt"], images=[item["img_tensor"]])
        meta, h_full = ninfer_clip.fetch_hidden_states(url, parts, layer=layer)
        t_types = meta.get("token_types") or []
        tok_texts = meta.get("token_texts") or []
        vis_pad = [i for i, t in enumerate(t_types) if int(t) != 0]
        if len(vis_pad) == item["vis_size"] and vis_pad:
            # Include flanking <|vision_start|> and <|vision_end|> to match minimax.py tags == 0
            s0 = vis_pad[0] - 1 if vis_pad[0] > 0 and (not tok_texts or tok_texts[vis_pad[0] - 1] == "<|vision_start|>") else vis_pad[0]
            s1 = vis_pad[-1] + 2 if vis_pad[-1] + 1 < h_full.shape[1] and (not tok_texts or tok_texts[vis_pad[-1] + 1] == "<|vision_end|>") else vis_pad[-1] + 1
            if (s1 - s0) == item["vis_block_len"]:
                x_vis_list.append(h_full[0, s0:s1].to(torch.bfloat16))
                y_vis_list.append(item["h_vis"].to(torch.bfloat16))
                vis_img_ids.extend([v_idx] * (s1 - s0))
                vis_matched_imgs += 1
                continue
        vis_mismatched_imgs += 1

    X_txt = torch.stack(x_txt_list, dim=0) if x_txt_list else torch.empty(0, 5120, dtype=torch.bfloat16)
    Y_txt = torch.stack(y_txt_list, dim=0) if y_txt_list else torch.empty(0, 5120, dtype=torch.bfloat16)
    X_vis = torch.cat(x_vis_list, dim=0) if x_vis_list else torch.empty(0, 5120, dtype=torch.bfloat16)
    Y_vis = torch.cat(y_vis_list, dim=0) if y_vis_list else torch.empty(0, 5120, dtype=torch.bfloat16)
    stats = {
        "n_prompts": len(subset),
        "n_exact": n_exact,
        "n_span": n_span,
        "n_unmatched": n_unmatched,
        "vis_matched_imgs": vis_matched_imgs,
        "vis_mismatched_imgs": vis_mismatched_imgs,
        "n_vis_tokens": int(X_vis.shape[0]),
        "seconds": round(time.perf_counter() - t0, 2),
        "exact_flags": torch.tensor(exact_flags, dtype=torch.bool),
        "prompt_ids": torch.tensor(prompt_ids, dtype=torch.long),
        "vis_img_ids": torch.tensor(vis_img_ids, dtype=torch.long),
    }
    return X_txt, Y_txt, X_vis, Y_vis, stats


# --- Ridge & Residual MLP fitting on GPU ---

def evaluate_predictions(y_pred, y_true):
    """Return (mean_cosine, r2) on [N, D] float32 tensors."""
    cos = F.cosine_similarity(y_pred, y_true, dim=-1).mean().item()
    ss_res = ((y_true - y_pred) ** 2).sum().item()
    ss_tot = ((y_true - y_true.mean(dim=0, keepdim=True)) ** 2).sum().item()
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return cos, r2


def fit_ridge(X_train, Y_train, X_test, Y_test, lambdas=(100.0, 500.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0, 300000.0), device="cuda"):
    """Closed-form standardized ridge regression on GPU."""
    # Compute standardization stats on train set in float64/float32
    Xt = X_train.float()
    Yt = Y_train.float()
    mean_in = Xt.mean(dim=0)
    std_in = Xt.std(dim=0).clamp(min=1e-6)
    mean_out = Yt.mean(dim=0)
    std_out = Yt.std(dim=0).clamp(min=1e-6)

    # Accumulate Gram matrices X^T X and X^T Y in chunks on GPU (float64 for numerical rock-solidity)
    d_in = Xt.shape[1]
    d_out = Yt.shape[1]
    xtx = torch.zeros(d_in, d_in, dtype=torch.float64, device=device)
    xty = torch.zeros(d_in, d_out, dtype=torch.float64, device=device)
    m_in = mean_in.to(device)
    s_in = std_in.to(device)
    m_out = mean_out.to(device)
    s_out = std_out.to(device)

    chunk = 8192
    for i in range(0, Xt.shape[0], chunk):
        xb = (Xt[i : i + chunk].to(device) - m_in) / s_in
        yb = (Yt[i : i + chunk].to(device) - m_out) / s_out
        xtx.add_((xb.T @ xb).double())
        xty.add_((xb.T @ yb).double())

    X_te_n = ((X_test.float().to(device) - m_in) / s_in).float()
    Y_te = Y_test.float().to(device)

    best = None
    for lam in lambdas:
        A64 = xtx.clone()
        A64.diagonal().add_(float(lam))
        L, info = torch.linalg.cholesky_ex(A64.float())
        if int(info.item()) == 0:
            W = torch.cholesky_solve(xty.float(), L)
            if not torch.isfinite(W).all():
                W = torch.linalg.solve(A64, xty).float()
        else:
            W = torch.linalg.solve(A64, xty).float()
        del A64
        pred = (X_te_n @ W) * s_out + m_out
        cos, r2 = evaluate_predictions(pred, Y_te)
        if best is None or cos > best["cos"]:
            best = {
                "W": W.cpu(),
                "lambda": float(lam),
                "cos": cos,
                "r2": r2,
                "mean_in": mean_in,
                "std_in": std_in,
                "mean_out": mean_out,
                "std_out": std_out,
            }
    return best


def train_residual_mlp(X_train, Y_train, X_test, Y_test, ridge, hidden_dim=8192, epochs=20, batch_size=2048, lr=1e-3, device="cuda"):
    """Train a 2-layer GELU residual MLP predicting (y_n - x_n @ W) in standardized space."""
    m_in = ridge["mean_in"].to(device)
    s_in = ridge["std_in"].to(device)
    m_out = ridge["mean_out"].to(device)
    s_out = ridge["std_out"].to(device)
    W = ridge["W"].to(device)

    d_in, d_out = W.shape
    net = torch.nn.Sequential(
        torch.nn.Linear(d_in, hidden_dim),
        torch.nn.GELU(),
        torch.nn.Linear(hidden_dim, d_out),
    ).to(device)
    # Zero-init final projection so the residual starts cleanly at the Ridge solution!
    torch.nn.init.zeros_(net[2].weight)
    torch.nn.init.zeros_(net[2].bias)

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # Pre-standardize on GPU (60k x 5120 bf16 is only ~600 MB!)
    X_tr_n = ((X_train.to(device=device, dtype=torch.float32) - m_in) / s_in).to(torch.bfloat16)
    Y_tr_n = ((Y_train.to(device=device, dtype=torch.float32) - m_out) / s_out).to(torch.bfloat16)
    X_te_n = ((X_test.to(device=device, dtype=torch.float32) - m_in) / s_in)
    Y_te = Y_test.to(device=device, dtype=torch.float32)

    n = X_tr_n.shape[0]
    best_cos = ridge["cos"]
    best_r2 = ridge["r2"]
    best_state = None
    t0 = time.perf_counter()

    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb = X_tr_n[idx].float()
            yb = Y_tr_n[idx].float()
            with torch.no_grad():
                base = xb @ W
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred_n = base + net(xb)
                # Combined standardized MSE + cosine loss (matches both scale and direction)
                mse = F.mse_loss(pred_n.float(), yb)
                cos_loss = 1.0 - F.cosine_similarity(pred_n.float() * s_out + m_out, yb * s_out + m_out, dim=-1).mean()
                loss = mse + 0.5 * cos_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        sched.step()

        net.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, X_te_n.shape[0], 4096):
                xb = X_te_n[i : i + 4096]
                yn = (xb @ W) + net(xb)
                preds.append(yn * s_out + m_out)
            y_pred = torch.cat(preds, dim=0)
            cos, r2 = evaluate_predictions(y_pred, Y_te)
        if cos > best_cos:
            best_cos = cos
            best_r2 = r2
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        if (ep + 1) % 5 == 0 or ep + 1 == epochs:
            print(f"  [MLP] epoch {ep + 1:2d}/{epochs} | test cos={cos:.4f} r2={r2:.4f} (best={best_cos:.4f}, {time.perf_counter() - t0:.1f}s)", flush=True)

    return best_state, best_cos, best_r2


def save_clipproj(path, ridge, sink_out, tap, meta_extra, mlp_state=None, vis_ridge=None, vis_mlp_state=None):
    tensors = {
        "W": ridge["W"].half().contiguous(),
        "mean_in": ridge["mean_in"].half().contiguous(),
        "std_in": ridge["std_in"].half().contiguous(),
        "mean_out": ridge["mean_out"].half().contiguous(),
        "std_out": ridge["std_out"].half().contiguous(),
        "sink_out": sink_out.half().contiguous(),
    }
    if mlp_state is not None:
        for k, v in mlp_state.items():
            tensors["mlp." + k] = v.half().contiguous()
    if vis_ridge is not None:
        tensors["W_vis"] = vis_ridge["W"].half().contiguous()
        tensors["mean_in_vis"] = vis_ridge["mean_in"].half().contiguous()
        tensors["std_in_vis"] = vis_ridge["std_in"].half().contiguous()
        tensors["mean_out_vis"] = vis_ridge["mean_out"].half().contiguous()
        tensors["std_out_vis"] = vis_ridge["std_out"].half().contiguous()
    if vis_mlp_state is not None:
        for k, v in vis_mlp_state.items():
            tensors["mlp_vis." + k] = v.half().contiguous()
    meta = {
        "tap": str(int(tap)),
        "d_in": str(int(ridge["W"].shape[0])),
        "d_out": str(int(ridge["W"].shape[1])),
        "lambda": str(ridge["lambda"]),
        "source_model": "ninfer:qwen3.8-27b",
        "target_model": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    }
    meta.update({k: str(v) for k, v in meta_extra.items()})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file(tensors, path, metadata=meta)
    print(f"[SAVE] {path} ({os.path.getsize(path) / (1024 * 1024):.1f} MB)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ninfer-url", default=ninfer_clip.DEFAULT_URL)
    parser.add_argument("--clip-32b", default="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")
    parser.add_argument("--prompts-json", default="", help="optional JSON list of prompts or {prompt:...} dicts")
    parser.add_argument("--n-prompts", type=int, default=800)
    parser.add_argument("--sweep-prompts", type=int, default=100)
    parser.add_argument("--sweep-vision", type=int, default=20)
    parser.add_argument("--n-vision", type=int, default=80)
    parser.add_argument("--sweep-layers", default="23,31,39,43,47,51,55,59,63,-1")
    parser.add_argument("--train-layer", type=int, default=None)
    parser.add_argument("--mlp-hidden", type=int, default=8192)
    parser.add_argument("--mlp-epochs", type=int, default=20)
    parser.add_argument("--out-dir", default="/opt/ComfyUI/models/clip_projections")
    parser.add_argument("--out-prefix", default="mmh3-qwen38-27b-v2")
    args = parser.parse_args()

    if args.prompts_json and os.path.isfile(args.prompts_json):
        with open(args.prompts_json, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        prompts = [x["prompt"] if isinstance(x, dict) else str(x) for x in loaded]
        random.Random(42).shuffle(prompts)
        prompts = prompts[: args.n_prompts]
        print(f"[CORPUS] Loaded {len(prompts)} diverse prompts (shuffled) from {args.prompts_json}", flush=True)
    else:
        prompts = build_prompt_corpus(args.n_prompts)
        print(f"[CORPUS] Generated {len(prompts)} synthetic prompts", flush=True)

    raw_imgs = collect_sample_images(args.n_vision)
    vision_items = [(name, img, prompts[i % len(prompts)]) for i, (name, img) in enumerate(raw_imgs)]

    # 1. Extract 32B targets once, then unload 32B
    text_targets, vision_targets, sink_out = extract_32b_targets(args.clip_32b, prompts, vision_items)

    # 2. Layer sweep across BOTH text and vision
    best_layer = args.train_layer
    if best_layer is None:
        layers = [int(x.strip()) for x in args.sweep_layers.split(",") if x.strip()]
        vis_sub = vision_targets[: args.sweep_vision]
        print(f"\n=== Layer Tap Sweep ({args.sweep_prompts} text + {len(vis_sub)} vision, layers={layers}) ===", flush=True)
        sweep_results = []
        n_train_p = int(args.sweep_prompts * 0.8)
        n_train_v = max(1, int(len(vis_sub) * 0.8))
        for layer in layers:
            X_txt, Y_txt, X_vis, Y_vis, st = collect_27b_aligned(
                args.ninfer_url, layer, text_targets, vis_sub, max_text=args.sweep_prompts
            )
            tr_t = st["prompt_ids"] < n_train_p
            res_t = fit_ridge(X_txt[tr_t], Y_txt[tr_t], X_txt[~tr_t], Y_txt[~tr_t])
            v_str = "n/a"
            v_cos = 0.0
            if X_vis.shape[0] > 0:
                tr_v = st["vis_img_ids"] < n_train_v
                if tr_v.any() and (~tr_v).any():
                    res_v = fit_ridge(X_vis[tr_v], Y_vis[tr_v], X_vis[~tr_v], Y_vis[~tr_v])
                    v_cos = res_v["cos"]
                    v_str = f"cos={res_v['cos']:.4f} r2={res_v['r2']:.4f}"
            sweep_results.append((layer, res_t["cos"], res_t["r2"], v_cos, res_t["lambda"]))
            label = "final" if layer < 0 else f"L{layer:02d}"
            print(
                f"  {label:5s} | text cos={res_t['cos']:.4f} r2={res_t['r2']:.4f} (lam={res_t['lambda']:4.0f}) | "
                f"vis {v_str} ({st['vis_matched_imgs']}/{len(vis_sub)} imgs) | "
                f"exact={st['n_exact']} span={st['n_span']} unm={st['n_unmatched']} ({st['seconds']}s)",
                flush=True,
            )
        # Pick highest combined (0.65 text + 0.35 vision if vision present)
        sweep_results.sort(key=lambda r: (0.65 * r[1] + 0.35 * r[3]) if r[3] > 0 else r[1], reverse=True)
        best_layer = sweep_results[0][0]
        print(f"--> Best layer: {best_layer} (text cos={sweep_results[0][1]:.4f}, vis cos={sweep_results[0][3]:.4f})\n", flush=True)

    # 3. Full extraction on winning layer (text + vision)
    print(f"=== Full Calibration on layer={best_layer} ({len(text_targets)} text + {len(vision_targets)} vision) ===", flush=True)
    X_txt, Y_txt, X_vis, Y_vis, st = collect_27b_aligned(args.ninfer_url, best_layer, text_targets, vision_targets)
    print(
        f"[27B] Collected in {st['seconds']}s: text exact={st['n_exact']}, span={st['n_span']}, "
        f"unmatched={st['n_unmatched']}; vision={st['n_vis_tokens']} tokens "
        f"({st['vis_matched_imgs']}/{len(vision_targets)} imgs matched)",
        flush=True,
    )

    n_train_p = int(len(text_targets) * 0.9)
    tr_txt = st["prompt_ids"] < n_train_p
    te_txt = ~tr_txt
    X_tr_txt, Y_tr_txt = X_txt[tr_txt], Y_txt[tr_txt]
    X_te_txt, Y_te_txt = X_txt[te_txt], Y_txt[te_txt]

    has_vis = X_vis.shape[0] > 0
    if has_vis:
        n_train_v = max(1, int(len(vision_targets) * 0.85))
        tr_vis = st["vis_img_ids"] < n_train_v
        te_vis = ~tr_vis
        X_tr_vis, Y_tr_vis = X_vis[tr_vis], Y_vis[tr_vis]
        X_te_vis, Y_te_vis = X_vis[te_vis], Y_vis[te_vis]

    # Compare Joint Ridge vs Dual (Separate Text + Vision) Ridge!
    if has_vis:
        joint = fit_ridge(
            torch.cat([X_tr_txt, X_tr_vis], dim=0),
            torch.cat([Y_tr_txt, Y_tr_vis], dim=0),
            X_te_txt,
            Y_te_txt,
        )
        dev = "cuda"
        xv = (X_te_vis.float().to(dev) - joint["mean_in"].to(dev)) / joint["std_in"].to(dev)
        pv = (xv @ joint["W"].to(dev)) * joint["std_out"].to(dev) + joint["mean_out"].to(dev)
        jv_cos, jv_r2 = evaluate_predictions(pv, Y_te_vis.float().to(dev))
        print(f"[Joint Ridge] text cos={joint['cos']:.4f} r2={joint['r2']:.4f} | vis cos={jv_cos:.4f} r2={jv_r2:.4f}", flush=True)

    ridge_txt = fit_ridge(X_tr_txt, Y_tr_txt, X_te_txt, Y_te_txt)
    print(f"[Dual Text Ridge]   cos={ridge_txt['cos']:.4f} r2={ridge_txt['r2']:.4f} (lam={ridge_txt['lambda']})", flush=True)

    ridge_vis = None
    if has_vis and X_tr_vis.shape[0] > 0 and X_te_vis.shape[0] > 0:
        ridge_vis = fit_ridge(X_tr_vis, Y_tr_vis, X_te_vis, Y_te_vis)
        print(f"[Dual Vision Ridge] cos={ridge_vis['cos']:.4f} r2={ridge_vis['r2']:.4f} (lam={ridge_vis['lambda']})", flush=True)

    ridge_path = os.path.join(args.out_dir, f"{args.out_prefix}-dual-ridge.safetensors")
    save_clipproj(
        ridge_path,
        ridge_txt,
        sink_out,
        best_layer,
        {
            "n_train_prompts": len(text_targets),
            "n_train_tokens": int(X_tr_txt.shape[0]),
            "cos_test": round(ridge_txt["cos"], 6),
            "r2_test": round(ridge_txt["r2"], 6),
            "cos_test_vis": round(ridge_vis["cos"], 6) if ridge_vis else 0.0,
            "r2_test_vis": round(ridge_vis["r2"], 6) if ridge_vis else 0.0,
        },
        vis_ridge=ridge_vis,
    )

    # Train Residual MLPs for BOTH text and vision
    print(f"\n=== Training Text Residual MLP (5120 -> {args.mlp_hidden} -> 5120, {args.mlp_epochs} epochs) ===", flush=True)
    mlp_txt, mlp_txt_cos, mlp_txt_r2 = train_residual_mlp(
        X_tr_txt, Y_tr_txt, X_te_txt, Y_te_txt, ridge_txt, hidden_dim=args.mlp_hidden, epochs=args.mlp_epochs
    )
    mlp_vis, mlp_vis_cos, mlp_vis_r2 = None, 0.0, 0.0
    if ridge_vis is not None:
        print(f"\n=== Training Vision Residual MLP (5120 -> {args.mlp_hidden} -> 5120, {args.mlp_epochs} epochs) ===", flush=True)
        mlp_vis, mlp_vis_cos, mlp_vis_r2 = train_residual_mlp(
            X_tr_vis, Y_tr_vis, X_te_vis, Y_te_vis, ridge_vis, hidden_dim=args.mlp_hidden, epochs=args.mlp_epochs
        )

    mlp_path = os.path.join(args.out_dir, f"{args.out_prefix}-dual-mlp.safetensors")
    save_clipproj(
        mlp_path,
        ridge_txt,
        sink_out,
        best_layer,
        {
            "n_train_prompts": len(text_targets),
            "n_train_tokens": int(X_tr_txt.shape[0]),
            "cos_test": round(mlp_txt_cos, 6),
            "r2_test": round(mlp_txt_r2, 6),
            "cos_test_vis": round(mlp_vis_cos, 6) if mlp_vis else 0.0,
            "r2_test_vis": round(mlp_vis_r2, 6) if mlp_vis else 0.0,
            "mlp_hidden": args.mlp_hidden,
        },
        mlp_state=mlp_txt,
        vis_ridge=ridge_vis,
        vis_mlp_state=mlp_vis,
    )
    print("\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
