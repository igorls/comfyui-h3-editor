"""Pure math, mask, pose-gating, windowed relay, layer-mapping, and zero-halo/zero-wisp plate
compositing helpers for H3 Editor masked character replacement and pose-guided motion transfer.

No ComfyUI import at module level so every function is unit-testable directly under plain Python.
"""

import gzip
import json
import math
import os
import re
import numpy as np

# Windowed relay geometry on H3's exact 17k + 5 temporal grid (1-4-4-4-4 latent spans):
#   124 frames = 37 latent steps (5 + 17*7)
#    22 frames =  7 latent steps (5 + 17*1, exact 1-4-4-4-4-1-4 boundary)
#   102 stride = 30 latent steps (17*6, so step 30..36 of window k align 1:1 with step 0..6 of window k+1)
WINDOW_FRAMES = 124
OVERLAP_FRAMES = 22
WINDOW_STRIDE = WINDOW_FRAMES - OVERLAP_FRAMES  # 102
SINGLE_MAX_FRAMES = 141  # 5 + 17*8 (~5.88s)


def snap_grid_frames(frame_count):
    """Snap a positive frame count upward onto H3's legal 5 + 17k grid."""
    n = max(5, int(frame_count))
    return 5 + 17 * math.ceil((n - 5) / 17)


def plan_relay_windows(frame_count, *, single_max=SINGLE_MAX_FRAMES, window_frames=WINDOW_FRAMES,
                       overlap_frames=OVERLAP_FRAMES):
    """Plan single-pass or seamless latent-locked relay windows covering `frame_count` frames.

    Clips up to `single_max` (141 frames / ~5.88s) run in one window. Longer clips (up to 30s / 720f)
    tile into 124-frame windows sharing 22 latent-locked frames (stride 102), keeping peak VRAM
    constant (~124f) and wall-clock strictly linear O(T) instead of O(T^2).
    """
    total = max(5, int(frame_count))
    if total <= int(single_max):
        return [{"start": 0, "length": snap_grid_frames(total), "overlap": 0}]

    stride = int(window_frames) - int(overlap_frames)
    if stride <= 0 or stride % 17 != 0:
        raise ValueError("window stride must be a positive multiple of 17 frames")

    windows = []
    pos = 0
    while pos < total:
        overlap = 0 if pos == 0 else int(overlap_frames)
        remaining = total - pos
        if remaining <= int(single_max):
            length = max(39, snap_grid_frames(remaining))
            windows.append({"start": pos, "length": length, "overlap": overlap})
            break
        windows.append({"start": pos, "length": int(window_frames), "overlap": overlap})
        pos += stride
    return windows


def join_window_frames(prev_frames, next_frames, overlap_frames=OVERLAP_FRAMES, blend_frames=4):
    """Stitch two decoded window RGB batches [N, H, W, C] across a latent-locked overlap.

    The last `overlap_frames` of `prev_frames` align 1:1 with the first `overlap_frames` of
    `next_frames`. Blending `blend_frames` at the midpoint of the locked overlap eliminates even
    1-LSB VAE tiling seams while preserving exact output frame count `N_prev + N_next - overlap`.
    """
    a = np.asarray(prev_frames, dtype=np.float32)
    b = np.asarray(next_frames, dtype=np.float32)
    ov = max(0, min(int(overlap_frames), a.shape[0], b.shape[0]))
    if ov == 0:
        return np.concatenate([a, b], axis=0)

    blend = max(0, min(int(blend_frames), ov))
    mid = ov // 2
    cut = mid - blend // 2
    head_end = a.shape[0] - ov + cut

    parts = [a[:head_end]]
    if blend > 0:
        weights = (np.arange(1, blend + 1, dtype=np.float32) / float(blend + 1)).reshape((blend, 1, 1, 1))
        mixed = a[head_end:head_end + blend] * (1.0 - weights) + b[cut:cut + blend] * weights
        parts.append(mixed)
    parts.append(b[cut + blend:])
    return np.concatenate(parts, axis=0)


def flash_controlnet_layer_map(injection_layers):
    """Map standard 50-block H3 Fun ControlNet injection layers onto 49-block HyperFlow Flash
    (where standard transformer block 9 was pruned during distillation).

    Supports both Union v1 (0, 10, 20, 30, 40) -> (0, 9, 19, 29, 39)
    and Union v2.0 (0, 5, 10, ..., 45) -> (0, 5, 9, 14, 19, 24, 29, 34, 39, 44).
    """
    mapped = []
    for idx in injection_layers:
        idx = int(idx)
        if idx == 9:
            raise ValueError("Cannot map standard H3 block 9 onto 49-block HyperFlow Flash (pruned block)")
        mapped.append(idx if idx < 9 else idx - 1)
    return tuple(mapped)


def gate_bboxes(bboxes, frame_count, height, width, min_area_frac=0.005):
    """Filter tiny/empty detections and inject a zero-area sentinel box on absent frames.

    Upstream ComfyUI SDPoseKeypointExtractor falls back to full-image [0, 0, W, H] inference
    whenever `img_bboxes` is empty, hallucinating ghost skeletons on empty scenes/skylines.
    Injecting a zero-area box forces its `if x2 <= x1: continue` skip path, yielding clean
    empty guide frames without modifying upstream ComfyUI.
    """
    frame_area = max(1.0, float(height) * float(width))
    if not isinstance(bboxes, list):
        norm = [[bboxes]] * frame_count
    elif len(bboxes) == 0:
        norm = [[] for _ in range(frame_count)]
    else:
        norm = bboxes

    gated = []
    present = []
    sentinel = [{"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0}]
    for i in range(frame_count):
        fb = norm[min(i, len(norm) - 1)] if norm else []
        kept = [
            b for b in (fb or [])
            if isinstance(b, dict) and (float(b.get("width", 0.0)) * float(b.get("height", 0.0))) >= min_area_frac * frame_area
        ]
        if kept:
            gated.append(kept)
            present.append(True)
        else:
            gated.append(sentinel)
            present.append(False)
    return gated, present


def filter_pose_keypoints(kp_frames, min_valid_body_kps=2, score_thresh=0.3):
    """Drop partial/phantom pose detections with fewer than `min_valid_body_kps` confident body joints."""
    if min_valid_body_kps <= 0:
        return kp_frames
    cleaned = []
    for frame in kp_frames:
        entry = dict(frame)
        people = []
        for person in frame.get("people") or []:
            raw = person.get("pose_keypoints_2d")
            if not raw or len(raw) < 54:
                continue
            body = np.asarray(raw[:54], dtype=np.float32).reshape(18, 3)
            if int((body[:, 2] >= score_thresh).sum()) >= min_valid_body_kps:
                people.append(person)
        entry["people"] = people
        cleaned.append(entry)
    return cleaned


def expand_mask_safe(masks_np, expansion_px):
    """Outward Euclidean dilation of a [N, H, W] float/bool mask that preserves empty exit frames.

    Unlike h3-relay's H3RelayExpandMask, all-zero frames (after a subject exits frame) remain
    cleanly all-zero instead of raising or forcing mask recovery.
    """
    from scipy.ndimage import distance_transform_edt

    arr = np.asarray(masks_np)
    if arr.ndim == 2:
        arr = arr[None]
    expansion_px = int(max(0, expansion_px))
    out = np.empty(arr.shape, dtype=np.float32)
    for i in range(arr.shape[0]):
        binary = arr[i] > 0.5
        if expansion_px > 0 and binary.any():
            out[i] = (distance_transform_edt(~binary) <= expansion_px).astype(np.float32)
        else:
            out[i] = binary.astype(np.float32)
    return out


def _resize_mask_nearest_or_bilinear(mask_np, height, width):
    """Resize [N, Hm, Wm] mask to [N, height, width] bool array cleanly."""
    from PIL import Image

    arr = np.asarray(mask_np, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None]
    if arr.shape[1] == height and arr.shape[2] == width:
        return arr > 0.5
    out = np.empty((arr.shape[0], height, width), dtype=bool)
    for i in range(arr.shape[0]):
        im = Image.fromarray(np.clip(arr[i] * 255.0, 0, 255).astype(np.uint8))
        out[i] = np.asarray(im.resize((width, height), Image.BILINEAR)) > 127
    return out


def _align_frames(arr, target_n):
    if arr.shape[0] == target_n:
        return arr
    if arr.shape[0] > target_n:
        return arr[:target_n]
    pad = np.repeat(arr[-1:], target_n - arr.shape[0], axis=0)
    return np.concatenate([arr, pad], axis=0)


def composite_plate_harmonized(
    generated_np,
    source_np,
    source_mask_np,
    gen_mask_np=None,
    src_expand_px=18,
    gen_expand_px=4,
    feather_sigma=4.0,
    harm_sigma=24.0,
    collar_outer_px=36,
):
    """Zero-halo, zero-flyaway source-plate compositor in [0, 1] float32 RGB space ([N, H, W, 3]).

    Solves both failure modes discovered in controlled benchmarks:
      1. Wide synthesis-mask compositing (68px) leaves a visible ceiling/wall tone halo where
         neither performer existed.
      2. Over-tight symmetric compositing (5px around M_src) pastes back translucent source hair
         flyaways extending 6-16px outside SAM3's binary source mask.

    How this solves both simultaneously:
      - Asymmetric union mask: dilate M_src by `src_expand_px` (default 18px) to completely cover
        source hair flyaways and motion-blurred hands, unioned with tight `M_gen` (`gen_expand_px`
        = 4px) for the generated character.
      - Spatially-varying normalized-convolution illumination harmonization: estimates the local
        smooth RGB offset Delta(x, y) = G_sigma * (W_bg * (src - gen)) / (G_sigma * W_bg + eps)
        from a clean background collar (`1..collar_outer_px` outside the union mask) and applies it
        ONLY to non-character pixels `(1 - soft(M_gen))`, seamlessly matching dark ceiling corners
        and bright doorframes alike without tinting the generated character.
    """
    from scipy.ndimage import distance_transform_edt, gaussian_filter

    gen = np.asarray(generated_np, dtype=np.float32)[..., :3]
    N, H, W, _ = gen.shape
    src = _align_frames(np.asarray(source_np, dtype=np.float32)[..., :3], N)
    if src.shape[1] != H or src.shape[2] != W:
        from PIL import Image

        resized = np.empty((N, H, W, 3), dtype=np.float32)
        for i in range(N):
            im = Image.fromarray(np.clip(src[i] * 255.0, 0, 255).astype(np.uint8))
            resized[i] = np.asarray(im.resize((W, H), Image.BICUBIC), dtype=np.float32) / 255.0
        src = resized

    m_src = _resize_mask_nearest_or_bilinear(_align_frames(np.asarray(source_mask_np, dtype=np.float32), N), H, W)
    if gen_mask_np is not None:
        m_gen = _resize_mask_nearest_or_bilinear(_align_frames(np.asarray(gen_mask_np, dtype=np.float32), N), H, W)
    else:
        m_gen = None

    out = np.empty_like(gen)
    scale = max(0.5, float(W) / 544.0)
    src_r = int(round(max(0, src_expand_px) * scale))
    gen_r = int(round(max(0, gen_expand_px) * scale))
    f_sigma = float(max(0.0, feather_sigma)) * scale
    h_sigma = float(max(4.0, harm_sigma)) * scale
    c_outer = max(8, int(round(collar_outer_px * scale)))

    for i in range(N):
        ms = m_src[i]
        mg = m_gen[i] if m_gen is not None else ms
        if not ms.any() and not mg.any():
            out[i] = src[i]
            continue

        ms_dil = (distance_transform_edt(~ms) <= src_r) if (src_r > 0 and ms.any()) else ms
        mg_dil = (distance_transform_edt(~mg) <= gen_r) if (gen_r > 0 and mg.any()) else mg
        mu = ms_dil | mg_dil

        alpha = gaussian_filter(mu.astype(np.float32), sigma=f_sigma) if f_sigma > 0 else mu.astype(np.float32)
        alpha = np.clip(alpha, 0.0, 1.0)[..., None]

        if m_gen is not None and c_outer > 2:
            dist_u = distance_transform_edt(~mu)
            collar = ((dist_u >= 1.0) & (dist_u <= c_outer)).astype(np.float32)
            if float(collar.sum()) >= 50.0:
                diff = (src[i] - gen[i]) * collar[..., None]
                w_blur = gaussian_filter(collar, sigma=h_sigma) + 1e-5
                delta = np.stack(
                    [gaussian_filter(diff[..., c], sigma=h_sigma) / w_blur for c in range(3)],
                    axis=-1,
                )
                delta = np.clip(delta, -0.15, 0.15)
                char_soft = gaussian_filter(mg.astype(np.float32), sigma=max(1.5, 2.5 * scale))
                char_soft = np.clip(char_soft, 0.0, 1.0)[..., None]
                gen_frame = np.clip(gen[i] + delta * (1.0 - char_soft), 0.0, 1.0)
            else:
                gen_frame = gen[i]
        else:
            gen_frame = gen[i]

        out[i] = np.clip(gen_frame * alpha + src[i] * (1.0 - alpha), 0.0, 1.0)
    return out


def classify_character_style(image_or_path):
    """Classify a character reference still as 'preserve' (cel-shaded/2D anime) or 'photoreal' (photo/realistic).

    Accepts a filesystem path or an HxWx3 RGB numpy array (uint8 or float [0, 1]).
    Cel-shaded 2D illustrations have both a very high flat-region fraction (low gradient)
    and very low interior micro-texture (Laplacian) compared to camera photos / realistic renders.
    """
    if isinstance(image_or_path, (str, bytes)) or hasattr(image_or_path, "__fspath__"):
        from PIL import Image

        with Image.open(image_or_path) as im:
            im = im.convert("RGB")
            im.thumbnail((512, 512))
            arr = np.asarray(im, dtype=np.float32)
    else:
        arr = np.asarray(image_or_path, dtype=np.float32)[..., :3]
        if arr.size and float(arr.max()) <= 1.05:
            arr = arr * 255.0
    if arr.ndim != 3 or arr.shape[0] < 8 or arr.shape[1] < 8:
        return "photoreal"
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    fg = (gray < 246.0) & (gray > 8.0)
    if float(fg.mean()) < 0.05:
        fg = np.ones_like(gray, dtype=bool)
    gx = np.abs(np.diff(gray, axis=1, append=gray[:, -1:]))
    gy = np.abs(np.diff(gray, axis=0, append=gray[-1:, :]))
    grad = np.hypot(gx, gy)
    lap = np.abs(
        4.0 * gray[1:-1, 1:-1]
        - gray[:-2, 1:-1]
        - gray[2:, 1:-1]
        - gray[1:-1, :-2]
        - gray[1:-1, 2:]
    )
    non_edge = fg[1:-1, 1:-1] & (grad[1:-1, 1:-1] < 18.0)
    flat_frac = float((grad[fg] < 1.5).mean())
    p75_lap = float(np.percentile(lap[non_edge], 75)) if non_edge.any() else 0.0
    if flat_frac >= 0.58 and p75_lap <= 3.5:
        return "preserve"
    return "photoreal"


def latent_swap_score(sampled_video, source_video, video_mask):
    """Measure normalized latent change inside the edited subject mask.

    Returns None when the mask has no active cells (e.g. subject absent), otherwise
    mean(|sampled - source|) / (std(source) + 1e-6) over masked latent cells.
    """
    samp = np.asarray(sampled_video, dtype=np.float32)
    src = np.asarray(source_video, dtype=np.float32)
    mask = np.asarray(video_mask, dtype=np.float32)
    if samp.shape != src.shape:
        raise ValueError("sampled and source latents must share shape (%s vs %s)" % (samp.shape, src.shape))
    active = np.broadcast_to(mask > 0.5, samp.shape)
    if not active.any():
        return None
    diff = float(np.abs(samp[active] - src[active]).mean())
    scale = float(src[active].std()) + 1e-6
    return diff / scale


def apply_overlap_mask(video_mask, ov_steps, tail_ramp=0.45):
    """Lock the carried overlap steps [0 .. ov_steps-2] at 0.0 while softening the final overlap
    step [ov_steps-1] inside the subject mask to avoid a hard 0->1 scene-cut cliff at frame 22."""
    ov = int(ov_steps)
    if ov <= 0:
        return video_mask
    out = video_mask.clone() if hasattr(video_mask, "clone") else np.array(video_mask, copy=True)
    if ov == 1:
        out[:, :, :1] = 0.0
        return out
    out[:, :, :ov - 1] = 0.0
    ramp = max(0.0, min(1.0, float(tail_ramp)))
    out[:, :, ov - 1:ov] = out[:, :, ov - 1:ov] * ramp
    return out


def reflect_pad_indices(length, target_length):
    """Frame indices of length `target_length` that smoothly reflect past `length - 1` instead of freezing."""
    length, target_length = int(length), int(target_length)
    if target_length <= length:
        return list(range(target_length))
    if length <= 1:
        return [0] * target_length
    indices = list(range(length))
    period = 2 * (length - 1)
    for k in range(length, target_length):
        pos = k % period
        indices.append(pos if pos < length else period - pos)
    return indices


# ---------------------------------------------------------------------------
# Persistent sidecar caches for source segmentation masks, pose keypoints,
# and isolated character cutouts so re-renders and split clips skip repeating
# SAM3 / RT-DETR / SDPose on already-analyzed frames.
# ---------------------------------------------------------------------------

def _cache_slug(text, fallback="person"):
    cleaned = re.sub(r"[^a-z0-9]+", "_", str(text or "").strip().lower()).strip("_")
    return cleaned[:40] or fallback


def _file_sig(path):
    st = os.stat(path)
    return [int(st.st_size), int(st.st_mtime_ns)]


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def mask_cache_path(source_path, subject="person"):
    stem = os.path.splitext(source_path)[0]
    return f"{stem}.mask.{_cache_slug(subject)}.npz"


def pose_cache_path(source_path):
    stem = os.path.splitext(source_path)[0]
    return f"{stem}.pose.json.gz"


def cutout_cache_path(character_path, subject="person", backdrop="white"):
    stem = os.path.splitext(character_path)[0]
    return f"{stem}.iso.{_cache_slug(subject)}.{_cache_slug(backdrop, 'white')}.png"


def _read_mask_cache_meta(cache_path, source_path, subject, detect_threshold, track_threshold):
    if not source_path or not os.path.exists(source_path) or not os.path.exists(cache_path):
        return None, None
    sig = _file_sig(source_path)
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"].item()))
            if (
                meta.get("version") != 1
                or meta.get("sig") != sig
                or meta.get("subject") != str(subject or "person")
                or abs(float(meta.get("detect_threshold", -1)) - float(detect_threshold)) > 1e-4
                or abs(float(meta.get("track_threshold", -1)) - float(track_threshold)) > 1e-4
            ):
                return None, None
            present = np.asarray(data["present"], dtype=bool)
            return meta, present
    except Exception:
        return None, None


def has_mask_cache(source_path, first, stop, subject="person", detect_threshold=0.35, track_threshold=0.5):
    """Fast check whether cached SAM3 raw masks already cover frames [first, stop)."""
    first, stop = int(first), int(stop)
    if first < 0 or stop <= first:
        return False
    cache_path = mask_cache_path(source_path, subject)
    meta, present = _read_mask_cache_meta(cache_path, source_path, subject, detect_threshold, track_threshold)
    if meta is None or stop > len(present):
        return False
    return bool(present[first:stop].all())


def load_mask_cache(source_path, first, stop, subject="person", detect_threshold=0.35, track_threshold=0.5):
    """Load cached raw masks [stop - first, H, W] float32 if frames [first, stop) are all present."""
    first, stop = int(first), int(stop)
    if first < 0 or stop <= first or not source_path or not os.path.exists(source_path):
        return None
    cache_path = mask_cache_path(source_path, subject)
    if not os.path.exists(cache_path):
        return None
    sig = _file_sig(source_path)
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"].item()))
            if (
                meta.get("version") != 1
                or meta.get("sig") != sig
                or meta.get("subject") != str(subject or "person")
                or abs(float(meta.get("detect_threshold", -1)) - float(detect_threshold)) > 1e-4
                or abs(float(meta.get("track_threshold", -1)) - float(track_threshold)) > 1e-4
            ):
                return None
            present = np.asarray(data["present"], dtype=bool)
            if stop > len(present) or not present[first:stop].all():
                return None
            h, w = int(meta["height"]), int(meta["width"])
            packed = data["packed"][first:stop]
            bits = np.unpackbits(packed, axis=1)[:, :h * w]
            return bits.reshape((stop - first, h, w)).astype(np.float32)
    except Exception:
        return None


def save_mask_cache(source_path, first, masks_np, total_frames=0, subject="person",
                    detect_threshold=0.35, track_threshold=0.5):
    """Merge newly tracked raw masks [N, H, W] at offset `first` into the bit-packed sidecar cache."""
    if not source_path or not os.path.exists(source_path):
        return None
    arr = np.asarray(masks_np)
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim != 3 or arr.shape[0] == 0:
        return None
    first = int(first)
    n, h, w = arr.shape
    total = max(int(total_frames or 0), first + n)
    sig = _file_sig(source_path)
    subject_str = str(subject or "person")
    new_packed = np.packbits((arr > 0.5).reshape(n, h * w).astype(np.uint8), axis=1)
    cols = new_packed.shape[1]

    cache_path = mask_cache_path(source_path, subject_str)
    present = np.zeros(total, dtype=bool)
    packed = np.zeros((total, cols), dtype=np.uint8)

    if os.path.exists(cache_path):
        try:
            with np.load(cache_path, allow_pickle=False) as old:
                old_meta = json.loads(str(old["meta"].item()))
                if (
                    old_meta.get("version") == 1
                    and old_meta.get("sig") == sig
                    and old_meta.get("subject") == subject_str
                    and int(old_meta.get("height", -1)) == h
                    and int(old_meta.get("width", -1)) == w
                    and abs(float(old_meta.get("detect_threshold", -1)) - float(detect_threshold)) <= 1e-4
                    and abs(float(old_meta.get("track_threshold", -1)) - float(track_threshold)) <= 1e-4
                ):
                    old_present = np.asarray(old["present"], dtype=bool)
                    old_packed = np.asarray(old["packed"], dtype=np.uint8)
                    old_total = len(old_present)
                    if old_total > total:
                        total = old_total
                        present = np.zeros(total, dtype=bool)
                        packed = np.zeros((total, cols), dtype=np.uint8)
                    present[:old_total] = old_present
                    packed[:old_total] = old_packed
        except Exception:
            pass

    packed[first:first + n] = new_packed
    present[first:first + n] = True
    meta = {
        "version": 1, "sig": sig, "frames": total, "height": h, "width": w,
        "subject": subject_str, "detect_threshold": float(detect_threshold), "track_threshold": float(track_threshold),
    }
    tmp = cache_path + ".tmp.npz"
    try:
        np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), present=present, packed=packed)
        os.replace(tmp, cache_path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
    return cache_path


def _read_pose_cache(source_path, min_area_frac=0.005):
    if not source_path or not os.path.exists(source_path):
        return None
    cache_path = pose_cache_path(source_path)
    if not os.path.exists(cache_path):
        return None
    sig = _file_sig(source_path)
    try:
        with gzip.open(cache_path, "rt", encoding="utf-8") as f:
            payload = json.load(f)
        if (
            payload.get("version") != 1
            or payload.get("sig") != sig
            or abs(float(payload.get("min_area_frac", -1)) - float(min_area_frac)) > 1e-5
        ):
            return None
        return payload
    except Exception:
        return None


def has_pose_cache(source_path, first, stop, min_area_frac=0.005):
    """Fast check whether cached SDPose keypoints already cover frames [first, stop)."""
    first, stop = int(first), int(stop)
    if first < 0 or stop <= first:
        return False
    payload = _read_pose_cache(source_path, min_area_frac=min_area_frac)
    if not payload:
        return False
    present = payload.get("present") or []
    if stop > len(present):
        return False
    return all(present[first:stop])


def load_pose_cache(source_path, first, stop, min_area_frac=0.005):
    """Load raw cached pose keypoints for frames [first, stop), or None on miss."""
    first, stop = int(first), int(stop)
    if first < 0 or stop <= first:
        return None
    payload = _read_pose_cache(source_path, min_area_frac=min_area_frac)
    if not payload:
        return None
    present = payload.get("present") or []
    keypoints = payload.get("keypoints") or []
    if stop > len(present) or stop > len(keypoints) or not all(present[first:stop]):
        return None
    return keypoints[first:stop]


def save_pose_cache(source_path, first, kp_frames, total_frames=0, min_area_frac=0.005):
    """Merge raw SDPose keypoint frames at offset `first` into the gzipped sidecar cache."""
    if not source_path or not os.path.exists(source_path) or not kp_frames:
        return None
    first = int(first)
    n = len(kp_frames)
    total = max(int(total_frames or 0), first + n)
    sig = _file_sig(source_path)
    old = _read_pose_cache(source_path, min_area_frac=min_area_frac)
    if old and isinstance(old.get("present"), list) and isinstance(old.get("keypoints"), list):
        old_total = len(old["present"])
        total = max(total, old_total)
        present = list(old["present"]) + [False] * (total - old_total)
        keypoints = list(old["keypoints"]) + [None] * (total - old_total)
    else:
        present = [False] * total
        keypoints = [None] * total
    safe_frames = _json_safe(list(kp_frames))
    for i, frame in enumerate(safe_frames):
        present[first + i] = True
        keypoints[first + i] = frame
    payload = {
        "version": 1, "sig": sig, "frames": total,
        "min_area_frac": float(min_area_frac), "present": present, "keypoints": keypoints,
    }
    cache_path = pose_cache_path(source_path)
    tmp = cache_path + ".tmp.gz"
    try:
        with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=5) as f:
            json.dump(payload, f)
        os.replace(tmp, cache_path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
    return cache_path


def has_cutout_cache(character_path, subject="person", backdrop="white", threshold=0.35):
    """Check PNG header metadata to see whether an isolated character cutout is cached and fresh."""
    if not character_path or not os.path.exists(character_path):
        return False
    cache_path = cutout_cache_path(character_path, subject, backdrop)
    if not os.path.exists(cache_path):
        return False
    sig = json.dumps(_file_sig(character_path))
    try:
        from PIL import Image

        with Image.open(cache_path) as im:
            return (
                im.info.get("h3_sig") == sig
                and abs(float(im.info.get("h3_threshold", -1)) - float(threshold)) <= 1e-4
            )
    except Exception:
        return False


def load_cutout_cache(character_path, subject="person", backdrop="white", threshold=0.35):
    """Load cached isolated character cutout [1, H, W, 3] float32 in [0, 1], or None on miss."""
    if not has_cutout_cache(character_path, subject, backdrop, threshold):
        return None
    cache_path = cutout_cache_path(character_path, subject, backdrop)
    try:
        from PIL import Image

        with Image.open(cache_path) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
            return arr[None]
    except Exception:
        return None


def save_cutout_cache(character_path, rgb_np, subject="person", backdrop="white", threshold=0.35):
    """Save isolated character cutout RGB [H, W, 3] (float [0, 1] or uint8) with source signature in PNG metadata."""
    if not character_path or not os.path.exists(character_path):
        return None
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo

    arr = np.asarray(rgb_np, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    arr = arr[..., :3]
    if arr.size and float(arr.max()) <= 1.05:
        arr = np.clip(arr * 255.0, 0, 255)
    img = Image.fromarray(arr.astype(np.uint8))
    info = PngInfo()
    info.add_text("h3_sig", json.dumps(_file_sig(character_path)))
    info.add_text("h3_threshold", str(float(threshold)))
    cache_path = cutout_cache_path(character_path, subject, backdrop)
    tmp = cache_path + ".tmp.png"
    try:
        img.save(tmp, format="PNG", pnginfo=info)
        os.replace(tmp, cache_path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
    return cache_path


def _bool_spans(present):
    """Collapse a boolean per-frame sequence into contiguous [start, end] frame spans."""
    spans = []
    start = None
    for idx, flag in enumerate(present):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            spans.append([start, idx])
            start = None
    if start is not None:
        spans.append([start, len(present)])
    return spans


def mask_cache_summary(source_path, subject="person", detect_threshold=0.35, track_threshold=0.5):
    """Return coverage/size metadata for a source video's cached SAM3 masks, or None if missing/stale."""
    cache_path = mask_cache_path(source_path, subject)
    meta, present = _read_mask_cache_meta(cache_path, source_path, subject, detect_threshold, track_threshold)
    if meta is None:
        return None
    return {
        "path": cache_path,
        "file": os.path.basename(cache_path),
        "size_bytes": int(os.path.getsize(cache_path)),
        "frames": int(len(present)),
        "cached_frames": int(present.sum()),
        "height": int(meta["height"]),
        "width": int(meta["width"]),
        "spans": _bool_spans(present.tolist()),
        "subject": meta["subject"],
    }


def pose_cache_summary(source_path, min_area_frac=0.005):
    """Return coverage/size metadata for a source video's cached SDPose keypoints, or None if missing/stale."""
    payload = _read_pose_cache(source_path, min_area_frac=min_area_frac)
    if not payload:
        return None
    cache_path = pose_cache_path(source_path)
    present = payload.get("present") or []
    keypoints = payload.get("keypoints") or []
    detected = sum(
        1 for flag, frame in zip(present, keypoints)
        if flag and isinstance(frame, dict) and len(frame.get("people") or []) > 0
    )
    return {
        "path": cache_path,
        "file": os.path.basename(cache_path),
        "size_bytes": int(os.path.getsize(cache_path)),
        "frames": int(len(present)),
        "cached_frames": int(sum(1 for flag in present if flag)),
        "detected_frames": int(detected),
        "spans": _bool_spans(present),
    }


def cutout_cache_summary(character_path, subject="person", backdrop="white", threshold=0.35):
    """Return metadata for a cached isolated character cutout PNG, or None if missing/stale."""
    if not has_cutout_cache(character_path, subject=subject, backdrop=backdrop, threshold=threshold):
        return None
    cache_path = cutout_cache_path(character_path, subject, backdrop)
    return {
        "path": cache_path,
        "file": os.path.basename(cache_path),
        "size_bytes": int(os.path.getsize(cache_path)),
        "subject": str(subject or "person"),
        "backdrop": str(backdrop or "white"),
    }


def sample_frame_indices(first, stop, count=6):
    """Pick up to `count` evenly spaced frame indices inside [first, stop)."""
    first, stop = int(first), int(stop)
    span = max(0, stop - first)
    if span == 0:
        return []
    if span <= count:
        return list(range(first, stop))
    return [first + int(round(i * (span - 1) / float(count - 1))) for i in range(count)]


def _mask_outline(binary):
    """1px boolean perimeter of a 2D boolean mask."""
    if not binary.any():
        return np.zeros_like(binary, dtype=bool)
    p = np.pad(binary, 1, mode="constant", constant_values=False)
    eroded = p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return binary & ~eroded


def _compose_sheet(tiles, labels, gap=4):
    """Pack PIL RGB tiles into a clean dark contact sheet with frame labels and return JPEG bytes."""
    import io
    from PIL import Image, ImageDraw

    if not tiles:
        raise ValueError("no preview frames available")
    tw, th = tiles[0].size
    cols = min(len(tiles), 3 if th > tw else 6)
    rows = math.ceil(len(tiles) / float(cols))
    canvas = Image.new("RGB", (cols * tw + (cols + 1) * gap, rows * th + (rows + 1) * gap), (18, 20, 24))
    draw = ImageDraw.Draw(canvas)
    for idx, (tile, label) in enumerate(zip(tiles, labels)):
        r, c = divmod(idx, cols)
        x = gap + c * (tw + gap)
        y = gap + r * (th + gap)
        canvas.paste(tile, (x, y))
        if label:
            pad_w = max(44, len(label) * 7 + 10)
            draw.rectangle([x + 4, y + 4, x + 4 + pad_w, y + 20], fill=(16, 18, 22))
            draw.text((x + 8, y + 6), label, fill=(235, 238, 245))
    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=88)
    return buf.getvalue()


def render_mask_sheet(rgb_frames, frame_indices, raw_masks, expand_px=32, tile_width=220):
    """Render a JPEG contact sheet showing SAM3 raw masks (cyan) + expanded synthesis halo (amber)."""
    from PIL import Image

    tiles, labels = [], []
    for rgb, fidx, raw in zip(rgb_frames, frame_indices, raw_masks):
        arr = np.asarray(rgb, dtype=np.float32)[..., :3]
        if arr.max() <= 1.05:
            arr = arr * 255.0
        h, w = arr.shape[:2]
        tw = min(int(tile_width), w)
        th = max(2, int(round(h * (tw / float(w)))))
        base = np.asarray(Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).resize((tw, th), Image.BILINEAR),
                          dtype=np.float32)
        m_raw = _resize_mask_nearest_or_bilinear(raw, th, tw)[0]
        scaled_exp = max(0, int(round(float(expand_px) * (tw / 384.0))))
        m_exp = expand_mask_safe(m_raw.astype(np.float32), scaled_exp)[0] > 0.5
        halo = m_exp & ~m_raw
        # Soft amber tint on expanded dilation ring, crisp cyan tint on raw SAM3 subject mask
        out = base.copy()
        out[halo] = out[halo] * 0.72 + np.array([245.0, 158.0, 11.0], dtype=np.float32) * 0.28
        out[m_raw] = out[m_raw] * 0.62 + np.array([56.0, 189.0, 248.0], dtype=np.float32) * 0.38
        out[_mask_outline(m_exp)] = [251.0, 191.0, 36.0]
        out[_mask_outline(m_raw)] = [125.0, 211.0, 252.0]
        tiles.append(Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)))
        labels.append(f"f{int(fidx)}" + ("" if m_raw.any() else " · empty"))
    return _compose_sheet(tiles, labels)


OPENPOSE_LIMBS = (
    (1, 2, (255, 85, 85)), (1, 5, (255, 170, 50)), (2, 3, (255, 120, 60)), (3, 4, (255, 215, 60)),
    (5, 6, (160, 235, 70)), (6, 7, (70, 230, 140)), (1, 8, (60, 210, 240)), (8, 9, (60, 160, 255)),
    (9, 10, (90, 120, 255)), (1, 11, (160, 110, 255)), (11, 12, (215, 95, 255)), (12, 13, (255, 95, 200)),
    (1, 0, (250, 250, 250)), (0, 14, (255, 140, 220)), (14, 16, (255, 110, 180)),
    (0, 15, (140, 210, 255)), (15, 17, (110, 180, 255)),
)


def render_pose_sheet(rgb_frames, frame_indices, kp_frames, score_thresh=0.3, tile_width=220):
    """Render a JPEG contact sheet drawing cached OpenPose skeletons over dimmed source frames."""
    from PIL import Image, ImageDraw

    tiles, labels = [], []
    for rgb, fidx, frame in zip(rgb_frames, frame_indices, kp_frames):
        arr = np.asarray(rgb, dtype=np.float32)[..., :3]
        if arr.max() <= 1.05:
            arr = arr * 255.0
        h, w = arr.shape[:2]
        tw = min(int(tile_width), w)
        th = max(2, int(round(h * (tw / float(w)))))
        dimmed = np.clip(arr * 0.45, 0, 255).astype(np.uint8)
        tile = Image.fromarray(dimmed).resize((tw, th), Image.BILINEAR)
        draw = ImageDraw.Draw(tile)
        people = (frame.get("people") or []) if isinstance(frame, dict) else []
        cw = float((frame or {}).get("canvas_width") or w)
        ch = float((frame or {}).get("canvas_height") or h)
        sx, sy = tw / max(1.0, cw), th / max(1.0, ch)
        for person in people:
            raw = person.get("pose_keypoints_2d") or []
            if len(raw) < 54:
                continue
            pts = [(float(raw[i * 3]), float(raw[i * 3 + 1]), float(raw[i * 3 + 2])) for i in range(18)]
            # Normalized [0..1] vs pixel coordinates
            max_coord = max((max(abs(x), abs(y)) for x, y, c in pts if c >= score_thresh), default=0.0)
            px_sx, px_sy = (tw, th) if 0.0 < max_coord <= 1.5 else (sx, sy)
            for a, b, color in OPENPOSE_LIMBS:
                xa, ya, ca = pts[a]
                xb, yb, cb = pts[b]
                if ca >= score_thresh and cb >= score_thresh:
                    draw.line([(xa * px_sx, ya * px_sy), (xb * px_sx, yb * px_sy)], fill=color, width=2)
            for x, y, c in pts:
                if c >= score_thresh:
                    px, py = x * px_sx, y * px_sy
                    draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=(255, 255, 255))
        tiles.append(tile)
        labels.append(f"f{int(fidx)}" + ("" if people else " · no pose"))
    return _compose_sheet(tiles, labels)
