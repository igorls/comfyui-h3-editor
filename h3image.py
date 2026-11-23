"""Official MiniMax-H3-Image (>= 4 MP) static props, turnaround plates, and paired state variants. No Comfy import."""

import math

try:
    from . import sprite
except ImportError:
    import sprite

UNET = "minimax_h3_image_pruned_int8_convrot.safetensors"
TEXT = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VAE = "minimax_h3_image_vae_fp16.safetensors"
PH3_UNETS = ("p-h3-image_int8_convrot.safetensors", "p-h3-image_bf16.safetensors")
PH3_BASE_UNETS = ("minimax_h3_ref2va_pruned_int8_convrot.safetensors",)
PH3_LORAS = ("p-h3-image_lora_bf16.safetensors",)
PH3_TEXTS = ("p-h3-image_text_encoder_int8_convrot.safetensors", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")
PH3_VAES = ("p-h3-image_vae_bf16.safetensors", "p-h3-image_vae_int8_convrot.safetensors")
PH3_SIGMAS = "1.0, 0.984753, 0.967905, 0.951688, 0.925372, 0.881923, 0.810083, 0.528718, 0.0"
STEPS = 6
MIN_MEGAPIXELS = 4.0
MIN_PIXELS = int(MIN_MEGAPIXELS * 1_000_000)

SIZES = {
    "square": (2048, 2048),      # 4.19 MP — ideal for 42-deg RPG isometric props & chibi plates
    "portrait": (1664, 2496),    # 4.15 MP — full-body standing characters
    "landscape": (2496, 1664),   # 4.15 MP — wide props or quadrupeds
}

TURNAROUND_VIEWS = ("front", "side", "back", "left")


def validate_native_size(width, height):
    """Enforce MiniMax-H3-Image's >= 4 MP operating range and 32-px alignment."""
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError("H3-Image width and height must be positive whole numbers")
    if width % 32 != 0 or height % 32 != 0:
        raise ValueError("H3-Image width and height must be multiples of 32 (got %sx%s)" % (width, height))
    if width * height < MIN_PIXELS:
        raise ValueError(
            "MiniMax-H3-Image degrades below %.1f MP (got %sx%s = %.2f MP). Use >= 2048x2048 or 1664x2496."
            % (MIN_MEGAPIXELS, width, height, (width * height) / 1_000_000.0)
        )
    return width, height


def resolve_size(size=None):
    """Accept a preset name ('square', 'portrait', 'landscape') or a (width, height) pair."""
    if size is None:
        return SIZES["square"]
    if isinstance(size, str):
        if size not in SIZES:
            raise ValueError("choose an H3-Image size preset from %s" % (", ".join(SIZES),))
        return SIZES[size]
    if isinstance(size, (tuple, list)) and len(size) == 2:
        return validate_native_size(size[0], size[1])
    raise ValueError("size must be a preset name or a (width, height) pair")


def prop_prompt(description, style="", camera="rpg", style_ref=True, top_canopy=False):
    """Style-locked 4 MP static game prop prompt on pure white, shadow-free."""
    desc = str(description or "").strip().rstrip(".")
    if not desc:
        raise ValueError("describe the game prop to generate")
    if camera not in sprite.CAMERAS:
        raise ValueError("camera must be one of %s" % (", ".join(sprite.CAMERAS),))
    parts = [desc + "."]
    if style_ref:
        parts.append(
            "Match the exact art style, line weight, dark ink outline thickness, cel shading, and warm color palette of <Picture 1>."
        )
    if style and style.strip():
        parts.append(style.strip().rstrip(".") + " style.")
    if camera == "rpg":
        parts.append(
            "Elevated 42-degree top-down isometric RPG camera angle with visible top planes and grounded base."
        )
        if top_canopy:
            parts.append(
                "Looking down onto the rounded upper crown and top foliage canopy from a 42-degree elevated isometric angle."
            )
    elif camera == "overhead":
        parts.append("Straight-down 90-degree top-down orthographic camera.")
    else:
        parts.append("Eye-level orthographic side view.")
    parts.append(
        "Single isolated game prop centered on a plain pure white background with clear margins on all sides. "
        "No ground plane, no floor, no drop shadow, no text, no extra objects."
    )
    return " ".join(parts)


def state_variant_prompt(description, style="", style_ref=True):
    """Paired state-variant edit prompt (e.g., lantern_unlit -> lantern_lit) keeping pixel-locked silhouette."""
    desc = str(description or "").strip().rstrip(".")
    if not desc:
        raise ValueError("describe the state change for the prop variant")
    if style_ref:
        ref_line = (
            "Edit the exact prop shown in <Picture 2>, keeping the exact art style and ink lineart of <Picture 1>. "
            "Preserve the exact silhouette, camera angle, geometry, proportions, and position of <Picture 2> pixel-for-pixel."
        )
    else:
        ref_line = (
            "Edit the exact prop shown in <Picture 1>. "
            "Preserve its exact silhouette, camera angle, geometry, proportions, and position pixel-for-pixel."
        )
    parts = [ref_line, "State change: " + desc + "."]
    if style and style.strip():
        parts.append(style.strip().rstrip(".") + " style.")
    parts.append("Plain pure white background with clear margins, no floor, no drop shadow, no text.")
    return " ".join(parts)


def turnaround_prompt(view, description="", style="", camera="rpg", proportions="adult", holding="", has_side=False):
    """Neutral standing turnaround prompt (never mid-run) so multi-view plates do not bias rear/diagonal loops."""
    if view not in TURNAROUND_VIEWS:
        raise ValueError("turnaround view must be one of %s" % (", ".join(TURNAROUND_VIEWS),))
    if camera not in sprite.CAMERAS:
        raise ValueError("camera must be one of %s" % (", ".join(sprite.CAMERAS),))
    poses = {
        "front": "strict front view facing directly toward the viewer in a calm neutral standing pose",
        "side": "strict right-facing side profile in a calm neutral standing pose with feet together (NOT running)",
        "left": "strict left-facing side profile in a calm neutral standing pose with feet together (NOT running)",
        "back": "strict rear back view facing directly away from the viewer in a calm neutral standing pose, showing the back of the head, hair, outfit and footwear with no face visible",
    }
    refs = "<Picture 1> (front view) and <Picture 2> (additional reference)" if has_side else "<Picture 1> (front view)"
    parts = ["Create a single full-body %s of the exact same character shown in %s." % (poses[view], refs)]
    if description and description.strip():
        parts.append(description.strip().rstrip(".") + ".")
    if style and style.strip():
        parts.append(style.strip().rstrip(".") + " style.")
    if proportions == "chibi":
        parts.append("Keep the exact 2.5-head chibi proportions and silhouette height.")
    if camera == "rpg":
        parts.append("Elevated 42-degree top-down isometric RPG camera angle matching <Picture 1>.")
    elif camera == "overhead":
        parts.append("Straight-down 90-degree overhead orthographic camera matching <Picture 1>.")
    else:
        parts.append("Exact same level camera height and framing as <Picture 1>.")
    if holding and holding.strip():
        parts.append("Keep %s clearly visible in the same hand." % holding.strip())
    parts.append(
        "Same exact height, colors, outfit, footwear, and proportions. "
        "Centered on a plain pure white background, head to feet fully in frame with clear margins. "
        "No floor, no shadow, no text, no extra figures."
    )
    return " ".join(parts)


def _still_node(asset):
    if isinstance(asset, str):
        file = asset
    elif isinstance(asset, dict) and isinstance(asset.get("file"), str) and asset["file"]:
        file = "%s/%s" % (asset["subfolder"], asset["file"]) if asset.get("subfolder") else asset["file"]
    else:
        raise ValueError("reference image must be a file path or {file, subfolder} asset")
    return {"class_type": "H3LoadStill", "inputs": {"file": file}}


def conditioning_node(available=None):
    """Return the installed H3-Image conditioning node ID (PH3ImageConditioning in current ComfyUI)."""
    if available is not None:
        if "node:PH3ImageConditioning" in available:
            return "PH3ImageConditioning"
        if "node:MiniMaxH3ImageConditioning" in available:
            return "MiniMaxH3ImageConditioning"
    return "PH3ImageConditioning"


def _pick(available, candidates, fallback):
    if available:
        for name in candidates:
            if name in available:
                return name
    return fallback


def prop_graph(prompt, width=2048, height=2048, seed=0, filename_prefix="props/prop",
               style_ref=None, base_prop=None, models=None, bg_model=None, available=None):
    """Build a ComfyUI graph for Official MiniMax-H3-Image / P-H3-Image (>= 4 MP) + optional BiRefNet + H3DefringeMatte."""
    validate_native_size(int(width), int(height))
    models = models or {}
    cond_cls = conditioning_node(available)
    ph3 = cond_cls == "PH3ImageConditioning"
    lora = None
    if ph3:
        unet = models.get("image_unet") or _pick(available, PH3_UNETS, None)
        if not unet:
            lora = _pick(available, PH3_LORAS, PH3_LORAS[0])
            unet = _pick(available, PH3_BASE_UNETS, PH3_BASE_UNETS[0])
        clip = models.get("clip") or _pick(available, PH3_TEXTS, PH3_TEXTS[0])
        vae = models.get("image_vae") or _pick(available, PH3_VAES, PH3_VAES[0])
    else:
        unet = models.get("image_unet") or UNET
        clip = models.get("clip") or TEXT
        vae = models.get("image_vae") or VAE
    cond_in = {"clip": ["2", 0], "prompt": str(prompt), "width": int(width), "height": int(height)}
    graph = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip, "type": "minimax", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
    }
    model_ref = ["1", 0]
    if lora:
        graph["5"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": model_ref, "lora_name": lora, "strength_model": 1.0},
        }
        model_ref = ["5", 0]
    if ph3:
        graph["4"] = {
            "class_type": "MiniMaxH3SigmaShift",
            "inputs": {"model": model_ref, "shift_video": 1.0, "shift_audio": 1.0},
        }
        model_ref = ["4", 0]
    refs = [r for r in (style_ref, base_prop) if r is not None]
    if refs:
        cond_in["vae"] = ["3", 0]
        for idx, asset in enumerate(refs, start=1):
            nid = str(40 + idx)
            graph[nid] = _still_node(asset)
            cond_in["images.image_%d" % idx] = [nid, 0]
    graph["10"] = {"class_type": cond_cls, "inputs": cond_in}
    if ph3:
        graph.update({
            "11": {"class_type": "BasicGuider", "inputs": {"model": model_ref, "conditioning": ["10", 0]}},
            "12": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
            "13": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}},
            "14": {"class_type": "ManualSigmas", "inputs": {"sigmas": PH3_SIGMAS}},
            "15": {"class_type": "SamplerCustomAdvanced", "inputs": {
                "noise": ["13", 0], "guider": ["11", 0], "sampler": ["12", 0],
                "sigmas": ["14", 0], "latent_image": ["10", 1],
            }},
            "16": {"class_type": "VAEDecode", "inputs": {"samples": ["15", 0], "vae": ["3", 0]}},
        })
    else:
        graph.update({
            "11": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["10", 0]}},
            "15": {"class_type": "KSampler", "inputs": {
                "model": model_ref, "positive": ["10", 0], "negative": ["11", 0], "latent_image": ["10", 1],
                "seed": int(seed), "steps": STEPS, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
            }},
            "16": {"class_type": "VAEDecode", "inputs": {"samples": ["15", 0], "vae": ["3", 0]}},
        })
    out_image = ["16", 0]
    if bg_model:
        graph["20"] = {"class_type": "LoadBackgroundRemovalModel", "inputs": {"bg_removal_name": bg_model}}
        graph["21"] = {"class_type": "RemoveBackground", "inputs": {"bg_removal_model": ["20", 0], "image": ["16", 0]}}
        graph["22"] = {"class_type": "H3DefringeMatte", "inputs": {"images": ["16", 0], "mask": ["21", 0]}}
        out_image = ["22", 0]
    graph["25"] = {"class_type": "SaveImage", "inputs": {"images": out_image, "filename_prefix": filename_prefix}}
    return graph


def turnaround_graph(view, front_ref, side_ref=None, description="", style="", camera="rpg", proportions="adult",
                     holding="", width=2048, height=2048, seed=0, filename_prefix="turnaround/plate",
                     models=None, bg_model=None, available=None):
    """Build a 4 MP H3-Image turnaround plate derivation graph from a front (and optional side) plate."""
    if front_ref is None:
        raise ValueError("turnaround derivation needs a front reference image")
    text = turnaround_prompt(
        view, description=description, style=style, camera=camera, proportions=proportions,
        holding=holding, has_side=side_ref is not None,
    )
    return prop_graph(
        text, width=width, height=height, seed=seed, filename_prefix=filename_prefix,
        style_ref=front_ref, base_prop=side_ref,
        models=models, bg_model=bg_model, available=available,
    )


def paired_alignment_iou(base_rgba, variant_rgba, alpha_threshold=128):
    """Silhouette IoU between two RGBA prop states (e.g. unlit vs. lit). Production target: >= 0.95."""
    import numpy as np
    from PIL import Image

    a = base_rgba.convert("RGBA")
    b = variant_rgba.convert("RGBA")
    if b.size != a.size:
        b = b.resize(a.size, Image.Resampling.BILINEAR)
    mask_a = np.asarray(a.getchannel("A")) >= int(alpha_threshold)
    mask_b = np.asarray(b.getchannel("A")) >= int(alpha_threshold)
    union = int((mask_a | mask_b).sum())
    if union == 0:
        return 0.0
    inter = int((mask_a & mask_b).sum())
    return round(float(inter) / float(union), 4)


def resize_premultiplied(im, output_size, pivot=None, pad=4):
    """Downscale an RGBA prop cleanly with premultiplied RGBa Lanczos, aligned to an optional ground pivot."""
    from PIL import Image

    out_w, out_h = sprite.validate_tile_size(output_size)
    rgba = im.convert("RGBA")
    bbox = rgba.getchannel("A").point(lambda v: 255 if v > 16 else 0).getbbox()
    if bbox is None:
        raise ValueError("prop has no visible foreground silhouette")
    crop = rgba.crop(bbox)
    cw, ch = crop.size
    if pivot is not None:
        px, py = int(pivot[0]), int(pivot[1])
        avail_w = max(2, 2 * min(px - pad, out_w - px - pad))
        avail_h = max(2, py - pad)
    else:
        px, py = out_w // 2, out_h - pad
        avail_w = max(2, out_w - 2 * pad)
        avail_h = max(2, out_h - 2 * pad)
    scale = min(avail_w / float(cw), avail_h / float(ch))
    new_w = max(1, int(round(cw * scale)))
    new_h = max(1, int(round(ch * scale)))
    scaled = crop.convert("RGBa").resize((new_w, new_h), Image.Resampling.LANCZOS).convert("RGBA")
    canvas = Image.new("RGBA", (out_w, out_h), (0, 0, 0, 0))
    paste_x = max(0, min(out_w - new_w, int(round(px - new_w / 2.0))))
    paste_y = max(0, min(out_h - new_h, int(round(py - new_h))))
    canvas.paste(scaled, (paste_x, paste_y))
    return canvas


def process_prop(frame, mask=None, output_size=None, pivot=None, pad=4):
    """Cut a 4 MP prop out via sprite.with_matte() (by its matte, or by the backdrop colour alone) and downsample cleanly."""
    rgba = sprite.with_matte(frame, mask)
    if output_size is None:
        return rgba
    return resize_premultiplied(rgba, output_size, pivot=pivot, pad=pad)
