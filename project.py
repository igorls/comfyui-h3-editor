"""An editing project: clips with rendered takes, and a timeline that only points at them.

Moving or trimming on the timeline never touches a render. No Comfy import.
"""

import copy
import json
import math
import random
import re
import time
import uuid

try:
    from .prompting import compose_prompt, compose_reference_prompt
except ImportError:
    from prompting import compose_prompt, compose_reference_prompt

FPS = 24
JOINT = 51
ROOT = "h3take_projects"
QUALITIES = {
    "draft": {},
    # A final renders at the final size, H3's own size of about a megapixel. Its picture is
    # then upscaled to the refine size, where the model redraws its last steps. The Realism
    # Slider skips early structure steps on the first pass (start_sigma 0.75) and runs across
    # the whole refine pass at this strength.
    "final": {"refine": True, "lora": 1.2, "lora_start_sigma": 0.75},
}
# Picture sizes per project format. Draft is for trying things, final for the cut, and
# refine is the size a final comes out at.
FORMATS = {
    "16:9": {"draft": (640, 352), "final": (1344, 768), "refine": (1920, 1088)},
    "9:16": {"draft": (352, 640), "final": (768, 1344), "refine": (1088, 1920)},
    "1:1": {"draft": (480, 480), "final": (1024, 1024), "refine": (1440, 1440)},
}
# Sizes a project can ask for instead of the usual ones, for a GPU with less memory or for
# speed. A draft is small unless the project asks for medium. A final is full unless it asks
# for something else: full renders large and is then enlarged to the refine size by redrawing
# its last steps there, sharp renders large and reaches the refine size through a video
# upscaler instead, and the others come out as sampled.
SIZES = {
    "16:9": {"small": (640, 352), "medium": (960, 544), "large": (1344, 768)},
    "9:16": {"small": (352, 640), "medium": (544, 960), "large": (768, 1344)},
    "1:1": {"small": (480, 480), "medium": (768, 768), "large": (1024, 1024)},
}
SIZE_CHOICES = {"draft": ("small", "medium"), "final": ("full", "sharp", "large", "medium", "small")}
# A sharp final is enlarged two times by this upscaler, keeping all the detail it adds, and
# scaled down to the refine size. H3 is not asked to draw at a size it was not made for, so
# skin stays as it is at H3's own size, and the edges come out crisper than a redraw gives.
SHARP = "sharp"
SHARP_UPSCALER = "relayvsr"
SHARP_DETAIL = 1.0
LANGUAGE = "English"
# The range a recorded picture side must fall in before export trusts it.
MIN_SIDE, MAX_SIDE = 16, 8192


def frame_size(project, quality):
    """Width and height for a quality in this project's format. Imported footage counts as final."""
    sizes = FORMATS.get(project.get("format") or "16:9", FORMATS["16:9"])
    return sizes["draft" if quality == "draft" else "final"]


def render_size(project, quality):
    """Width and height a new render of this quality is sampled at, and whether it is then enlarged.

    The project's own choice under "sizes" where it makes one, or the usual size for its format.
    """
    quality = "draft" if quality == "draft" else "final"
    shape = project.get("format") if project.get("format") in FORMATS else "16:9"
    name = size_name(project, quality)
    if name == "full":
        return FORMATS[shape]["final"] + (True,)
    if name == SHARP:
        return FORMATS[shape]["final"] + (False,)
    return SIZES[shape][name] + (False,)


def size_name(project, quality):
    """The name of the size a new render of this quality uses: the project's choice, or the usual one."""
    quality = "draft" if quality == "draft" else "final"
    chosen = project.get("sizes")
    name = chosen.get(quality) if isinstance(chosen, dict) else None
    return name if name in SIZE_CHOICES[quality] else SIZE_CHOICES[quality][0]


def size_choices():
    """Every size a project can pick, per format and quality, for the editor: id, width, height, and the size it comes out at."""
    table = {}
    for shape in FORMATS:
        table[shape] = {}
        for quality, names in SIZE_CHOICES.items():
            rows = []
            for name in names:
                width, height, refined = render_size({"format": shape, "sizes": {quality: name}}, quality)
                rows.append({"id": name, "width": width, "height": height,
                             "out": list(FORMATS[shape]["refine"] if refined or name == SHARP else (width, height))})
            table[shape][quality] = rows
    return table


def take_size(project, take):
    """A take's picture size as two whole numbers in range, since it ends up in an ffmpeg filter."""
    try:
        if take.get("size"):
            width, height = (int(side) for side in take["size"])
        else:
            # Takes from before refined finals record how much they were upscaled.
            scale = int(take.get("upscale") or 1)
            width, height = (side * scale for side in frame_size(project, take.get("quality")))
    except (TypeError, ValueError):
        raise ValueError("take %s records a picture size that is not two numbers" % take.get("id", ""))
    if not (MIN_SIDE <= width <= MAX_SIDE and MIN_SIDE <= height <= MAX_SIDE):
        raise ValueError("take %s records a picture size of %sx%s, outside %s to %s"
                         % (take.get("id", ""), width, height, MIN_SIDE, MAX_SIDE))
    return width, height


def refine_size(project):
    """Width and height a final comes out at in this project's format."""
    return FORMATS.get(project.get("format") or "16:9", FORMATS["16:9"])["refine"]
# Renders use HyperFlow H3 Flash: its ref2va model when a clip has references, and its
# fl2va model when a clip is only a prompt. Its INT8 safetensors and GGUF files load as
# HyperFlow (see is_hyperflow). A final's picture is enlarged by MiniMax's learned latent
# upscaler and refined at that size, with the MMH3 Realism Slider LoRA on the model.
REALISM_LORAS = ("MMH3-Realism-V2.safetensors", "MMH3-RealismSlider-V1.safetensors")
MODELS = {
    "unet": "hyperflow/minimax_h3_ref2va_hyperflow_flash_14b-int8_convrot.safetensors",
    "unet_text": "hyperflow/minimax_h3_fl2va_hyperflow_flash_14b-int8_convrot.safetensors",
    "clip": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "vae": "minimax_h3_video_vae_fp16.safetensors",
    "audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
    "latent_upscaler": "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors",
    "lora": REALISM_LORAS[0],
}
# The last steps a final redraws at the refine size, for families whose own schedule has no
# good place to start them: HyperFlow's 6-step tail from its fourth step.
REFINE_TAIL = "0.675, 0.385, 0.159, 0.0"
# Checkpoint profiles, one per family of H3 checkpoints. A project picks one under "profile",
# and a clip can pick its own the same way.
# unet names the ref2va file and unet_text the fl2va file; each is a list of names the file
# may have, and a name ending in * matches any file it starts. A family with no fl2va renders
# prompt-only clips on its ref2va. A HyperFlow file (see is_hyperflow) loads with the pack's
# loader and its own sigmas; any other file loads with ComfyUI's loader and the simple
# scheduler. lora is a LoRA the family always samples with. A final's second pass starts at
# refine_from in its own sigmas, or runs refine_sigmas.
PROFILES = {
    "hyperflow": {
        "title": "HyperFlow H3 Flash",
        "hint": "The fastest. Distilled to 6 steps.",
        # The INT8 convrot safetensors first, in a hyperflow folder and then anywhere. Without
        # it, the Q8_0 GGUF in a hyperflow folder, then any other quantization, anywhere, in
        # name order. The first name is the one the editor asks for when nothing is installed.
        "unet": ["hyperflow/minimax_h3_ref2va_hyperflow_flash_14b-int8_convrot.safetensors",
                 "minimax_h3_ref2va_hyperflow_flash_14b-int8_convrot*",
                 "hyperflow/minimax_h3_ref2va_hyperflow_flash_14b-Q8_0.gguf",
                 "minimax_h3_ref2va_hyperflow_flash_14b-*"],
        "unet_text": ["hyperflow/minimax_h3_fl2va_hyperflow_flash_14b-int8_convrot.safetensors",
                      "minimax_h3_fl2va_hyperflow_flash_14b-int8_convrot*",
                      "hyperflow/minimax_h3_fl2va_hyperflow_flash_14b-Q8_0.gguf",
                      "minimax_h3_fl2va_hyperflow_flash_14b-*"],
        "steps": {"draft": 4, "final": 6}, "sampler": "euler", "refine_from": 3,
    },
    "h3_turbo": {
        "title": "MiniMax H3 Turbo",
        "hint": "The original H3 with a 6-step Turbo LoRA. About as fast as HyperFlow.",
        "unet": ["minimax_h3_ref2va_pruned_int8_convrot.safetensors"],
        "unet_text": ["minimax_h3_fl2va_pruned_int8_convrot.safetensors"],
        "lora": ["minimax_h3_turbo_v4_step600_ema.safetensors", 1.0],
        "steps": {"draft": 6, "final": 6}, "sampler": "euler", "refine_sigmas": REFINE_TAIL,
    },
    "h3": {
        "title": "MiniMax H3",
        "hint": "The original H3 at 20 steps. About twice as slow as HyperFlow.",
        "unet": ["minimax_h3_ref2va_pruned_int8_convrot.safetensors"],
        "unet_text": ["minimax_h3_fl2va_pruned_int8_convrot.safetensors"],
        "steps": {"draft": 10, "final": 20}, "sampler": "res_multistep", "refine_from": 17,
    },
    "singularity": {
        "title": "Singularity v2",
        "hint": "A sharp, expressive H3 fine-tune, ref2va only. 20 steps, about twice as slow.",
        "unet": ["Minimax-h3_Singularity_ref2va_v2_pruned_int8.safetensors"],
        "steps": {"draft": 10, "final": 20}, "sampler": "res_multistep", "refine_from": 17,
    },
    "dasiwa": {
        "title": "DaSiWa Hybrid v4",
        "hint": "A community H3 checkpoint, ref2va only. 20 steps, about twice as slow.",
        "unet": ["dasiwa_minimax_h3_ref2va_v4_pruned_int8.safetensors", "DasiwaMinimaxH3_dasiwaHybridV4_*"],
        "steps": {"draft": 10, "final": 20}, "sampler": "res_multistep", "refine_from": 17,
    },
    "dasiwa_turbo": {
        "title": "DaSiWa Hybrid v4 Turbo",
        "hint": "Its distilled version, 8 steps.",
        "unet": ["dasiwa_minimax_h3_ref2va_v4_pruned_turbo_int8.safetensors", "DasiwaMinimaxH3_dasiwaHybridV4Turbo*"],
        "steps": {"draft": 8, "final": 8}, "sampler": "euler", "refine_sigmas": REFINE_TAIL,
    },
    "dasiwa_v5": {
        "title": "DaSiWa GXS v5 Turbo",
        "hint": "A merged community H3 checkpoint with its turbo built in, ref2va only. 6 steps for drafts, 8 for finals.",
        "unet": ["dasiwa_minimax_h3_ref2va_v5xs_pruned_graviton_turbo_int8_row-wise_convrot_runtime_mixed.safetensors",
                 "dasiwa_minimax_h3_ref2va_v5xs_pruned_graviton_turbo*"],
        "steps": {"draft": 6, "final": 8}, "sampler": "euler", "refine_sigmas": REFINE_TAIL,
    },
    "eros": {
        "title": "10Eros Max beta 7",
        "hint": "A community H3 merge with an 8-step turbo built in, one file for every clip. 6 steps for drafts, 8 for finals.",
        "unet": ["10Eros_Max_h3_8stepTURBO_beta7_fix1-native_int8_protected_edges_HQ.safetensors", "10Eros_Max_h3_8stepTURBO_beta7*"],
        "steps": {"draft": 6, "final": 8}, "sampler": "euler", "refine_sigmas": REFINE_TAIL,
    },
    "prism": {
        "title": "PRISM (Singularity x 10Eros Max)",
        "hint": "A merge of two community H3 fine-tunes with its turbo built in, ref2va only. For short single shots: "
                "sharp and glossy, with more movement, but speech is less exact than on HyperFlow and extensions drift. "
                "6 steps for drafts, 8 for finals.",
        "unet": ["PRISM_DMAD_PDMD_int8.safetensors", "PRISM_DMAD_PDMD*"],
        "steps": {"draft": 6, "final": 8}, "sampler": "euler", "scheduler": "beta57", "refine_sigmas": REFINE_TAIL,
    },
}
DEFAULT_PROFILE = "hyperflow"
# A scheduler the editor makes itself: the beta schedule with these two numbers, which other packs add to ComfyUI
# under this name. The editor builds it with ComfyUI's own node, so a checkpoint can ask for it on any install.
BETA57 = "beta57"
BETA57_SHAPE = (0.5, 0.7)


def sigmas_node(scheduler, steps):
    """The node that makes a render's sigmas for a scheduler by name."""
    if scheduler == BETA57:
        return {"class_type": "BetaSamplingScheduler", "inputs": {
            "model": ["5", 0], "steps": steps, "alpha": BETA57_SHAPE[0], "beta": BETA57_SHAPE[1]}}
    return {"class_type": "BasicScheduler", "inputs": {"model": ["5", 0], "scheduler": scheduler or "simple", "steps": steps, "denoise": 1.0}}
# A project's own checkpoints: how many, the family one starts from, and the steps it may take.
CHECKPOINT_LIMIT = 12
# Checkpoints saved as presets are shared by every project. A project keeps its own copy of the ones it uses.
PRESET_LIMIT = 64
CHECKPOINT_BASE = "h3"
STEP_RANGE = (1, 100)
CHECKPOINT_FIELDS = ("id", "title", "base", "unet", "unet_text", "steps", "sampler", "scheduler", "lora")
# Models that older projects name but the editor no longer renders with.
RETIRED = {"fasth3_omniref_qad_nvfp4_comfy.safetensors"}


def custom_profile(row):
    """A project's own checkpoint as a profile: the family it is based on, with its own file and sampling.

    It keeps the family's LoRA and the way its finals are refined, and takes its own steps,
    sampler and scheduler where it names them.
    """
    base = PROFILES.get(row.get("base")) or PROFILES[CHECKPOINT_BASE]
    profile = dict(base, title=str(row.get("title") or row["unet"]), unet=[row["unet"]], custom=True,
                   hint="Your own checkpoint, on the settings of %s." % base["title"])
    profile.pop("unet_text", None)
    if isinstance(row.get("unet_text"), str) and row["unet_text"]:
        profile["unet_text"] = [row["unet_text"]]
    steps = dict(base["steps"])
    for quality in steps:
        value = row["steps"].get(quality) if isinstance(row.get("steps"), dict) else None
        if isinstance(value, int) and not isinstance(value, bool) and STEP_RANGE[0] <= value <= STEP_RANGE[1]:
            steps[quality] = value
    profile["steps"] = steps
    for key in ("sampler", "scheduler"):
        if isinstance(row.get(key), str) and row[key]:
            profile[key] = row[key]
    if row.get("lora") is False:
        # A checkpoint with its turbo built in takes the family's settings without the family's LoRA.
        profile.pop("lora", None)
    if "refine_from" in base:
        # A final redraws as many steps at the larger size as the family it is based on.
        profile["refine_from"] = max(1, steps["final"] - (base["steps"]["final"] - base["refine_from"]))
    return profile


def project_profiles(project):
    """Every profile a project can render with, by id: the built-in ones, then its own checkpoints."""
    profiles = dict(PROFILES)
    for row in project.get("checkpoints") or []:
        if (isinstance(row, dict) and isinstance(row.get("id"), str) and row["id"] not in PROFILES
                and isinstance(row.get("unet"), str) and row["unet"]):
            profiles[row["id"]] = custom_profile(row)
    return profiles


def project_profile(project, available=None, clip=None):
    """The checkpoint profile a project renders with, or the one a clip of it picks, and its id.

    A project that names no profile takes the first one whose files are installed, so a fresh
    install renders with whatever it has. available as in pick_file.
    """
    profiles = project_profiles(project)
    name = (clip or {}).get("profile")
    if not isinstance(name, str) or name not in profiles:
        name = project.get("profile")
    if not isinstance(name, str) or name not in profiles:
        name = installed_profile(available)
    return name, profiles[name]


def installed_profile(available=None):
    """The first profile, in menu order, whose files are installed. The default when none is, or when unknown."""
    if available is not None:
        for row in profile_status(available):
            if row["available"]:
                return row["id"]
    return DEFAULT_PROFILE


def installed(name, available=None):
    """Whether a model file is installed. Unknown counts as installed, so graphs stay as they were."""
    return available is None or name in available


def realism_lora(available=None):
    """The installed Realism Slider file: V2 first, falling back to V1 when only V1 is installed."""
    return pick_file(REALISM_LORAS, available)


def finals_status(available):
    """What a final leaves out because its file is not installed, in words for the editor."""
    notes = []
    if not installed(MODELS["latent_upscaler"], available):
        notes.append("Finals stay at H3's own size, because %s is not installed in models/latent_upscale_models."
                     % MODELS["latent_upscaler"])
    if realism_lora(available) is None:
        notes.append("Finals render without the Realism Slider, because %s is not installed in models/loras."
                     % MODELS["lora"])
    return notes


def realism_strength(project, size):
    """How strong the Realism Slider is on a render of this size: the project's own setting, or the size's."""
    value = project.get("realism")
    if not size.get("lora") or isinstance(value, bool) or not isinstance(value, (int, float)):
        return size.get("lora") or 0
    return max(REALISM_RANGE[0], min(REALISM_RANGE[1], float(value)))


def project_loras(project, available=None, clip=None):
    """The LoRAs a render samples with, in order: (file, strength, start_sigma).

    The project's, then the clip's own. A clip's row for a file the project also lists takes
    that row's place, so one clip can run a project LoRA at another strength or without it.
    Only the ones that are switched on, installed and not at zero. A project file can be
    written by hand, so anything that is not a LoRA row is passed over.
    """
    listed = []
    for row in list(project.get("loras") or []) + list((clip or {}).get("loras") or []):
        if not isinstance(row, dict) or not isinstance(row.get("file"), str):
            continue
        same = [n for n, other in enumerate(listed) if other["file"].replace("\\", "/") == row["file"].replace("\\", "/")]
        if same:
            listed[same[0]] = row
        else:
            listed.append(row)
    rows = []
    for row in listed:
        strength, start = row.get("strength", 1.0), row.get("start_sigma", 1.0)
        if row.get("on") is False or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in (strength, start)):
            continue
        strength = max(LORA_RANGE[0], min(LORA_RANGE[1], float(strength)))
        file = row["file"] if available is None else pick_file([row["file"].replace("\\", "/")], available)
        if file and strength:
            rows.append((file, strength, max(0.0, min(1.0, float(start)))))
    return rows[:LORA_LIMIT]


def lora_status(available):
    """What the editor needs to offer LoRAs: the Realism Slider's file, its usual strength, and whether it is installed."""
    found = realism_lora(available)
    return {"file": found or MODELS["lora"], "strength": QUALITIES["final"]["lora"], "installed": found is not None,
            "range": list(REALISM_RANGE), "lora_range": list(LORA_RANGE), "limit": LORA_LIMIT}


def pick_file(names, available=None):
    """The first of a profile's file names that is installed, or the first name when nothing is known.

    available is the set of model files ComfyUI lists, or None when it is not known. The file
    comes back as ComfyUI lists it, which is what its loaders take: on Windows a file in a
    folder is listed with a backslash, where a profile writes a slash.
    """
    if not names:
        return None
    if available is None:
        return names[0]
    listed = {f.replace("\\", "/"): f for f in available}
    for name in names:
        if name.endswith("*"):
            stem = name[:-1]
            # In file name order, whatever folder each is in.
            found = sorted((f for f in listed if f.rsplit("/", 1)[-1].startswith(stem)),
                           key=lambda f: (f.rsplit("/", 1)[-1], f))
            if found:
                return listed[found[0]]
        elif name in listed:
            return listed[name]
    return None


def is_hyperflow(name):
    """Whether a unet file is a HyperFlow H3 Flash checkpoint, by its name.

    Every .gguf is one, and so is a .safetensors with hyperflow in its file name: the INT8
    convrot files. The other INT8 safetensors are plain H3 checkpoints for ComfyUI's loader.
    """
    base = str(name or "").replace("\\", "/").rsplit("/", 1)[-1].lower()
    return base.endswith(".gguf") or (base.endswith(".safetensors") and "hyperflow" in base)


# H3's text encoder and VAEs are published in several precisions. A render takes the first one
# installed, in this order, unless the project names another: the last name of each list matches
# any other precision. PROJECT_ENCODERS maps what set_project calls them to the keys used here.
MODEL_CHOICES = {
    "clip": ["qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
             "qwen3vl_32b_minimax_h3_bf16.safetensors", "qwen3vl_32b_minimax_h3*"],
    "vae": ["minimax_h3_video_vae_fp16.safetensors", "minimax_h3_video_vae_int8_convrot.safetensors", "minimax_h3_video_vae*"],
    "audio_vae": ["minimax_h3_audio_vae_fp32.safetensors", "minimax_h3_audio_vae*"],
}
PROJECT_ENCODERS = {"text_encoder": "clip", "vae": "vae", "audio_vae": "audio_vae"}

# A Qwen3-VL 4B or 8B can stand in for the 32B text encoder through a ClipProj projection in
# models/clip_projections (see clipproj.py). The size comes from the file name here, where only
# names are known; the node checks the file itself. The projection names its size the same way.
PROJECTION_FOLDER = "clip_projections"
PROJECTION_PREFIX = PROJECTION_FOLDER + ":"
PROJECTION_ORDER = ["mmh3-{size}-ClipProj-v3.1.safetensors", "mmh3-{size}-ClipProj-v3.1-mlp.safetensors",
                    "mmh3-{size}-ClipProj-v3-mlp.safetensors", "mmh3-{size}-ClipProj-v3.safetensors", "mmh3-{size}-*"]


def small_encoder_size(name):
    """The size of a Qwen3-VL 4B or 8B text encoder file, "4b" or "8b", else None. A text-only Qwen3 does not count."""
    base = str(name or "").replace("\\", "/").rsplit("/", 1)[-1].lower().replace("-", "_")
    if "minimax" in base or not ("qwen3vl" in base or "qwen3_vl" in base):
        return None
    for size in ("4b", "8b"):
        if "_%s" % size in base or "vl%s" % size in base:
            return size
    return None


def projections_for(encoder, available):
    """The installed projections for a small text encoder's size, the one a render takes by itself first."""
    size = small_encoder_size(encoder)
    if not size or available is None:
        return []
    found = sorted((name[len(PROJECTION_PREFIX):] for name in available if str(name).startswith(PROJECTION_PREFIX)), key=str.lower)
    ordered = []
    for pattern in PROJECTION_ORDER:
        pattern = pattern.format(size=size).lower()
        for name in found:
            base = name.replace("\\", "/").rsplit("/", 1)[-1].lower()
            if name not in ordered and (base == pattern or (pattern.endswith("*") and base.startswith(pattern[:-1]))):
                ordered.append(name)
    return ordered


def projection_for(encoder, available, chosen=None):
    """The projection a small text encoder renders through: the chosen one when it is installed and fits, else
    the first installed for its size, or None when there is none."""
    listed = projections_for(encoder, available)
    if chosen:
        same = [name for name in listed if name.replace("\\", "/") == str(chosen).replace("\\", "/")]
        if same:
            return same[0]
    return listed[0] if listed else None



def encoder_status(available):
    """What the editor needs to offer a text encoder and a video VAE: the one a render takes by itself, and what is missing."""
    found = {name: pick_file(MODEL_CHOICES[key], available) for name, key in PROJECT_ENCODERS.items()}
    where = {"text_encoder": "models/text_encoders", "vae": "models/vae", "audio_vae": "models/vae"}
    missing = ["No MiniMax H3 %s is installed. Put %s in %s." % (name.replace("_", " "), MODEL_CHOICES[key][0], where[name])
               for name, key in PROJECT_ENCODERS.items() if found[name] is None]
    return {"auto": found, "missing": missing}


def text_encoder_node(encoder, available, chosen=None):
    """The graph node that loads the text encoder: ComfyUI's loader for the 32B, the pack's for a small one."""
    projection = projection_for(encoder, available, chosen)
    if projection:
        return {"class_type": "H3SmallTextEncoder", "inputs": {"clip_name": encoder, "projection": projection}}
    return {"class_type": "CLIPLoader", "inputs": {"clip_name": encoder, "type": "minimax", "device": "default"}}


def project_models(project, available=None, clip=None):
    """The models a render uses: the profile's, with any the project names, except retired ones."""
    _, profile = project_profile(project, available, clip)
    models = dict(MODELS)
    for key, names in MODEL_CHOICES.items():
        models[key] = pick_file(names, available) or names[0]
    models["unet"] = pick_file(profile["unet"], available) or profile["unet"][0]
    models["unet_text"] = pick_file(profile.get("unet_text"), available)
    models["lora"] = realism_lora(available) or MODELS["lora"]
    if profile.get("lora"):
        models["profile_lora"] = pick_file([profile["lora"][0]], available) or profile["lora"][0]
    named = {key: name for key, name in (project.get("models") or {}).items() if name and name not in RETIRED}
    for key in MODEL_CHOICES:
        # A text encoder or VAE the project names but this machine lacks gives way to an installed one.
        if key in named and isinstance(named[key], str) and available is not None:
            named[key] = pick_file([named[key].replace("\\", "/")], available)
            # A 4B or 8B encoder needs its projection; without one it gives way too.
            if named[key] is None or (key == "clip" and small_encoder_size(named[key]) and not projection_for(named[key], available)):
                del named[key]
    if "unet" in named and "unet_text" not in named:
        # A project that names its own model renders every clip on it.
        models["unet_text"] = None
    models.update(named)
    return models


def profile_status(available):
    """Every profile, with what it needs that is not installed. available as in pick_file."""
    rows = []
    for name, profile in PROFILES.items():
        missing = []
        if pick_file(profile["unet"], available) is None:
            missing.append(profile["unet"][0])
        if profile.get("lora") and pick_file([profile["lora"][0]], available) is None:
            missing.append(profile["lora"][0])
        rows.append({"id": name, "title": profile["title"], "hint": profile["hint"],
                     "available": not missing, "missing": missing, "steps": profile["steps"],
                     "sampler": profile["sampler"], "scheduler": profile.get("scheduler") or "simple",
                     "unet": pick_file(profile["unet"], available) or profile["unet"][0],
                     "unet_text": pick_file(profile.get("unet_text"), available),
                     "lora": profile["lora"][0] if profile.get("lora") else None})
    return rows


def slug(name):
    """A safe folder name. Refuses anything that would leave the projects folder."""
    cleaned = re.sub(r"[^A-Za-z0-9 _-]", "", str(name or "")).strip().replace(" ", "_")
    if not cleaned:
        raise ValueError("a project needs a name with letters or numbers")
    return cleaned[:64]


def clean_title(name):
    """A project's name as it is shown. It has to hold letters or numbers, like the folder made from it."""
    title = str(name or "").strip()[:64]
    slug(title)
    return title


def copy_name(title, taken):
    """A name for a duplicate that no project has yet: Title copy, Title copy 2 and so on. taken holds folder names."""
    used = {str(name).lower() for name in taken}
    number = 1
    while True:
        name = "%s copy" % title if number == 1 else "%s copy %s" % (title, number)
        if slug(name).lower() not in used:
            return name
        number += 1


def retarget(value, old, new):
    """A project's data with every path inside one project folder pointed at another, for a duplicate."""
    before, after = "%s/%s" % (ROOT, old), "%s/%s" % (ROOT, new)
    if isinstance(value, dict):
        return {key: retarget(inner, old, new) for key, inner in value.items()}
    if isinstance(value, list):
        return [retarget(inner, old, new) for inner in value]
    if isinstance(value, str):
        path = value.replace("\\", "/")
        if path == before or path.startswith(before + "/"):
            return after + path[len(before):]
    return value


def new_project(name):
    return {
        "name": str(name),
        "style": "Live-action, cinematic",
        "shift": 12.0,
        "clips": [],
        "timeline": [],
    }


def snap_frames(seconds):
    """Nearest legal H3 length, at least about a second."""
    frames = float(seconds) * FPS
    return 5 + 17 * max(1, int(round((frames - 5) / 17.0)))


def linked_length(link, seconds):
    """Take length for a linked clip, so every hold starts on a shared video and audio boundary."""
    hold = int(link["hold"])
    frames = float(seconds) * FPS
    if link["mode"] == "bridge":
        return hold + JOINT * max(2, int(round((frames + hold) / JOINT)))
    return hold + JOINT * max(1, int(round(frames / JOINT)))


def source_range(side, hold, tail):
    """The held part of a source clip, from what is actually on the timeline."""
    if tail:
        return max(side["first"], side["last"] - hold), side["last"]
    return side["first"], side["first"] + hold


def load_node(side, hold, tail):
    first, last = source_range(side, hold, tail)
    file = "%s/%s" % (side["subfolder"], side["file"]) if side.get("subfolder") else side["file"]
    return {"class_type": "H3LoadClip", "inputs": {"file": file, "first": first, "last": last}}


def clip_cast(project, clip):
    """The cast members pictured in a clip. A voice with no picture is never a subject."""
    chosen = clip.get("cast") or []
    return [c for c in project.get("cast", []) if c["id"] in chosen and c.get("file")]


def member_pictures(member):
    """A cast member's pictures as assets: the main one first, then the others, up to CAST_PICTURES."""
    if not member.get("file"):
        return []
    others = [p for p in member.get("pictures") or [] if isinstance(p, dict) and isinstance(p.get("file"), str) and p["file"]]
    return ([member] + others)[:CAST_PICTURES]


def cast_pictures(project, clip):
    """Each pictured cast member of a clip, with the pictures it brings to the shot.

    H3 takes H3_PICTURES pictures in all, pinned first and last frames included. Every member
    brings its main picture, and the room left is dealt out one more picture per member in turn,
    so one member with many pictures never crowds out another.
    """
    members = clip_cast(project, clip)
    have = [member_pictures(member) for member in members]
    room = H3_PICTURES - bool(frame_of(clip, "first")) - bool(frame_of(clip, "last"))
    taken = [0] * len(members)
    for _ in range(CAST_PICTURES):
        for index, pictures in enumerate(have):
            if taken[index] < len(pictures) and sum(taken) < room:
                taken[index] += 1
    return [(member, have[index][:taken[index]]) for index, member in enumerate(members) if taken[index]]


def clip_refmods(project, clip, available=None):
    """The RefMods a clip renders with, as (member, file, strength): one for each chosen cast member that has one.

    available holds refmod:<name> for every RefMod the pack offers, so a RefMod is left out when
    the pack or the file is missing. Unknown counts as installed, as for models.
    """
    chosen = clip.get("cast") or []
    rows = []
    for member in project.get("cast", []):
        mod = member.get("refmod")
        if member["id"] not in chosen or not isinstance(mod, dict) or not isinstance(mod.get("file"), str) or not mod["file"]:
            continue
        strength = mod.get("strength", 1.0)
        if isinstance(strength, bool) or not isinstance(strength, (int, float)):
            strength = 1.0
        strength = max(0.0, min(1.0, float(strength)))
        if strength and (available is None or REFMOD_PREFIX + mod["file"] in available):
            rows.append((member, mod["file"], strength))
    return rows[:REFMOD_SLOTS]


# How the RefMod pack stores a reference: pooled down, its default, or the whole encode.
REFMOD_EXTRACT = "MiniMaxH3RefModExtract"
REFMOD_MODES = {
    "compressed": {"extraction_preset": "manual", "mode": "Compressed Reference", "max_tokens": 5120},
    "whole": {"extraction_preset": "identity_encode", "mode": "Full Reference", "max_tokens": 8192},
}


def refmod_graph(project, cast_id, name=None, mode="compressed", available=None):
    """The name a RefMod of a cast member is saved under, and the graph that makes it from the member's pictures.

    It runs the RefMod pack's own extract node, so it needs that pack. Compressed is quick to
    render with; whole keeps more of the pictures and makes drafts several times slower.
    """
    person = find_cast(project, cast_id)
    pictures = member_pictures(person)
    if not pictures:
        raise ValueError("%s has no picture to make a RefMod from" % person["name"])
    if mode not in REFMOD_MODES:
        raise ValueError("mode must be one of %s" % ", ".join(REFMOD_MODES))
    name = re.sub(r"[^A-Za-z0-9_-]", "", str(name or person["name"]).strip().replace(" ", "_")).lower()[:48]
    if not name:
        raise ValueError("a RefMod needs a name with letters or numbers")
    settings = REFMOD_MODES[mode]
    graph = {"1": {"class_type": "VAELoader", "inputs": {"vae_name": project_models(project, available)["vae"]}}}
    extract = {"name": name, "mode": settings["mode"], "concept_type": "identity", "background_retention": 0.0,
               "ref_resolution": 1024, "pool_h": 16, "pool_w": 16, "latent_frames": 16, "identity": 500, "merge": False,
               "motion_only": False, "multiplier": 1, "max_tokens": settings["max_tokens"],
               "description": str(person.get("description") or ""), "save": True, "vae": ["1", 0],
               "extraction_preset": settings["extraction_preset"], "budget_policy": "truncate"}
    for index, asset in enumerate(pictures):
        node = str(10 + index)
        graph[node] = still_node(asset)
        extract["refs_image.ref_image_%s" % index] = [node, 0]
    graph["2"] = {"class_type": REFMOD_EXTRACT, "inputs": extract}
    return name, graph


def name_unpictured(text, members):
    """Write out cast members who have no picture: @Name becomes their name and, the first time, what they look like.

    A member with a picture is a numbered subject of the prompt. One that is only a RefMod has
    no number to point at, so the words carry who they are.
    """
    for member in sorted(members, key=lambda m: -len(m["name"])):
        who = str(member.get("description") or "").strip().rstrip(".,")
        seen = []

        def swap(match, member=member, who=who, seen=seen):
            first = not seen
            seen.append(1)
            return "%s, %s," % (member["name"], who) if first and who else member["name"]

        text = re.sub("@" + re.escape(member["name"]), swap, text, flags=re.IGNORECASE)
    return text


def clip_language(project, clip):
    return (clip.get("language") or project.get("language") or LANGUAGE).strip()


def clip_voice(project, clip):
    """The cast member whose voice the speaker uses: named with @ in Who speaks, and holding a voice.

    A clip that holds a track already has its sound, so it never takes a voice reference.
    """
    if not str(clip.get("dialogue", "")).strip() or clip.get("track"):
        return None
    speaker = str(clip.get("speaker", "")).lower()
    named = [p for p in project.get("cast", []) if p.get("voice") and "@" + p["name"].lower() in speaker]
    return max(named, key=lambda p: len(p["name"])) if named else None


def speaker_text(project, clip):
    """Who speaks, with any unseen cast member written out, since only pictured ones become subjects.

    The speaker ID goes right after the identity it belongs to, before anything like "off-screen".
    """
    text = str(clip.get("speaker", ""))
    pictured = {p["id"] for p in clip_cast(project, clip)}
    marked = []
    for person in sorted(project.get("cast", []), key=lambda p: -len(p["name"])):
        if person["id"] in pictured:
            continue
        words = (person.get("description") or person["name"]).strip().rstrip(".")

        def put(match, words=words):
            marked.append(True)
            return words + (" (S1)" if len(marked) == 1 else "")

        text = re.sub("@" + re.escape(person["name"]), put, text, flags=re.IGNORECASE)
    return text


def frame_of(clip, side):
    """A first or last frame. A clip that holds real frames there has no use for one."""
    link = clip.get("link")
    if link and (link["mode"] == "bridge" or (side == "first") == (link["mode"] == "extend_after")):
        return None
    return clip.get(side)


def has_references(project, clip):
    """Does the clip show a cast picture, pin a first or last frame, or speak with a cast voice."""
    return bool(clip_cast(project, clip) or frame_of(clip, "first") or frame_of(clip, "last")
                or clip_voice(project, clip))


def clip_prompt(project, clip):
    fields = {
        "camera": clip.get("camera", ""),
        "speaker": speaker_text(project, clip),
        "dialogue": clip.get("dialogue", ""),
        "sound": clip.get("sound", ""),
        "music": clip.get("music", ""),
        "language": clip_language(project, clip),
    }
    shown = cast_pictures(project, clip)
    cast = [dict(member, picture_count=len(pictures)) for member, pictures in shown]
    chosen = clip.get("cast") or []
    action = name_unpictured(clip.get("action", ""), [m for m in project.get("cast", [])
                                                      if m["id"] in chosen and m.get("refmod") and not m.get("file")])
    voice = clip_voice(project, clip)
    if has_references(project, clip):
        owner = None
        if voice:
            number = next((n for n, c in enumerate(cast, 1) if c["id"] == voice["id"]), None)
            owner = {"subject": number} if number else {"description": voice.get("description") or voice["name"]}
        return compose_reference_prompt(
            project.get("style", ""), action, subjects=cast,
            first_frame=bool(frame_of(clip, "first")), last_frame=bool(frame_of(clip, "last")), voice=owner, **fields,
        )
    return compose_prompt(project.get("style", ""), action, **fields)


def still_node(asset):
    file = "%s/%s" % (asset["subfolder"], asset["file"]) if asset.get("subfolder") else asset["file"]
    return {"class_type": "H3LoadStill", "inputs": {"file": file}}


def render_graph(project, folder, clip, take, quality, seed, available=None):
    """One clip as a ComfyUI graph, saved as its own file in the project.

    available is the set of model files ComfyUI lists, used to find a profile's files.
    """
    if clip.get("kind") == "footage":
        raise ValueError("%s is imported footage. Extend or bridge it to generate from it" % clip.get("name", "this clip"))
    if not str(clip.get("action", "")).strip():
        raise ValueError("describe what happens in this clip before rendering it")
    if quality not in QUALITIES:
        raise ValueError("quality must be one of %s" % (tuple(QUALITIES),))
    width, height, enlarged = render_size(project, quality)
    _, profile = project_profile(project, available, clip)
    size = dict(QUALITIES[quality], width=width, height=height, steps=profile["steps"][quality])
    if not enlarged:
        # The project asked for finals at the size they are sampled at.
        size.pop("refine", None)
    models = project_models(project, available, clip)
    references = has_references(project, clip)
    # A family's fl2va model takes a prompt alone. Anything with references goes to its ref2va model.
    unet = models["unet"] if references or not models.get("unet_text") else models["unet_text"]
    hyperflow = is_hyperflow(unet)
    link = clip.get("link")
    length = linked_length(link, clip.get("seconds", 5)) if link else snap_frames(clip.get("seconds", 5))
    graph = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "2": text_encoder_node(models["clip"], available, models.get("projection")),
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": models["vae"]}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": models["audio_vae"]}},
        "5": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["1", 0], "shift_video": project.get("shift", 12.0), "shift_audio": 3.0}},
        "10": {"class_type": "H3Take", "inputs": {
            "clip": ["2", 0], "vae": ["3", 0], "audio_vae": ["4", 0],
            "prompt": clip_prompt(project, clip), "width": size["width"], "height": size["height"],
            "length": length, "checkpoint": "ref2va", "anchor": "none", "mid_frames": "",
            "video_mode": "off", "hold_frames": 39, "strength": "hold", "keep": 0.0, "align": "av", "video_feather": 0,
            "audio_hold": "off", "audio_feather": 8, "ref_image_size": "match"}},
        "9": {"class_type": "H3LivePreview", "inputs": {
            "model": ["5", 0], "audio_vae": ["4", 0], "folder": "%s/%s/preview" % (ROOT, folder)}},
        "11": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": profile["sampler"]}},
        "12": sigmas_node(profile.get("scheduler"), size["steps"]),
        "13": {"class_type": "BasicGuider", "inputs": {"model": ["9", 0], "conditioning": ["10", 0]}},
        "14": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}},
        "15": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["14", 0], "guider": ["13", 0], "sampler": ["11", 0], "sigmas": ["12", 0], "latent_image": ["10", 1]}},
        "16": {"class_type": "VAEDecode", "inputs": {"samples": ["15", 0], "vae": ["3", 0]}},
        "17": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["15", 0], "vae": ["4", 0]}},
        "18": {"class_type": "CreateVideo", "inputs": {"images": ["16", 0], "fps": float(FPS), "audio": ["17", 0]}},
        "19": {"class_type": "SaveVideo", "inputs": {
            "video": ["18", 0], "filename_prefix": "%s/%s/clips/%s_%s" % (ROOT, folder, clip["id"], take),
            "format": "auto", "codec": "auto"}},
        # The take's first and last frames as sampled, for the clips that will extend or bridge it.
        "20": {"class_type": "H3SaveHold", "inputs": {
            "latent": ["15", 0], "file": "%s/%s/clips/%s_%s.hold.safetensors" % (ROOT, folder, clip["id"], take)}},
    }
    if unet == models.get("unet_text"):
        graph["10"]["inputs"]["checkpoint"] = "fl2va"
    if hyperflow:
        # Few-step and CFG-free, on its own two-time schedule.
        graph["1"] = {"class_type": "H3HyperFlowLoader", "inputs": {"unet_name": unet, "weights_in": "auto"}}
        graph["12"] = {"class_type": "H3HyperFlowSigmas", "inputs": {"model": ["5", 0], "steps": size["steps"]}}
    model = ["1", 0]
    if profile.get("lora") and models.get("profile_lora"):
        # A LoRA the family always samples with, such as a turbo LoRA.
        graph["94"] = {"class_type": "H3Lora", "inputs": {"model": model, "lora_name": models["profile_lora"],
                                                          "strength": profile["lora"][1]}}
        model = ["94", 0]
    realism = realism_strength(project, size)
    if realism and models.get("lora") and installed(models["lora"], available):
        # The Realism Slider, on the model both passes sample with. Without it a final still renders.
        lora_inputs = {"model": model, "lora_name": models["lora"], "strength": realism}
        if size.get("lora_start_sigma") is not None:
            lora_inputs["start_sigma"] = size["lora_start_sigma"]
        graph["95"] = {"class_type": "H3Lora", "inputs": lora_inputs}
        model = ["95", 0]
    for n, (file, strength, start) in enumerate(project_loras(project, available, clip)):
        # The project's own LoRAs and then the clip's, stacked in the order listed, on drafts and finals alike.
        node = str(LORA_NODE + n)
        graph[node] = {"class_type": "H3Lora", "inputs": {"model": model, "lora_name": file, "strength": strength}}
        if start < 1.0:
            # The LoRA sits out the first steps, which set the composition, and joins for the detail.
            graph[node]["inputs"]["start_sigma"] = start
        model = [node, 0]
    graph["5"]["inputs"]["model"] = model
    take_inputs = graph["10"]["inputs"]
    first, last = frame_of(clip, "first"), frame_of(clip, "last")
    if first:
        graph["40"] = still_node(first)
        take_inputs["start_image"] = ["40", 0]
    if last:
        graph["41"] = still_node(last)
        take_inputs["end_image"] = ["41", 0]
    take_inputs["anchor"] = "both" if first and last else "start" if first else "end" if last else "none"
    index = 0
    for _, pictures in cast_pictures(project, clip):
        # A member's pictures sit together, in the order the prompt numbers them.
        for asset in pictures:
            node = str(50 + index)
            graph[node] = still_node(asset)
            take_inputs["ref_images.ref_image_%s" % index] = [node, 0]
            index += 1
    voice = clip_voice(project, clip)
    if voice:
        sound = voice["voice"]
        file = "%s/%s" % (sound["subfolder"], sound["file"]) if sound.get("subfolder") else sound["file"]
        graph["60"] = {"class_type": "H3LoadSound", "inputs": {"file": file, "start": 0.0, "seconds": 30.0}}
        take_inputs["ref_audios.ref_audio_0"] = ["60", 0]
    if link:
        hold = int(link["hold"])
        graph["6"] = load_node(link["a"], hold, link["mode"] != "extend_before")
        take_inputs.update({
            "video_mode": link["mode"], "hold_frames": hold, "audio_hold": "span",
            "source_video": ["6", 0], "source_audio": ["6", 1], "source_hold": ["6", 2],
        })
        if link["mode"] == "bridge":
            graph["7"] = load_node(link["b"], hold, False)
            # A hold alone does not make the take arrive. The guide tells H3 what it must land on.
            graph["8"] = {"class_type": "ImageFromBatch", "inputs": {"image": ["7", 0], "batch_index": 0, "length": 1}}
            take_inputs.update({
                "end_video": ["7", 0], "end_audio": ["7", 1], "end_hold": ["7", 2],
                "mid_frames": str(length - hold), "mid_images.mid_image_0": ["8", 0],
            })
    track = clip.get("track")
    if track:
        # The picture is generated to this slice of the track, laid from the take's first frame.
        file = "%s/%s" % (track["subfolder"], track["file"]) if track.get("subfolder") else track["file"]
        graph["61"] = {"class_type": "H3LoadSound", "inputs": {
            "file": file, "start": float(track.get("from") or 0), "seconds": round(length / float(FPS) + 1.0, 3)}}
        take_inputs.update({"audio_hold": "full", "audio_align": "take", "source_audio": ["61", 0]})
        if link and link["mode"] == "bridge":
            take_inputs["end_audio"] = ["61", 0]
    if size.get("refine") and models.get("latent_upscaler") and installed(models["latent_upscaler"], available):
        # Without the latent upscaler a final stays at H3's own size.
        tail = profile.get("refine_sigmas") or profile["refine_from"]
        add_refine(graph, refine_size(project), tail, models["latent_upscaler"])
    if quality != "draft" and size_name(project, quality) == SHARP:
        add_sharp(graph, refine_size(project), available)
    mods = clip_refmods(project, clip, available)
    if mods:
        # Each chosen cast member's RefMod joins what H3 Take conditions on, in both passes of a final.
        loader = {"show_info": False}
        for slot in range(1, REFMOD_SLOTS + 1):
            _, file, strength = mods[slot - 1] if slot <= len(mods) else (None, "(none)", 1.0)
            loader.update({"mod_%s" % slot: file, "strength_%s" % slot: strength, "copies_%s" % slot: 1})
        graph["72"] = {"class_type": "MiniMaxH3RefModsLoader", "inputs": loader}
        guiders = sorted((key for key, node in graph.items() if node["class_type"] == "BasicGuider"), key=int)
        for number, key in enumerate(guiders):
            node = str(73 + number)
            graph[node] = {"class_type": "MiniMaxH3RefModApply", "inputs": {
                "conditioning": graph[key]["inputs"]["conditioning"], "mods": ["72", 0], "override": False, "retention": 1.0,
                "curve_direction": "constant", "scramble_seed": -1, "curve_shape": "linear", "curve_value": 1.0}}
            graph[key]["inputs"]["conditioning"] = [node, 0]
    return graph


def add_refine(graph, size, tail, upscaler):
    """Enlarge the sampled latent and redraw its last steps at that size, keeping the sound.

    A second take at the refine size gives the conditioning there (pinned frames and pictures
    at that size) and its held frames. The picture comes out at the refine size, and the sound
    is the first pass's. tail is the step in the final's own sigmas the redraw starts from, or
    the sigmas it runs, as text.
    """
    width, height = size
    take = copy.deepcopy(graph["10"])
    take["inputs"].update({"width": width, "height": height})
    graph.update({
        "80": take,
        "81": {"class_type": "LatentUpscaleModelLoader", "inputs": {"model_name": upscaler}},
        "82": {"class_type": "MiniMaxH3LatentUpscaleWithModel", "inputs": {
            "model": ["81", 0], "samples": ["15", 0], "width": width, "height": height}},
        "83": {"class_type": "H3RefineLatent", "inputs": {"upscaled": ["82", 0], "take": ["80", 1]}},
        "87": ({"class_type": "ManualSigmas", "inputs": {"sigmas": tail}} if isinstance(tail, str)
               else {"class_type": "SplitSigmas", "inputs": {"sigmas": ["12", 0], "step": tail}}),
        "88": {"class_type": "BasicGuider", "inputs": {"model": ["9", 0], "conditioning": ["80", 0]}},
        "90": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["14", 0], "guider": ["88", 0], "sampler": ["11", 0],
            "sigmas": ["87", 0 if isinstance(tail, str) else 1], "latent_image": ["83", 0]}},
        "91": {"class_type": "VAEDecode", "inputs": {"samples": ["90", 0], "vae": ["3", 0]}},
    })
    graph["18"]["inputs"]["images"] = ["91", 0]


def add_sharp(graph, size, available=None):
    """Enlarge the decoded picture two times with a video upscaler, then scale it down to the final's size.

    The picture is sampled once, at H3's own size. The sound is untouched.
    """
    _, files = pick_upscaler(SHARP_UPSCALER, available)
    width, height = size
    graph.update({
        "31": {"class_type": "UNETLoader", "inputs": {"unet_name": files["model"], "weight_dtype": "default"}},
        "32": {"class_type": "VAELoader", "inputs": {"vae_name": files["vae"]}},
        "33": {"class_type": "RelayVSRConditioning", "inputs": {"model": ["31", 0], "images": ["16", 0]}},
        "34": {"class_type": "KSampler", "inputs": {
            "model": ["31", 0], "seed": 0, "steps": 1, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
            "positive": ["33", 0], "negative": ["33", 0], "latent_image": ["33", 1], "denoise": 1.0}},
        "35": {"class_type": "VSRDecode", "inputs": {"vae": ["32", 0], "samples": ["34", 0], "images": ["16", 0], "detail": SHARP_DETAIL}},
        "36": {"class_type": "ImageScale", "inputs": {
            "image": ["35", 0], "upscale_method": "lanczos", "width": width, "height": height, "crop": "disabled"}},
    })
    graph["18"]["inputs"]["images"] = ["36", 0]


def active_take(clip):
    takes = clip.get("takes") or []
    for take in takes:
        if take.get("id") == clip.get("active"):
            return take
    return takes[-1] if takes else None


TRANSITIONS = {"dissolve": "fade", "black": "fadeblack", "wipe": "wipeleft"}


MIN_SPEED, MAX_SPEED = 0.25, 4.0


def clean_speed(value):
    """A timeline speed. The project file comes from the client, so anything else is refused."""
    speed = 1.0 if value in (None, "") else float(value)
    if not math.isfinite(speed) or not MIN_SPEED <= speed <= MAX_SPEED:
        raise ValueError("speed must be between %s and %s" % (MIN_SPEED, MAX_SPEED))
    return speed


def item_length(item):
    """Frames an item fills on the timeline, after its speed."""
    return max(1, int(round((item["out"] - item["in"]) / clean_speed(item.get("speed")))))


def tempo_chain(speed):
    """atempo only takes 0.5 to 2 at a time, so larger changes are chained."""
    speed = clean_speed(speed)
    steps = []
    while speed > 2.0:
        steps.append("atempo=2.0")
        speed /= 2.0
    while speed < 0.5:
        steps.append("atempo=0.5")
        speed /= 0.5
    if abs(speed - 1.0) > 1e-6:
        steps.append("atempo=%.5f" % speed)
    return ",".join(steps)


SOUND_FORMAT = "aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo"

# How hard an export is compressed: x264's quality number, where lower keeps more.
EXPORT_QUALITY = {"best": 12, "high": 16, "small": 23}
EXPORT_SCALES = (1.0, 0.75, 0.5)


def export_plan(project, scale=1.0):
    """What an export of the timeline will hold, before making it: its size and length, and what to know first.

    drafts counts the generated clips whose take on the timeline is a draft, missing the ones
    with no render (they are left out), and gaps the stretches of black between clips.
    """
    clips = {c["id"]: c for c in project.get("clips", [])}
    width = height = drafts = missing = 0
    for item in project.get("timeline", []):
        clip = clips.get(item.get("clip")) or {}
        take = active_take(clip)
        if not take:
            missing += 1
            continue
        if take.get("quality") == "draft" and clip.get("kind") != "footage":
            drafts += 1
        w, h = take_size(project, take)
        if w > width:
            width, height = w, h
    end = cursor = gaps = 0
    for item in ordered(project):
        if item["start"] > cursor:
            gaps += 1
        cursor = max(cursor, item_end(item))
        end = cursor
    effects = [e for e in project.get("effects") or [] if e["start"] < end]
    return {"width": scaled(width, scale), "height": scaled(height, scale), "frames": end, "seconds": round(end / float(FPS), 2),
            "clips": len(project.get("timeline", [])) - missing, "drafts": drafts, "missing": missing, "gaps": gaps,
            "effects": sum(1 for e in effects if active_take(e)),
            "effects_missing": sum(1 for e in effects if not active_take(e)),
            "effects_stale": sum(1 for e in effects if effect_stale(project, e))}


def scaled(side, scale):
    """A picture side at a part of its size, kept even, as video encoders need."""
    return max(2, 2 * int(round(side * scale / 2.0)))


def export_command(project, paths, out_path, width=None, height=None, scale=1.0, quality="high", sound_paths=None,
                   effect_paths=None):
    """ffmpeg arguments that join the timeline's rendered clips. Gaps become black and silence.

    paths maps a clip id to the file of its active take. Clips with no render
    are skipped. Speed and transitions live on the timeline and are applied here.
    sound_paths maps the id of an extra sound to its file; those are mixed under the cut,
    each from its place on the timeline and at its own gain, and stop where the cut ends.
    """
    clips = {c["id"]: c for c in project.get("clips", [])}
    items = sorted(
        (i for i in project.get("timeline", []) if i.get("clip") in paths),
        key=lambda i: i["start"],
    )
    if not items:
        raise ValueError("there are no rendered clips on the timeline to export")
    if width is None or height is None:
        # The cut takes the size of its largest picture, so upscaled finals keep their detail.
        width = height = 0
        for item in items:
            w, h = take_size(project, active_take(clips[item["clip"]]) or {})
            if w > width:
                width, height = w, h
    if scale not in EXPORT_SCALES:
        raise ValueError("scale must be one of %s" % ", ".join("%g" % s for s in EXPORT_SCALES))
    if quality not in EXPORT_QUALITY:
        raise ValueError("quality must be one of %s" % ", ".join(EXPORT_QUALITY))
    width, height = scaled(width, scale), scaled(height, scale)

    args = ["ffmpeg", "-y", "-v", "error"]
    chains = []
    finish = "fps=%s,scale=%s:%s:force_original_aspect_ratio=increase,crop=%s:%s,setsar=1,format=yuv420p,settb=1/%s" % (
        FPS, width, height, width, height, FPS)
    sound = SOUND_FORMAT

    def gap_segment(label, frames):
        seconds = frames / float(FPS)
        chains.append("color=c=black:s=%sx%s:r=%s:d=%.5f,format=yuv420p,setsar=1,settb=1/%s[%sv]" % (width, height, FPS, seconds, FPS, label))
        chains.append("anullsrc=r=48000:cl=stereo:d=%.5f,%s[%sa]" % (seconds, sound, label))

    video = audio = None
    cursor = 0
    step = 0

    def join(next_video, next_audio):
        nonlocal video, audio, step
        if video is None:
            video, audio = next_video, next_audio
            return
        chains.append("[%s][%s][%s][%s]concat=n=2:v=1:a=1[j%sv][j%sa]" % (video, audio, next_video, next_audio, step, step))
        chains.append("[j%sv]settb=1/%s[k%sv]" % (step, FPS, step))
        video, audio = "k%sv" % step, "j%sa" % step
        step += 1

    for index, item in enumerate(items):
        args += ["-i", paths[item["clip"]]]
        speed = clean_speed(item.get("speed"))
        first, last = int(item["in"]), int(item["out"])
        chains.append("[%s:v]trim=start_frame=%s:end_frame=%s,setpts=(PTS-STARTPTS)/%.5f,%s[v%s]" % (index, first, last, speed, finish, index))
        tempo = tempo_chain(speed)
        chains.append("[%s:a]atrim=start=%.5f:end=%.5f,asetpts=PTS-STARTPTS,%s%s[a%s]" % (
            index, first / float(FPS), last / float(FPS), tempo + "," if tempo else "", sound, index))

        overlap = cursor - item["start"]
        transition = item.get("transition")
        if transition and overlap > 0 and video is not None:
            seconds = overlap / float(FPS)
            chains.append("[%s][v%s]xfade=transition=%s:duration=%.5f:offset=%.5f[x%sv]" % (
                video, index, TRANSITIONS.get(transition.get("type"), "fade"), seconds, (cursor - overlap) / float(FPS), index))
            chains.append("[%s][a%s]acrossfade=d=%.5f[x%sa]" % (audio, index, seconds, index))
            video, audio = "x%sv" % index, "x%sa" % index
        else:
            gap = item["start"] - cursor
            if gap > 0:
                gap_segment("g%s" % index, gap)
                join("g%sv" % index, "g%sa" % index)
            join("v%s" % index, "a%s" % index)
        cursor = item["start"] + item_length(item)

    extra = []
    for row in project.get("sounds") or []:
        if row.get("id") not in (sound_paths or {}) or row.get("start", 0) >= cursor or not row.get("gain", 1.0):
            continue
        number = len(items) + len(extra)
        args += ["-i", sound_paths[row["id"]]]
        offset = float(row.get("offset") or 0.0)
        chains.append("[%s:a]atrim=start=%.5f:end=%.5f,asetpts=PTS-STARTPTS,volume=%.3f,%s,adelay=%d:all=1[s%s]" % (
            number, offset, offset + row["frames"] / float(FPS), float(row.get("gain", 1.0)), SOUND_FORMAT,
            int(round(row.get("start", 0) * 1000.0 / FPS)), number))
        extra.append("s%s" % number)
    if extra:
        # The cut sets the length: a sound that runs past its end is cut there.
        chains.append("[%s]%samix=inputs=%s:duration=first:dropout_transition=0:normalize=0[mix]" % (
            audio, "".join("[%s]" % label for label in extra), len(extra) + 1))
        audio = "mix"

    laid = 0
    for effect in sorted(project.get("effects") or [], key=lambda e: e["start"]):
        take = active_take(effect)
        if effect.get("id") not in (effect_paths or {}) or not take or effect["start"] >= cursor:
            continue
        # A VFX take is the finished picture for its span: it covers the cut there, and the sound below plays on.
        number = len(items) + len(extra) + laid
        args += ["-i", effect_paths[effect["id"]]]
        first = effect["start"] / float(FPS)
        last = min(cursor, effect["start"] + min(effect["frames"], int(take.get("frames") or effect["frames"])
                                                      - int(effect.get("skip") or 0))) / float(FPS)
        skip = int(effect.get("skip") or 0)
        trim = "trim=start_frame=%s,setpts=PTS-STARTPTS," % skip if skip else ""
        chains.append("[%s:v]%s%s,setpts=PTS-STARTPTS+%.5f/TB[e%s]" % (number, trim, finish, first, number))
        chains.append("[%s][e%s]overlay=eof_action=pass:enable='between(t,%.5f,%.5f)',settb=1/%s[o%s]" % (
            video, number, first, last - 0.5 / FPS, FPS, number))
        video = "o%s" % number
        laid += 1

    args += ["-filter_complex", ";".join(chains), "-map", "[%s]" % video, "-map", "[%s]" % audio,
             "-c:v", "libx264", "-crf", str(EXPORT_QUALITY[quality]), "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
             "-movflags", "+faststart", out_path]
    return args


RATING_FRAMES = 180 * FPS        # the longest stretch of the cut one rating looks at
RATINGS_KEPT = 40


def slice_project(project, first, last):
    """The part of the cut from frame first up to frame last, as a project of its own that starts at frame 0.

    Clips, sounds and VFX clips that cross an end are trimmed there. Used to export one range of the timeline.
    """
    first, last = int(first), int(last)
    if first < 0 or last <= first:
        raise ValueError("a range runs from a frame to a later one")
    out = copy.deepcopy(project)
    items = []
    for item in ordered(project):
        start, end = item["start"], item_end(item)
        if end <= first or start >= last:
            continue
        speed = clean_speed(item.get("speed"))
        head, tail = max(0, first - start), max(0, end - last)
        new = copy.deepcopy(item)
        new["in"] = item["in"] + int(round(head * speed))
        new["out"] = item["out"] - int(round(tail * speed))
        new["start"] = max(start, first) - first
        if head:
            new.pop("transition", None)      # what it came in over is outside the range
        if new["out"] - new["in"] >= 1:
            items.append(new)
    out["timeline"] = items
    sounds = []
    for sound in project.get("sounds") or []:
        start, end = sound.get("start", 0), sound.get("start", 0) + sound.get("frames", 0)
        if end <= first or start >= last:
            continue
        head, tail = max(0, first - start), max(0, end - last)
        new = copy.deepcopy(sound)
        new["offset"] = round(float(sound.get("offset") or 0.0) + head / float(FPS), 4)
        new["frames"] = sound["frames"] - head - tail
        new["start"] = max(start, first) - first
        if new["frames"] >= 1:
            sounds.append(new)
    out["sounds"] = sounds
    effects = []
    for effect in project.get("effects") or []:
        start, end = effect["start"], effect["start"] + effect["frames"]
        if end <= first or start >= last:
            continue
        head, tail = max(0, first - start), max(0, end - last)
        new = copy.deepcopy(effect)
        new["skip"] = head                   # frames of its take that fall before the range
        new["frames"] = effect["frames"] - head - tail
        new["start"] = max(start, first) - first
        if new["frames"] >= 1:
            effects.append(new)
    out["effects"] = effects
    return out


def add_rating(project, rating):
    """Keep a rating with the project, newest first, so a range can be compared with how it scored before."""
    rows = [dict(rating, id=new_id("r"))] + list(project.get("ratings") or [])
    project["ratings"] = rows[:RATINGS_KEPT]
    return project["ratings"][0]


# Extra sound. Music, narration or effects laid under the cut on lanes of their own, mixed in at export.

SOUND_LANES = 4
SOUND_FIELDS = ("name", "start", "offset", "frames", "gain", "lane")
SOUND_GAIN = (0.0, 2.0)


def find_sound(project, sound_id):
    for sound in project.get("sounds") or []:
        if sound["id"] == sound_id:
            return sound
    raise KeyError("no sound %s" % sound_id)


def clean_sound_fields(fields, seconds):
    """Where a sound sits and how it plays. seconds is the length of its file, which bounds what can be heard."""
    check_fields(fields, SOUND_FIELDS)
    out = {}
    if "name" in fields:
        if not isinstance(fields["name"], str) or not fields["name"].strip():
            raise ValueError("a sound's name must be text")
        out["name"] = fields["name"].strip()[:80]
    for key, what in (("start", "start is a timeline frame, 0 or later"), ("lane", "lane is a whole number from 0 to %s" % (SOUND_LANES - 1))):
        if key in fields:
            value = fields[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value) or value < 0 \
                    or (key == "lane" and value >= SOUND_LANES):
                raise ValueError(what)
            out[key] = int(value)
    if "gain" in fields:
        out["gain"] = clean_strength(fields["gain"], SOUND_GAIN, "gain")
    if "offset" in fields:
        offset = fields["offset"]
        if isinstance(offset, bool) or not isinstance(offset, (int, float)) or not 0 <= offset < seconds:
            raise ValueError("offset is the second of the file the sound starts from, from 0 to its length")
        out["offset"] = round(float(offset), 4)
    if "frames" in fields:
        frames = fields["frames"]
        if isinstance(frames, bool) or not isinstance(frames, (int, float)) or frames != int(frames) or frames < 1:
            raise ValueError("frames is how long the sound plays, in timeline frames")
        out["frames"] = int(frames)
    return out


def fit_sound(sound):
    """A sound never plays past the end of its file."""
    room = max(1, int(round((sound["seconds"] - sound.get("offset", 0.0)) * FPS)))
    sound["frames"] = max(1, min(int(sound.get("frames") or room), room))
    return sound


def add_sound(project, asset, seconds, fields=None):
    """Lay a sound file under the cut. asset is an uploaded sound, seconds its length."""
    asset = clean_asset(asset, "a sound")
    if asset is None:
        raise ValueError("a sound needs a file, and a subfolder if it has one")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not seconds > 0:
        raise ValueError("seconds is the length of the sound file")
    fields = clean_sound_fields(fields or {}, float(seconds))
    sound = {"id": new_id("s"), "name": "Sound %s" % (len(project.get("sounds") or []) + 1), "file": asset["file"],
             "seconds": round(float(seconds), 4), "start": 0, "offset": 0.0, "gain": 1.0, "lane": 0}
    if asset.get("subfolder"):
        sound["subfolder"] = asset["subfolder"]
    sound.update(fields)
    project.setdefault("sounds", []).append(fit_sound(sound))
    return sound


def update_sound(project, sound_id, fields):
    sound = find_sound(project, sound_id)
    sound.update(clean_sound_fields(fields, sound["seconds"]))
    return fit_sound(sound)


def remove_sound(project, sound_id):
    find_sound(project, sound_id)
    project["sounds"] = [s for s in project["sounds"] if s["id"] != sound_id]


# VFX. A clip drawn by code, on a lane of its own above the picture. Each one holds a small JavaScript module that
# draws every frame and can use the picture below it. The editor runs the module in the browser; a render bakes the
# finished picture for the clip's span into a take, which the export lays over the cut.

EFFECT_FIELDS = ("name", "start", "frames", "source", "params", "inputs", "data")
EFFECT_DATA_LIMIT = 2000000
EFFECT_INPUTS = 8
EFFECT_SOURCE_LIMIT = 200000
EFFECT_STARTER = """// A VFX clip is a module that draws each frame. frame(env, k) must depend only on k.
// env: canvas, width, height, fps, frames, params, THREE, below (the picture under this frame, or null)
export const meta = {
  controls: {
    text: { type: 'text', default: 'Title' },
    size: { type: 'number', min: 0.03, max: 0.3, step: 0.005, default: 0.09, label: 'Size' },
    color: { type: 'color', default: '#ece8df' },
  },
};

export function frame(env, k) {
  const c = env.canvas.getContext('2d');
  const { width: w, height: h } = env;
  c.clearRect(0, 0, w, h);
  if (env.below) c.drawImage(env.below, 0, 0, w, h);
  else { c.fillStyle = '#1b1e23'; c.fillRect(0, 0, w, h); }
  const t = Math.min(1, k / 12);                       // half a second in
  const ease = 1 - Math.pow(1 - t, 3);
  c.font = '700 ' + Math.round(h * env.params.size) + 'px "Segoe UI", system-ui, sans-serif';
  c.textAlign = 'center';
  c.textBaseline = 'middle';
  c.globalAlpha = ease;
  c.shadowColor = 'rgba(0,0,0,0.6)';
  c.shadowBlur = h * 0.02;
  c.fillStyle = env.params.color;
  c.fillText(env.params.text, w / 2, h * 0.5 + (1 - ease) * h * 0.04);
  c.globalAlpha = 1;
  c.shadowBlur = 0;
}
"""


def find_effect(project, effect_id):
    for effect in project.get("effects") or []:
        if effect["id"] == effect_id:
            return effect
    raise KeyError("no VFX clip %s" % effect_id)


def clean_effect_fields(fields):
    check_fields(fields, EFFECT_FIELDS)
    out = {}
    if "name" in fields:
        if not isinstance(fields["name"], str) or not fields["name"].strip():
            raise ValueError("a VFX clip's name must be text")
        out["name"] = fields["name"].strip()[:80]
    for key, least, what in (("start", 0, "start is a timeline frame, 0 or later"), ("frames", 1, "frames is how long the clip lasts, 1 or more")):
        if key in fields:
            value = fields[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value) or value < least:
                raise ValueError(what)
            out[key] = int(value)
    if "source" in fields:
        if not isinstance(fields["source"], str) or len(fields["source"]) > EFFECT_SOURCE_LIMIT:
            raise ValueError("source is the module's text, up to %s characters" % EFFECT_SOURCE_LIMIT)
        out["source"] = fields["source"]
    if "params" in fields:
        if not isinstance(fields["params"], dict):
            raise ValueError("params is an object of control values")
        out["params"] = copy.deepcopy(fields["params"])
    if "data" in fields:
        # Anything the code should read and cannot fetch: word timings for captions, numbers for a chart.
        try:
            size = len(json.dumps(fields["data"]))
        except (TypeError, ValueError):
            raise ValueError("data must be JSON: objects, lists, text, numbers")
        if size > EFFECT_DATA_LIMIT:
            raise ValueError("data is %s characters as JSON; the most is %s" % (size, EFFECT_DATA_LIMIT))
        out["data"] = copy.deepcopy(fields["data"])
    if "inputs" in fields:
        # Pictures the code may draw: a name it asks for by, and an uploaded file. The code cannot fetch anything itself.
        given = fields["inputs"]
        if not isinstance(given, dict) or len(given) > EFFECT_INPUTS:
            raise ValueError("inputs is an object of up to %s named pictures" % EFFECT_INPUTS)
        inputs = {}
        for name, asset in given.items():
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,31}", name):
                raise ValueError("an input's name is a word: letters, numbers and _")
            asset = clean_asset(asset, "the input %s" % name)
            if asset is None:
                raise ValueError("the input %s needs a file, and a subfolder if it has one" % name)
            inputs[name] = asset
        out["inputs"] = inputs
    return out


def check_effect_room(project, effect):
    """VFX clips share one lane, so two never cover the same frame."""
    first, last = effect["start"], effect["start"] + effect["frames"]
    for other in project.get("effects") or []:
        if other["id"] != effect["id"] and other["start"] < last and first < other["start"] + other["frames"]:
            raise ValueError("%s is already on the VFX lane from %s to %s" % (
                other.get("name") or "another clip", timecode(other["start"]), timecode(other["start"] + other["frames"])))


def add_effect(project, fields=None):
    """A new VFX clip. With no source it starts as a title drawn over the picture below."""
    fields = clean_effect_fields(fields or {})
    effect = {"id": new_id("x"), "name": "VFX %s" % (len(project.get("effects") or []) + 1), "start": 0, "frames": 3 * FPS,
              "source": EFFECT_STARTER, "params": {}, "rev": 1, "takes": [], "active": None}
    effect.update(fields)
    check_effect_room(project, effect)
    project.setdefault("effects", []).append(effect)
    return effect


def update_effect(project, effect_id, fields):
    effect = find_effect(project, effect_id)
    fields = clean_effect_fields(fields)
    before = {key: effect.get(key) for key in ("frames", "source", "params", "inputs", "data")}
    check_effect_room(project, dict(effect, **fields))   # the lane, with the new place and length
    effect.update(fields)
    if any(effect.get(key) != value for key, value in before.items()):
        effect["rev"] = int(effect.get("rev") or 0) + 1    # what it draws changed, so its take is out of date
    return effect


def remove_effect(project, effect_id):
    find_effect(project, effect_id)
    project["effects"] = [e for e in project["effects"] if e["id"] != effect_id]


def request_effect_render(project, effect_id):
    """Ask for a VFX clip to be rendered. The code runs in a browser, so an open editor picks the request up and does it."""
    effect = find_effect(project, effect_id)
    effect["queued"] = True
    effect.pop("error", None)
    return effect


def claim_effect_render(project, effect_id, job):
    """One editor takes a waiting request, so two open editors never render the same clip. True if this one got it."""
    effect = find_effect(project, effect_id)
    if effect.get("queued") is not True:
        return False
    effect["queued"] = {"job": str(job)}
    return True


def effect_below(project, effect):
    """What lies under a VFX clip on the picture lane, as far as its take depends on it."""
    first, last = effect["start"], effect["start"] + effect["frames"]
    clips = {c["id"]: c for c in project.get("clips", [])}
    rows = []
    for item in ordered(project):
        if item["start"] < last and first < item_end(item):
            take = active_take(clips.get(item.get("clip")) or {})
            transition = item.get("transition") or {}
            rows.append([item["clip"], take["id"] if take else None, item["start"], item["in"], item["out"],
                         clean_speed(item.get("speed")), transition.get("type"), transition.get("frames")])
    return rows


def effect_stamp(project, effect):
    """What a take of a VFX clip was made from. A take whose stamp no longer matches is out of date."""
    return {"rev": int(effect.get("rev") or 0), "start": effect["start"], "below": effect_below(project, effect)}


def effect_stale(project, effect):
    take = active_take(effect)
    return bool(take) and take.get("stamp") != effect_stamp(project, effect)


def accept_effect_take(project, effect_id, take):
    """A finished render becomes the VFX clip's active take, stamped with what it was made from."""
    effect = find_effect(project, effect_id)
    take = dict(take, stamp=effect_stamp(project, effect))
    effect.setdefault("takes", []).append(take)
    effect["active"] = take["id"]
    effect.pop("error", None)
    effect.pop("queued", None)
    return effect


def effects_waiting(project):
    """VFX clips with a render asked for and not yet done."""
    return [e for e in project.get("effects") or [] if e.get("queued")]


# Editing operations. The editor, the HTTP API and agents share these, so the rules live in one place.

HOLD = 39
MERGE_FRAMES = 63
CLIP_FIELDS = ("name", "action", "camera", "speaker", "dialogue", "sound", "music", "seconds", "cast", "seed",
               "first", "last", "language", "track", "bin", "profile", "loras")
TEXT_FIELDS = ("name", "action", "camera", "speaker", "dialogue", "sound", "music", "language")
PROJECT_FIELDS = ("style", "format", "language", "profile", "loras", "realism", "checkpoints", "text_encoder", "text_projection", "vae", "audio_vae", "sizes")
# The model files set_project names, and their keys in the project's models. A projection only
# applies to a 4B or 8B text encoder; one that does not fit it gives way to one that does.
PROJECT_MODEL_FIELDS = {**PROJECT_ENCODERS, "text_projection": "projection"}
# A project's own LoRAs: how many, how strong, and where their nodes start in a render graph.
LORA_LIMIT = 8
LORA_RANGE = (-2.0, 2.0)
REALISM_RANGE = (0.0, 3.0)
LORA_NODE = 100
CAST_FIELDS = ("name", "description", "voice", "pictures", "refmod")
# A RefMod is a reference saved by the ComfyUI-MiniMaxH3Mod pack. The editor renders with one only
# when that pack is installed: its loader takes this many, and the model list names each as refmod:<name>.
REFMOD_SLOTS = 8
REFMOD_PREFIX = "refmod:"
# A cast member can have several pictures, the main one included. H3 takes this many pictures in one shot.
CAST_PICTURES = 8
H3_PICTURES = 9
BIN_FIELDS = ("name",)
LINK_MODES = ("extend_after", "extend_before", "bridge", "merge")
LINK_TITLES = {"extend_after": "After ", "extend_before": "Before ", "bridge": "Bridge from ", "merge": "Merge from "}
SAVED_FIELDS = ("action", "camera", "speaker", "dialogue", "sound", "music", "seconds", "cast", "track")


def new_id(prefix):
    return prefix + uuid.uuid4().hex[:8]


def timecode(frames):
    f = max(0, int(round(frames)))
    return "%02d:%02d:%02d" % (f // FPS // 60, (f // FPS) % 60, f % FPS)


def find_clip(project, clip_id):
    for clip in project.get("clips", []):
        if clip["id"] == clip_id:
            return clip
    raise KeyError("no clip %s" % clip_id)


def find_item(project, item_id):
    for item in project.get("timeline", []):
        if item["id"] == item_id:
            return item
    raise KeyError("no timeline item %s" % item_id)


def find_cast(project, cast_id):
    for person in project.get("cast", []):
        if person["id"] == cast_id:
            return person
    raise KeyError("no cast member %s" % cast_id)


def find_bin(project, bin_id):
    for found in project.get("bins", []):
        if found["id"] == bin_id:
            return found
    raise KeyError("no bin %s" % bin_id)


def item_end(item):
    return item["start"] + item_length(item)


def scene_end(project):
    return max([item_end(i) for i in project.get("timeline", [])] or [0])


def ordered(project):
    return sorted(project.get("timeline", []), key=lambda i: i["start"])


def settle(project, moved=None):
    """Clips never overlap on the picture track, except where a transition says so."""
    items = sorted(project.get("timeline", []), key=lambda i: (i["start"], 0 if i is moved else 1))
    end = 0
    for item in items:
        floor = end - ((item.get("transition") or {}).get("frames") or 0)
        if item["start"] < floor:
            item["start"] = floor
        end = max(end, item_end(item))


def shift_from(project, frame, by, skip=None):
    for other in project.get("timeline", []):
        if other is not skip and other["start"] >= frame:
            other["start"] += by


def linked_range(link, frames):
    """The part of a linked render the timeline shows. Held frames are already on screen in the source."""
    if not link:
        return 0, frames
    hold = int(link["hold"])
    if link["mode"] == "extend_after":
        return hold, frames
    if link["mode"] == "extend_before":
        return 0, frames - hold
    return hold, frames - hold


def planned_frames(clip):
    """What a render of this clip will be, so its place on the timeline is right before it exists."""
    take = active_take(clip)
    if take:
        return take["frames"]
    if clip.get("link"):
        return linked_length(clip["link"], clip.get("seconds", 5))
    return snap_frames(clip.get("seconds", 5))


def check_fields(fields, allowed):
    if not isinstance(fields, dict):
        raise ValueError("fields must be an object")
    unknown = sorted(set(fields) - set(allowed))
    if unknown:
        raise ValueError("unknown fields: %s. Allowed: %s" % (", ".join(unknown), ", ".join(allowed)))


def clean_asset(value, what):
    if value is None:
        return None
    if not isinstance(value, dict) or not isinstance(value.get("file"), str) or not value["file"]:
        raise ValueError("%s needs a file, and a subfolder if it has one" % what)
    return {"file": value["file"], "subfolder": str(value.get("subfolder") or "")}


def clean_clip_fields(fields):
    check_fields(fields, CLIP_FIELDS)
    out = dict(fields)
    if "seconds" in out:
        seconds = float(out["seconds"])
        if not math.isfinite(seconds) or not 0.5 <= seconds <= 60:
            raise ValueError("seconds must be between 0.5 and 60")
        out["seconds"] = seconds
    if "seed" in out:
        out["seed"] = None if out["seed"] in (None, "") else int(out["seed"])
    if "cast" in out:
        if not isinstance(out["cast"], list) or not all(isinstance(c, str) for c in out["cast"]):
            raise ValueError("cast is a list of cast ids")
    for name in TEXT_FIELDS:
        if name in out and not isinstance(out[name], str):
            raise ValueError("%s must be text" % name)
    for side in ("first", "last"):
        if side in out:
            out[side] = clean_asset(out[side], "a %s frame" % side)
    if "track" in out:
        out["track"] = clean_track(out["track"])
    if out.get("profile") is not None and not isinstance(out["profile"], str):
        raise ValueError("profile is a profile id, or null for the project's")
    if "bin" in out and out["bin"] is not None and not isinstance(out["bin"], str):
        raise ValueError("bin is a bin id, or null for none")
    if out.get("loras") is not None:
        out["loras"] = clean_loras(out["loras"], "a clip")
    return out


def check_profile(project, fields):
    """A clip can only pick a profile the project can render with."""
    profiles = project_profiles(project)
    if fields.get("profile") is not None and fields["profile"] not in profiles:
        raise ValueError("profile must be one of %s, or null for the project's" % ", ".join(profiles))


def check_bin(project, fields):
    """A clip can only be filed in a bin the project has."""
    if fields.get("bin") is not None:
        find_bin(project, fields["bin"])


def clean_track(value):
    """A sound the picture is generated to: an asset, and the second of it where the clip's first frame sits."""
    if value is None:
        return None
    asset = clean_asset(value, "a track")
    start = float(value.get("from") or 0)
    if not math.isfinite(start) or start < 0:
        raise ValueError("a track's from is the second of the sound where the clip starts, 0 or more")
    asset["from"] = round(start, 6)
    return asset


def continued_track(source, item, mode, frames):
    """The slice of the source's track a linked clip holds, so one long performance carries on across clips."""
    track = source.get("track")
    if not track:
        return None
    if clean_speed(item.get("speed")) != 1:
        raise ValueError("set %s to 100%% speed to continue its track" % source["name"])
    if mode == "extend_before":
        start = track["from"] + (item["in"] - (frames - HOLD)) / float(FPS)
        if start < -1e-6:
            raise ValueError("the track starts %.2f s too late for this extension. Use fewer seconds" % -start)
    else:
        start = track["from"] + (item["out"] - HOLD) / float(FPS)
    return dict(track, **{"from": round(max(0.0, start), 6)})


def place_clip(project, clip_id, start=None):
    """Put a clip on the timeline, at the end unless a frame is given. Later clips make room."""
    clip = find_clip(project, clip_id)
    first, last = linked_range(clip.get("link"), planned_frames(clip))
    at = scene_end(project) if start is None else max(0, int(start))
    item = {"id": new_id("i"), "clip": clip["id"], "start": at, "in": first, "out": last}
    project.setdefault("timeline", []).append(item)
    settle(project, item)
    return item


def add_clip(project, fields, start=None, place=True):
    fields = clean_clip_fields(fields or {})
    check_bin(project, fields)
    check_profile(project, fields)
    number = len(project.get("clips", [])) + 1
    clip = {"id": new_id("c"), "name": "Clip %s" % number, "action": "", "camera": "", "speaker": "",
            "dialogue": "", "sound": "", "music": "", "seconds": 5, "takes": [], "active": None}
    clip.update({k: v for k, v in fields.items() if v is not None or k == "seed"})
    project.setdefault("clips", []).append(clip)
    item = place_clip(project, clip["id"], start) if place else None
    return clip, item


def update_clip(project, clip_id, fields):
    clip = find_clip(project, clip_id)
    fields = clean_clip_fields(fields)
    check_bin(project, fields)
    check_profile(project, fields)
    before = planned_frames(clip)
    for key, value in fields.items():
        if value is None and key in ("first", "last", "track", "bin", "profile", "loras"):
            clip.pop(key, None)
        else:
            clip[key] = value
    if not clip.get("takes") and planned_frames(clip) != before:
        # An unrendered clip changes length with its seconds. Later clips follow it.
        first, last = linked_range(clip.get("link"), planned_frames(clip))
        for item in project.get("timeline", []):
            if item["clip"] == clip_id:
                end = item_end(item)
                item["in"], item["out"] = first, last
                shift_from(project, end, item_end(item) - end, skip=item)
        settle(project)
    return clip


def remove_clip(project, clip_id):
    """The clip and its places on the timeline go. Its render files stay, so older revisions still play."""
    find_clip(project, clip_id)
    project["clips"] = [c for c in project["clips"] if c["id"] != clip_id]
    project["timeline"] = [i for i in project.get("timeline", []) if i["clip"] != clip_id]
    if project.get("notes"):
        project["notes"] = [n for n in project["notes"] if not (isinstance(n, dict) and n.get("clip") == clip_id)]


def clean_bin_name(value, fallback):
    if value is None:
        return fallback
    if not isinstance(value, str):
        raise ValueError("a bin's name must be text")
    return value.strip()[:80] or fallback


def add_bin(project, fields=None):
    """A folder in the Clips panel. A clip files itself in one with its bin field."""
    fields = fields or {}
    check_fields(fields, BIN_FIELDS)
    taken = {b["name"] for b in project.setdefault("bins", [])}
    number = len(taken) + 1
    while "Bin %s" % number in taken:
        number += 1
    found = {"id": new_id("b"), "name": clean_bin_name(fields.get("name"), "Bin %s" % number)}
    project["bins"].append(found)
    return found


def update_bin(project, bin_id, fields):
    check_fields(fields, BIN_FIELDS)
    found = find_bin(project, bin_id)
    if "name" in fields:
        found["name"] = clean_bin_name(fields["name"], found["name"])
    return found


def remove_bin(project, bin_id):
    """The bin goes. Its clips stay, loose in the Clips panel."""
    find_bin(project, bin_id)
    project["bins"] = [b for b in project["bins"] if b["id"] != bin_id]
    for clip in project.get("clips", []):
        if clip.get("bin") == bin_id:
            del clip["bin"]


def previous_item(project, item):
    earlier = [o for o in ordered(project) if o is not item and o["start"] < item["start"]]
    return earlier[-1] if earlier else None


def following_item(project, item):
    later = [o for o in ordered(project) if o is not item and o["start"] >= item_end(item)]
    return later[0] if later else None


def set_speed(project, item_id, speed):
    """Speed is non-destructive. Clips after this one move with its new end."""
    item = find_item(project, item_id)
    old_end = item_end(item)
    item["speed"] = clean_speed(speed)
    shift = item_end(item) - old_end
    for other in project.get("timeline", []):
        if other is not item and other["start"] >= old_end - ((other.get("transition") or {}).get("frames") or 0):
            other["start"] += shift
    return item


def set_transition(project, item_id, kind, frames=12):
    """A transition overlaps this clip with the one before. The clip and everything after it slide back."""
    item = find_item(project, item_id)
    before = previous_item(project, item)
    old = (item.get("transition") or {}).get("frames") or 0
    if kind in (None, "none"):
        kind = "none"
    elif kind not in TRANSITIONS:
        raise ValueError("transition must be none or one of %s" % ", ".join(TRANSITIONS))
    if kind != "none" and (not before or item_end(before) < item["start"] + old):
        raise ValueError("put this clip right after another one to add a transition")
    limit = min(item_length(before), item_length(item)) - 1 if before else 0
    length = 0 if kind == "none" else max(2, min(int(frames), limit))
    shift = length - old
    for other in project.get("timeline", []):
        if other is item or other["start"] > item["start"]:
            other["start"] -= shift
    if kind == "none":
        item.pop("transition", None)
    else:
        item["transition"] = {"type": kind, "frames": length}
    return item


def update_item(project, item_id, fields, ripple=False):
    """Move or trim a timeline item. With ripple, later clips follow its new end."""
    check_fields(fields, ("start", "in", "out", "speed", "transition"))
    item = find_item(project, item_id)
    take = active_take(find_clip(project, item["clip"]))
    old_end = item_end(item)
    changed = {k: int(fields[k]) for k in ("start", "in", "out") if k in fields}
    start = changed.get("start", item["start"])
    first = changed.get("in", item["in"])
    last = changed.get("out", item["out"])
    if start < 0 or first < 0 or last - first < 6:
        raise ValueError("an item needs a start of 0 or more and at least 6 frames between in and out")
    if take and last > take["frames"]:
        raise ValueError("out is past the end of the take, which has %s frames" % take["frames"])
    item.update({"start": start, "in": first, "out": last})
    if ripple and "start" not in changed:
        shift_from(project, old_end, item_end(item) - old_end, skip=item)
    if "speed" in fields:
        set_speed(project, item_id, fields["speed"])
    if "transition" in fields:
        transition = fields["transition"] or {}
        if isinstance(transition, str):
            transition = {"type": transition}
        set_transition(project, item_id, transition.get("type", "none"), transition.get("frames", 12))
    settle(project, item)
    return item


def remove_item(project, item_id, ripple=False):
    item = find_item(project, item_id)
    end = item_end(item)
    shift = item_length(item) - ((item.get("transition") or {}).get("frames") or 0)
    project["timeline"] = [i for i in project["timeline"] if i is not item]
    if ripple:
        for other in project["timeline"]:
            if other["start"] >= end - ((other.get("transition") or {}).get("frames") or 0):
                other["start"] -= shift


def split_item(project, item_id, frame):
    """Cut an item in two at a timeline frame. Both halves keep pointing at the same take."""
    item = find_item(project, item_id)
    frame = int(frame)
    if not item["start"] < frame < item_end(item):
        raise ValueError("frame %s is not inside that item, which runs from %s to %s" % (frame, item["start"], item_end(item)))
    cut = int(round(item["in"] + (frame - item["start"]) * clean_speed(item.get("speed"))))
    right = {"id": new_id("i"), "clip": item["clip"], "start": frame, "in": cut, "out": item["out"]}
    if item.get("speed"):
        right["speed"] = item["speed"]
    item["out"] = cut
    project["timeline"].append(right)
    return right


def gap_at(project, frame):
    end = 0
    for item in ordered(project):
        if end <= frame < item["start"]:
            return end, item["start"]
        end = max(end, item_end(item))
    return None


def close_gap(project, frame=None):
    """Close the gap at a frame, or every gap when no frame is given. Returns the frames removed."""
    if frame is not None:
        gap = gap_at(project, int(frame))
        if not gap:
            raise ValueError("there is no gap at frame %s" % frame)
        shift_from(project, gap[1], gap[0] - gap[1])
        return gap[1] - gap[0]
    removed = 0
    while True:
        gap = next((g for g in (gap_at(project, i["start"] - 1) for i in ordered(project) if i["start"] > 0) if g), None)
        if not gap:
            return removed
        shift_from(project, gap[1], gap[0] - gap[1])
        removed += gap[1] - gap[0]


def side_of(project, item):
    take = active_take(find_clip(project, item["clip"]))
    if not take:
        raise ValueError("render %s before joining another clip to it" % find_clip(project, item["clip"])["name"])
    return {"clip": item["clip"], "item": item["id"], "file": take["file"], "subfolder": take.get("subfolder"),
            "first": item["in"], "last": item["out"]}


def link_sides(link):
    return [s for s in ((link or {}).get("a"), (link or {}).get("b")) if s]


def side_changed(project, side):
    item = next((i for i in project.get("timeline", []) if i["id"] == side["item"]), None)
    clip = next((c for c in project.get("clips", []) if c["id"] == side["clip"]), None)
    take = active_take(clip) if clip else None
    if not item or not take:
        return False
    return take["file"] != side["file"] or item["in"] != side["first"] or item["out"] != side["last"]


def touches(project, clip, side):
    """Only a join that still exists can show, so a clip that moved away from its source is left alone."""
    source = next((i for i in project.get("timeline", []) if i["id"] == side["item"]), None)
    if not source:
        return False
    return any(i["clip"] == clip["id"] and (item_end(i) == source["start"] or item_end(source) == i["start"])
               for i in project.get("timeline", []))


def is_stale(project, clip):
    return bool(clip.get("takes")) and any(side_changed(project, s) and touches(project, clip, s) for s in link_sides(clip.get("link")))


def refresh_link(project, clip):
    """A linked clip always renders from what its sources show on the timeline right now."""
    for side in link_sides(clip.get("link")):
        item = next((i for i in project.get("timeline", []) if i["id"] == side["item"]), None)
        source = next((c for c in project.get("clips", []) if item and c["id"] == item["clip"]), None)
        if item and source and active_take(source):
            side.update(side_of(project, item))


def add_linked(project, item_id, mode, seconds=None, action=""):
    """Extend a clip forward or back, bridge it to the next one, or merge the two with a generated crossing."""
    if mode not in LINK_MODES:
        raise ValueError("mode must be one of %s" % ", ".join(LINK_MODES))
    item = find_item(project, item_id)
    source = find_clip(project, item["clip"])
    side = side_of(project, item)
    if item_length(item) < HOLD:
        raise ValueError("the clip needs at least %s frames on the timeline" % HOLD)
    nxt = following_item(project, item) if mode in ("bridge", "merge") else None
    if mode in ("bridge", "merge"):
        if not nxt or not active_take(find_clip(project, nxt["clip"])):
            raise ValueError("%s needs a rendered clip after this one" % mode)
        if item_length(nxt) < HOLD:
            raise ValueError("the next clip needs at least %s frames on the timeline" % HOLD)
    if mode == "merge":
        away = (MERGE_FRAMES + 1) // 2
        rest = MERGE_FRAMES - away
        if nxt["start"] != item_end(item):
            raise ValueError("put the two clips side by side to merge them")
        if clean_speed(item.get("speed")) != 1 or clean_speed(nxt.get("speed")) != 1:
            raise ValueError("set both clips to 100% speed to merge them")
        if item["out"] - item["in"] < HOLD + away or nxt["out"] - nxt["in"] < HOLD + rest:
            raise ValueError("both clips need at least %s frames on the timeline to merge" % (HOLD + away))
        item["out"] -= away
        nxt["in"] += rest
        nxt["start"] += rest
        side = side_of(project, item)
        seconds = 2.625

    link = {"mode": "bridge" if mode == "merge" else mode, "hold": HOLD, "a": side}
    if nxt:
        link["b"] = side_of(project, nxt)
    seconds = float(seconds if seconds is not None else 4.25)
    if not math.isfinite(seconds) or not 0.5 <= seconds <= 60:
        raise ValueError("seconds must be between 0.5 and 60")
    default = "One continuous shot that moves naturally from the opening frames into the closing frames." if mode == "merge" else ""
    clip = {"id": new_id("c"), "name": LINK_TITLES[mode] + source["name"], "action": action or default,
            "camera": "", "speaker": "", "dialogue": "", "sound": "", "music": "", "seconds": seconds,
            "takes": [], "active": None, "link": link}
    if source.get("cast"):
        clip["cast"] = list(source["cast"])
    if source.get("bin"):
        clip["bin"] = source["bin"]
    track = continued_track(source, item, link["mode"], linked_length(link, seconds))
    if track:
        clip["track"] = track
    project["clips"].append(clip)
    first, last = linked_range(link, linked_length(link, seconds))
    length = last - first
    start = item_end(item)
    if mode == "extend_before":
        start = item["start"]
        shift_from(project, item["start"], length)
    elif mode == "extend_after":
        shift_from(project, start, length)
    elif mode == "bridge":
        shift_from(project, nxt["start"], start + length - nxt["start"])
    placed = {"id": new_id("i"), "clip": clip["id"], "start": start, "in": first, "out": last}
    project["timeline"].append(placed)
    return clip, placed


def choose_take(project, clip_id, take_id):
    clip = find_clip(project, clip_id)
    take = next((t for t in clip.get("takes", []) if t["id"] == take_id), None)
    if take is None:
        raise KeyError("no take %s on clip %s" % (take_id, clip_id))
    clip["active"] = take_id
    _, last = linked_range(clip.get("link"), take["frames"])
    for item in project.get("timeline", []):
        if item["clip"] == clip_id:
            item["out"] = min(item["out"], last)
    return clip


def accept_take(project, clip_id, take):
    """A finished render becomes the clip's active take. A first render sets its timeline range."""
    clip = find_clip(project, clip_id)
    fresh = not clip.get("takes")
    clip.setdefault("takes", []).append(take)
    clip["active"] = take["id"]
    clip.pop("error", None)
    first, last = linked_range(clip.get("link"), take["frames"])
    for item in project.get("timeline", []):
        if item["clip"] != clip_id:
            continue
        if fresh:
            end = item_end(item)
            item["in"], item["out"] = first, last
            shift_from(project, end, item_end(item) - end, skip=item)
        else:
            item["out"] = min(item["out"], last)
    settle(project)
    return clip


def clean_strength(value, limits, what):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not limits[0] <= value <= limits[1]:
        raise ValueError("%s must be a number from %g to %g" % (what, limits[0], limits[1]))
    return round(float(value), 2)


def clean_loras(value, owner="a project"):
    """LoRAs as an agent or the editor writes them, for a project or for one clip.

    Each is a file from models/loras, a strength, whether it is on, and optionally start_sigma:
    where in the schedule it joins, from 1 (the first step, the default) down to 0.
    """
    if not isinstance(value, list):
        raise ValueError("loras must be a list of {file, strength}")
    if len(value) > LORA_LIMIT:
        raise ValueError("%s takes up to %s LoRAs" % (owner, LORA_LIMIT))
    rows = []
    for row in value:
        if not isinstance(row, dict) or not isinstance(row.get("file"), str) or not row["file"].strip():
            raise ValueError("each LoRA needs a file, as models/loras lists it")
        if any(r["file"] == row["file"] for r in rows):
            raise ValueError("%s is listed twice" % row["file"])
        unknown = sorted(set(row) - {"id", "file", "strength", "on", "start_sigma"})
        if unknown:
            raise ValueError("unknown LoRA fields: %s. Allowed: file, strength, on, start_sigma" % ", ".join(unknown))
        clean = {"id": str(row.get("id") or new_id("l")), "file": row["file"],
                 "strength": clean_strength(row.get("strength", 1.0), LORA_RANGE, "a LoRA's strength"),
                 "on": row.get("on") is not False}
        start = clean_strength(row.get("start_sigma", 1.0), (0.0, 1.0), "a LoRA's start_sigma")
        if start < 1.0:
            clean["start_sigma"] = start
        rows.append(clean)
    return rows


def clean_checkpoints(value, limit=CHECKPOINT_LIMIT):
    """A project's own checkpoints as an agent or the editor writes them.

    Each is a ref2va file from models/diffusion_models, the built-in family whose settings it
    starts from, and whatever it changes: steps for drafts and finals, sampler, scheduler, and
    an fl2va file for clips that are only a prompt.
    """
    if not isinstance(value, list):
        raise ValueError("checkpoints must be a list of {title, unet, base}")
    if len(value) > limit:
        raise ValueError("there is room for %s checkpoints here" % limit)
    rows = []
    for row in value:
        if not isinstance(row, dict) or not isinstance(row.get("unet"), str) or not row["unet"].strip():
            raise ValueError("each checkpoint needs unet, a file as models/diffusion_models lists it")
        check_fields(row, CHECKPOINT_FIELDS)
        base = row.get("base") or CHECKPOINT_BASE
        if base not in PROFILES:
            raise ValueError("base must be one of %s" % ", ".join(PROFILES))
        made = {"id": row.get("id") or new_id("m"), "unet": row["unet"], "base": base,
                "title": str(row.get("title") or "").strip() or row["unet"].replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]}
        if not isinstance(made["id"], str) or made["id"] in PROFILES or any(r["id"] == made["id"] for r in rows):
            raise ValueError("a checkpoint's id must be its own, and not the id of a built-in profile")
        if row.get("unet_text"):
            if not isinstance(row["unet_text"], str):
                raise ValueError("unet_text is a file as models/diffusion_models lists it")
            made["unet_text"] = row["unet_text"]
        if row.get("steps") is not None:
            steps = row["steps"]
            if not isinstance(steps, dict) or set(steps) - set(QUALITIES):
                raise ValueError("steps is {draft, final}")
            for quality, count in steps.items():
                if isinstance(count, bool) or not isinstance(count, int) or not STEP_RANGE[0] <= count <= STEP_RANGE[1]:
                    raise ValueError("steps must be whole numbers from %s to %s" % STEP_RANGE)
            made["steps"] = dict(steps)
        if row.get("lora") is not None:
            if not isinstance(row["lora"], bool):
                raise ValueError("lora is true to keep the base's LoRA, or false to render without it")
            if row["lora"] is False:
                made["lora"] = False
        for key in ("sampler", "scheduler"):
            if row.get(key):
                if not isinstance(row[key], str):
                    raise ValueError("%s must be a name ComfyUI knows, such as %s" % (key, "euler" if key == "sampler" else "simple"))
                made[key] = row[key]
        rows.append(made)
    return rows


def clean_sizes(value):
    """The sizes a project asks for, without the ones that are the usual size anyway. None asks for none."""
    if value is None:
        return {}
    if not isinstance(value, dict) or any(key not in SIZE_CHOICES for key in value):
        raise ValueError("sizes takes draft and final, such as {\"draft\": \"medium\", \"final\": \"large\"}")
    chosen = {}
    for quality, name in value.items():
        if name is None:
            continue
        if name not in SIZE_CHOICES[quality]:
            raise ValueError("the %s size must be one of %s" % (quality, ", ".join(SIZE_CHOICES[quality])))
        if name != SIZE_CHOICES[quality][0]:
            chosen[quality] = name
    return chosen


def set_project(project, fields):
    check_fields(fields, PROJECT_FIELDS)
    # The project's own checkpoints first, so the same call can pick one as its profile.
    for key, value in sorted(fields.items(), key=lambda pair: pair[0] != "checkpoints"):
        if key == "checkpoints":
            project["checkpoints"] = clean_checkpoints(value)
            continue
        if key == "loras":
            project["loras"] = clean_loras(value)
            continue
        if key in PROJECT_MODEL_FIELDS:
            # A file as ComfyUI lists it, or None for the first one installed.
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError("%s is a file name, or null to take the first one installed" % key)
            chosen = dict(project.get("models") or {})
            if value is None:
                chosen.pop(PROJECT_MODEL_FIELDS[key], None)
            else:
                chosen[PROJECT_MODEL_FIELDS[key]] = value
            if chosen:
                project["models"] = chosen
            else:
                project.pop("models", None)
            continue
        if key == "sizes":
            project.pop("sizes", None)
            chosen = clean_sizes(value)
            if chosen:
                project["sizes"] = chosen
            continue
        if key == "realism":
            # None goes back to the usual strength.
            if value is None:
                project.pop("realism", None)
            else:
                project["realism"] = clean_strength(value, REALISM_RANGE, "realism")
            continue
        if not isinstance(value, str):
            raise ValueError("%s must be text" % key)
        if key == "format" and value not in FORMATS:
            raise ValueError("format must be one of %s" % ", ".join(FORMATS))
        if key == "profile" and value not in project_profiles(project):
            raise ValueError("profile must be one of %s" % ", ".join(project_profiles(project)))
        if key == "language" and not value.strip():
            raise ValueError("language names the language people speak, such as English or Portuguese")
        project[key] = value.strip() if key == "language" else value


def clean_pictures(value, person):
    """More pictures of a cast member, beside the main one: other angles, other light, the whole figure."""
    if not isinstance(value, list):
        raise ValueError("pictures is a list of picture assets")
    if not person.get("file"):
        raise ValueError("%s needs a main picture before it takes more" % person.get("name", "a cast member"))
    if len(value) > CAST_PICTURES - 1:
        raise ValueError("a cast member takes up to %s pictures, the main one included" % CAST_PICTURES)
    return [clean_asset(asset, "a picture") for asset in value]


def clean_refmod(value):
    """A cast member's RefMod: its name as the RefMod pack's loader lists it, and how strong it is, from 0 to 1."""
    if value is None:
        return None
    if not isinstance(value, dict) or not isinstance(value.get("file"), str) or not value["file"].strip():
        raise ValueError("a RefMod needs file, its name as Load H3 RefMods lists it")
    check_fields(value, ("file", "strength"))
    return {"file": value["file"], "strength": clean_strength(value.get("strength", 1.0), (0.0, 1.0), "a RefMod's strength")}


def add_cast(project, fields):
    """A person or place from a picture, a voice with no picture, a RefMod, or any of them together."""
    check_fields({k: v for k, v in fields.items() if k not in ("file", "subfolder")}, CAST_FIELDS)
    voice = clean_asset(fields.get("voice"), "a voice")
    refmod = clean_refmod(fields.get("refmod"))
    asset = clean_asset(fields, "a cast member") if fields.get("file") or not (voice or refmod) else None
    project.setdefault("cast", [])
    person = {"id": new_id("p"), "name": str(fields.get("name") or "Person %s" % (len(project["cast"]) + 1)),
              "description": str(fields.get("description") or "")}
    if asset:
        person.update(asset)
    if voice:
        person["voice"] = voice
    if fields.get("pictures"):
        person["pictures"] = clean_pictures(fields["pictures"], person)
    if refmod:
        person["refmod"] = refmod
    project["cast"].append(person)
    return person


def update_cast(project, cast_id, fields):
    check_fields(fields, CAST_FIELDS)
    person = find_cast(project, cast_id)
    for key, value in fields.items():
        if key == "voice":
            if value is None and not (person.get("file") or person.get("refmod")):
                raise ValueError("%s is only a voice. Remove the cast member instead" % person["name"])
            if value is None:
                person.pop("voice", None)
            else:
                person["voice"] = clean_asset(value, "a voice")
            continue
        if key == "pictures":
            person["pictures"] = clean_pictures(value or [], person)
            continue
        if key == "refmod":
            if value is None and not (person.get("file") or person.get("voice")):
                raise ValueError("%s is only a RefMod. Remove the cast member instead" % person["name"])
            if value is None:
                person.pop("refmod", None)
            else:
                person["refmod"] = clean_refmod(value)
            continue
        if not isinstance(value, str):
            raise ValueError("%s must be text" % key)
        person[key] = value.strip() or person[key] if key == "name" else value
    return person


def add_footage(project, name, take, start=None, place=True, bin_id=None):
    """Real video or a picture, already made into a take. It edits like any clip but is never rendered.

    A bin removed while the file was being converted is let go, so the import is never lost.
    """
    clip = {"id": take.pop("clip", None) or new_id("c"), "kind": "footage", "name": str(name or "Footage"),
            "action": "", "camera": "", "speaker": "", "dialogue": "", "sound": "", "music": "",
            "seconds": round(take["frames"] / float(FPS), 3), "takes": [take], "active": take["id"]}
    if bin_id and any(b["id"] == bin_id for b in project.get("bins", [])):
        clip["bin"] = bin_id
    project.setdefault("clips", []).append(clip)
    item = place_clip(project, clip["id"], start) if place else None
    return clip, item


# The cast catalog: cast members kept once per ComfyUI install, for any project to bring in. A project
# takes its own copy, linked to the catalog's entry by id and version. The link also keeps what the copy
# was when it last matched the catalog, so the editor can tell a newer catalog entry from a change made
# in the project.

CATALOG_KEEP = ("name", "description", "file", "subfolder", "pictures", "voice", "refmod")
CATALOG_LIMIT = 500


def cast_snapshot(person):
    """What a cast member is, without its id, its link or anything a project adds."""
    return {k: copy.deepcopy(person[k]) for k in CATALOG_KEEP if person.get(k) not in (None, "", [])}


def find_catalog(catalog, entry_id):
    for entry in catalog:
        if entry.get("id") == entry_id:
            return entry
    raise KeyError("no cast member %s in the catalog" % entry_id)


def catalog_state(person, catalog):
    """How a project's cast member stands against the catalog: None when it is not from it, else
    current, newer (the catalog changed since), changed (the project changed it since) or missing."""
    link = person.get("catalog")
    if not isinstance(link, dict) or not link.get("id"):
        return None
    entry = next((e for e in catalog if e.get("id") == link["id"]), None)
    if entry is None:
        return "missing"
    if int(entry.get("version", 1)) > int(link.get("version", 0)):
        return "newer"
    return "changed" if cast_edited(person) else "current"


def cast_edited(person):
    """Whether a linked cast member was changed in its project since it last matched the catalog."""
    link = person.get("catalog")
    return isinstance(link, dict) and link.get("synced") is not None and link["synced"] != cast_snapshot(person)


def save_to_catalog(catalog, person, assets):
    """Put a cast member in the catalog, or write it over the entry it came from. assets maps each of its
    pictures and voice, as (subfolder, file), to the catalog's own copy. Returns the entry."""
    snapshot = moved_assets(cast_snapshot(person), assets)
    link = person.get("catalog") if isinstance(person.get("catalog"), dict) else {}
    entry = next((e for e in catalog if e.get("id") == link.get("id")), None)
    if entry is None:
        if len(catalog) >= CATALOG_LIMIT:
            raise ValueError("the catalog holds up to %s cast members" % CATALOG_LIMIT)
        entry = {"id": new_id("k"), "version": 0}
        catalog.append(entry)
    for key in CATALOG_KEEP:
        entry.pop(key, None)
    entry.update(snapshot)
    entry["version"] = int(entry.get("version", 0)) + 1
    person["catalog"] = {"id": entry["id"], "version": entry["version"], "synced": cast_snapshot(person)}
    return entry


def cast_from_catalog(project, entry, assets, cast_id=None):
    """Bring a catalog entry into a project as a linked copy, or bring a linked member up to date with it.
    assets maps the catalog's files to the project's own copies."""
    fields = moved_assets({k: copy.deepcopy(entry[k]) for k in CATALOG_KEEP if k in entry}, assets)
    if cast_id is None:
        person = add_cast(project, {k: v for k, v in fields.items()})
    else:
        person = find_cast(project, cast_id)
        name = person.get("name")
        for key in CATALOG_KEEP:
            person.pop(key, None)
        person.update(fields)
        person.setdefault("name", name)
        person.setdefault("description", "")
    person["catalog"] = {"id": entry["id"], "version": int(entry.get("version", 1)), "synced": cast_snapshot(person)}
    return person


def cast_assets(fields):
    """Every file a cast member uses, as (subfolder, file)."""
    found = []
    if fields.get("file"):
        found.append((fields.get("subfolder") or "", fields["file"]))
    for asset in (fields.get("pictures") or []) + ([fields["voice"]] if fields.get("voice") else []):
        if isinstance(asset, dict) and asset.get("file"):
            found.append((asset.get("subfolder") or "", asset["file"]))
    return found


def moved_assets(fields, assets):
    """The same fields, with each file swapped for its copy in assets."""
    def move(asset):
        new = assets.get((asset.get("subfolder") or "", asset["file"]))
        return dict(asset, **new) if new else asset
    out = dict(fields)
    if out.get("file"):
        moved = move({"file": out["file"], "subfolder": out.get("subfolder") or ""})
        out["file"], out["subfolder"] = moved["file"], moved["subfolder"]
    if out.get("pictures"):
        out["pictures"] = [move(p) for p in out["pictures"]]
    if out.get("voice"):
        out["voice"] = move(out["voice"])
    return out


def remove_cast(project, cast_id):
    find_cast(project, cast_id)
    project["cast"] = [c for c in project["cast"] if c["id"] != cast_id]
    for clip in project.get("clips", []):
        if cast_id in (clip.get("cast") or []):
            clip["cast"] = [c for c in clip["cast"] if c != cast_id]


# Renders. The server queues the graph; these keep the project's record of it.

def render_models(graph):
    """The models a render graph samples with, so a take records what made it."""
    used = {"unet": graph["1"]["inputs"]["unet_name"], "checkpoint": graph["10"]["inputs"]["checkpoint"],
            "text_encoder": graph["2"]["inputs"]["clip_name"], "vae": graph["3"]["inputs"]["vae_name"]}
    if "projection" in graph["2"]["inputs"]:
        used["text_projection"] = graph["2"]["inputs"]["projection"]
    if "72" in graph:
        slots = graph["72"]["inputs"]
        used["refmods"] = [{"file": slots["mod_%s" % n], "strength": slots["strength_%s" % n]}
                           for n in range(1, REFMOD_SLOTS + 1) if slots.get("mod_%s" % n) not in (None, "(none)")]
    if "81" in graph:
        used["upscaler"] = graph["81"]["inputs"]["model_name"]
    if "94" in graph:
        used["profile_lora"] = graph["94"]["inputs"]["lora_name"]
        used["profile_lora_strength"] = graph["94"]["inputs"]["strength"]
    if "95" in graph:
        used["lora"] = graph["95"]["inputs"]["lora_name"]
        used["lora_strength"] = graph["95"]["inputs"]["strength"]
    extra = [graph[n]["inputs"] for n in sorted((n for n in graph if n.isdigit() and int(n) >= LORA_NODE), key=int)
             if graph[n]["class_type"] == "H3Lora"]
    if extra:
        used["loras"] = [dict({"file": i["lora_name"], "strength": i["strength"]},
                              **({"start_sigma": i["start_sigma"]} if "start_sigma" in i else {})) for i in extra]
    return used


# Upscaling. A finished final can be enlarged two times by a video upscaler ComfyUI runs natively. The result is a
# new take of the clip, at twice the size, with the same sound. Which upscaler is the user's choice, among the ones
# whose files are installed: they differ a great deal in what they do to a picture and in what they cost.
UPSCALERS = {
    "relayvsr": {
        "title": "RelayVSR",
        "about": "Fast: seconds a clip, about 16 GB of GPU memory. Sharpens the edges that are there; it does not add new texture.",
        "files": {"model": "relayvsr_x2_lightx2v_int8_convrot.safetensors", "vae": "relayvsr_flash_vae_x2_fp32.safetensors"},
        "source": "Comfy-Org/RelayVSR", "detail": True,
    },
    "seedvr2": {
        "title": "SeedVR2",
        "about": "Slow: minutes a clip, and far more GPU memory. Redraws fine texture, so it looks like a larger picture; it can change a face a little.",
        "files": {"model": "seedvr2_3b_int8_convrot.safetensors", "vae": "seedvr2_ema_vae_fp16.safetensors"},
        "source": "Comfy-Org/SeedVR2", "detail": False,
    },
}
UPSCALE_QUALITY = "upscale"
UPSCALE_MOST = 240          # frames: the enlarged frames are all held in memory at once, about 100 MB each at 4K
# RelayVSR only: how much of the detail it adds is kept, from 0 (a plain enlargement) to 1 (all of it). At 1 skin gets
# pores and shine that can make a face look older, so the default keeps half.
UPSCALE_DETAIL = 0.5


def clean_detail(value):
    if value in (None, ""):
        return UPSCALE_DETAIL
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ValueError("detail must be a number from 0 to 1")
    return round(float(value), 2)


def upscaler_files(name, available=None):
    """An upscaler's two files as ComfyUI lists them (they may sit in a subfolder), or None for one that is missing."""
    found = {}
    for kind, file in UPSCALERS[name]["files"].items():
        if available is None:
            found[kind] = file
        else:
            found[kind] = next((entry for entry in sorted(available) if entry.replace("\\", "/").split("/")[-1] == file), None)
    return found


def upscaler_status(available=None):
    """The upscalers on offer, each with whether its files are installed and what is missing."""
    rows = []
    for name, entry in UPSCALERS.items():
        files = upscaler_files(name, available)
        missing = [entry["files"][kind] for kind in ("model", "vae") if not files[kind]]
        rows.append({"id": name, "title": entry["title"], "about": entry["about"], "detail": entry["detail"], "installed": not missing,
                     "missing": missing, "source": entry["source"]})
    return rows


def pick_upscaler(name, available=None):
    """The upscaler to use: the one asked for, or the first that is installed. Its files come back with it."""
    if name in (None, ""):
        name = next((row["id"] for row in upscaler_status(available) if row["installed"]), next(iter(UPSCALERS)))
    if name not in UPSCALERS:
        raise ValueError("upscaler must be one of %s" % ", ".join(UPSCALERS))
    files = upscaler_files(name, available)
    missing = [UPSCALERS[name]["files"][kind] for kind in ("model", "vae") if not files[kind]]
    if missing:
        raise ValueError("Upscaling with %s needs %s in ComfyUI's models folder (diffusion_models and vae), from %s on Hugging Face"
                         % (UPSCALERS[name]["title"], " and ".join(missing), UPSCALERS[name]["source"]))
    return name, files


def upscale_graph(folder, clip, take, source, name, files, detail=UPSCALE_DETAIL):
    """The graph that enlarges one take of a clip two times with the named upscaler and saves it as another take."""
    file = "%s/%s" % (source["subfolder"], source["file"]) if source.get("subfolder") else source["file"]
    graph = {
        "30": {"class_type": "H3LoadClip", "inputs": {"file": file, "first": 0, "last": 0}},
        "31": {"class_type": "UNETLoader", "inputs": {"unet_name": files["model"], "weight_dtype": "default"}},
        "32": {"class_type": "VAELoader", "inputs": {"vae_name": files["vae"]}},
        "19": {"class_type": "SaveVideo", "inputs": {
            "video": ["18", 0], "filename_prefix": "%s/%s/clips/%s_%s" % (ROOT, folder, clip["id"], take),
            "format": "auto", "codec": "auto"}},
    }
    if name == "relayvsr":
        # A large model writes reference latents for sparse keyframes and a small decoder carries them over every frame.
        graph.update({
            "33": {"class_type": "RelayVSRConditioning", "inputs": {"model": ["31", 0], "images": ["30", 0]}},
            "34": {"class_type": "KSampler", "inputs": {
                "model": ["31", 0], "seed": 0, "steps": 1, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
                "positive": ["33", 0], "negative": ["33", 0], "latent_image": ["33", 1], "denoise": 1.0}},
            "35": {"class_type": "VSRDecode", "inputs": {"vae": ["32", 0], "samples": ["34", 0], "images": ["30", 0], "detail": detail}},
            "18": {"class_type": "CreateVideo", "inputs": {"images": ["35", 0], "fps": float(FPS), "audio": ["30", 1]}},
        })
    else:
        # SeedVR2 redraws the plainly enlarged picture in one step. It is encoded and decoded in tiles and sampled in
        # chunks of frames that fit the GPU, since a whole clip at this size does not.
        graph.update({
            "40": {"class_type": "ResizeImageMaskNode", "inputs": {
                "input": ["30", 0], "resize_type": "scale by multiplier", "resize_type.multiplier": 2.0, "scale_method": "lanczos"}},
            "41": {"class_type": "SeedVR2Preprocess", "inputs": {"resized_images": ["40", 0]}},
            "42": {"class_type": "VAEEncodeTiled", "inputs": {
                "pixels": ["41", 0], "vae": ["32", 0], "tile_size": 512, "overlap": 128, "temporal_size": 64, "temporal_overlap": 8}},
            "43": {"class_type": "SeedVR2TemporalChunk", "inputs": {"latent": ["42", 0], "temporal_overlap": 2, "chunking_mode": "auto"}},
            "44": {"class_type": "SeedVR2Conditioning", "inputs": {"model": ["31", 0], "vae_conditioning": ["43", 0]}},
            "34": {"class_type": "KSampler", "inputs": {
                "model": ["31", 0], "seed": 0, "steps": 1, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
                "positive": ["44", 0], "negative": ["44", 1], "latent_image": ["43", 0], "denoise": 1.0}},
            "45": {"class_type": "SeedVR2TemporalMerge", "inputs": {"latents": ["34", 0], "temporal_overlap": ["43", 1]}},
            "46": {"class_type": "VAEDecodeTiled", "inputs": {
                "samples": ["45", 0], "vae": ["32", 0], "tile_size": 512, "overlap": 128, "temporal_size": 64, "temporal_overlap": 8}},
            # The colours are matched back to the take, so an upscaled clip sits beside its neighbours.
            "47": {"class_type": "SeedVR2PostProcessing", "inputs": {
                "images": ["46", 0], "original_resized_images": ["40", 0], "color_correction_method": "lab"}},
            "18": {"class_type": "CreateVideo", "inputs": {"images": ["47", 0], "fps": float(FPS), "audio": ["30", 1]}},
        })
    return graph


def plan_upscale(project, folder, clip_id, available=None, detail=None, upscaler=None):
    """The graph that enlarges a clip's active take two times, and the job that will make the result a take."""
    clip = find_clip(project, clip_id)
    if any(j["clip"] == clip_id for j in project.get("pending", [])):
        raise ValueError("%s is already rendering" % clip["name"])
    source = active_take(clip)
    if not source:
        raise ValueError("%s has no render to upscale" % clip["name"])
    if source.get("upscaled"):
        raise ValueError("%s is already upscaled. Choose the take it was made from to upscale again" % clip["name"])
    if source.get("quality") == "draft" and clip.get("kind") != "footage":
        raise ValueError("%s is a draft. Render a final first, then upscale it" % clip["name"])
    if int(source.get("frames") or 0) > UPSCALE_MOST:
        raise ValueError("%s is %s frames. Upscaling holds every enlarged frame in memory, so it takes up to %s at a time"
                         % (clip["name"], source.get("frames"), UPSCALE_MOST))
    name, files = pick_upscaler(upscaler, available)
    width, height = take_size(project, source)
    if not (2 * width <= MAX_SIDE and 2 * height <= MAX_SIDE):
        raise ValueError("%s is %sx%s: twice that is larger than the editor handles" % (clip["name"], width, height))
    detail = clean_detail(detail)
    take = new_id("t")
    job = {"clip": clip_id, "take": take, "quality": "final", "seed": int(source.get("seed") or 0), "saved": copy.deepcopy(source.get("saved")),
           "size": [2 * width, 2 * height], "upscaled": source["id"], "upscaler": name,
           # The take keeps what made its picture, plus the upscaler. A batch groups it by the model it loads.
           "models": dict(source.get("models") or {}, video_upscaler=files["model"]), "loads": files["model"]}
    if UPSCALERS[name]["detail"]:
        job["detail"] = detail
    return job, upscale_graph(folder, clip, take, source, name, files, detail)


def plan_render(project, folder, clip_id, quality, seed=None, available=None, detail=None, upscaler=None):
    """The graph for a new take, and the job that will turn its file into a take. detail and upscaler are for an upscale only."""
    if quality == UPSCALE_QUALITY:
        return plan_upscale(project, folder, clip_id, available, detail, upscaler)
    clip = find_clip(project, clip_id)
    if clip.get("kind") == "footage":
        raise ValueError("%s is imported footage. Extend or bridge it to generate from it" % clip["name"])
    if any(j["clip"] == clip_id for j in project.get("pending", [])):
        raise ValueError("%s is already rendering" % clip["name"])
    refresh_link(project, clip)
    if seed in (None, ""):
        seed = clip.get("seed")
    if seed in (None, ""):
        seed = random.randrange(2 ** 31)
    take = new_id("t")
    graph = render_graph(project, folder, clip, take, quality, int(seed), available)
    job = {"clip": clip_id, "take": take, "quality": quality, "seed": int(seed),
           "saved": {k: copy.deepcopy(clip.get(k)) for k in SAVED_FIELDS},
           "models": dict(render_models(graph), profile=project_profile(project, available, clip)[0],
                          profile_title=project_profile(project, available, clip)[1]["title"],
                          steps=graph["12"]["inputs"]["steps"], sampler=graph["11"]["inputs"]["sampler_name"])}
    if "91" in graph:
        job["size"] = [graph["80"]["inputs"]["width"], graph["80"]["inputs"]["height"]]
    elif "36" in graph:
        # A sharp final: the take records the upscaler that brought it to this size.
        job["size"] = [graph["36"]["inputs"]["width"], graph["36"]["inputs"]["height"]]
        job["upscaler"], job["detail"] = SHARP_UPSCALER, SHARP_DETAIL
        job["models"]["video_upscaler"] = graph["31"]["inputs"]["unet_name"]
    elif tuple(render_size(project, quality)[:2]) != tuple(frame_size(project, quality)):
        # A size the project chose. A take that records none is at the usual size for its format.
        job["size"] = [graph["10"]["inputs"]["width"], graph["10"]["inputs"]["height"]]
    if clip.get("track"):
        # Once rendered, the take's sound is replaced by this exact slice of the track.
        job["track"] = dict(clip["track"], seconds=graph["10"]["inputs"]["length"] / float(FPS))
    return job, graph


def add_pending(project, job):
    find_clip(project, job["clip"]).pop("error", None)
    project.setdefault("pending", []).append(job)


def pop_pending(project, job_id):
    pending = project.get("pending") or []
    job = next((j for j in pending if j.get("id") == job_id), None)
    project["pending"] = [j for j in pending if j.get("id") != job_id]
    return job


def finish_render(project, job_id, file, info):
    """Attach a finished render. Returns the clip, or None when the clip was removed meanwhile."""
    job = pop_pending(project, job_id)
    if not job:
        return None
    try:
        find_clip(project, job["clip"])
    except KeyError:
        return None
    take = {"id": job["take"], "file": file["filename"], "subfolder": file.get("subfolder") or "",
            "poster": info["poster"], "frames": int(info["frames"]), "quality": job["quality"],
            "seed": job["seed"], "saved": job.get("saved")}
    for key in ("models", "upscale", "size", "upscaled", "upscaler", "detail"):
        if job.get(key):
            take[key] = job[key]
    if info.get("strip"):
        take["strip"] = info["strip"]
    return accept_take(project, job["clip"], take)


def fail_render(project, job_id, message=None):
    job = pop_pending(project, job_id)
    if job and message:
        clip = next((c for c in project.get("clips", []) if c["id"] == job["clip"]), None)
        if clip is not None:
            clip["error"] = str(message)[:400]
    return job


# Merging. Two writers can change a project at once: the editor, an agent, and finished renders.

MISSING = object()   # a key or element that one side does not have
UNKNOWN = object()   # no common version to compare against
THEIRS_ON_CONFLICT = ("active", "error", "revision")


def _copy(value):
    return value if value is MISSING else copy.deepcopy(value)


def _is_id_list(value):
    return isinstance(value, list) and all(isinstance(x, dict) and "id" in x for x in value)


def _merge_lists(base, mine, theirs, keep_theirs=False):
    """Lists of things with ids merge by id. A removal wins over an edit, unless keep_theirs protects theirs.

    Without a common version a removal cannot be told from an addition, so nothing is dropped.
    """
    unknown = base is UNKNOWN
    base_by = {x["id"]: x for x in base} if _is_id_list(base) else {}
    mine_by = {x["id"]: x for x in mine}
    theirs_by = {x["id"]: x for x in theirs}
    out = []
    for element in theirs:
        key = element["id"]
        if key in mine_by:
            out.append(merge(UNKNOWN if unknown else base_by.get(key, MISSING), mine_by[key], element))
        elif unknown or keep_theirs or key not in base_by:
            out.append(copy.deepcopy(element))
    for element in mine:
        if element["id"] not in theirs_by and (unknown or element["id"] not in base_by):
            out.append(copy.deepcopy(element))
    if not keep_theirs and _is_id_list(base) and [x["id"] for x in theirs] == [x["id"] for x in base]:
        # Only mine changed what the list holds, so its order stands: an undone removal comes back in its place.
        rank = {x["id"]: n for n, x in enumerate(mine)}
        out.sort(key=lambda x: rank.get(x["id"], len(rank)))
    return out


def merge(base, mine, theirs, key=None):
    """Three-way merge of a project, or of any part of one.

    base is the version both sides started from, or UNKNOWN. Renders and the render queue belong
    to the server, so a stale copy can never drop a take or a job.
    """
    if key == "pending":
        return _copy(theirs)
    if key == "takes":
        return _merge_lists(base if base is UNKNOWN or isinstance(base, list) else [],
                            mine if isinstance(mine, list) else [], theirs if isinstance(theirs, list) else [],
                            keep_theirs=True)
    if mine == theirs:
        return _copy(theirs)
    if base is not UNKNOWN and mine == base:
        return _copy(theirs)
    nested = (isinstance(mine, dict) and isinstance(theirs, dict)) or (_is_id_list(mine) and _is_id_list(theirs))
    if base is not UNKNOWN and theirs == base and not nested:
        return _copy(mine)
    # Containers are always walked, even when only one side changed, so takes and the queue stay protected inside them.
    if isinstance(mine, dict) and isinstance(theirs, dict):
        out = {}
        for name in list(theirs) + [n for n in mine if n not in theirs]:
            known = base.get(name, MISSING) if isinstance(base, dict) else (UNKNOWN if base is UNKNOWN else MISSING)
            value = merge(known, mine.get(name, MISSING), theirs.get(name, MISSING), name)
            if value is not MISSING:
                out[name] = value
        return out
    if _is_id_list(mine) and _is_id_list(theirs):
        return _merge_lists(base, mine, theirs)
    if mine is MISSING or theirs is MISSING:
        # One side removed what the other changed. The removal wins, unless there is nothing to tell them apart.
        return _copy(theirs if mine is MISSING else mine) if base is UNKNOWN else MISSING
    return _copy(theirs if key in THEIRS_ON_CONFLICT else mine)


# Notes a reviewer pins to a frame of a take: what should change there, with a pin, a box or a
# drawing on the picture. A note belongs to the clip and the take it was written on, and counts
# frames inside that take, so it stays on its frame when the clip moves on the timeline or is trimmed.

NOTE_FIELDS = ("clip", "take", "frame", "range", "shape", "text", "author", "still")
NOTE_TEXT = 2000
NOTE_PATHS, NOTE_POINTS = 40, 2000


def _fraction(value, what):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
        raise ValueError("%s is a fraction of the frame, from 0 to 1" % what)
    return round(float(value), 5)


def clean_shape(value):
    """Where on the picture a note points: a pin, a box or freehand strokes, in fractions of the frame from its top left."""
    if value is None:
        return None
    if not isinstance(value, dict) or value.get("type") not in ("pin", "rect", "stroke"):
        raise ValueError('shape is {"type": "pin", "x", "y"}, {"type": "rect", "x", "y", "w", "h"} or {"type": "stroke", "paths"}')
    if value["type"] == "pin":
        return {"type": "pin", "x": _fraction(value.get("x"), "x"), "y": _fraction(value.get("y"), "y")}
    if value["type"] == "rect":
        shape = {"type": "rect"}
        for key in ("x", "y", "w", "h"):
            shape[key] = _fraction(value.get(key), key)
        if shape["x"] + shape["w"] > 1.00001 or shape["y"] + shape["h"] > 1.00001:
            raise ValueError("the box runs past the edge of the frame")
        return shape
    paths = value.get("paths")
    if not isinstance(paths, list) or not paths or len(paths) > NOTE_PATHS:
        raise ValueError("a stroke shape has 1 to %s paths" % NOTE_PATHS)
    made = []
    for path in paths:
        if not isinstance(path, list) or not path or len(path) > NOTE_POINTS:
            raise ValueError("each path has 1 to %s points" % NOTE_POINTS)
        made.append([[_fraction(p[0], "x"), _fraction(p[1], "y")] for p in path
                     if isinstance(p, (list, tuple)) and len(p) == 2])
    return {"type": "stroke", "paths": made}


def find_note(project, note_id):
    for note in project.get("notes") or []:
        if isinstance(note, dict) and note.get("id") == note_id:
            return note
    raise KeyError("no note %s" % note_id)


def add_note(project, fields):
    """A note on a frame of a clip's take. The take is the clip's active one unless named."""
    check_fields(fields, NOTE_FIELDS)
    clip = find_clip(project, fields.get("clip"))
    takes = {t["id"]: t for t in clip.get("takes") or []}
    take = takes.get(fields.get("take")) if fields.get("take") else active_take(clip)
    if take is None:
        raise ValueError("%s has no take %s" % (clip["name"], fields["take"]) if fields.get("take")
                         else "%s has no render yet, so it has no frames to write a note on" % clip["name"])
    last = max(0, int(take.get("frames") or 1) - 1)

    def frame_of(value, what):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= last:
            raise ValueError("%s is a frame of the take, from 0 to %s" % (what, last))
        return value

    text = str(fields.get("text") or "").strip()
    if not text:
        raise ValueError("a note needs text: what should change at that frame")
    note = {"id": new_id("n"), "clip": clip["id"], "take": take["id"], "frame": frame_of(fields.get("frame"), "frame"),
            "range": None, "shape": clean_shape(fields.get("shape")), "text": text[:NOTE_TEXT], "status": "open",
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    span = fields.get("range")
    if span is not None:
        if not isinstance(span, dict):
            raise ValueError('range is {"from", "to"}, frames of the take')
        start, end = frame_of(span.get("from"), "range from"), frame_of(span.get("to"), "range to")
        if end < start:
            raise ValueError("a range ends after it starts")
        note["range"] = {"from": start, "to": end}
        note["frame"] = start
    if fields.get("author"):
        note["author"] = str(fields["author"]).strip()[:60]
    if isinstance(fields.get("still"), dict) and fields["still"].get("file"):
        note["still"] = {"file": str(fields["still"]["file"]), "subfolder": str(fields["still"].get("subfolder") or "")}
    project.setdefault("notes", []).append(note)
    return note


def resolve_note(project, note_id, resolution=""):
    """Close a note, with a line on what was done about it."""
    note = find_note(project, note_id)
    note["status"] = "resolved"
    note["resolved"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if str(resolution or "").strip():
        note["resolution"] = str(resolution).strip()[:NOTE_TEXT]
    return note


def reopen_note(project, note_id):
    note = find_note(project, note_id)
    note["status"] = "open"
    note.pop("resolved", None)
    return note


def remove_note(project, note_id):
    find_note(project, note_id)
    project["notes"] = [n for n in project.get("notes") or [] if not (isinstance(n, dict) and n.get("id") == note_id)]


def note_place(project, note):
    """The timeline frame where a note's frame is on screen now, or None when its frame is not on the timeline."""
    frame = note.get("frame")
    if not isinstance(frame, int):
        return None
    for item in ordered(project):
        if item.get("clip") == note.get("clip") and item["in"] <= frame < item["out"]:
            return int(round(item["start"] + (frame - item["in"]) / clean_speed(item.get("speed"))))
    return None


def shape_words(shape):
    """A note's shape in words, for a reader with no picture."""
    if not isinstance(shape, dict):
        return "the whole frame"
    pct = lambda v: "%d%%" % round(v * 100)
    if shape.get("type") == "pin":
        return "a pin at %s across, %s down" % (pct(shape["x"]), pct(shape["y"]))
    if shape.get("type") == "rect":
        return "a box from %s to %s across and %s to %s down" % (
            pct(shape["x"]), pct(shape["x"] + shape["w"]), pct(shape["y"]), pct(shape["y"] + shape["h"]))
    points = [p for path in shape.get("paths") or [] for p in path]
    if not points:
        return "a drawing"
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return "a drawing within %s to %s across and %s to %s down" % (pct(min(xs)), pct(max(xs)), pct(min(ys)), pct(max(ys)))


def describe_notes(project):
    """Every note, open ones first, each with where it is on the timeline now."""
    clips = {c["id"]: c for c in project.get("clips", [])}
    rows = []
    for note in project.get("notes") or []:
        if not isinstance(note, dict) or note.get("clip") not in clips or not note.get("id"):
            continue
        clip = clips[note["clip"]]
        place = note_place(project, note)
        active = active_take(clip)
        row = {"note": note["id"], "status": note.get("status") or "open", "clip": clip["id"], "name": clip["name"],
               "take": note.get("take"), "frame": note.get("frame"), "range": note.get("range"),
               "at": timecode(place) if place is not None else None, "timeline_frame": place,
               "text": note.get("text", ""), "where": shape_words(note.get("shape"))}
        if active is not None and note.get("take") != active["id"]:
            row["earlier_take"] = True
        for key in ("author", "created", "resolved", "resolution", "still"):
            if note.get(key):
                row[key] = note[key]
        rows.append(row)
    rows.sort(key=lambda r: (r["status"] != "open", r["timeline_frame"] is None, r["timeline_frame"] or 0))
    return rows


# A short account of a project, for agents and for checking work.

def clip_state(project, clip):
    if any(j["clip"] == clip["id"] for j in project.get("pending", [])):
        return "rendering"
    if clip.get("error"):
        return "failed"
    if not clip.get("takes"):
        return "not rendered"
    if is_stale(project, clip):
        return "source changed"
    return (active_take(clip) or {}).get("quality", "draft")


def describe(project):
    profiles = project_profiles(project)
    clips = []
    for clip in project.get("clips", []):
        take = active_take(clip)
        entry = {"id": clip["id"], "name": clip["name"], "state": clip_state(project, clip),
                 "seconds": clip.get("seconds"), "action": clip.get("action", ""), "takes": len(clip.get("takes") or [])}
        if take:
            entry.update({"take": take["id"], "frames": take["frames"], "take_seed": take.get("seed"), "quality": take.get("quality")})
        if clip.get("seed") is not None:
            entry["seed"] = clip["seed"]
        if clip.get("link"):
            entry["link"] = clip["link"]["mode"]
        if clip.get("kind") == "footage":
            entry["footage"] = True
        if clip.get("bin"):
            entry["bin"] = clip["bin"]
        if clip.get("profile") in profiles:
            entry["profile"] = clip["profile"]
        if clip.get("track"):
            entry["track"] = {"file": clip["track"]["file"], "from": clip["track"]["from"]}
        if clip.get("error"):
            entry["error"] = clip["error"]
        for name in ("speaker", "dialogue", "cast"):
            if clip.get(name):
                entry[name] = clip[name]
        clips.append(entry)
    names = {c["id"]: c["name"] for c in project.get("clips", [])}
    timeline = []
    for item in ordered(project):
        entry = {"item": item["id"], "clip": item["clip"], "name": names.get(item["clip"], "?"),
                 "start": item["start"], "end": item_end(item), "at": timecode(item["start"]),
                 "in": item["in"], "out": item["out"]}
        if clean_speed(item.get("speed")) != 1:
            entry["speed"] = item["speed"]
        if item.get("transition"):
            entry["transition"] = item["transition"]
        timeline.append(entry)
    gaps = []
    end = 0
    for item in ordered(project):
        if item["start"] > end:
            gaps.append({"start": end, "end": item["start"]})
        end = max(end, item_end(item))
    sounds = [{"sound": s["id"], "name": s.get("name", ""), "lane": s.get("lane", 0), "start": s.get("start", 0),
               "end": s.get("start", 0) + s.get("frames", 0), "at": timecode(s.get("start", 0)),
               "offset": s.get("offset", 0.0), "gain": s.get("gain", 1.0)}
              for s in sorted(project.get("sounds") or [], key=lambda s: (s.get("lane", 0), s.get("start", 0)))]
    return {
        "sounds": sounds,
        "effects": [{"effect": e["id"], "name": e.get("name", ""), "start": e["start"], "frames": e["frames"],
                     "at": timecode(e["start"]), "state": "waiting for an open editor to render it" if e.get("queued") is True else
                     "rendering" if e.get("queued") else "not rendered" if not active_take(e) else
                     "out of date" if effect_stale(project, e) else "rendered",
                     **({"error": e["error"]} if e.get("error") else {})}
                    for e in sorted(project.get("effects") or [], key=lambda e: e["start"])],
        "name": project.get("name"), "revision": project.get("revision", 0), "style": project.get("style", ""),
        "format": project.get("format") or "16:9", "language": project.get("language") or LANGUAGE,
        "profile": project_profile(project)[0],
        "sizes": {quality: dict(zip(("width", "height"), render_size(project, quality)[:2]),
                                name=(project.get("sizes") or {}).get(quality) if isinstance(project.get("sizes"), dict)
                                and (project.get("sizes") or {}).get(quality) in SIZE_CHOICES[quality] else SIZE_CHOICES[quality][0])
                  for quality in SIZE_CHOICES},
        "checkpoints": [{"id": name, "title": profile["title"], "unet": profile["unet"][0], "steps": profile["steps"],
                         "sampler": profile["sampler"], "scheduler": profile.get("scheduler") or "simple"}
                        for name, profile in profiles.items() if profile.get("custom")],
        "loras": [{"file": r["file"], "strength": r.get("strength", 1.0), "on": r.get("on") is not False}
                  for r in project.get("loras") or [] if isinstance(r, dict) and r.get("file")],
        "frames": scene_end(project), "length": timecode(scene_end(project)), "fps": FPS,
        "cast": [{"id": p["id"], "name": p["name"], "description": p.get("description", ""),
                  "picture": bool(p.get("file")), "pictures": len(member_pictures(p)), "voice": bool(p.get("voice")),
                  **({"refmod": p["refmod"]["file"]} if isinstance(p.get("refmod"), dict) and p["refmod"].get("file") else {}),
                  **({"catalog": p["catalog"]["id"]} if isinstance(p.get("catalog"), dict) and p["catalog"].get("id") else {})}
                 for p in project.get("cast", [])],
        "bins": [{"id": b["id"], "name": b["name"], "clips": sum(1 for c in project.get("clips", []) if c.get("bin") == b["id"])}
                 for b in project.get("bins", [])],
        "clips": clips, "timeline": timeline, "gaps": gaps, "notes": describe_notes(project),
        "rendering": [{"job": j.get("id"), "clip": j["clip"], "quality": j["quality"]} for j in project.get("pending", [])],
    }


# A batch of operations, as an agent sends them. Later operations can name what earlier ones made.

def _resolve(value, names, kind):
    if not isinstance(value, str) or not value.startswith("$"):
        return value
    name, _, part = value[1:].partition(".")
    if name not in names:
        raise ValueError("nothing earlier in this batch is called $%s" % name)
    made = names[name]
    found = made.get(part or kind) or made.get("clip" if kind == "clip" else "item")
    if not found:
        raise ValueError("$%s has no %s" % (name, part or kind))
    return found


def apply_op(project, op, names):
    """One edit. Returns what it made, so an agent can refer to it. Renders are queued by the server."""
    if not isinstance(op, dict) or not op.get("op"):
        raise ValueError("each operation needs an op")
    kind = op["op"]
    clip_id = _resolve(op.get("clip"), names, "clip")
    item_id = _resolve(op.get("item"), names, "item")
    fields = op.get("fields")
    if isinstance(fields, dict) and isinstance(fields.get("cast"), list):
        # A cast member added earlier in the batch can be named in a clip's cast list.
        op = dict(op, fields=dict(fields, cast=[_resolve(c, names, "cast") for c in fields["cast"]]))
    if isinstance(fields, dict) and fields.get("bin") is not None:
        op = dict(op, fields=dict(op["fields"], bin=_resolve(fields["bin"], names, "bin")))
    bin_id = _resolve(op.get("bin"), names, "bin")
    if kind == "add_clip":
        clip, item = add_clip(project, op.get("fields") or {}, op.get("start"), op.get("place", True))
        result = {"clip": clip["id"], "item": item["id"] if item else None}
    elif kind == "update_clip":
        update_clip(project, clip_id, op.get("fields") or {})
        result = {"clip": clip_id}
    elif kind == "remove_clip":
        remove_clip(project, clip_id)
        result = {"clip": clip_id}
    elif kind == "place_clip":
        result = {"clip": clip_id, "item": place_clip(project, clip_id, op.get("start"))["id"]}
    elif kind == "update_item":
        update_item(project, item_id, op.get("fields") or {}, bool(op.get("ripple")))
        result = {"item": item_id}
    elif kind == "remove_item":
        remove_item(project, item_id, bool(op.get("ripple")))
        result = {"item": item_id}
    elif kind == "split_item":
        result = {"item": item_id, "right": split_item(project, item_id, op["frame"])["id"]}
    elif kind == "close_gap":
        result = {"removed": close_gap(project, op.get("frame"))}
    elif kind in LINK_MODES:
        clip, item = add_linked(project, item_id, kind, op.get("seconds"), op.get("action") or "")
        if op.get("fields"):
            update_clip(project, clip["id"], op["fields"])
        result = {"clip": clip["id"], "item": item["id"]}
    elif kind == "choose_take":
        choose_take(project, clip_id, op["take"])
        result = {"clip": clip_id}
    elif kind == "set_project":
        set_project(project, op.get("fields") or {})
        result = {}
    elif kind == "add_cast":
        result = {"cast": add_cast(project, op.get("fields") or {})["id"]}
    elif kind == "update_cast":
        update_cast(project, _resolve(op.get("cast"), names, "cast"), op.get("fields") or {})
        result = {}
    elif kind == "remove_cast":
        remove_cast(project, _resolve(op.get("cast"), names, "cast"))
        result = {}
    elif kind == "add_bin":
        result = {"bin": add_bin(project, op.get("fields") or {})["id"]}
    elif kind == "update_bin":
        update_bin(project, bin_id, op.get("fields") or {})
        result = {"bin": bin_id}
    elif kind == "remove_bin":
        remove_bin(project, bin_id)
        result = {"bin": bin_id}
    elif kind == "add_sound":
        result = {"sound": add_sound(project, op.get("asset"), op.get("seconds"), op.get("fields") or {})["id"]}
    elif kind == "update_sound":
        sound_id = _resolve(op.get("sound"), names, "sound")
        update_sound(project, sound_id, op.get("fields") or {})
        result = {"sound": sound_id}
    elif kind == "remove_sound":
        sound_id = _resolve(op.get("sound"), names, "sound")
        remove_sound(project, sound_id)
        result = {"sound": sound_id}
    elif kind == "add_note":
        fields = dict(op.get("fields") or {})
        if "clip" in fields:
            fields["clip"] = _resolve(fields["clip"], names, "clip")
        result = {"note": add_note(project, fields)["id"]}
    elif kind == "resolve_note":
        note_id = _resolve(op.get("note"), names, "note")
        resolve_note(project, note_id, op.get("resolution") or "")
        result = {"note": note_id}
    elif kind == "reopen_note":
        note_id = _resolve(op.get("note"), names, "note")
        reopen_note(project, note_id)
        result = {"note": note_id}
    elif kind == "remove_note":
        note_id = _resolve(op.get("note"), names, "note")
        remove_note(project, note_id)
        result = {"note": note_id}
    elif kind == "add_effect":
        result = {"effect": add_effect(project, op.get("fields") or {})["id"]}
    elif kind == "update_effect":
        effect_id = _resolve(op.get("effect"), names, "effect")
        update_effect(project, effect_id, op.get("fields") or {})
        result = {"effect": effect_id}
    elif kind == "render_effect":
        effect_id = _resolve(op.get("effect"), names, "effect")
        request_effect_render(project, effect_id)
        result = {"effect": effect_id}
    elif kind == "remove_effect":
        effect_id = _resolve(op.get("effect"), names, "effect")
        remove_effect(project, effect_id)
        result = {"effect": effect_id}
    else:
        raise ValueError("unknown op %s" % kind)
    if op.get("as"):
        names[str(op["as"])] = result
    return result
