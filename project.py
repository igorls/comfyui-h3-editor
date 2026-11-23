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
    from . import h3image
    from . import sprite
    from . import holds
    from . import style_timeline
    from . import vosr2
    from . import motion_replace
    from . import patches as patching
    from .prompting import compose_prompt, compose_reference_prompt
except ImportError:
    import h3image
    import sprite
    import holds
    import style_timeline
    import vosr2
    import motion_replace
    import patches as patching
    from prompting import compose_prompt, compose_reference_prompt

FPS = 24
JOINT = 51
ROOT = "h3take_projects"
QUALITIES = {
    "draft": {},
    # A final renders at the final size, H3's own size of about a megapixel. Its picture is
    # then upscaled to the refine size, where the model redraws its last steps. The Realism
    # Slider skips the first steps of the first pass, which set the framing and pose, and runs
    # across the whole refine pass at this strength. With the video shift of 12, HyperFlow's six
    # steps sit at sigma 1.0, 0.983, 0.915, 0.675, 0.385 and 0.159: 0.93 leaves out the first
    # two. At 0.75 it only joined for the last three, where it changed next to nothing.
    "final": {"refine": True, "lora": 1.2, "lora_start_sigma": 0.93},
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
# A project can also write a size of its own, as "1024x576", for a draft or a final. Each side
# is a multiple of 32, which H3's latent needs, in this range, and the picture comes out as
# sampled. Any shape goes: the cut takes the shape of its largest picture.
SIDE_STEP, SIDE_MIN, SIDE_MAX = 32, 256, 2048
# What the editor offers first when someone asks for a size of their own: about 0.6 megapixels.
OWN_SIZES = {"16:9": (1024, 576), "9:16": (576, 1024), "1:1": (768, 768)}
FORMAT_PRESETS = (*FORMATS, "3:2", "4:3", "7:5", "21:9")
# A sharp final is enlarged two times by this upscaler, keeping all the detail it adds, and
# scaled down to the refine size. H3 is not asked to draw at a size it was not made for, so
# skin stays as it is at H3's own size, and the edges come out crisper than a redraw gives.
SHARP = "sharp"
SHARP_UPSCALER = "relayvsr"
SHARP_DETAIL = 1.0
RELAYVSR_MODEL = "relayvsr_x2_lightx2v_int8_convrot.safetensors"
RELAYVSR_VAE = "relayvsr_flash_vae_x2_fp32.safetensors"
LANGUAGE = "English"
# The range a recorded picture side must fall in before export trusts it.
MIN_SIDE, MAX_SIDE = 16, 8192


def clean_format(value):
    match = re.fullmatch(r"\s*(\d{1,5})\s*:\s*(\d{1,5})\s*", value) if isinstance(value, str) else None
    if match:
        wide, high = map(int, match.groups())
        if high and SIDE_MIN / SIDE_MAX <= wide / high <= SIDE_MAX / SIDE_MIN:
            divisor = math.gcd(wide, high)
            reduced = "%s:%s" % (wide // divisor, high // divisor)
            return "21:9" if reduced == "7:3" else reduced
    raise ValueError("format must be a ratio such as 16:9, 9:16, 1:1 or 4:3, between 1:8 and 8:1")


def project_format(project):
    try:
        return clean_format(project.get("format"))
    except ValueError:
        return "16:9"


def fitted_size(shape, target, *, fit=False):
    """Resolve a ratio and a longest side or pixel budget onto H3's 32-pixel grid."""
    if not isinstance(target, dict) or set(target) not in ({"longest_side"}, {"megapixels"}):
        raise ValueError("a size target takes either longest_side or megapixels")
    key, value = next(iter(target.items()))
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= SIDE_MAX ** 2:
        raise ValueError("%s must be a positive number within the supported picture size" % key)
    wide, high = map(int, shape.split(":"))
    aspect = max(wide, high) / min(wide, high)
    longest = value if key == "longest_side" else math.sqrt(value * 1_000_000 * aspect)
    if fit:
        longest = max(SIDE_MIN * aspect, min(SIDE_MAX, longest))
    elif longest < SIDE_MIN * aspect - 1e-8 or longest > SIDE_MAX + 1e-8:
        raise ValueError("this target at %s needs a side outside %s–%s pixels" % (shape, SIDE_MIN, SIDE_MAX))
    sides = tuple(SIDE_STEP * math.floor(n / SIDE_STEP + 0.5) for n in (longest, longest / aspect))
    return sides if wide >= high else sides[::-1]


def format_sizes(shape):
    if shape in FORMATS:
        return {**SIZES[shape], "refine": FORMATS[shape]["refine"], "own": OWN_SIZES[shape]}
    return {name: fitted_size(shape, {"megapixels": mp}, fit=True)
            for name, mp in (("small", 0.23), ("medium", 0.52), ("large", 1.04), ("refine", 2.07), ("own", 0.59))}


def frame_size(project, quality):
    """Width and height for a quality in this project's format. Imported footage counts as final."""
    sizes = format_sizes(project_format(project))
    return sizes["small" if quality == "draft" else "large"]


def render_size(project, quality):
    """Width and height a new render of this quality is sampled at, and whether it is then enlarged.

    The project's own choice under "sizes" where it makes one, or the usual size for its format.
    """
    quality = "draft" if quality == "draft" else "final"
    sizes = format_sizes(project_format(project))
    name = size_name(project, quality)
    if name in ("full", SHARP):
        return sizes["large"] + (name == "full",)
    return (own_size(name) or sizes[name]) + (False,)


def own_size(name):
    """Width and height of a size written as "1024x576", or None when it is not one a render can use."""
    found = re.fullmatch(r"(\d{1,5})x(\d{1,5})", name) if isinstance(name, str) else None
    if not found:
        return None
    sides = int(found.group(1)), int(found.group(2))
    return sides if all(SIDE_MIN <= side <= SIDE_MAX and side % SIDE_STEP == 0 for side in sides) else None


def size_name(project, quality):
    """The name of the size a new render of this quality uses: the project's choice, or the usual one.

    A size of the project's own is its name, "1024x576".
    """
    quality = "draft" if quality == "draft" else "final"
    chosen = project.get("sizes")
    name = chosen.get(quality) if isinstance(chosen, dict) else None
    return name if name in SIZE_CHOICES[quality] or own_size(name) else SIZE_CHOICES[quality][0]


def size_choices(shapes=FORMAT_PRESETS):
    """Every size a project can pick, per format and quality, for the editor: id, width, height, and the size it comes out at."""
    table = {}
    for shape in shapes:
        sizes = format_sizes(shape)
        table[shape] = {}
        for quality, names in SIZE_CHOICES.items():
            rows = []
            for name in names:
                width, height, refined = render_size({"format": shape, "sizes": {quality: name}}, quality)
                rows.append({"id": name, "width": width, "height": height,
                             "out": list(sizes["refine"] if refined or name == SHARP else (width, height))})
            table[shape][quality] = rows
        # What a size of the project's own may be, and the one to offer first.
        table[shape]["own"] = {"step": SIDE_STEP, "min": SIDE_MIN, "max": SIDE_MAX, "first": list(sizes["own"])}
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
    return format_sizes(project_format(project))["refine"]
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
# scheduler. lora is a LoRA the family always samples with, loaded by the pack's node or by the
# node lora_node names, and lora_text the one its fl2va file takes in its place. sigmas is the
# schedule a family samples on in place of a scheduler's.
# finals names the profile that renders a family's finals. A final's second pass starts at
# refine_from in its own sigmas, or runs refine_sigmas, or refine_sigmas_text on the fl2va file
# of a family that names one.
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
    "pi_h3": {
        "title": "pi-H3",
        "hint": "The original H3 with the 4-step pi-H3 LoRA, for drafts. Its finals render on HyperFlow H3 Flash.",
        "unet": ["minimax_h3_ref2va_pruned_int8_convrot.safetensors"],
        "unet_text": ["minimax_h3_fl2va_pruned_int8_convrot.safetensors"],
        # The adapter in ComfyUI's own layout. Besides its LoRA pairs it replaces whole tensors, which only
        # ComfyUI's loader applies, and it samples on the four sigmas it was distilled for.
        "lora": ["minimax_pi_h3_4step_fl2va_pruned_rank_64_fp16.safetensors", 1.0], "lora_node": "LoraLoaderModelOnly",
        "sigmas": "0.99654, 0.96445, 0.89930, 0.69565, 0.0",
        # At the refine size its own last step draws harsh skin, so a final, or a draft being finished, goes to
        # HyperFlow, which takes the draft's latents as they are. Without HyperFlow it refines with its last step.
        "finals": "hyperflow", "refine_sigmas": "0.69565, 0.0",
        "steps": {"draft": 4, "final": 6}, "sampler": "euler",
    },
    "lightx2v": {
        "title": "LightX2V H3 Turbo",
        "hint": "The original H3 with LightX2V's 8-step Turbo LoRAs, one for each of its two files. "
                "4 steps for drafts, 8 for finals.",
        "unet": ["minimax_h3_ref2va_pruned_int8_convrot.safetensors"],
        "unet_text": ["minimax_h3_fl2va_pruned_int8_convrot.safetensors"],
        # LightX2V distils each H3 file on its own: lora is for clips with references, on the ref2va file, and
        # lora_text for prompt-only clips, on the fl2va file. Plain LoRA pairs, which the pack's node reads.
        "lora": ["minimax_h3_ref2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors", 1.0],
        "lora_text": ["minimax_h3_fl2v_turbo_8step_v1.2_768p_comfyui_bf16.safetensors", 1.0],
        # The 8-step LoRAs hold a 640x352 draft at 4 steps, so a draft takes half the steps of a final.
        "steps": {"draft": 4, "final": 8}, "sampler": "euler", "refine_sigmas": REFINE_TAIL,
        # The fl2v v1.2 LoRA is only good for the high sigmas of its own schedule. Asked for a step from 0.385 or
        # below, as the usual tail does, it leaves a dot on every token: a grid 16 pixels apart over the picture.
        # So its refine is one step, from the last sigma of its 8 steps at shift 12. The ref2v LoRA takes the
        # usual tail cleanly.
        "refine_sigmas_text": "0.632, 0.0",
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
# The keys of a profile that name a LoRA it always samples with.
PROFILE_LORAS = ("lora", "lora_text")
# A scheduler the editor makes itself: the beta schedule with these two numbers, which other packs add to ComfyUI
# under this name. The editor builds it with ComfyUI's own node, so a checkpoint can ask for it on any install.
BETA57 = "beta57"
BETA57_SHAPE = (0.5, 0.7)


def sigma_steps(node):
    """How many steps a sigmas node runs: its own count, or one fewer than the sigmas it lists."""
    inputs = node["inputs"]
    if "steps" in inputs:
        return inputs["steps"]
    return max(1, len([part for part in str(inputs.get("sigmas", "")).split(",") if part.strip()]) - 1)


def lora_names(profile, key="lora"):
    """The names a profile's LoRA may be installed under: as written, then that file in any folder.

    key is lora, or lora_text for the one a family samples its prompt-only clips with.
    """
    name = profile[key][0]
    return [name, name.rsplit(".", 1)[0] + "*"]


def sigmas_node(scheduler, steps, sigmas=None):
    """The node that makes a render's sigmas: a profile's own list, or a scheduler by name."""
    if sigmas:
        return {"class_type": "ManualSigmas", "inputs": {"sigmas": sigmas}}
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
        # A checkpoint with its turbo built in takes the family's settings without the family's LoRAs.
        profile.pop("lora", None)
        profile.pop("lora_text", None)
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
        if (clip or {}).get("sprite") is not None and not (project.get("models") or {}).get("unet"):
            for candidate in profile_status(available):
                if candidate["id"] in ("hyperflow", "h3_turbo") and candidate["available"]:
                    name = candidate["id"]
                    break
    if not isinstance(name, str) or name not in profiles:
        name = installed_profile(available)
    return name, profiles[name]


def render_profile(project, available=None, clip=None, quality="draft"):
    """The profile one render samples with, and its id.

    A draft takes the clip's or the project's profile. So does a final, unless that profile hands its finals to
    another one whose files are installed.
    """
    name, profile = project_profile(project, available, clip)
    other = profile.get("finals")
    if quality != "draft" and other in PROFILES:
        if available is None or next(row for row in profile_status(available) if row["id"] == other)["available"]:
            return other, PROFILES[other]
    return name, profile


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


def project_patches(project, known=None):
    """The model patches a render samples with, in order: (the project's row, the node's definition).

    known maps the ids of the installed patch nodes to their definitions (patches.definition),
    or is None when the installed nodes are not known. Only the patches that are switched on
    and installed: a node's inputs and their defaults come from its definition, so without
    one there is nothing to wire. A project file can be written by hand, so anything that is
    not a patch row is passed over.
    """
    rows = []
    for row in project.get("patches") or []:
        if isinstance(row, dict) and isinstance(row.get("node"), str) and row.get("on") is not False and row["node"] in (known or {}):
            rows.append((row, known[row["node"]]))
    return rows[:PATCH_LIMIT]


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


# The files the editor's Get the models panel downloads into a fresh install: what the default profile
# drafts with, and the latent upscaler its finals are enlarged with. Each one's needs is the list a
# render picks its file from (MODEL_CHOICES, PROFILES, MODELS), so a file counts as installed exactly
# when a render would find it, and file is that list's first name, the one a render asks for when none
# is there. Each url is pinned to a commit of its repository, so the bytes are the ones sha256 names.
# Only these urls are ever fetched. The Realism Slider is on Civitai, which asks a signed-in account to
# download it, so the panel links to its page instead (manual).
MINIMAX_LICENSE = ("MiniMax H3 Community License Agreement", "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE")
HYPERFLOW_LICENSE = ("MiniMax H3 Community License Agreement", "https://huggingface.co/VideoRebirthAI/hyperflow-h3-flash/blob/main/LICENSE")
SAM3_LICENSE = ("SAM License", "https://huggingface.co/Comfy-Org/sam3.1/blob/main/LICENSE")
SWAP_LORA_LICENSE = ("MiniMax H3 Community License Agreement", "https://huggingface.co/akatz-ai/MiniMax-H3-Character-Swap-LoRA/blob/main/LICENSE")
MIT_SDPOSE_LICENSE = ("MIT License", "https://huggingface.co/Comfy-Org/SDPose")
APACHE_RELAYVSR_LICENSE = ("Apache License 2.0", "https://huggingface.co/Comfy-Org/RelayVSR")
SETUP_REPOS = {
    "Comfy-Org/MiniMax-H3": "4b53a35191437de05370da548e59aeabe2a5a09a",
    "VideoRebirthAI/hyperflow-h3-flash": "6834923e1a9ca4d15a2349a622b6677bea1404cb",
    "LBH-123-AI/Minimax_h3_latent_Upscaler": "85a0187096046badc0a71ab6bd6be182ab06c0f4",
    "Comfy-Org/sam3.1": "7bb8374780a725b4353ed31f3a9395c9742b5621",
    "akatz-ai/MiniMax-H3-Character-Swap-LoRA": "96ea3407b230b6f7df71c58c70e31df96bd7e533",
    "Comfy-Org/SDPose": "acb43fbe8142b54d9ebd52b720b81dd4e14c8d81",
    "Comfy-Org/RelayVSR": "202c35eb0a5bb64b7017d5a09ea9acea87b7a081",
}
SETUP_FILES = [
    {"id": "text_encoder", "title": "Text encoder, Qwen3-VL 32B", "needs": MODEL_CHOICES["clip"], "folder": "text_encoders",
     "repo": "Comfy-Org/MiniMax-H3", "path": "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
     "bytes": 15687142551, "sha256": "35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6",
     "license": MINIMAX_LICENSE, "for": "draft"},
    {"id": "vae", "title": "Video VAE", "needs": MODEL_CHOICES["vae"], "folder": "vae",
     "repo": "Comfy-Org/MiniMax-H3", "path": "vae/minimax_h3_video_vae_fp16.safetensors",
     "bytes": 5207808496, "sha256": "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522",
     "license": MINIMAX_LICENSE, "for": "draft"},
    {"id": "audio_vae", "title": "Audio VAE", "needs": MODEL_CHOICES["audio_vae"], "folder": "vae",
     "repo": "Comfy-Org/MiniMax-H3", "path": "vae/minimax_h3_audio_vae_fp32.safetensors",
     "bytes": 605254808, "sha256": "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48",
     "license": MINIMAX_LICENSE, "for": "draft"},
    {"id": "unet", "title": "HyperFlow H3 Flash, for clips with references", "needs": PROFILES[DEFAULT_PROFILE]["unet"],
     "folder": "diffusion_models", "repo": "VideoRebirthAI/hyperflow-h3-flash",
     "path": "ref2va_int8_convrot_comfyui/minimax_h3_ref2va_hyperflow_flash_14b-int8_convrot.safetensors",
     "bytes": 16433000936, "sha256": "38782037e343b49803dec6a912cf66cde54cafcac9e14a34fd4df831a83bcead",
     "license": HYPERFLOW_LICENSE, "for": "draft"},
    {"id": "unet_text", "title": "HyperFlow H3 Flash, for clips that are only a prompt", "needs": PROFILES[DEFAULT_PROFILE]["unet_text"],
     "folder": "diffusion_models", "repo": "VideoRebirthAI/hyperflow-h3-flash",
     "path": "fl2va_int8_convrot_comfyui/minimax_h3_fl2va_hyperflow_flash_14b-int8_convrot.safetensors",
     "bytes": 16005009056, "sha256": "d38f0c0c46824091fa448cd8f4a68b5d3832317867d5ddc02305eaa650710604",
     "license": HYPERFLOW_LICENSE, "for": "draft"},
    {"id": "latent_upscaler", "title": "Latent upscaler, for finals at up to 1920 pixels", "needs": [MODELS["latent_upscaler"]],
     "folder": "latent_upscale_models", "repo": "LBH-123-AI/Minimax_h3_latent_Upscaler",
     "path": "minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors",
     "bytes": 690592992, "sha256": "4f57821f5837f32f7142b67d815606dbd7550f194e5c769f7d6c3f83b146a5e6",
     "license": ("Apache License 2.0", "https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler"), "for": "final"},
    {"id": "realism", "title": "MMH3 Realism Slider LoRA, for natural skin in finals", "needs": list(REALISM_LORAS),
     "folder": "loras", "bytes": 74572040, "manual": "https://civitai.com/models/3041489",
     "license": ("Its terms on Civitai", "https://civitai.com/models/3041489"), "for": "final"},
]


def setup_url(entry):
    """Where a setup file downloads from: its path at the pinned commit of its Hugging Face repository."""
    return "https://huggingface.co/%s/resolve/%s/%s" % (entry["repo"], SETUP_REPOS[entry["repo"]], entry["path"])


def setup_status(available):
    """The setup files, each with whether a render finds one of its names, and what is left to download.

    available as in pick_file. ready says whether drafts can render with the default profile once the
    missing files are in; manual lists the missing files the panel can only link to.
    """
    rows = []
    for entry in SETUP_FILES:
        found = pick_file(entry["needs"], available)
        row = {"id": entry["id"], "title": entry["title"], "file": entry["needs"][0], "folder": entry["folder"],
               "bytes": entry["bytes"], "for": entry["for"], "installed": found is not None, "found": found,
               "license": {"name": entry["license"][0], "url": entry["license"][1]}}
        if entry.get("manual"):
            row["page"] = entry["manual"]
        else:
            row["source"] = "https://huggingface.co/%s" % entry["repo"]
        rows.append(row)
    missing = [row for row in rows if not row["installed"] and "page" not in row]
    return {"profile": PROFILES[DEFAULT_PROFILE]["title"], "files": rows, "missing": [row["id"] for row in missing],
            "bytes": sum(row["bytes"] for row in missing),
            "manual": [row["id"] for row in rows if not row["installed"] and "page" in row],
            "ready": all(row["installed"] for row in rows if row["for"] == "draft")}



# The Universal Conditioner pack reads the prompt on a language model server and maps it to H3's
# conditioning through an adapter in models/universal_conditioner, so no text encoder is loaded here.
# A project names the adapter and the server's address; the pack lists the folder only when installed.
CONDITIONER_FOLDER = "universal_conditioner"
CONDITIONER_PREFIX = CONDITIONER_FOLDER + ":"
CONDITIONER_NODE = "UniCondEncoder"
CONDITIONER_URL = "http://127.0.0.1:8000"


def conditioners_for(available):
    """The installed Universal Conditioner adapters made for MiniMax H3: its files are named unicond-h3-*.

    The pack also lists models/clip_projections under its folder, so the name is what tells an adapter from
    a small text encoder's projection."""
    if available is None:
        return []
    found = sorted((name[len(CONDITIONER_PREFIX):] for name in available if str(name).startswith(CONDITIONER_PREFIX)),
                   key=str.lower)
    return [name for name in found if name.replace("\\", "/").rsplit("/", 1)[-1].lower().startswith("unicond-h3")]


def conditioner_for(available, chosen):
    """The adapter a project names, as installed, or None. With no list of what is installed it is taken as named."""
    if not chosen:
        return None
    if available is None:
        return chosen
    same = [name for name in conditioners_for(available) if name.replace("\\", "/") == str(chosen).replace("\\", "/")]
    return same[0] if same else None


def text_encoder_node(encoder, available, chosen=None, conditioner=None, url=None):
    """The graph node that gives a render its text encoder: ComfyUI's loader for the 32B, the pack's for a
    small one, or the Universal Conditioner's for an encoder on a server."""
    if conditioner:
        return {"class_type": CONDITIONER_NODE, "inputs": {
            "url": url or CONDITIONER_URL, "model": "", "layer": -1, "adapter": conditioner, "profile": "auto"}}
    projection = projection_for(encoder, available, chosen)
    if projection:
        return {"class_type": "H3SmallTextEncoder", "inputs": {"clip_name": encoder, "projection": projection}}
    return {"class_type": "CLIPLoader", "inputs": {"clip_name": encoder, "type": "minimax", "device": "default"}}


def project_models(project, available=None, clip=None, quality="draft"):
    """The models a render uses: the profile's, with any the project names, except retired ones."""
    _, profile = render_profile(project, available, clip, quality)
    models = dict(MODELS)
    for key, names in MODEL_CHOICES.items():
        models[key] = pick_file(names, available) or names[0]
    models["unet"] = pick_file(profile["unet"], available) or profile["unet"][0]
    models["unet_text"] = pick_file(profile.get("unet_text"), available)
    models["lora"] = realism_lora(available) or MODELS["lora"]
    for key in PROFILE_LORAS:
        if profile.get(key):
            models["profile_" + key] = pick_file(lora_names(profile, key), available) or profile[key][0]
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
    # An adapter this machine lacks gives way to the text encoder file, and its address goes with it.
    named["conditioner"] = conditioner_for(available, named.get("conditioner"))
    if not named["conditioner"]:
        del named["conditioner"]
        named.pop("conditioner_url", None)
    models.update(named)
    return models


def profile_status(available):
    """Every profile, with what it needs that is not installed. available as in pick_file."""
    rows = []
    for name, profile in PROFILES.items():
        missing = []
        if pick_file(profile["unet"], available) is None:
            missing.append(profile["unet"][0])
        for key in PROFILE_LORAS:
            if profile.get(key) and pick_file(lora_names(profile, key), available) is None:
                missing.append(profile[key][0])
        rows.append({"id": name, "title": profile["title"], "hint": profile["hint"],
                     "available": not missing, "missing": missing, "steps": profile["steps"],
                     "sampler": profile["sampler"], "scheduler": profile.get("scheduler") or "simple",
                     "unet": pick_file(profile["unet"], available) or profile["unet"][0],
                     "unet_text": pick_file(profile.get("unet_text"), available),
                     "lora": profile["lora"][0] if profile.get("lora") else None,
                     "lora_text": profile["lora_text"][0] if profile.get("lora_text") else None})
    return rows


MOTION_POSE = "sdpose_wholebody_fp16.safetensors"
MOTION_DETECTOR = "rt_detr_v4-x-hgnet_fp16.safetensors"
MOTION_CONTROL = "minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors"
MOTION_CONTROL_V2 = "minimax_h3_fun_controlnet_union_2.0_pruned_int8_convrot.safetensors"
MOTION_SAM3 = "sam3.1_multiplex_fp16.safetensors"
MOTION_SWAP_LORA = "h3_character_swap_v1.5.safetensors"
MOTION_MODES = ("replace", "rescene", "choreography")
MOTION_STYLES = ("auto", "preserve", "photoreal")
MOTION_PRESETS = ("standard", "realism_fast", "realism", "fast_sharp")
H3_IMAGE_SAMPLER_NODE = "H3ImageSampler"
# Hardware-verified <=32 GB RTX 5090 motion realism recipes (F1 / F1-Fast / F2 / M5 / M7):
MOTION_REALISM_P1_SAMPLER = {"texture_from": 0.55, "eta": 0.50, "detail": 0.07}
MOTION_REALISM_P2_SAMPLER = {
    "realism_fast": {"texture_from": 0.55, "eta": 0.50, "detail": 0.07},
    "realism": {"texture_from": 0.70, "eta": 0.50, "detail": 0.07},
}
MOTION_REALISM_SIGMAS = {
    "realism_fast": "0.50, 0.20, 0.0",
    "realism": "0.675, 0.385, 0.159, 0.0",
}
MOTION_REALISM_P1_LORA = 0.60
MOTION_REALISM_P2_LORA = 0.55
MOTION_SWAP_END_SIGMA = 0.45
MOTION_FAST_SHARP_DETAIL = 0.85
MOTION_MAX_SECONDS = 30
SAM3_NODES = ("CheckpointLoaderSimple", "SAM3_Detect", "SAM3_VideoTrack", "SAM3_TrackToMask")
MOTION_NODES = ("CheckpointLoaderSimple", "RTDETR_detect", "SDPoseKeypointExtractor",
                "SDPoseDrawKeypoints", "ModelPatchLoader", "MiniMaxH3FunControlNetApply")

MOTION_SETUP_GROUPS = ("replace", "pose", "sharp")
MOTION_SETUP_FILES = [
    {"id": "sam3", "group": "replace", "title": "SAM 3.1 subject segmentation & tracking",
     "needs": [MOTION_SAM3], "check_prefix": "motion-checkpoints:", "folder": "checkpoints",
     "repo": "Comfy-Org/sam3.1", "path": "checkpoints/sam3.1_multiplex_fp16.safetensors",
     "bytes": 1745546848, "sha256": "9ba99c92703c2e8b4f47de2d34a539bb8e18923049e238b780d70dbe6368eb03",
     "license": SAM3_LICENSE, "for": "replace"},
    {"id": "swap_lora", "group": "replace", "title": "H3 Character Swap LoRA v1.5",
     "needs": [MOTION_SWAP_LORA], "folder": "loras",
     "repo": "akatz-ai/MiniMax-H3-Character-Swap-LoRA", "path": "h3_character_swap_v1.5.safetensors",
     "bytes": 155110792, "sha256": "8f2c3c3eba161df8d0c6099598b4730f641f5ce01e04cd16d2427cff8a35218f",
     "license": SWAP_LORA_LICENSE, "for": "replace"},
    {"id": "rtdetr", "group": "pose", "title": "RT-DETR person detector",
     "needs": [MOTION_DETECTOR], "folder": "diffusion_models",
     "repo": "Comfy-Org/SDPose", "path": "diffusion_models/rt_detr_v4-x-hgnet_fp16.safetensors",
     "bytes": 123968978, "sha256": "581f9af9bbabb664d1891cbccd823308b176ecd409146f954dfa39af3bec2476",
     "license": MIT_SDPOSE_LICENSE, "for": "pose"},
    {"id": "sdpose", "group": "pose", "title": "SDPose wholebody keypoint extractor",
     "needs": [MOTION_POSE], "check_prefix": "motion-checkpoints:", "folder": "checkpoints",
     "repo": "Comfy-Org/SDPose", "path": "checkpoints/sdpose_wholebody_fp16.safetensors",
     "bytes": 1916645792, "sha256": "63d01f9a7494560693b24767f4469d59c9d3266b31ff0a253e74d1e611442721",
     "license": MIT_SDPOSE_LICENSE, "for": "pose"},
    {"id": "fun_controlnet_v2", "group": "pose", "title": "MiniMax H3 Fun ControlNet Union 2.0",
     "needs": [MOTION_CONTROL_V2, MOTION_CONTROL], "check_prefix": "motion-control:", "folder": "model_patches",
     "repo": "Comfy-Org/MiniMax-H3", "path": "model_patches/minimax_h3_fun_controlnet_union_2.0_pruned_int8_convrot.safetensors",
     "bytes": 4531220608, "sha256": "890af58bb350f0c2f6c409b1c67ea8f4196654297d037769b72de1e6f504c612",
     "license": MINIMAX_LICENSE, "for": "pose"},
    {"id": "relayvsr_model", "group": "sharp", "title": "RelayVSR 2x video upscaler",
     "needs": [RELAYVSR_MODEL], "folder": "diffusion_models",
     "repo": "Comfy-Org/RelayVSR", "path": "diffusion_models/relayvsr_x2_lightx2v_int8_convrot.safetensors",
     "bytes": 5163295248, "sha256": "240e75e0c17cf2845d559b25254a230ff7458f81449d7a8c5e7e3fb8275c7b3c",
     "license": APACHE_RELAYVSR_LICENSE, "for": "sharp"},
    {"id": "relayvsr_vae", "group": "sharp", "title": "RelayVSR Flash VAE 2x",
     "needs": [RELAYVSR_VAE], "folder": "vae",
     "repo": "Comfy-Org/RelayVSR", "path": "vae/relayvsr_flash_vae_x2_fp32.safetensors",
     "bytes": 127759464, "sha256": "b5bb0339acb5b52545911ea29a51b4c5b805776a779222b5a14591571f42e21c",
     "license": APACHE_RELAYVSR_LICENSE, "for": "sharp"},
]


def _motion_file_found(entry, available):
    prefix = entry.get("check_prefix")
    if prefix:
        files = available or ()
        candidates = [name[len(prefix):] for name in sorted(files) if name.startswith(prefix)]
        return next((name for base in entry["needs"] for name in candidates
                     if name.replace("\\", "/").rsplit("/", 1)[-1] == base), None)
    exact = pick_file(entry["needs"], available)
    if exact is not None or available is None:
        return exact
    # Also match files stored inside subfolders (e.g. relayvsr/relayvsr_x2_lightx2v_int8_convrot.safetensors).
    candidates = [name for name in sorted(available) if ":" not in name]
    return next((name for base in entry["needs"] for name in candidates
                 if name.replace("\\", "/").rsplit("/", 1)[-1] == base), None)


def motion_setup_status(available, groups=None):
    """Status of optional Hugging Face motion transfer models, grouped into replace, pose and sharp."""
    selected = list(MOTION_SETUP_GROUPS) if not groups else [g for g in groups if g in MOTION_SETUP_GROUPS]
    if groups is not None and not selected:
        raise ValueError("groups must name one or more of %s" % (", ".join(MOTION_SETUP_GROUPS),))
    rows = []
    for entry in MOTION_SETUP_FILES:
        found = _motion_file_found(entry, available)
        rows.append({
            "id": entry["id"], "group": entry["group"], "title": entry["title"],
            "file": entry["needs"][0], "folder": entry["folder"], "bytes": entry["bytes"],
            "for": entry["for"], "installed": found is not None, "found": found,
            "source": "https://huggingface.co/%s" % entry["repo"],
            "license": {"name": entry["license"][0], "url": entry["license"][1]},
        })
    by_group = {}
    for g in MOTION_SETUP_GROUPS:
        grows = [r for r in rows if r["group"] == g]
        gmiss = [r for r in grows if not r["installed"]]
        by_group[g] = {"installed": not gmiss, "missing": [r["id"] for r in gmiss],
                       "bytes": sum(r["bytes"] for r in gmiss)}
    missing = [r for r in rows if r["group"] in selected and not r["installed"]]
    return {
        "files": rows,
        "groups": by_group,
        "selected": selected,
        "missing": [r["id"] for r in missing],
        "bytes": sum(r["bytes"] for r in missing),
        "ready": by_group["replace"]["installed"],
    }


def motion_status(available):
    """Capabilities for masked character replacement (`replace`) and pose-guided choreography (`choreography`)."""
    files = available or ()

    def model(folder, *names):
        prefix = folder + ":"
        candidates = [name[len(prefix):] for name in sorted(files) if name.startswith(prefix)]
        return next((name for base in names for name in candidates
                     if name.replace("\\", "/").rsplit("/", 1)[-1] == base), None)

    sam3 = model("motion-checkpoints", MOTION_SAM3)
    swap_lora = pick_file([MOTION_SWAP_LORA], available)
    pose = model("motion-checkpoints", MOTION_POSE)
    control = model("motion-control", MOTION_CONTROL_V2, MOTION_CONTROL)
    detector = pick_file([MOTION_DETECTOR.removesuffix(".safetensors") + "*"], available)
    base = pick_file(PROFILES["h3"]["unet"], available)
    flash = pick_file(PROFILES["hyperflow"]["unet"], available)
    profile = "h3_turbo" if pick_file([PROFILES["h3_turbo"]["lora"][0]], available) else "h3"

    replace_missing = []
    if available is not None and "sam3-nodes" not in files:
        replace_missing.append("native ComfyUI SAM3 nodes")
    if not sam3:
        replace_missing.append(MOTION_SAM3)
    if not (flash or base):
        replace_missing.append(PROFILES["hyperflow"]["unet"][0])
    replace_ready = not replace_missing

    pose_missing = []
    if available is not None and "motion-nodes" not in files:
        pose_missing.append("the native ComfyUI pose and H3 ControlNet nodes")
    pose_missing.extend(label for found, label in (
        (pose, MOTION_POSE), (detector, MOTION_DETECTOR),
        (control, MOTION_CONTROL_V2), (base, PROFILES["h3"]["unet"][0]),
    ) if not found)
    pose_ready = not pose_missing

    relayvsr = UPSCALERS["relayvsr"]["files"]
    sharp_ready = bool(installed(relayvsr["model"], available) and installed(relayvsr["vae"], available))

    rescene_missing = [] if (flash or base) else [PROFILES["hyperflow"]["unet"][0]]
    rescene_ready = not rescene_missing
    h3_sampler_ready = bool(available is None or "h3-image-sampler" in files)

    available_any = replace_ready or rescene_ready or pose_ready
    missing = [] if available_any else list(dict.fromkeys(replace_missing + pose_missing))
    default_mode = (
        "replace" if replace_ready
        else ("choreography" if pose_ready else ("rescene" if rescene_ready else "replace"))
    )
    return {
        "available": available_any,
        "missing": missing,
        "replace": replace_ready,
        "replace_missing": replace_missing,
        "rescene": rescene_ready,
        "rescene_missing": rescene_missing,
        "pose_control": pose_ready,
        "pose_missing": pose_missing,
        "sharp_ready": sharp_ready,
        "h3_image_sampler": h3_sampler_ready,
        "default_mode": default_mode,
        "profile": profile,
        "model": PROFILES[profile]["title"],
        "pose": pose,
        "detector": detector,
        "control": control,
        "sam3": sam3,
        "swap_lora": swap_lora,
    }


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
    if link["mode"] == "bridge" and not arrives_on_pin(link):
        end = end_hold(link)
        return hold + JOINT * max(2 if end == hold else 1, int(round((frames + end) / JOINT)))
    return hold + JOINT * max(1, int(round(frames / JOINT)))


def end_hold(link):
    """How many frames of the next clip a bridge holds at its own end.

    A bridge holds the last 39 frames of the first clip. Of the next clip it holds a short run, END_HOLD frames,
    with their sound: H3 cuts to a long run of another shot's frames, and one pinned frame carries no movement and
    no sound, so the take lands on it dark, still and silent. A short run is the middle that works: on two test
    pairs over three seeds each the cut into the next clip scored 97 to 99 for the picture and 78 to 97 for the
    sound, against 71 to 97 and 0 to 90 for a pin. A link without end_hold holds as much as of the first clip.
    """
    return int(link.get("end_hold") or link["hold"])


def arrives_on_pin(link):
    """Does a bridge arrive on the next clip's first frame, pinned as its own last frame.

    A bridge can reach the next clip in two ways. Holding the next clip's first frames at its end, as it holds the
    first clip's last frames at its start, keeps their latents exactly, but H3 then treats them as another shot:
    in tests it followed the prompt and cut to them a few frames early, in 8 of 9 pairs of different shots. With
    only the next clip's first frame pinned, the take moves into that frame in one shot. So that is how a bridge
    arrives unless it was made to hold: holding is still the better way into a clip that starts in mid-movement,
    where one frame does not say where things are going.
    """
    return bool(link) and link.get("mode") == "bridge" and link.get("arrive") == "pin"


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
    so one member with many pictures never crowds out another. For directional sprite clips whose
    cast pictures carry view tags ('front', 'side', 'back', 'left'), only heading-matched views
    are routed so side profiles do not pull rear or diagonal loops.
    """
    members = clip_cast(project, clip)
    config = clip.get("sprite") if isinstance(clip.get("sprite"), dict) else None
    direction = config.get("direction") if config else None
    have = [
        sprite.filter_cast_pictures(member_pictures(member), direction) if direction else member_pictures(member)
        for member in members
    ]
    first_pin = bool(frame_of(clip, "first"))
    last_pin = bool(frame_of(clip, "last"))
    pinned = frame_of(clip, "first") or frame_of(clip, "last")
    if direction and pinned:
        # The clip's own pinned frame shows the character in its heading. A member with view-tagged pictures and
        # none for this heading (every diagonal, or west with no left view) would bring views of other headings,
        # and H3 turns the body to them mid-take: a north-east idle pinned at both ends swung to the back view
        # and returned. The pinned picture stands in as that member's picture.
        have = [
            [pinned] if pictures and not sprite.heading_matched(pictures, direction) else pictures
            for pictures in have
        ]
    if config and not first_pin and not last_pin and (config.get("motion") == "idle" or config.get("animation")):
        if any(sprite.heading_matched(pictures, direction) for pictures in have):
            first_pin = last_pin = True
    room = H3_PICTURES - first_pin - last_pin
    taken = [0] * len(members)
    for _ in range(CAST_PICTURES):
        for index, pictures in enumerate(have):
            if taken[index] < len(pictures) and sum(taken) < room:
                taken[index] += 1
    return [(member, have[index][:taken[index]]) for index, member in enumerate(members) if taken[index]]


def sprite_auto_pin(project, clip):
    """Return the primary direction-matched plate to auto-pin at start & end for sprite idles and one-shot actions."""
    config = clip.get("sprite") if isinstance(clip.get("sprite"), dict) else None
    if not config or not (config.get("motion") == "idle" or config.get("animation")):
        return None
    if frame_of(clip, "first") or frame_of(clip, "last"):
        return None
    shown = cast_pictures(project, clip)
    # A view of another heading pinned at both ends would hold the whole take on that heading: the back view
    # for a north-east idle. Only a picture of this heading, or an untagged one, is pinned.
    for _, pictures in shown:
        if sprite.heading_matched(pictures, config.get("direction")):
            return pictures[0]
    return None


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
                or clip_voice(project, clip) or (clip.get("sprite") is not None and not clip["sprite"].get("motion")))


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
    is_sprite = clip.get("sprite") is not None
    sprite_cfg = sprite.settings(clip["sprite"]) if is_sprite else None
    if is_sprite:
        action = action + " " + sprite.instruction(sprite_cfg)
    # Sprite Drafts with character references lock their visual style to the cast plates
    # (or an explicit sprite.style), never inheriting a video project's live-action/camera look.
    if is_sprite and sprite_cfg.get("style"):
        style = sprite_cfg["style"]
    elif is_sprite and cast:
        style = ""
    else:
        style = project.get("style", "")
    if has_references(project, clip):
        owner = None
        if voice:
            number = next((n for n, c in enumerate(cast, 1) if c["id"] == voice["id"]), None)
            owner = {"subject": number} if number else {"description": voice.get("description") or voice["name"]}
        auto_pin = sprite_auto_pin(project, clip)
        sp_pin = (clip.get("sprite") or {}).get("self_pin") if isinstance(clip.get("sprite"), dict) else None
        return compose_reference_prompt(
            style, action, subjects=cast,
            first_frame=bool(frame_of(clip, "first") or auto_pin or sp_pin),
            last_frame=bool(frame_of(clip, "last") or auto_pin) or arrives_on_pin(clip.get("link")), voice=owner, **fields,
        )
    return compose_prompt(style, action, **fields)


def still_node(asset, backdrop=None, fit=None):
    """Load a picture. backdrop "white" lays a transparent one on white; fit (width, height) shows it whole at that size."""
    file = "%s/%s" % (asset["subfolder"], asset["file"]) if asset.get("subfolder") else asset["file"]
    inputs = {"file": file}
    if backdrop:
        inputs["backdrop"] = backdrop
    if fit:
        inputs["width"], inputs["height"] = int(fit[0]), int(fit[1])
    return {"class_type": "H3LoadStill", "inputs": inputs}


def motion_prompt(motion, user_prompt="", ref_count=1, *, refine_pass=False):
    """Build the clean positive conditioning prompt for masked replacement, pure re-scene motion, or choreography."""
    extra = str(user_prompt or "").strip()
    mode = motion.get("mode") or "choreography"
    preset = motion.get("preset") or "standard"
    pics = "<Picture 1> and <Picture 2>" if int(ref_count or 1) >= 2 else "<Picture 1>"

    if mode == "rescene":
        if refine_pass:
            if motion.get("style") == "photoreal":
                prefix = (
                    f"The exact person shown in {pics}, faithfully preserving their facial structure, eyes, eyebrows, "
                    "lips, natural skin tone, and hair from every angle. "
                    "Authentic unretouched handheld camera video, natural matte skin texture with real pores, "
                    "fine flyaway hair, soft natural shadows, and subtle realistic camera grain.\n"
                )
            else:
                prefix = (
                    f"The exact character shown in {pics}, faithfully preserving their exact art style, face, hair, "
                    "and proportions in a clean, cohesive scene.\n"
                )
            return prefix + extra
        pose_clause = (
            " Do NOT stay locked in a static reference pose — keep both arms free to follow every gesture in <Video 1>."
            if float(motion.get("strength") or 0.0) > 0.0 else ""
        )
        if motion.get("style") == "photoreal":
            prefix = (
                f"Transfer ONLY the body movement, dance choreography, arm gestures, head tilts, and subtle mouth and "
                f"facial expressions from <Video 1> to the exact person shown in {pics}, starting from the very first frame.{pose_clause} "
                "Completely ignore and discard the original background, lighting, and outfit from <Video 1>. "
                "Keep the new scene, lighting, outfit, and person continuously consistent throughout the entire clip. "
                "Authentic unretouched handheld camera video, natural matte skin texture with real pores, "
                "subtle skin tone variation, fine flyaway hair, soft natural shadows, and realistic video grain.\n"
            )
        else:
            prefix = (
                f"Transfer ONLY the body movement, choreography, gestures, and facial expressions from <Video 1> to the "
                f"character shown in {pics}, starting from the very first frame.{pose_clause} "
                "Completely ignore and discard the original background and outfit from <Video 1>, preserving "
                f"{pics}'s exact art style, face, and hair in the requested scene.\n"
            )
        return prefix + extra

    if mode == "replace":
        if motion.get("style") == "photoreal":
            if preset != "standard":
                prefix = (
                    f"Replace the person in <Video 1> from the first frame to the last frame with the exact person shown in "
                    f"{pics}, faithfully preserving their facial structure, eyes, eyebrows, lips, hair, and likeness from every angle. "
                    "Match every body movement, expression, mouth micro-movement, hand gesture, framing, and natural lighting of <Video 1>. "
                    "Authentic unretouched smartphone camera video, natural matte skin texture with real pores, subtle skin tone variation, "
                    "fine flyaway hair, soft natural shadows, and realistic video grain. "
                    "Leave the original background cleanly visible whenever the person moves.\n"
                )
            else:
                prefix = (
                    "Completely replace the person in <Video 1> from the first frame to the last frame with the "
                    "photorealistic person in <Picture 1>, preserving <Picture 1>'s exact face, hair, and outfit "
                    "while matching the scene's lighting, shadows, framing, camera movement, and motion. "
                    "Leave the source background cleanly visible whenever the person exits.\n"
                )
        else:
            prefix = (
                f"Completely replace the person in <Video 1> from the first frame to the last frame with the "
                f"character in {pics}, matching {pics}'s exact art style, face, hair, and outfit "
                "while preserving the scene and motion. Leave the source background cleanly visible whenever the person exits.\n"
            )
        return prefix + extra
    return (
        f"{pics} defines the performer's face, hair, body and outfit. "
        "The performer follows the pose guide's choreography and timing in a continuous full-body shot.\n"
        + extra
    )


def resolve_motion_runtime(motion, available=None, detect_style=None):
    """Validate motion settings and resolve installed checkpoints for a render graph."""
    if not isinstance(motion, dict):
        raise ValueError("motion transfer needs a character image and settings")
    character = clean_asset(motion.get("character"), "the character image")
    if not character:
        raise ValueError("choose a character image for motion transfer")
    settings = motion_status(available)
    if not settings["available"]:
        raise ValueError("motion transfer needs " + ", ".join(settings["missing"]))
    mode = motion.get("mode") or settings["default_mode"]
    if mode not in MOTION_MODES:
        raise ValueError("motion mode must be one of %s" % (", ".join(MOTION_MODES),))
    preset = motion.get("preset") or "standard"
    if preset not in MOTION_PRESETS:
        raise ValueError("motion preset must be one of %s" % (", ".join(MOTION_PRESETS),))
    style = motion.get("style") or "auto"
    if style not in MOTION_STYLES:
        raise ValueError("character style must be one of %s" % (", ".join(MOTION_STYLES),))
    if style == "auto":
        detected = detect_style(character) if detect_style is not None else None
        style = "preserve" if detected == "preserve" else "photoreal"

    if mode == "rescene":
        if not settings["rescene"]:
            raise ValueError("new-scene motion transfer needs " + ", ".join(settings["rescene_missing"]))
        pose_strength = clean_strength(motion.get("strength", 0.0), (0.0, 2.0), "motion strength")
        if pose_strength > 0.0 and not settings["pose_control"]:
            raise ValueError("motion transfer needs " + ", ".join(settings["pose_missing"]))
        out = {
            "mode": "rescene",
            "preset": preset,
            "style": style,
            "character": character,
            "strength": pose_strength,
            "sam3": settings["sam3"],
            "isolate": bool(motion.get("isolate", False)),
            "plate": False,
            "swap_lora": None,
            "swap_lora_strength": 0.0,
            "h3_image_sampler": bool(settings["h3_image_sampler"]),
        }
        if pose_strength > 0.0:
            out.update({k: settings[k] for k in ("pose", "detector", "control")})
        if motion.get("sharp") or preset == "fast_sharp":
            out["sharp"] = True
            if preset == "fast_sharp":
                out["sharp_detail"] = MOTION_FAST_SHARP_DETAIL
        return out

    if mode == "replace":
        if not settings["replace"]:
            raise ValueError("character replacement needs " + ", ".join(settings["replace_missing"]))
        pose_strength = clean_strength(motion.get("strength", 0.0), (0.0, 2.0), "motion strength")
        if pose_strength > 0.0 and not settings["pose_control"]:
            raise ValueError("pose reinforcement needs " + ", ".join(settings["pose_missing"]))
        photoreal_default_swap = 0.85 if preset != "standard" else 1.0
        default_swap = (photoreal_default_swap if style == "photoreal" else 0.5) if settings["swap_lora"] else 0.0
        raw_swap = motion.get("swap_lora")
        swap_strength = clean_strength(
            default_swap if raw_swap is None else raw_swap, (0.0, 1.5), "character swap LoRA strength"
        )
        if swap_strength > 0.0 and not settings["swap_lora"]:
            raise ValueError("character swap LoRA needs " + MOTION_SWAP_LORA)
        expand_raw = motion.get("expand_px", 32)
        if isinstance(expand_raw, bool) or not isinstance(expand_raw, (int, float)) or not 0 <= int(expand_raw) <= 128:
            raise ValueError("mask expansion must be between 0 and 128 pixels")
        out = {
            "mode": "replace",
            "preset": preset,
            "style": style,
            "character": character,
            "strength": pose_strength,
            "sam3": settings["sam3"],
            "isolate": bool(motion.get("isolate", True)),
            "plate": bool(motion.get("plate", True)),
            "expand_px": int(expand_raw),
            "swap_lora": settings["swap_lora"] if swap_strength > 0.0 else None,
            "swap_lora_strength": swap_strength,
            "h3_image_sampler": bool(settings["h3_image_sampler"]),
        }
        if pose_strength > 0.0:
            out.update({k: settings[k] for k in ("pose", "detector", "control")})
        if motion.get("sharp") or preset == "fast_sharp":
            out["sharp"] = True
            if preset == "fast_sharp":
                out["sharp_detail"] = MOTION_FAST_SHARP_DETAIL
        return out

    if not settings["pose_control"]:
        raise ValueError("motion transfer needs " + ", ".join(settings["pose_missing"]))
    out = {k: settings[k] for k in ("pose", "detector", "control", "profile")}
    out.update(
        mode="choreography",
        preset=preset,
        style=style,
        character=character,
        strength=clean_strength(motion.get("strength", 1.0), (0.1, 2.0), "motion strength"),
        sam3=settings["sam3"],
        isolate=bool(motion.get("isolate", bool(settings["sam3"]))),
        plate=False,
        h3_image_sampler=bool(settings["h3_image_sampler"]),
    )
    if motion.get("sharp") or preset == "fast_sharp":
        out["sharp"] = True
        if preset == "fast_sharp":
            out["sharp_detail"] = MOTION_FAST_SHARP_DETAIL
    return out


def wire_motion_transfer(graph, motion, *, hyperflow=False, width=384, frame_count=None, prompt_text="",
                         refine_prompt_text="", seed=0, source_info=None, cache_hits=None,
                         extra_characters=None, refine_cfg=None):
    """Wire single-pass or automatic multi-window pure-latent relay motion transfer onto `graph`."""
    mode = motion.get("mode") or "choreography"
    preset = motion.get("preset") or "standard"
    exact_frames = int(frame_count or graph["10"]["inputs"]["length"])
    if refine_cfg and refine_cfg.get("multi_refine"):
        windows = motion_replace.plan_relay_windows(
            exact_frames,
            single_max=motion_replace.REALISM_WINDOW_FRAMES,
            window_frames=motion_replace.REALISM_WINDOW_FRAMES,
            overlap_frames=motion_replace.OVERLAP_FRAMES,
        )
    else:
        windows = motion_replace.plan_relay_windows(exact_frames)
    multi = len(windows) > 1
    multi_refine = bool(multi and refine_cfg and refine_cfg.get("multi_refine"))
    hits = cache_hits or {}

    graph["214"] = still_node(motion["character"])
    char_ref = ["214", 0]
    char_file = "/".join(filter(None, (motion["character"].get("subfolder"), motion["character"].get("file"))))

    sam3_ckpt = motion.get("sam3")
    want_isolate = bool(sam3_ckpt and motion.get("isolate"))
    need_sam3 = bool(sam3_ckpt) and (
        (mode == "replace" and (motion.get("plate", True) or not hits.get("mask")))
        or (want_isolate and not hits.get("cutout"))
    )
    if need_sam3:
        graph["238"] = dict(class_type="CheckpointLoaderSimple", inputs={"ckpt_name": sam3_ckpt})
    if want_isolate:
        iso_inputs = {
            "image": ["214", 0], "subject": "person", "backdrop": "white", "threshold": 0.35,
            "character_file": char_file,
        }
        if "238" in graph:
            iso_inputs["sam_model"] = ["238", 0]
            iso_inputs["sam_clip"] = ["238", 1]
        graph["239"] = dict(class_type="H3IsolateCharacter", inputs=iso_inputs)
        char_ref = ["239", 0]

    graph["10"]["inputs"]["ref_images.ref_image_0"] = char_ref
    for idx, extra_asset in enumerate((extra_characters or [])[:1]):
        nid = str(270 + idx)
        graph[nid] = still_node(extra_asset)
        graph["10"]["inputs"]["ref_images.ref_image_%d" % (idx + 1)] = [nid, 0]
    graph["10"]["inputs"]["prompt"] = prompt_text
    graph["10"]["inputs"]["length"] = windows[0]["length"]

    pose_strength = float(motion.get("strength", 0.0 if mode in ("replace", "rescene") else 1.0))
    use_pose = pose_strength > 0.0 and motion.get("detector") and motion.get("pose") and motion.get("control")

    # Keep a clean model reference (pre-swap_lora and pre-ControlNet) for Pass-2 1088p Refine.
    pre_swap_model = list(graph["5"]["inputs"]["model"])
    clean_refine_model = ["5", 0] if use_pose else (["9", 0] if "9" in graph else ["5", 0])

    if mode == "replace":
        # Decode full-resolution RGB pixels cleanly (also supports >15s clips for windowed relay).
        graph["213"]["inputs"]["reference_only"] = False
        graph["213"]["inputs"]["frames_only"] = True

        base_expand = int(motion.get("expand_px", 32))
        scaled_expand = max(0, min(128, int(round(base_expand * (float(width) / 384.0)))))
        track_inputs = {
            "images": ["213", 0], "subject": "person", "expansion_px": scaled_expand,
            "detect_threshold": 0.35, "track_threshold": 0.5,
        }
        if "238" in graph:
            track_inputs["sam_model"] = ["238", 0]
            track_inputs["sam_clip"] = ["238", 1]
        if source_info:
            track_inputs.update({
                "source_file": source_info["file"],
                "first": int(source_info["first"]),
                "total_frames": int(source_info["total_frames"]),
            })
        graph["240"] = dict(class_type="H3TrackSubjectMask", inputs=track_inputs)

        swap_strength = float(motion.get("swap_lora_strength", 0.0))
        if swap_strength > 0.0 and motion.get("swap_lora"):
            swap_inputs = {
                "model": pre_swap_model, "lora_name": motion["swap_lora"],
                "strength": swap_strength,
            }
            if preset != "standard":
                # Sigma-split swap_lora to high-sigma structure steps only (1.0 -> 0.45), freeing low-sigma steps for crisp skin.
                swap_inputs["start_sigma"] = 1.0
                swap_inputs["end_sigma"] = MOTION_SWAP_END_SIGMA
                graph["231"] = {
                    "class_type": "MiniMaxH3SigmaShift",
                    "inputs": {
                        "model": pre_swap_model,
                        "shift_video": graph["5"]["inputs"]["shift_video"],
                        "shift_audio": graph["5"]["inputs"]["shift_audio"],
                    },
                }
                clean_refine_model = ["231", 0]
            graph["230"] = dict(class_type="H3Lora", inputs=swap_inputs)
            graph["5"]["inputs"]["model"] = ["230", 0]
    elif mode == "rescene" or multi:
        graph["213"]["inputs"]["reference_only"] = False
        graph["213"]["inputs"]["frames_only"] = True

    if refine_cfg is not None:
        refine_cfg["refine_model"] = clean_refine_model

    if use_pose:
        pose_inputs = {
            "image": ["213", 0], "batch_size": 8,
            "min_area_frac": 0.005, "min_valid_body_kps": 2, "kp_score_thresh": 0.3,
        }
        if source_info:
            pose_inputs.update({
                "source_file": source_info["file"],
                "first": int(source_info["first"]),
                "total_frames": int(source_info["total_frames"]),
            })
        if not (hits.get("pose") and source_info):
            graph["215"] = dict(class_type="UNETLoader", inputs={"unet_name": motion["detector"], "weight_dtype": "default"})
            graph["216"] = dict(class_type="RTDETR_detect", inputs={
                "model": ["215", 0], "image": ["213", 0], "threshold": 0.5, "class_name": "person", "max_detections": 1})
            graph["217"] = dict(class_type="CheckpointLoaderSimple", inputs={"ckpt_name": motion["pose"]})
            pose_inputs.update({"model": ["217", 0], "vae": ["217", 2], "bboxes": ["216", 0]})
        graph["218"] = dict(class_type="H3GatedSDPose", inputs=pose_inputs)
        draw_face = bool(mode == "rescene" or preset != "standard")
        graph["219"] = dict(class_type="SDPoseDrawKeypoints", inputs={
            "keypoints": ["218", 0], "draw_body": True, "draw_head": True, "draw_hands": True, "draw_feet": True,
            "draw_face": draw_face, "stick_width": 4, "face_point_size": 2, "score_threshold": 0.3})
        graph["220"] = dict(class_type="ModelPatchLoader", inputs={"name": motion["control"]})

    base_guider_model = list(graph["13"]["inputs"]["model"])
    ctrl_node = "H3FlashControlNetApply" if hyperflow else "MiniMaxH3FunControlNetApply"
    ctrl_end = 0.80 if (mode == "rescene" or preset != "standard") else 1.0

    def attach_swap_guard(model_ref):
        swap_lora_name = motion.get("swap_lora") or ""
        cur_swap = float(motion.get("swap_lora_strength", 0.0))
        boost = max(0.0, min(0.35, 1.25 - cur_swap)) if swap_lora_name else 0.0
        graph["248"] = dict(class_type="H3SwapGuard", inputs={
            "sampled": ["15", 0],
            "masked_source": ["245", 0],
            "model": list(model_ref),
            "conditioning": ["10", 0],
            "sampler": ["11", 0],
            "sigmas": ["12", 0],
            "seed": int(seed),
            "swap_lora": swap_lora_name,
            "boost_strength": round(boost, 2),
            "min_change": 0.30 if preset != "standard" else 0.42,
        })
        graph["16"]["inputs"]["samples"] = ["248", 0]
        return "248"

    if not multi:
        if mode == "replace":
            # Always align ref_video_0 and source mask to exact 5 + 17k grid length so
            # MiniMaxH3ReferenceToVideo never truncates the tail and dissolves the character.
            graph["260"] = dict(class_type="H3SliceBatch", inputs={
                "start": 0, "length": windows[0]["length"], "images": ["213", 0], "mask": ["240", 1],
            })
            graph["10"]["inputs"]["ref_videos.ref_video_0"] = ["260", 0]
            graph["245"] = dict(class_type="H3ApplyEditMask", inputs={
                "latent": ["10", 1], "vae": ["3", 0], "source_video": ["260", 0], "mask": ["260", 1],
                "overlap_frames": 0,
            })
            graph["15"]["inputs"]["latent_image"] = ["245", 0]
        elif mode == "rescene":
            # M5 / M7: Unmasked grid-aligned RGB <Video 1> (NO edit mask, NO source plate, NO swap_lora!).
            graph["260"] = dict(class_type="H3SliceBatch", inputs={
                "start": 0, "length": windows[0]["length"], "images": ["213", 0],
            })
            graph["10"]["inputs"]["ref_videos.ref_video_0"] = ["260", 0]
        if use_pose:
            if mode == "rescene":
                graph["261"] = dict(class_type="H3SliceBatch", inputs={
                    "start": 0, "length": windows[0]["length"], "images": ["219", 0],
                })
                ctrl_vid = ["261", 0]
            else:
                ctrl_vid = ["219", 0]
            graph["221"] = dict(class_type=ctrl_node, inputs={
                "model": graph["9"]["inputs"]["model"], "vae": ["3", 0], "model_patch": ["220", 0],
                "control_video": ctrl_vid, "strength": pose_strength, "start_percent": 0.0, "end_percent": ctrl_end,
            })
            graph["9"]["inputs"]["model"] = ["221", 0]
        if mode == "replace":
            attach_swap_guard(graph["13"]["inputs"]["model"])
        return windows

    # Multi-window pure-latent relay (>141 frames / >107f for realism refine -> up to 30s):
    # Window 0 uses the primary take/sampler/decode nodes (10/13/15/16) with sliced inputs.
    w0 = windows[0]
    w0_slice_inputs = {"start": w0["start"], "length": w0["length"], "images": ["213", 0]}
    if mode == "replace":
        w0_slice_inputs["mask"] = ["240", 1]
    graph["260"] = dict(class_type="H3SliceBatch", inputs=w0_slice_inputs)
    if mode == "replace":
        graph["10"]["inputs"]["ref_videos.ref_video_0"] = ["260", 0]
        graph["245"] = dict(class_type="H3ApplyEditMask", inputs={
            "latent": ["10", 1], "vae": ["3", 0], "source_video": ["260", 0], "mask": ["260", 1],
            "overlap_frames": 0,
        })
        graph["15"]["inputs"]["latent_image"] = ["245", 0]
    elif mode == "rescene":
        graph["10"]["inputs"]["ref_videos.ref_video_0"] = ["260", 0]
    if use_pose:
        graph["261"] = dict(class_type="H3SliceBatch", inputs={"start": w0["start"], "length": w0["length"], "images": ["219", 0]})
        graph["221"] = dict(class_type=ctrl_node, inputs={
            "model": base_guider_model, "vae": ["3", 0], "model_patch": ["220", 0],
            "control_video": ["261", 0], "strength": pose_strength, "start_percent": 0.0, "end_percent": ctrl_end,
        })
        graph["13"]["inputs"]["model"] = ["221", 0]

    prev_sampler = attach_swap_guard(graph["13"]["inputs"]["model"]) if mode == "replace" else "15"
    prev_decoded = ["16", 0]
    prev_p2_sampler = None
    base_take_inputs = copy.deepcopy(graph["10"]["inputs"])

    if multi_refine:
        # Window 0 Pass-2 1088p Refine (shared 3D upscaler "81", H3ImageSampler "86", ManualSigmas "87").
        ref_w, ref_h = refine_cfg["size"]
        p2_base_inputs = copy.deepcopy(base_take_inputs)
        p2_base_inputs.update({"width": ref_w, "height": ref_h, "length": w0["length"]})
        if mode == "rescene":
            p2_base_inputs.pop("ref_videos.ref_video_0", None)
            p2_base_inputs["prompt"] = refine_prompt_text or prompt_text
        graph["80"] = dict(class_type="H3Take", inputs=p2_base_inputs)
        w0_p2_take_latent = ["80", 1]
        if mode == "replace":
            graph["246"] = dict(class_type="H3ApplyEditMask", inputs={
                "latent": ["80", 1], "vae": ["3", 0], "source_video": ["260", 0], "mask": ["260", 1],
                "overlap_frames": 0,
            })
            w0_p2_take_latent = ["246", 0]
        graph["81"] = {"class_type": "LatentUpscaleModelLoader", "inputs": {"model_name": refine_cfg["upscaler"]}}
        graph["82"] = {"class_type": "MiniMaxH3LatentUpscaleWithModel", "inputs": {
            "model": ["81", 0], "samples": [prev_sampler, 0], "width": ref_w, "height": ref_h}}
        graph["83"] = {"class_type": "H3RefineLatent", "inputs": {"upscaled": ["82", 0], "take": w0_p2_take_latent}}
        graph["86"] = (
            {"class_type": H3_IMAGE_SAMPLER_NODE, "inputs": dict(refine_cfg["sampler_cfg"])}
            if refine_cfg.get("use_h3_sampler") else copy.deepcopy(graph["11"])
        )
        graph["87"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": refine_cfg["sigmas"]}}
        graph["88"] = {"class_type": "BasicGuider", "inputs": {"model": clean_refine_model, "conditioning": ["80", 0]}}
        graph["90"] = {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["14", 0], "guider": ["88", 0], "sampler": ["86", 0],
            "sigmas": ["87", 0], "latent_image": ["83", 0]}}
        graph["91"] = {"class_type": "VAEDecode", "inputs": {"samples": ["90", 0], "vae": ["3", 0]}}
        graph.pop("16", None)
        prev_p2_sampler = "90"
        prev_decoded = ["91", 0]
        refine_cfg["applied"] = True

    win_stride_ids = 40 if multi_refine else 20
    for idx, win in enumerate(windows[1:], start=1):
        base = 300 + win_stride_ids * idx
        k_slice = str(base)
        k_pslice = str(base + 1)
        k_ctrl = str(base + 2)
        k_take = str(base + 3)
        k_mask = str(base + 4)
        k_guider = str(base + 5)
        k_sampler = str(base + 6)
        k_decode = str(base + 7)
        k_join = str(base + 8)
        k_noise = str(base + 9)

        slice_inputs = {"start": win["start"], "length": win["length"], "images": ["213", 0]}
        if mode == "replace":
            slice_inputs["mask"] = ["240", 1]
        graph[k_slice] = dict(class_type="H3SliceBatch", inputs=slice_inputs)

        win_model = base_guider_model
        if use_pose:
            graph[k_pslice] = dict(class_type="H3SliceBatch", inputs={
                "start": win["start"], "length": win["length"], "images": ["219", 0],
            })
            graph[k_ctrl] = dict(class_type=ctrl_node, inputs={
                "model": base_guider_model, "vae": ["3", 0], "model_patch": ["220", 0],
                "control_video": [k_pslice, 0], "strength": pose_strength, "start_percent": 0.0, "end_percent": ctrl_end,
            })
            win_model = [k_ctrl, 0]

        win_take_inputs = copy.deepcopy(base_take_inputs)
        win_take_inputs["length"] = win["length"]
        if mode == "replace":
            win_take_inputs["ref_videos.ref_video_0"] = [k_slice, 0]
        elif mode == "rescene":
            # M5 / M7 multi-window: carry prior 768p latent (22f overlap) AND feed sliced unmasked RGB <Video 1>!
            win_take_inputs.update({
                "ref_videos.ref_video_0": [k_slice, 0],
                "carry": [prev_sampler, 0], "hold_frames": win["overlap"],
                "align": "video", "audio_hold": "off", "video_mode": "off",
            })
        else:
            # Choreography mode carries the prior window's last 22 latent frames directly via H3Take.carry.
            win_take_inputs.update({
                "carry": [prev_sampler, 0], "hold_frames": win["overlap"],
                "align": "video", "audio_hold": "off", "video_mode": "off",
            })
        graph[k_take] = dict(class_type="H3Take", inputs=win_take_inputs)

        if mode == "replace":
            graph[k_mask] = dict(class_type="H3ApplyEditMask", inputs={
                "latent": [k_take, 1], "vae": ["3", 0],
                "source_video": [k_slice, 0], "mask": [k_slice, 1],
                "prev_latent": [prev_sampler, 0], "overlap_frames": win["overlap"],
            })
            win_latent = [k_mask, 0]
        else:
            win_latent = [k_take, 1]

        graph[k_noise] = {"class_type": "RandomNoise", "inputs": {"noise_seed": (int(seed) + idx) & 0x7fffffff}}
        graph[k_guider] = dict(class_type="BasicGuider", inputs={"model": win_model, "conditioning": [k_take, 0]})
        graph[k_sampler] = dict(class_type="SamplerCustomAdvanced", inputs={
            "noise": [k_noise, 0], "guider": [k_guider, 0], "sampler": ["11", 0],
            "sigmas": ["12", 0], "latent_image": win_latent,
        })
        is_last = idx == len(windows) - 1

        if multi_refine:
            # Double latent carry: Pass-2 window i carries prior 1088p refined latent (22f locked in 1088p latent space!).
            ref_w, ref_h = refine_cfg["size"]
            k_p2_take = str(base + 20)
            k_p2_mask = str(base + 21)
            k_p2_up = str(base + 22)
            k_p2_reflat = str(base + 23)
            k_p2_guider = str(base + 24)
            k_p2_sampler = str(base + 25)
            k_p2_decode = str(base + 26)
            k_p2_join = str(base + 27)

            p2_take_inputs = copy.deepcopy(graph["80"]["inputs"])
            p2_take_inputs["length"] = win["length"]
            if mode == "replace":
                p2_take_inputs["ref_videos.ref_video_0"] = [k_slice, 0]
            else:
                p2_take_inputs.update({
                    "carry": [prev_p2_sampler, 0], "hold_frames": win["overlap"],
                    "align": "video", "audio_hold": "off", "video_mode": "off",
                })
            graph[k_p2_take] = dict(class_type="H3Take", inputs=p2_take_inputs)
            if mode == "replace":
                graph[k_p2_mask] = dict(class_type="H3ApplyEditMask", inputs={
                    "latent": [k_p2_take, 1], "vae": ["3", 0],
                    "source_video": [k_slice, 0], "mask": [k_slice, 1],
                    "prev_latent": [prev_p2_sampler, 0], "overlap_frames": win["overlap"],
                })
                p2_take_latent = [k_p2_mask, 0]
            else:
                p2_take_latent = [k_p2_take, 1]
            graph[k_p2_up] = {"class_type": "MiniMaxH3LatentUpscaleWithModel", "inputs": {
                "model": ["81", 0], "samples": [k_sampler, 0], "width": ref_w, "height": ref_h}}
            graph[k_p2_reflat] = {"class_type": "H3RefineLatent", "inputs": {
                "upscaled": [k_p2_up, 0], "take": p2_take_latent}}
            graph[k_p2_guider] = dict(class_type="BasicGuider", inputs={
                "model": clean_refine_model, "conditioning": [k_p2_take, 0]})
            graph[k_p2_sampler] = dict(class_type="SamplerCustomAdvanced", inputs={
                "noise": [k_noise, 0], "guider": [k_p2_guider, 0], "sampler": ["86", 0],
                "sigmas": ["87", 0], "latent_image": [k_p2_reflat, 0]})
            graph[k_p2_decode] = dict(class_type="VAEDecode", inputs={"samples": [k_p2_sampler, 0], "vae": ["3", 0]})
            graph[k_p2_join] = dict(class_type="H3JoinWindow", inputs={
                "prev_images": prev_decoded, "next_images": [k_p2_decode, 0],
                "overlap_frames": win["overlap"], "blend_frames": 6,
                "trim_to": exact_frames if is_last else 0,
            })
            prev_sampler = k_sampler
            prev_p2_sampler = k_p2_sampler
            prev_decoded = [k_p2_join, 0]
        else:
            graph[k_decode] = dict(class_type="VAEDecode", inputs={"samples": [k_sampler, 0], "vae": ["3", 0]})
            graph[k_join] = dict(class_type="H3JoinWindow", inputs={
                "prev_images": prev_decoded, "next_images": [k_decode, 0],
                "overlap_frames": win["overlap"], "blend_frames": 4,
                "trim_to": exact_frames if is_last else 0,
            })
            prev_sampler = k_sampler
            prev_decoded = [k_join, 0]

    graph["18"]["inputs"]["images"] = prev_decoded
    graph["18"]["inputs"].pop("audio", None)
    graph.pop("17", None)
    graph.pop("20", None)
    return windows


def wire_motion_plate(graph, motion):
    """Keep latent masking across single-window refine and optionally composite the zero-halo source plate."""
    if (motion.get("mode") or "choreography") != "replace":
        return
    if "248" in graph:
        # Ensure single-window refine, audio decode, and hold save all read the swap-verified Window 0 latent.
        if "82" in graph:
            graph["82"]["inputs"]["samples"] = ["248", 0]
        if "17" in graph:
            graph["17"]["inputs"]["samples"] = ["248", 0]
        if "20" in graph:
            graph["20"]["inputs"]["latent"] = ["248", 0]
    if "83" in graph and "80" in graph and "240" in graph and "246" not in graph:
        # Keep the source background locked across a single-window final's refine pass too.
        src_ref = ["260", 0] if "260" in graph else ["213", 0]
        mask_ref = ["260", 1] if "260" in graph else ["240", 1]
        graph["246"] = dict(class_type="H3ApplyEditMask", inputs={
            "latent": ["80", 1], "vae": ["3", 0], "source_video": src_ref, "mask": mask_ref,
            "overlap_frames": 0,
        })
        graph["83"]["inputs"]["take"] = ["246", 0]
    if not motion.get("plate", True):
        return
    decoded = list(graph["18"]["inputs"]["images"])
    plate_inputs = {
        "generated": decoded, "source": ["213", 0], "source_mask": ["240", 0],
        "subject": "person", "src_expand_px": 18, "gen_expand_px": 4, "feather_sigma": 4.0,
    }
    if "238" in graph:
        plate_inputs["sam_model"] = ["238", 0]
        plate_inputs["sam_clip"] = ["238", 1]
    graph["250"] = dict(class_type="H3CompositeSourcePlate", inputs=plate_inputs)
    graph["18"]["inputs"]["images"] = ["250", 0]


def render_graph(project, folder, clip, take, quality, seed, available=None, patches=None, finish=None, edit=None,
                 finish_window=None, check_cache=None):
    """One clip as a ComfyUI graph, saved as its own file in the project.

    available is the set of model files ComfyUI lists, used to find a profile's files.
    patches is the installed patch nodes by id, as project_patches takes them.
    finish is a draft take of the clip to finish: the final's first pass is that draft's own latents, enlarged and
    refined at the refine size, so the shot keeps the draft's framing, motion and sound.
    finish_window is an optional (grid_in, grid_len) tuple to slice and refine only part of finish's latents.
    """
    if clip.get("kind") == "footage":
        raise ValueError("%s is imported footage. Extend or bridge it to generate from it" % clip.get("name", "this clip"))
    active_motion = (edit.get("motion") if edit else None) or clip.get("motion")
    if not str(clip.get("action", "")).strip() and not active_motion:
        raise ValueError("describe what happens in this clip before rendering it")
    if quality not in QUALITIES:
        raise ValueError("quality must be one of %s" % (tuple(QUALITIES),))
    motion_src = None
    if edit and edit.get("motion"):
        motion_src = {"file": edit["file"], "subfolder": edit.get("subfolder") or "",
                      "in": int(edit["in"]), "out": int(edit["out"]),
                      "frames": int(edit.get("frames") or edit["out"])}
    elif clip.get("motion"):
        src = clip["motion"].get("source") or {}
        motion_src = {"file": src["file"], "subfolder": src.get("subfolder") or "",
                      "in": int(src["in"]), "out": int(src["out"]),
                      "frames": int(src.get("frames") or src["out"])}
    exact_motion_frames = (motion_src["out"] - motion_src["in"]) if motion_src else 0
    motion_mode = (active_motion.get("mode") or "choreography") if active_motion else None
    motion_preset = (active_motion.get("preset") or "standard") if active_motion else "standard"
    is_realism_refine = bool(active_motion and motion_preset in ("realism_fast", "realism"))
    is_fast_sharp = bool(active_motion and motion_preset == "fast_sharp")
    is_realism_preset = bool(is_realism_refine or is_fast_sharp)
    single_max = (
        motion_replace.REALISM_WINDOW_FRAMES
        if (is_realism_refine and quality != "draft")
        else motion_replace.SINGLE_MAX_FRAMES
    )
    multi_window = bool(active_motion and exact_motion_frames > single_max)

    width, height, enlarged = render_size(project, quality)
    config = sprite.settings(clip.get("sprite"))
    if config:
        if clip.get("link"):
            raise ValueError("Sprite Draft needs an independent shot")
        width, height = sprite.canvas_size(config, quality)
        enlarged = False
    is_replace_motion = bool(active_motion and motion_mode == "replace")
    is_rescene_motion = bool(active_motion and motion_mode == "rescene")
    if (is_replace_motion or is_rescene_motion) and not config and quality == "draft":
        # Video-referenced motion transfer needs ~0.5 MP (medium, e.g. 544x960) so drafts
        # commit cleanly from frame 0 instead of sticking to <Video 1>'s original performer.
        med_w, med_h = format_sizes(project_format(project))["medium"]
        if width * height < med_w * med_h:
            width, height = med_w, med_h
    use_sharp = not config and quality != "draft" and (
        is_fast_sharp
        or (not is_realism_refine and (size_name(project, quality) == SHARP or bool(active_motion and active_motion.get("sharp"))))
    )
    if is_realism_preset and not config and quality != "draft":
        # Hardware-verified RTX 5090 <=32 GB realism presets (F1 / F1-Fast / F2 / M5 / M7) sample Pass-1 at
        # large (~768x1344) so facial geometry and skin micro-texture have enough latent tokens.
        width, height = format_sizes(project_format(project))["large"]
        enlarged = bool(is_realism_refine)
    elif use_sharp:
        if is_replace_motion:
            # Legacy masked replacement on Sharp (RelayVSR 2x) samples at medium (544x960 / 960x544) before
            # exact 2x RelayVSR to 1088x1920 / 1920x1088.
            width, height = format_sizes(project_format(project))["medium"]
        enlarged = False
    if multi_window and not is_realism_refine:
        # Legacy multi-window relay tiles 124f windows; skip single-pass latent refine (use Sharp RelayVSR instead).
        enlarged = False
    _, profile = render_profile(project, available, clip, quality)
    size = dict(QUALITIES[quality], width=width, height=height, steps=profile["steps"][quality])
    if not enlarged:
        # The project asked for finals at the size they are sampled at.
        size.pop("refine", None)
    models = project_models(project, available, clip, quality)
    references = bool(edit) or bool(active_motion) or has_references(project, clip)
    # A family's fl2va model takes a prompt alone. Anything with references goes to its ref2va model.
    unet = models["unet"] if references or not models.get("unet_text") else models["unet_text"]
    hyperflow = is_hyperflow(unet)
    if (is_replace_motion or is_rescene_motion) and not config and hyperflow and size["steps"] < 6:
        size["steps"] = 6
    link = clip.get("link")
    length = linked_length(link, clip.get("seconds", 5)) if link else snap_frames(clip.get("seconds", 5))
    if finish_window is not None:
        length = int(finish_window[1])
    elif edit:
        length = motion_replace.snap_grid_frames(edit["out"] - edit["in"])
    elif motion_src:
        length = motion_replace.snap_grid_frames(exact_motion_frames)
    if config:
        length = config.get("frames", sprite.FRAMES)
    guides = clean_mid_guides(clip, length)
    graph = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "2": text_encoder_node(models["clip"], available, models.get("projection"),
                               models.get("conditioner"), models.get("conditioner_url")),
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
        "12": sigmas_node(profile.get("scheduler"), size["steps"], profile.get("sigmas")),
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
        # The take's latents as sampled, picture and sound, for the clips that will extend or bridge it.
        "20": {"class_type": "H3SaveHold", "inputs": {
            "latent": ["15", 0], "file": "%s/%s/clips/%s_%s.hold.safetensors" % (ROOT, folder, clip["id"], take)}},
    }
    if unet == models.get("unet_text"):
        graph["10"]["inputs"]["checkpoint"] = "fl2va"
    if hyperflow:
        # Few-step and CFG-free, on its own two-time schedule.
        graph["1"] = {"class_type": "H3HyperFlowLoader", "inputs": {"unet_name": unet, "weights_in": "auto"}}
        graph["12"] = {"class_type": "H3HyperFlowSigmas", "inputs": {"model": ["5", 0], "steps": size["steps"]}}
        if is_realism_preset and active_motion.get("h3_image_sampler"):
            # H3ImageSampler (res_2s_rkmk2e stochastic exponential Heun) injects organic matte skin/hair texture.
            graph["11"] = {"class_type": H3_IMAGE_SAMPLER_NODE, "inputs": dict(MOTION_REALISM_P1_SAMPLER)}
    model = ["1", 0]
    # A LoRA the family always samples with, such as a turbo LoRA. A family with one LoRA for each of its two
    # files gives the fl2va file its own.
    family_lora = "lora_text" if graph["10"]["inputs"]["checkpoint"] == "fl2va" and profile.get("lora_text") else "lora"
    if profile.get(family_lora) and models.get("profile_" + family_lora):
        name, strength = models["profile_" + family_lora], profile[family_lora][1]
        if profile.get("lora_node"):
            graph["94"] = {"class_type": profile["lora_node"], "inputs": {
                "model": model, "lora_name": name, "strength_model": strength}}
        else:
            graph["94"] = {"class_type": "H3Lora", "inputs": {"model": model, "lora_name": name, "strength": strength}}
        model = ["94", 0]
    # A linked clip holds frames of a final that already went through the Realism Slider, and carries their look
    # on. At full strength the slider adds its texture on top of that texture, and a chain of extensions grows
    # harsher with each one: on a chorus sung as a head and four extensions, 5, 18, 33 and 46% harsher than the
    # head. With none, and the held frames' far end redrawn (CONTEXT_NOISE), the skin turned glossy and waxy over
    # the chain. At half: +1% by the third clip and +12% by the fifth, with the skin's texture kept.
    # Do not inject photographic Realism Slider grain onto stylized character replacements.
    stylized_motion = bool(
        active_motion and motion_mode in ("replace", "rescene")
        and active_motion.get("style", "preserve") != "photoreal"
    )
    realism = 0.0 if stylized_motion else realism_strength(project, size) * (LINKED_REALISM if clip.get("link") else 1.0)
    if is_realism_preset and not stylized_motion and realism > 0.0:
        # Cap Realism Slider at 0.55 (start_sigma=0.85) on motion realism presets to avoid glossy cheek highlights.
        realism = min(realism, MOTION_REALISM_P2_LORA)
    if realism and models.get("lora") and installed(models["lora"], available):
        # The Realism Slider, on the model both passes sample with. Without it a final still renders.
        lora_inputs = {"model": model, "lora_name": models["lora"], "strength": realism}
        if is_realism_preset:
            lora_inputs["start_sigma"] = 0.85
        elif size.get("lora_start_sigma") is not None:
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
    for n, (row, known) in enumerate(project_patches(project, patches)):
        # The project's model patches, after every LoRA and before the shift, so both passes of a final sample
        # with them. The editor knows nothing about what one does: it wires the node as its definition says.
        node = str(PATCH_NODE + n)
        graph[node] = {"class_type": known["node"], "inputs": dict(patching.values(row, known), **{known["model"]: model})}
        model = [node, 0]
    graph["5"]["inputs"]["model"] = model
    take_inputs = graph["10"]["inputs"]
    motion_refine_cfg = None
    motion_refine_prompt = ""
    if active_motion and motion_src:
        # Load real look-ahead frames up to the grid-snapped render length when the source video has them,
        # so MiniMaxH3ReferenceToVideo never truncates <Video 1> and dissolves the subject at the tail.
        has_latent_up = bool(models.get("latent_upscaler") and installed(models["latent_upscaler"], available))
        want_multi_refine = bool(multi_window and is_realism_refine and quality != "draft" and has_latent_up)
        if want_multi_refine:
            windows_needed = motion_replace.plan_relay_windows(
                exact_motion_frames,
                single_max=motion_replace.REALISM_WINDOW_FRAMES,
                window_frames=motion_replace.REALISM_WINDOW_FRAMES,
                overlap_frames=motion_replace.OVERLAP_FRAMES,
            )
            motion_refine_cfg = {
                "multi_refine": True,
                "size": refine_size(project),
                "upscaler": models["latent_upscaler"],
                "sigmas": MOTION_REALISM_SIGMAS[motion_preset],
                "sampler_cfg": MOTION_REALISM_P2_SAMPLER[motion_preset],
                "use_h3_sampler": bool(hyperflow and active_motion.get("h3_image_sampler")),
            }
        else:
            windows_needed = motion_replace.plan_relay_windows(exact_motion_frames)
            if is_realism_refine:
                motion_refine_cfg = {"multi_refine": False}
        needed_frames = windows_needed[-1]["start"] + windows_needed[-1]["length"]
        src_total = max(motion_src["out"], int(motion_src.get("frames") or motion_src["out"]))
        load_last = min(src_total, motion_src["in"] + needed_frames)
        src_file = "/".join(filter(None, (motion_src["subfolder"], motion_src["file"])))
        graph["213"] = dict(class_type="H3LoadClip", inputs={
            "file": src_file,
            "first": motion_src["in"], "last": load_last, "reference_only": True,
        })
        source_info = {"file": src_file, "first": motion_src["in"], "last": load_last, "total_frames": src_total}
        cache_hits = check_cache(source_info, active_motion.get("character"), active_motion) if check_cache else None
        extra_chars = []
        if is_realism_preset and active_motion.get("cast"):
            member = next((p for p in (project.get("cast") or []) if p.get("id") == active_motion["cast"]), None)
            primary = active_motion.get("character") or {}
            primary_key = (primary.get("subfolder") or "", primary.get("file") or "")
            if member:
                pics = []
                if member.get("file"):
                    pics.append({"file": member["file"], "subfolder": member.get("subfolder") or ""})
                for p in member.get("pictures") or []:
                    if isinstance(p, dict) and p.get("file"):
                        pics.append({"file": p["file"], "subfolder": p.get("subfolder") or ""})
                for p in pics:
                    if (p["subfolder"], p["file"]) != primary_key:
                        extra_chars.append(p)
                        break
        ref_count = 1 + len(extra_chars)
        user_notes = edit["prompt"] if edit else str(clip.get("action") or "")
        prompt_text = motion_prompt(active_motion, user_notes, ref_count=ref_count)
        motion_refine_prompt = motion_prompt(active_motion, user_notes, ref_count=ref_count, refine_pass=True)
        wire_motion_transfer(
            graph, active_motion, hyperflow=hyperflow, width=size["width"],
            frame_count=exact_motion_frames, prompt_text=prompt_text,
            refine_prompt_text=motion_refine_prompt, seed=int(seed),
            source_info=source_info, cache_hits=cache_hits,
            extra_characters=extra_chars, refine_cfg=motion_refine_cfg,
        )
    elif edit:
        graph["213"] = dict(class_type="H3LoadClip", inputs={
            "file": "/".join(filter(None, (edit["subfolder"], edit["file"]))),
            "first": edit["in"], "last": edit["out"], "reference_only": True,
        })
        take_inputs["ref_videos.ref_video_0"] = ["213", 0]
        take_inputs["prompt"] = (
            "Edit <Video 1>. Preserve its motion, timing and framing wherever the requested change allows.\n"
            + ("<Picture 1> is the replacement first frame. Carry its visual changes through the shot.\n"
               if clip.get("first") else "")
            + edit["prompt"]
        )
    if config and not config.get("motion"):
        guide = config.get("guide")
        if guide:
            file = "/".join(filter(None, [guide["subfolder"], guide["file"]]))
            graph["210"] = {"class_type": "H3LoadClip", "inputs": {"file": file, "first": 0, "last": length}}
        else:
            guide_in = {k: config[k] for k in ("direction", "camera", "poses")}
            if length != sprite.FRAMES:
                guide_in["frames"] = length
            if config.get("canvas", "portrait") != "portrait":
                guide_in["canvas"] = config["canvas"]
            graph["210"] = {"class_type": "H3SpriteGuide", "inputs": guide_in}
        take_inputs["ref_videos.ref_video_0"] = ["210", 0]
    first, last = frame_of(clip, "first"), frame_of(clip, "last")
    auto_pin = sprite_auto_pin(project, clip) if config else None
    sp_pin = config.get("self_pin") if config else None
    if auto_pin and not first and not last:
        first = last = auto_pin
    elif sp_pin and not first:
        first = sp_pin
    # A sprite is drawn on white, and its pictures are often cutouts. Loaded as they are, their transparency turns
    # black and the take fades from black to the white the prompt asks for. A pinned picture of another shape than
    # the canvas (a portrait picture on a square canvas) is shown whole, where it would be cropped to fill it.
    sprite_still = dict(backdrop="white") if config else {}
    sprite_pin = dict(sprite_still, fit=(size["width"], size["height"])) if config else {}
    if first:
        graph["40"] = still_node(first, **sprite_pin)
        take_inputs["start_image"] = ["40", 0]
    if last:
        graph["41"] = still_node(last, **sprite_pin)
        take_inputs["end_image"] = ["41", 0]
    take_inputs["anchor"] = "both" if first and last else "start" if first else "end" if last else "none"
    if sp_pin and not guides:
        stride = sp_pin["stride"]
        sp_frames = [i for i in (stride, 2 * stride) if 0 < i < length]
        if sp_frames:
            graph["209"] = still_node(sp_pin, **sprite_pin)
            take_inputs["mid_frames"] = ", ".join(str(i) for i in sp_frames)
            for idx, _ in enumerate(sp_frames):
                take_inputs["mid_images.mid_image_%s" % idx] = ["209", 0]
    index = 0
    if not active_motion:
        for _, pictures in cast_pictures(project, clip):
            # A member's pictures sit together, in the order the prompt numbers them.
            for asset in pictures:
                node = str(50 + index)
                graph[node] = still_node(asset, **sprite_still)
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
            # The held frames far from the join are partly redrawn, so a chain does not pile up small shifts.
            "context_noise": CONTEXT_NOISE,
        })
        if arrives_on_pin(link):
            # The take carries on from the first clip as an extension does, and ends on the next clip's first
            # frame, pinned. H3 moves into a pinned frame in one shot, where it cuts to a run of held frames.
            at = int(link["b"]["first"])
            graph["7"] = load_node(dict(link["b"], last=at + 1), 1, True)
            take_inputs.update({"video_mode": "extend_after", "anchor": "end", "end_image": ["7", 0]})
        elif link["mode"] == "bridge":
            end = end_hold(link)
            graph["7"] = load_node(link["b"], end, False)
            take_inputs.update({"end_video": ["7", 0], "end_audio": ["7", 1], "end_hold": ["7", 2]})
            if end != hold:
                # A short run of the next clip, picture and sound: enough to say where things are and how they
                # move, and too little for H3 to take for another shot. It needs no guide frame beside it.
                take_inputs["end_hold_frames"] = end
            else:
                # A hold alone does not make the take arrive. The guide tells H3 what it must land on.
                graph["8"] = {"class_type": "ImageFromBatch", "inputs": {"image": ["7", 0], "batch_index": 0, "length": 1}}
                take_inputs.update({"mid_frames": str(length - hold), "mid_images.mid_image_0": ["8", 0]})
    if guides:
        # A bridge with a full hold uses its own landing guide. User guides are earlier
        # in the generated span. Keep the images in the same order as their frame numbers.
        pins = []
        for index, guide in enumerate(guides):
            node = str(200 + index)
            graph[node] = still_node(guide)
            pins.append((guide["frame"], [node, 0]))
        if take_inputs["mid_frames"]:
            pins.append((int(take_inputs["mid_frames"]), take_inputs["mid_images.mid_image_0"]))
        take_inputs["mid_frames"] = ", ".join(str(frame) for frame, _ in pins)
        for index, (_, image) in enumerate(pins):
            take_inputs["mid_images.mid_image_%s" % index] = image
    track = clip.get("track")
    if track:
        # The picture is generated to this slice of the track, laid from the take's first frame.
        file = "%s/%s" % (track["subfolder"], track["file"]) if track.get("subfolder") else track["file"]
        graph["61"] = {"class_type": "H3LoadSound", "inputs": {
            "file": file, "start": float(track.get("from") or 0), "seconds": round(length / float(FPS) + 1.0, 3)}}
        take_inputs.update({"audio_hold": "full", "audio_align": "take", "source_audio": ["61", 0]})
        if link and link["mode"] == "bridge" and not arrives_on_pin(link):
            take_inputs["end_audio"] = ["61", 0]
    if finish and not config and size_name(project, quality) == SHARP:
        raise ValueError("Sharp uses a video upscaler. Choose Full, Large, Medium, Small or your own Final size to finish a draft")
    already_refined = bool(motion_refine_cfg and motion_refine_cfg.get("applied"))
    if (size.get("refine") or finish) and not already_refined and models.get("latent_upscaler") and installed(models["latent_upscaler"], available):
        # Without the latent upscaler a final stays at H3's own size.
        tail = (
            MOTION_REALISM_SIGMAS[motion_preset]
            if (is_realism_refine and hyperflow)
            # A family whose fl2va LoRA needs another tail than its ref2va one names it.
            else ((family_lora == "lora_text" and profile.get("refine_sigmas_text"))
                  or profile.get("refine_sigmas") or profile["refine_from"])
        )
        add_refine(graph, refine_size(project) if enlarged else (width, height), tail, models["latent_upscaler"])
        if "248" in graph:
            # Feed the SwapGuard-verified Pass-1 latent into Pass-2 upscaling.
            graph["82"]["inputs"]["samples"] = ["248", 0]
        if is_replace_motion and "260" in graph:
            # Keep background noise_mask=0 during Pass-2 1088p Refine for clean masked replacement.
            graph["246"] = dict(class_type="H3ApplyEditMask", inputs={
                "latent": ["80", 1], "vae": ["3", 0], "source_video": ["260", 0], "mask": ["260", 1],
                "overlap_frames": 0,
            })
            graph["83"]["inputs"]["take"] = ["246", 0]
        if is_realism_refine:
            # F1 / F1-Fast / M5 / M7 Pass-2 Refine: ZERO swap_lora, ZERO ControlNet, H3ImageSampler micro-texture!
            if motion_refine_cfg and motion_refine_cfg.get("refine_model"):
                graph["88"]["inputs"]["model"] = list(motion_refine_cfg["refine_model"])
            if hyperflow and active_motion.get("h3_image_sampler"):
                graph["86"] = {"class_type": H3_IMAGE_SAMPLER_NODE, "inputs": dict(MOTION_REALISM_P2_SAMPLER[motion_preset])}
                graph["90"]["inputs"]["sampler"] = ["86", 0]
            if is_rescene_motion:
                graph["80"]["inputs"].pop("ref_videos.ref_video_0", None)
                graph["80"]["inputs"]["prompt"] = motion_refine_prompt
        if length > LONG_TAKE:
            add_refine_windows(graph)
    sharp_detail = float(active_motion.get("sharp_detail", SHARP_DETAIL)) if active_motion else SHARP_DETAIL
    if use_sharp and is_fast_sharp:
        # F2 recipe: upscale 768p generation via RelayVSR 2x (detail=0.85) FIRST, then composite lossless 1080p plate!
        add_sharp(graph, refine_size(project), available, detail=sharp_detail)
        if active_motion:
            wire_motion_plate(graph, active_motion)
    else:
        if active_motion:
            wire_motion_plate(graph, active_motion)
        if use_sharp:
            add_sharp(graph, refine_size(project), available, detail=sharp_detail)
    if finish:
        start_from_take(graph, clip, finish, length, finish_window)
    markers = style_timeline.schedule(clip.get("style_markers"), length, project.get("style", ""))
    if markers:
        # Each style prefixes the same shot text, so content, cast and speech stay constant.
        shot = clip_prompt(dict(project, style="__h3_style__"), clip)
        for node in ("10", "80"):
            if node in graph:
                graph[node]["inputs"].update(prompt=shot, style_markers=json.dumps(markers, ensure_ascii=False))
        for node in ("13", "88"):
            if node in graph:
                graph[node]["class_type"] = "H3StyleGuider"
    mods = clip_refmods(project, clip, available)
    if mods:
        # Each chosen cast member's RefMod joins what H3 Take conditions on, in both passes of a final.
        loader = {"show_info": False}
        for slot in range(1, REFMOD_SLOTS + 1):
            _, file, strength = mods[slot - 1] if slot <= len(mods) else (None, "(none)", 1.0)
            loader.update({"mod_%s" % slot: file, "strength_%s" % slot: strength, "copies_%s" % slot: 1})
        graph["72"] = {"class_type": "MiniMaxH3RefModsLoader", "inputs": loader}
        guiders = sorted((key for key, node in graph.items() if node["class_type"] in ("BasicGuider", "H3StyleGuider")), key=int)
        for number, key in enumerate(guiders):
            node = str(73 + number)
            graph[node] = {"class_type": "MiniMaxH3RefModApply", "inputs": {
                "conditioning": graph[key]["inputs"]["conditioning"], "mods": ["72", 0], "override": False, "retention": 1.0,
                "curve_direction": "constant", "scramble_seed": -1, "curve_shape": "linear", "curve_value": 1.0}}
            graph[key]["inputs"]["conditioning"] = [node, 0]
    if config:
        # Sprite drafts are silent and display the finished loop. Skip live preview, audio decode,
        # and hold latent saving unless explicitly refining a draft later.
        for guider in ("13", "88"):
            if guider in graph:
                graph[guider]["inputs"]["model"] = graph["9"]["inputs"]["model"]
        graph.pop("9")
        graph.pop("17", None)
        if "18" in graph:
            graph["18"]["inputs"].pop("audio", None)
        if not config.get("refine"):
            graph.pop("20", None)
        layout = (finish or {}).get("sprite", {}).get("layout")
        sheet_in = {
            "images": list(graph["18"]["inputs"]["images"]), "poses": config["poses"],
            "filename_prefix": graph["19"]["inputs"]["filename_prefix"],
            "layout": json.dumps(layout) if layout else "",
        }
        if config.get("canvas", "portrait") != "portrait":
            sheet_in["canvas"] = config["canvas"]
        if config.get("tile_size"):
            sheet_in["tile_width"], sheet_in["tile_height"] = config["tile_size"]
        graph["211"] = {"class_type": "H3SpriteSheet", "inputs": sheet_in}
        if config.get("motion") == "idle":
            graph["211"]["inputs"]["idle"] = True
        if config.get("animation"):
            graph["211"]["inputs"]["full_action"] = True
        if config.get("alpha") == "hard":
            graph["211"]["inputs"]["alpha"] = "hard"
        if config.get("cutout") != "edge":
            matte = sprite.matte_model(available)
            if matte:
                graph["212"] = {"class_type": "LoadBackgroundRemovalModel", "inputs": {"bg_removal_name": matte}}
                graph["211"]["inputs"]["bg_model"] = ["212", 0]
    if finish_window is not None:
        # Partial finishes and edit-draft finishes leave the full draft's latents intact on the draft take.
        graph.pop("20", None)
    elif edit:
        if quality == "draft" and not active_motion and "20" in graph:
            # Keep the draft's sampled latents so later partial/full finals can refine them without re-sampling;
            # partial=True stops extend/bridge holds from misaligning them against the assembled video.
            graph["20"]["inputs"]["partial"] = True
        else:
            # The finished take contains source frames around the edit, so the sampler's latents don't line up.
            graph.pop("20", None)
    return graph


def start_from_take(graph, clip, source, length, window=None):
    """Make a final's refine pass start from a rendered draft's latents in place of its own first pass.

    The draft's sampled latents, saved beside its video, take the place of the first pass's: they are enlarged and
    refined at the refine size as a final's would be, and their sound is the take's sound.
    When window=(grid_in, grid_len) is given, only that on-grid slice is loaded and refined.
    """
    if "90" not in graph:
        raise ValueError("finishing a draft needs the latent upscaler: install the latent upscaler from Get the models")
    if window is None and int(source.get("frames") or 0) != int(length):
        raise ValueError("%s runs %s frames now and its draft %s. Render a new draft to finish"
                         % (clip.get("name", "This clip"), length, source.get("frames")))
    file = "%s/%s" % (source["subfolder"], source["file"]) if source.get("subfolder") else source["file"]
    inputs = {"file": file}
    if window is not None:
        grid_in, grid_len = int(window[0]), int(window[1])
        inputs["first"] = grid_in
        inputs["last"] = grid_in + grid_len
    graph["15"] = {"class_type": "H3LoadTake", "inputs": inputs}
    # The first pass's sampler, and its decode, which the refine replaces, have nothing left to do.
    for unused in ("13", "16"):
        graph.pop(unused, None)


# How far the held frames furthest from a join are redrawn, as a mask value, falling to nothing at the join (see
# policy.hold_curve). A take that holds every frame frozen copies them as they are and adds its own small lean on
# top, and over a chain the leans add up: on HyperFlow, a head clip and four extensions at final size came out 4, 16,
# 28 and 31% harsher than the head, with shinier, more orange skin. At 0.95: -3, 3, 6 and 3%, with the joins
# scoring the same. 0.99 let more drift through, 0.8 about as much as 0.95.
CONTEXT_NOISE = 0.95
# The share of the Realism Slider's strength an extension, bridge or merge runs (see render_graph): 0.6 at the
# usual 1.2.
LINKED_REALISM = 0.5


def add_refine(graph, size, tail, upscaler):
    """Enlarge the sampled latent and redraw its last steps at that size, keeping the sound.

    A second take at the refine size gives the conditioning there (pinned frames and pictures
    at that size) and its held frames: the refined latents of the clip it joins, when that clip
    was refined at this size, and its video otherwise. The picture comes out at the refine size,
    and the sound is the first pass's. tail is the step in the final's own sigmas the redraw starts from, or
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
    # The refined picture is saved with the take as well: a clip that extends this one holds it in its own refine.
    graph["20"]["inputs"]["final"] = ["90", 0]


# A long take is one generation, not a chain: nothing is ever conditioned on a finished clip, so nothing piles up,
# and the place stays one place. Its first pass draws the whole shot at once. Sampled in context windows instead, each
# window drew its own room and the windows dissolved into one another (a 20.75 s chorus: 44% of the background changed).
# Its refine redraws only the last steps of a settled picture, so it can go in windows: on a 20.75 s take at
# 1088 x 1920 the refine's three steps took 22 minutes whole (50 GB of GPU memory in use at the peak) and the whole
# final 15 minutes in windows of 124 frames, the same picture to the pore, the place, the skin and the lips held.
LONG_TAKE = 249                 # frames (10.4 s): a refine longer than this goes in windows
REFINE_WINDOW = (124, 39)       # frames each window and frames two windows share, both 17n + 5
CONTEXT_WINDOWS = "MiniMaxH3ContextWindows"


def add_refine_windows(graph):
    """Put ComfyUI's context windows on the refine pass's model, and only there."""
    length, overlap = REFINE_WINDOW
    graph["99"] = {"class_type": CONTEXT_WINDOWS, "inputs": {
        "model": graph["88"]["inputs"]["model"], "context_length": length, "context_overlap": overlap,
        "fuse_method": "pyramid", "freenoise": True, "split_conds_to_windows": False}}
    graph["88"]["inputs"]["model"] = ["99", 0]


def without_nodes(graph, missing):
    """The graph with each node of a class in missing taken out, its users given what it was given: for model patches
    a ComfyUI too old to have does not need (context windows)."""
    out = copy.deepcopy(graph)
    for key in [k for k, node in out.items() if node.get("class_type") in missing]:
        source = out.pop(key)["inputs"].get("model")
        for node in out.values():
            for name, value in node.get("inputs", {}).items():
                if isinstance(value, list) and len(value) == 2 and value[0] == key:
                    node["inputs"][name] = source
    return out


def add_sharp(graph, size, available=None, *, detail=SHARP_DETAIL):
    """Enlarge the decoded picture two times with a video upscaler, then scale it down to the final's size.

    The picture is sampled once, at H3's own size. The sound is untouched.
    """
    _, files = pick_upscaler(SHARP_UPSCALER, available)
    width, height = size
    picture = list(graph["18"]["inputs"]["images"])
    graph.update({
        "31": {"class_type": "UNETLoader", "inputs": {"unet_name": files["model"], "weight_dtype": "default"}},
        "32": {"class_type": "VAELoader", "inputs": {"vae_name": files["vae"]}},
        "33": {"class_type": "RelayVSRConditioning", "inputs": {"model": ["31", 0], "images": picture}},
        "34": {"class_type": "KSampler", "inputs": {
            "model": ["31", 0], "seed": 0, "steps": 1, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
            "positive": ["33", 0], "negative": ["33", 0], "latent_image": ["33", 1], "denoise": 1.0}},
        "35": {"class_type": "VSRDecode", "inputs": {"vae": ["32", 0], "samples": ["34", 0], "images": picture, "detail": float(detail)}},
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


def project_card(project):
    """What a list of projects shows for one of them: its name, shape, length, and a picture to know it by.

    The picture is the first rendered clip on the timeline, or failing that the first rendered clip anywhere.
    """
    clips = {clip.get("id"): clip for clip in project.get("clips") or [] if isinstance(clip, dict)}
    frames, placed, picture = 0, 0, None
    for item in sorted((i for i in project.get("timeline") or [] if isinstance(i, dict)), key=lambda i: i.get("start") or 0):
        try:
            frames = max(frames, int(item.get("start") or 0) + item_length(item))
        except (KeyError, TypeError, ValueError):
            continue
        placed += 1
        picture = picture or take_picture(clips.get(item.get("clip")))
    for clip in clips.values():
        picture = picture or take_picture(clip)
    return {"name": str(project.get("name") or ""), "format": project_format(project),
            "clips": placed, "frames": frames, "picture": picture, "rendering": len(project.get("pending") or [])}


def take_picture(clip):
    take = active_take(clip or {})
    if not take or not take.get("poster"):
        return None
    return {"file": take["poster"], "subfolder": take.get("subfolder") or ""}


TRANSITIONS = {"dissolve": "fade", "black": "fadeblack", "wipe": "wipeleft"}


MIN_SPEED, MAX_SPEED = 0.25, 4.0


def clean_speed(value):
    """A timeline speed. The project file comes from the client, so anything else is refused."""
    speed = 1.0 if value in (None, "") else float(value)
    if not math.isfinite(speed) or not MIN_SPEED <= speed <= MAX_SPEED:
        raise ValueError("speed must be between %s and %s" % (MIN_SPEED, MAX_SPEED))
    return speed


# How loud a clip plays where it sits on the timeline: 1 is as rendered, 0 is silent, and
# above 1 it is louder in the export. The same range as an added sound's.
ITEM_GAIN = (0.0, 2.0)


def clean_gain(value):
    """A timeline item's volume. The project file comes from the client, so anything else is refused."""
    gain = 1.0 if value in (None, "") else value
    if isinstance(gain, bool) or not isinstance(gain, (int, float)) or not math.isfinite(gain) \
            or not ITEM_GAIN[0] <= gain <= ITEM_GAIN[1]:
        raise ValueError("volume must be a number from %g to %g" % ITEM_GAIN)
    return round(float(gain), 2)


def clean_crop(value):
    """An optional rectangle in the source picture, independent of take resolution."""
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"x", "y", "w", "h"}:
        raise ValueError("crop needs x, y, w and h in fractions of the source picture")
    if any(isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n)
           or not 0 <= n <= 1 for n in value.values()):
        raise ValueError("crop coordinates must be numbers from 0 to 1")
    if value["w"] <= 0 or value["h"] <= 0 or value["x"] + value["w"] > 1.000000001 or value["y"] + value["h"] > 1.000000001:
        raise ValueError("crop needs a positive area inside the picture")
    return {k: value[k] for k in ("x", "y", "w", "h")}


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


def export_window(project, span):
    if not isinstance(span, dict) or set(span) != {"from", "to"} or any(type(n) is not int for n in span.values()):
        raise ValueError("range needs integer timeline frames from and to (exclusive)")
    end = scene_end(project)
    if not 0 <= span["from"] < span["to"] <= end:
        raise ValueError("range must fall inside the cut, from 0 to %s frames" % end)
    return span["from"], span["to"]


def merge_ranges(ranges):
    """Merge overlapping or adjacent [start, end] frame spans into sorted disjoint pairs."""
    spans = sorted((int(a), int(b)) for a, b in (ranges or []) if int(b) > int(a))
    merged = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def range_covered(ranges, first, last):
    """Whether frames [first, last) are completely covered by ranges."""
    first, last = int(first), int(last)
    return any(a <= first and b >= last for a, b in merge_ranges(ranges))


def take_range_status(take, first, last):
    """Quality status of frames [first, last) inside a take: 'final', 'draft', 'partial', or 'footage'."""
    if not take:
        return None
    first, last = int(first), int(last)
    quality = take.get("quality")
    if quality == "draft":
        edit = take.get("edit")
        if edit:
            lo, hi = max(first, int(edit["in"])), min(last, int(edit["out"]))
            if lo >= hi:
                return "footage"
            return "draft" if lo == first and hi == last else "partial"
        return "draft"
    if quality == "final":
        ranges = take.get("final_ranges")
        if ranges:
            merged = merge_ranges(ranges)
            if any(a <= first and b >= last for a, b in merged):
                return "final"
            if any(max(first, a) < min(last, b) for a, b in merged):
                return "partial"
            return "draft"
        return "final"
    return quality or "footage"


def export_plan(project, scale=1.0, span=None):
    """What an export of the timeline will hold, before making it: its size and length, and what to know first.

    drafts counts the generated clips whose take on the timeline is a draft, missing the ones
    with no render (they are left out), and gaps the stretches of black between clips.
    """
    clips = {c["id"]: c for c in project.get("clips", [])}
    first, last = export_window(project, span) if span is not None else (0, None)
    width = height = drafts = missing = placed = 0
    for item in project.get("timeline", []):
        in_range = last is None or (item["start"] < last and first < item_end(item))
        placed += int(in_range)
        clip = clips.get(item.get("clip")) or {}
        take = active_take(clip)
        if not take:
            missing += int(in_range)
            continue
        status = take_range_status(take, item["in"], item["out"])
        if in_range and status in ("draft", "partial") and (clip.get("kind") != "footage" or take.get("edit")):
            drafts += 1
        w, h = take_size(project, take)
        if w > width:
            width, height = w, h
    end = cursor = gaps = 0
    for item in ordered(project):
        if item["start"] > max(cursor, first) and (last is None or cursor < last):
            gaps += 1
        cursor = max(cursor, item_end(item))
        end = cursor
    stop = end if last is None else last
    effects = [e for e in project.get("effects") or [] if e["start"] < stop and e["start"] + e["frames"] > first]
    return {"width": scaled(width, scale), "height": scaled(height, scale), "frames": stop - first, "seconds": round((stop - first) / float(FPS), 2),
            **({"range": {"from": first, "to": stop}} if span is not None else {}),
            "clips": placed - missing, "drafts": drafts, "missing": missing, "gaps": gaps,
            "effects": sum(1 for e in effects if active_take(e)),
            "effects_missing": sum(1 for e in effects if not active_take(e)),
            "effects_stale": sum(1 for e in effects if effect_stale(project, e))}


def scaled(side, scale):
    """A picture side at a part of its size, kept even, as video encoders need."""
    return max(2, 2 * int(round(side * scale / 2.0)))


def export_command(project, paths, out_path, width=None, height=None, scale=1.0, quality="high", sound_paths=None,
                   effect_paths=None, span=None):
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
    # Keep the full cut's output dimensions. Move the working clock to the start
    # of the earliest contributing media, preserving phase within each placement.
    window = export_window(project, span) if span is not None else None
    if window:
        first, last = window
        items = [i for i in items if i["start"] < last and item_end(i) > first]
        sounds = [s for s in project.get("sounds") or [] if s["start"] < last and s["start"] + s["frames"] > first]
        effects = [e for e in project.get("effects") or [] if e["start"] < last and e["start"] + e["frames"] > first]
        origin = min([first] + [row["start"] for row in (*items, *sounds, *effects)])
        items = [dict(i, start=i["start"] - origin) for i in items]
        project = dict(project, sounds=[dict(s, start=s["start"] - origin) for s in sounds],
                       effects=[dict(e, start=e["start"] - origin) for e in effects])
        window = first - origin, last - origin

    args = ["ffmpeg", "-y", "-v", "error"]
    chains = []
    finish = "fps=%s,scale=%s:%s:force_original_aspect_ratio=increase,crop=%s:%s,setsar=1,format=yuv420p,settb=1/%s" % (
        FPS, width, height, width, height, FPS)
    sound = SOUND_FORMAT

    def gap_segment(label, frames):
        # color rounds duration up to the next frame. Bound the source, then select the exact
        # frame count (a decimal approximation of e.g. 10/24 alone can produce 11 frames).
        seconds = frames / float(FPS)
        chains.append("color=c=black:s=%sx%s:r=%s:d=%.8f,trim=end_frame=%d,format=yuv420p,setsar=1,settb=1/%s[%sv]" % (
            width, height, FPS, seconds, frames, FPS, label))
        chains.append("anullsrc=r=48000:cl=stereo:d=%.8f,atrim=end_sample=%d,%s[%sa]" % (
            seconds, frames * 48000 // FPS, sound, label))

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
        crop = clean_crop(item.get("crop"))
        crop_filter = ""
        if crop:
            crop_filter = ("crop=w='max(1,round(iw*%(w).12f))':h='max(1,round(ih*%(h).12f))':"
                           "x='round(iw*%(x).12f)':y='round(ih*%(y).12f)':exact=1," % crop)
        chains.append("[%s:v]trim=start_frame=%s:end_frame=%s,setpts=(PTS-STARTPTS)/%.5f,%s%s[v%s]" % (
            index, first, last, speed, crop_filter, finish, index))
        tempo = tempo_chain(speed)
        gain = clean_gain(item.get("gain"))
        frames = item_length(item)
        samples = frames * 48000 // FPS
        take = active_take(clips[item["clip"]]) or {}
        source_frames = take.get("frames", last)
        envelope = sound_envelope_filter({**item, "frames": frames, "seconds": source_frames / FPS,
                                          "offset": first / FPS, "rate": speed,
                                          **{key: min(item.get("_fade_frames", frames), item.get(key, 0))
                                             for key in ("fade_in", "fade_out")},
                                          "gain_points": [p for p in item.get("gain_points", []) if p["frame"] <= source_frames]})
        chains.append("[%s:a]atrim=start=%.5f:end=%.5f,asetpts=PTS-STARTPTS,%s%s%s%s,apad=whole_len=%d,atrim=end_sample=%d[a%s]" % (
            index, first / float(FPS), last / float(FPS), tempo + "," if tempo else "",
            "volume=%.3f," % gain if gain != 1 else "", sound, envelope, samples, samples, index))

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

    # Unrendered placements can also trail the last rendered picture. Keep the timeline clock
    # running across them, including any soundtrack or VFX that plays there.
    timeline_end = window[1] if window else scene_end(project)
    if timeline_end > cursor:
        gap_segment("tail", timeline_end - cursor)
        join("tailv", "taila")
        cursor = timeline_end

    picture_track = audio_track_filter(project, 0)
    if picture_track:
        chains.append("[%s]%s[track0]" % (audio, picture_track[1:]))
        audio = "track0"

    extra = []
    for row in project.get("sounds") or []:
        if row.get("id") not in (sound_paths or {}) or row.get("start", 0) >= cursor or not row.get("gain", 1.0):
            continue
        number = len(items) + len(extra)
        args += ["-i", sound_paths[row["id"]]]
        offset = float(row.get("offset") or 0.0)
        # At 48 kHz each timeline frame is exactly 2,000 samples.
        delay = row.get("start", 0) * (48000 // FPS)
        delay_arg = str(delay // 48) if delay % 48 == 0 else "%dS" % delay
        chains.append("[%s:a]atrim=start=%.8f:end=%.8f,asetpts=PTS-STARTPTS,volume=%.3f,%s%s%s,adelay=%s:all=1[s%s]" % (
            number, offset, offset + row["frames"] / float(FPS), float(row.get("gain", 1.0)), SOUND_FORMAT,
            sound_envelope_filter(row), audio_track_filter(project, (row.get("lane") or 0) + 1, row.get("start", 0)),
            delay_arg, number))
        extra.append("s%s" % number)
    if extra:
        # A packet cut by atrim can be short. FFmpeg 6's amix can stop early when that
        # stream shares its source with the video. Regroup samples without padding the tail.
        chains.append("[%s]asetnsamples=n=1024:p=0[cut_audio]" % audio)
        # The cut sets the length: a sound that runs past its end is cut there.
        chains.append("[cut_audio]%samix=inputs=%s:duration=first:dropout_transition=0:normalize=0[mix]" % (
            "".join("[%s]" % label for label in extra), len(extra) + 1))
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

    if window:
        first, last = window
        # Select after composition to keep the clocks of transitions, VFX and sound envelopes.
        chains.append("[%s]trim=start_frame=%d:end_frame=%d,setpts=PTS-STARTPTS[rangev]" % (video, first, last))
        chains.append("[%s]atrim=start_sample=%d:end_sample=%d,asetpts=PTS-STARTPTS[rangea]" % (
            audio, first * 48000 // FPS, last * 48000 // FPS))
        video, audio = "rangev", "rangea"
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
    if out.get("audio_tracks"):
        out["_audio_track_origin"] = project.get("_audio_track_origin", 0) + first
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
        if item.get("fade_in") or item.get("fade_out"):
            new["_fade_from"] = item.get("_fade_from", 0) + head
            new["_fade_frames"] = item.get("_fade_frames", item_length(item))
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
        new["offset"] = float(sound.get("offset") or 0.0) + head / float(FPS)
        if sound.get("fade_in") or sound.get("fade_out"):
            # An export range hears the middle of the original fades. Only an actual edit makes new edges.
            new["_fade_from"] = sound.get("_fade_from", 0) + head
            new["_fade_frames"] = sound.get("_fade_frames", sound["frames"])
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
SOUND_FIELDS = ("name", "start", "offset", "frames", "gain", "lane", "fade_in", "fade_out", "gain_points")
SOUND_GAIN = (0.0, 2.0)
GAIN_POINT_LIMIT = 64


def sound_frame(value, limit, field):
    if type(value) is not int or not 0 <= value <= limit:
        raise ValueError("%s must be a whole number from 0 to %d" % (field, limit))
    return value


def clean_gain_points(points, limit, gain_range=(0, 1), digits=2):
    if not isinstance(points, list) or len(points) > GAIN_POINT_LIMIT:
        raise ValueError("gain_points must be a list of up to %d points" % GAIN_POINT_LIMIT)
    result = []
    for point in points:
        if not isinstance(point, dict) or set(point) != {"frame", "gain"}:
            raise ValueError("each gain point needs a source frame and gain")
        result.append({"frame": sound_frame(point["frame"], limit, "gain point frame"),
                       "gain": clean_strength(point["gain"], gain_range, "gain point", digits)})
    result.sort(key=lambda p: p["frame"])
    if any(a["frame"] == b["frame"] for a, b in zip(result, result[1:])):
        raise ValueError("gain points must have different frames")
    return result


def clean_audio_tracks(tracks):
    lanes = {str(lane) for lane in range(SOUND_LANES + 1)}
    if not isinstance(tracks, dict) or not tracks.keys() <= lanes:
        raise ValueError("audio tracks must be keyed by lane number from 0 to %d" % SOUND_LANES)
    result = {}
    for key, track in tracks.items():
        check_fields(track, ("gain", "gain_points"))
        gain = clean_strength(track.get("gain", 1), SOUND_GAIN, "track gain", 6)
        points = clean_gain_points(track.get("gain_points", []), 2**31 - 1, SOUND_GAIN, 6)
        if gain != 1 or points:
            result[key] = {"gain": gain, "gain_points": points}
    return result


def gain_curve_expression(points, position):
    """Piecewise-linear gain with constant values outside the first and last points."""
    if len(points) == 1:
        return "%g" % points[0]["gain"]

    def segment(lo, hi):
        if hi == lo + 1:
            left, right = points[lo], points[hi]
            delta = right["gain"] - left["gain"]
            if not delta:
                return "%g" % left["gain"]
            return "(%g+%g*clip((%s-%d)/%d,0,1))" % (
                left["gain"], delta, position, left["frame"], right["frame"] - left["frame"])
        mid = (lo + hi) // 2
        return "if(lt(%s,%d),%s,%s)" % (position, points[mid]["frame"], segment(lo, mid), segment(mid, hi))

    return segment(0, len(points) - 1)


def audio_track_filter(project, lane, start=0):
    track = (project.get("audio_tracks") or {}).get(str(lane))
    if not track:
        return ""
    gain = track.get("gain", 1)
    points = track.get("gain_points", [])
    if not points:
        return ",volume=%g" % gain if gain != 1 else ""
    position = "(t*%d+%d)" % (FPS, project.get("_audio_track_origin", 0) + start)
    return ",aeval=exprs='val(ch)*%g*%s':c=same" % (gain, gain_curve_expression(points, position))


def sound_envelope_filter(sound):
    """Sample-accurate gain over the source file, multiplied by fades at the edges of the sound."""
    if not any(sound.get(key) for key in ("gain_points", "fade_in", "fade_out")):
        return ""
    limit = max(1, math.floor(sound["seconds"] * FPS + 0.5))
    points = clean_gain_points(sound.get("gain_points", []), limit)
    factors = []
    if points:
        if points[0]["frame"] > 0:
            points.insert(0, {"frame": 0, "gain": 1})
        if points[-1]["frame"] < limit:
            points.append({"frame": limit, "gain": 1})
        pos = "(t*%g+%.8f)" % (FPS * sound.get("rate", 1), sound.get("offset", 0) * FPS)
        factors.append(gain_curve_expression(points, pos))
    frames = sound.get("_fade_frames", sound["frames"])
    head = sound.get("_fade_from", 0)
    for key in ("fade_in", "fade_out"):
        length = min(frames, sound_frame(sound.get(key, 0), max(limit, frames), key))
        if length:
            progress = "(t*%d+%d)" % (FPS, head) if key == "fade_in" else "(%d-t*%d)" % (frames - head, FPS)
            factors.append("clip(%s/%d,0,1)" % (progress, length))
    return ",aeval=exprs='val(ch)*%s':c=same" % "*".join(factors)


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
    limit = max(1, math.floor(seconds * FPS + 0.5))
    for key in ("fade_in", "fade_out"):
        if key in fields:
            out[key] = sound_frame(fields[key], limit, key)
    if "gain_points" in fields:
        out["gain_points"] = clean_gain_points(fields["gain_points"], limit)
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
            if item.get("crop"):
                rows[-1].append(clean_crop(item["crop"]))
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
# A linked clip holds HOLD frames of its source, and both files then show them. The timeline cuts from one file to
# the other this many frames inside them, not at their edge. A decoder draws the last frames of a file with nothing
# after them and the first with nothing before them, so a file's last 3 frames and its first 7 never quite match the
# same frames decoded in the middle of the other file (measured on 16 joins at a file's end: 1.5 dB under the rest
# on the last frame, 5 dB at worst, level again from the fourth; at a file's start 6 to 9 dB on the first frame and
# level from the eighth). The held sound is also softened over the 8 steps, 5 frames, next to the new material, so
# at the edge of the hold the two waves have drifted apart and the cut can click. 9 frames is clear of all three,
# and is 15 whole steps of the sound.
HANDOVER = 9
MERGE_FRAMES = 63
CLIP_FIELDS = ("name", "action", "camera", "speaker", "dialogue", "sound", "music", "seconds", "cast", "seed",
               "first", "last", "language", "track", "bin", "profile", "loras", "style_markers", "mid_guides", "sprite", "motion")
TEXT_FIELDS = ("name", "action", "camera", "speaker", "dialogue", "sound", "music", "language")
PROJECT_FIELDS = ("style", "format", "language", "profile", "loras", "patches", "realism", "checkpoints", "text_encoder", "text_projection", "text_conditioner",
                  "text_conditioner_url", "vae", "audio_vae", "sizes")
# The model files set_project names, and their keys in the project's models. A projection only
# applies to a 4B or 8B text encoder; one that does not fit it gives way to one that does.
PROJECT_MODEL_FIELDS = {**PROJECT_ENCODERS, "text_projection": "projection",
                        "text_conditioner": "conditioner", "text_conditioner_url": "conditioner_url"}
# A project's own LoRAs: how many, how strong, and where their nodes start in a render graph.
LORA_LIMIT = 8
LORA_RANGE = (-2.0, 2.0)
REALISM_RANGE = (0.0, 3.0)
LORA_NODE = 100
# A project's model patches: how many, and where their nodes start in a render graph, clear of the LoRAs.
PATCH_LIMIT = patching.LIMIT
PATCH_NODE = 120
CAST_FIELDS = ("name", "description", "voice", "pictures", "refmod", "view")
# A RefMod is a reference saved by the ComfyUI-MiniMaxH3Mod pack. The editor renders with one only
# when that pack is installed: its loader takes this many, and the model list names each as refmod:<name>.
REFMOD_SLOTS = 8
REFMOD_PREFIX = "refmod:"
# A cast member can have several pictures, the main one included. H3 takes this many pictures in one shot.
CAST_PICTURES = 8
H3_PICTURES = 9
PICTURES = (".png", ".jpg", ".jpeg", ".webp")
BIN_FIELDS = ("name",)
LINK_MODES = ("extend_after", "extend_before", "bridge", "merge")
# How a bridge reaches the next clip: on a short run of its first frames (see end_hold), on its first frame alone,
# pinned (see arrives_on_pin), or on as many frames as it holds of the first clip.
ARRIVALS = ("short", "pin", "hold")
# The short run: 22 frames, a 17-frame group of the latent and the 5 frames before the next, so a take of 39 + 51k
# frames can end on it. The cut then sits a handover inside it, clear of the frames a decoder draws badly.
END_HOLD = 22
# A bridge that arrives on a pinned frame gets 153 new frames unless told otherwise. A move between two framings
# needs the room: at 102 new frames a push-in from a wide shot to a close-up came out as one shot in 0 of 6 takes
# (it dissolved or cut instead), and at 153 in 4 of 6.
BRIDGE_SECONDS = 6.375
LINK_TITLES = {"extend_after": "After ", "extend_before": "Before ", "bridge": "Bridge from ", "merge": "Merge from "}
SAVED_FIELDS = ("action", "camera", "speaker", "dialogue", "sound", "music", "seconds", "cast", "track", "style_markers", "mid_guides", "sprite", "motion")


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


def link_handover(link):
    """How many of the held frames a linked clip plays itself, at each side it joins. See HANDOVER."""
    return int((link or {}).get("handover") or 0)


def linked_range(link, frames):
    """The part of a linked render the timeline shows.

    Held frames are already on screen in the source, so the clip starts after them, or ends before them. A link
    with a handover plays the last few of them itself, and its source stops that much earlier.
    """
    if not link:
        return 0, frames
    hold, over = int(link["hold"]), link_handover(link)
    if link["mode"] == "extend_after":
        return hold - over, frames
    if link["mode"] == "extend_before":
        return 0, frames - hold + over
    if arrives_on_pin(link):
        # The last frame is the next clip's own first frame, which that clip shows itself.
        return hold - over, frames - 1
    return hold - over, frames - end_hold(link) + over


def planned_frames(clip):
    """What a render of this clip will be, so its place on the timeline is right before it exists."""
    take = active_take(clip)
    if take:
        return take["frames"]
    if clip.get("motion"):
        src = clip["motion"].get("source") or {}
        span = int(src.get("out") or 0) - int(src.get("in") or 0)
        if span > 0:
            return span
    if clip.get("link"):
        return linked_length(clip["link"], clip.get("seconds", 5))
    if clip.get("sprite") is not None:
        return clip["sprite"].get("frames", sprite.FRAMES) if isinstance(clip["sprite"], dict) else sprite.FRAMES
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


def clean_mid_guides(clip, length=None):
    """Validated authored guides in source-frame order. Linked shots reserve their held frames."""
    rows = clip.get("mid_guides")
    if rows is None:
        return []
    if not isinstance(rows, list):
        raise ValueError("mid_guides must be a list of pictures with frame, file and subfolder")
    if not rows:
        return []
    if clip.get("kind") == "footage":
        raise ValueError("mid-frame guides belong to generated clips")
    link = clip.get("link") or {}
    seconds = clip.get("seconds", 5)
    sprite_len = clip["sprite"].get("frames", sprite.FRAMES) if isinstance(clip.get("sprite"), dict) else sprite.FRAMES
    length = length or (linked_length(link, seconds) if link else sprite_len if clip.get("sprite") else snap_frames(seconds))
    first, last, limit = 1, length - 2, 8
    if link.get("mode") in ("extend_after", "bridge"):
        first = link["hold"]
    if link.get("mode") == "extend_before":
        last = length - link["hold"] - 1
    if link.get("mode") == "bridge" and not arrives_on_pin(link):
        last = length - end_hold(link) - 1
        limit -= int(end_hold(link) == link["hold"])  # H3 Take has eight mid image sockets.
    if len(rows) > limit:
        raise ValueError("this shot takes up to %s mid-frame guides" % limit)
    found = {}
    for row in rows:
        if not isinstance(row, dict) or type(row.get("frame")) is not int or not first <= row["frame"] <= last:
            raise ValueError("mid-frame guides must be on generated frames %s to %s of this shot" % (first, last))
        frame = row["frame"]
        if frame in found:
            raise ValueError("there is already a mid-frame guide at frame %s" % frame)
        asset = clean_asset(row, "a mid-frame guide")
        if not asset["file"].lower().endswith(PICTURES):
            raise ValueError("mid-frame guides use PNG, JPEG or WebP pictures")
        found[frame] = dict(asset, frame=frame)
    return [found[frame] for frame in sorted(found)]


def clean_clip_fields(fields):
    check_fields(fields, CLIP_FIELDS)
    out = dict(fields)
    if "sprite" in out:
        out["sprite"] = sprite.settings(out["sprite"])
    if "style_markers" in out:
        out["style_markers"] = style_timeline.clean_markers(out["style_markers"])
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
    if "motion" in out:
        out["motion"] = clean_motion_field(out["motion"])
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


def clean_motion_field(value):
    """Validate a clip's stored motion transfer recipe (or None to clear it)."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("motion must be an object")
    src = value.get("source")
    if not isinstance(src, dict):
        raise ValueError("motion needs a source take and frame range")
    asset = clean_asset(src, "motion source")
    first, last = src.get("in"), src.get("out")
    if type(first) is not int or type(last) is not int or first < 0 or last <= first:
        raise ValueError("motion source needs integer in and out frames")
    if not 2 * FPS <= last - first <= MOTION_MAX_SECONDS * FPS:
        raise ValueError("motion transfer takes between 2 and %s seconds" % MOTION_MAX_SECONDS)
    character = clean_asset(value.get("character"), "the character image")
    if not character:
        raise ValueError("choose a character image for motion transfer")
    mode = value.get("mode") or "replace"
    if mode not in MOTION_MODES:
        raise ValueError("motion mode must be one of %s" % (", ".join(MOTION_MODES),))
    preset = value.get("preset") or "standard"
    if preset not in MOTION_PRESETS:
        raise ValueError("motion preset must be one of %s" % (", ".join(MOTION_PRESETS),))
    style = value.get("style") or "auto"
    if style not in MOTION_STYLES:
        raise ValueError("character style must be one of %s" % (", ".join(MOTION_STYLES),))
    strength_limits = (0.0, 2.0) if mode in ("replace", "rescene") else (0.1, 2.0)
    strength_default = 0.0 if mode in ("replace", "rescene") else 1.0
    strength = clean_strength(value.get("strength", strength_default), strength_limits, "motion strength")
    expand_raw = value.get("expand_px", 32)
    if isinstance(expand_raw, bool) or not isinstance(expand_raw, (int, float)) or not 0 <= int(expand_raw) <= 128:
        raise ValueError("mask expansion must be between 0 and 128 pixels")
    cleaned = {
        "source": {
            "clip": str(src.get("clip") or ""),
            "take": str(src.get("take") or ""),
            "file": asset["file"],
            "subfolder": asset["subfolder"],
            "in": first,
            "out": last,
            "frames": int(src.get("frames") or last),
        },
        "character": character,
        "mode": mode,
        "style": style,
        "strength": strength,
        "isolate": bool(value.get("isolate", mode != "rescene")),
        "plate": bool(value.get("plate", mode == "replace")) if mode == "replace" else False,
        "expand_px": int(expand_raw),
    }
    if preset != "standard":
        cleaned["preset"] = preset
    if value.get("cast"):
        cleaned["cast"] = str(value["cast"])
    if value.get("swap_lora") is not None:
        cleaned["swap_lora"] = clean_strength(value["swap_lora"], (0.0, 1.5), "character swap LoRA strength")
    if value.get("sharp") or preset == "fast_sharp":
        cleaned["sharp"] = True
    return cleaned


def add_motion_clip(project, clip_id, *, take=None, first=0, last=None, motion=None, action="",
                    name=None, item_id=None, start=None, place=True, replace_item=False):
    """Create a first-class normal clip driven by motion transfer.

    With place=False (default in the dialog), the clip is added to Clips without modifying the timeline.
    With replace_item=True, the selected timeline placement is swapped to the new clip in place (no ripple).
    With place=True and replace_item=False, the clip is inserted on the timeline after the source item.
    """
    source_clip = find_clip(project, clip_id)
    takes = source_clip.get("takes") or []
    source_take = (next((t for t in takes if t["id"] == take), None) if take else active_take(source_clip))
    if not source_take:
        raise ValueError("choose a clip with a rendered or imported take for motion transfer")
    if not isinstance(motion, dict):
        raise ValueError("motion transfer needs a character image or cast member")
    total_frames = int(source_take["frames"])
    in_frame = 0 if first is None else first
    out_frame = total_frames if last is None else last
    if type(in_frame) is not int or type(out_frame) is not int or not 0 <= in_frame < out_frame <= total_frames:
        raise ValueError("the motion range must fall inside the source take")
    if not 2 * FPS <= out_frame - in_frame <= MOTION_MAX_SECONDS * FPS:
        raise ValueError("select between 2 and %s seconds of the source take" % MOTION_MAX_SECONDS)

    cast_id = motion.get("cast")
    cast_member = None
    if cast_id:
        cast_member = find_cast(project, cast_id)
        pic = ({"file": cast_member["file"], "subfolder": cast_member.get("subfolder") or ""}
               if cast_member.get("file")
               else next(({"file": p["file"], "subfolder": p.get("subfolder") or ""}
                          for p in (cast_member.get("pictures") or []) if isinstance(p, dict) and p.get("file")), None))
        if not pic:
            raise ValueError("%s has no picture to use for motion transfer" % cast_member["name"])
        character = pic
    else:
        character = clean_asset(motion.get("character"), "the character image")
        if not character:
            raise ValueError("choose a character image or cast member for motion transfer")

    spec = clean_motion_field({
        **motion,
        "character": character,
        **({"cast": cast_member["id"]} if cast_member else {}),
        "source": {
            "clip": source_clip["id"],
            "take": source_take["id"],
            "file": source_take["file"],
            "subfolder": source_take.get("subfolder") or "",
            "in": in_frame,
            "out": out_frame,
            "frames": total_frames,
        },
    })
    default_name = ("%s · %s" % (cast_member["name"], source_clip["name"])) if cast_member else ("Motion · %s" % source_clip["name"])
    clip_name = str(name or "").strip()[:80] or default_name[:80]
    clip = {
        "id": new_id("c"),
        "name": clip_name,
        "action": str(action or "").strip(),
        "camera": "", "speaker": "", "dialogue": "", "sound": "", "music": "",
        "seconds": round((out_frame - in_frame) / float(FPS), 3),
        "takes": [],
        "active": None,
        "motion": spec,
    }
    if cast_member:
        clip["cast"] = [cast_member["id"]]
    if source_clip.get("bin"):
        clip["bin"] = source_clip["bin"]
    project.setdefault("clips", []).append(clip)

    placed = None
    span = out_frame - in_frame
    if replace_item:
        anchor_item = None
        if item_id:
            anchor_item = next((i for i in project.get("timeline", []) if i["id"] == item_id), None)
        if anchor_item is None:
            anchor_item = next((i for i in ordered(project) if i["clip"] == source_clip["id"]), None)
        if anchor_item is not None:
            anchor_item["clip"] = clip["id"]
            anchor_item["in"] = 0
            anchor_item["out"] = span
            anchor_item.pop("take", None)
            anchor_item.pop("speed", None)
            settle(project)
            placed = anchor_item
    elif place:
        anchor_item = None
        if item_id:
            anchor_item = next((i for i in project.get("timeline", []) if i["id"] == item_id), None)
        if anchor_item is None:
            anchor_item = next((i for i in ordered(project) if i["clip"] == source_clip["id"]), None)
        if start is not None:
            at = max(0, int(start))
            shift_from(project, at, span)
        elif anchor_item is not None:
            at = item_end(anchor_item)
            shift_from(project, at, span)
        else:
            at = scene_end(project)
        placed = {"id": new_id("i"), "clip": clip["id"], "start": at, "in": 0, "out": span}
        project.setdefault("timeline", []).append(placed)
        settle(project)
    return clip, placed


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


def continued_track(source, item, mode, frames, side=None):
    """The slice of the source's track a linked clip holds, so one long performance carries on across clips.

    side is the link's side for the source: the held frames are counted from it, not from where the item is cut,
    which a handover puts a few frames inside them.
    """
    track = source.get("track")
    if not track:
        return None
    if clean_speed(item.get("speed")) != 1:
        raise ValueError("set %s to 100%% speed to continue its track" % source["name"])
    first, last = (side["first"], side["last"]) if side else (item["in"], item["out"])
    if mode == "extend_before":
        start = track["from"] + (first - (frames - HOLD)) / float(FPS)
        if start < -1e-6:
            raise ValueError("the track starts %.2f s too late for this extension. Use fewer seconds" % -start)
    else:
        start = track["from"] + (last - HOLD) / float(FPS)
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
    if "mid_guides" in fields:
        clip["mid_guides"] = clean_mid_guides(clip)
    project.setdefault("clips", []).append(clip)
    item = place_clip(project, clip["id"], start) if place else None
    return clip, item


def update_clip(project, clip_id, fields):
    clip = find_clip(project, clip_id)
    fields = clean_clip_fields(fields)
    check_bin(project, fields)
    check_profile(project, fields)
    before = planned_frames(clip)
    if fields.get("sprite") and (clip.get("link") or clip.get("kind") == "footage"):
        raise ValueError("Sprite Draft needs an independent generated shot")
    if "mid_guides" in fields or (("seconds" in fields or "sprite" in fields) and clip.get("mid_guides")):
        fields["mid_guides"] = clean_mid_guides(dict(clip, **fields))
    for key, value in fields.items():
        if value is None and key in ("first", "last", "track", "bin", "profile", "loras", "sprite", "motion"):
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
    check_fields(fields, ("start", "in", "out", "speed", "transition", "gain", "crop", "fade_in", "fade_out", "gain_points"))
    item = find_item(project, item_id)
    if "crop" in fields:
        crop = clean_crop(fields["crop"])
        if crop:
            item["crop"] = crop
        else:
            item.pop("crop", None)
    if "gain" in fields:
        gain = clean_gain(fields["gain"])
        if gain == 1:
            item.pop("gain", None)
        else:
            item["gain"] = gain
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
    for key in ("fade_in", "fade_out"):
        if key in fields:
            item[key] = sound_frame(fields[key], item_length(item), key)
    if "gain_points" in fields:
        item["gain_points"] = clean_gain_points(fields["gain_points"], take["frames"] if take else last)
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
    if "gain" in item:
        right["gain"] = item["gain"]
    if "crop" in item:
        right["crop"] = dict(item["crop"])
    for key in ("fade_in", "fade_out", "gain_points"):
        if key in item:
            right[key] = copy.deepcopy(item[key])
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


def is_tail(link, key):
    """Does this side of a link hold the end of its source. Otherwise it holds the start."""
    return key == "a" and link["mode"] != "extend_before"


def side_cut(link, key):
    """The cut of its source a side joins, "out" or "in", and the frame that cut sits at while the join is whole.

    A side records the frames that are held: they end at its last, or start at its first. The source is cut a
    handover inside them, and the linked clip plays the rest of them itself.
    """
    side, over = link[key], link_handover(link)
    if is_tail(link, key):
        return "out", side["last"] - over
    if key == "b" and arrives_on_pin(link):
        # Nothing of the next clip is held, so it stays cut where it is: its first frame is the pin.
        return "in", side["first"]
    return "in", side["first"] + over


def side_changed(project, link, key):
    side = link[key]
    item = next((i for i in project.get("timeline", []) if i["id"] == side["item"]), None)
    clip = next((c for c in project.get("clips", []) if c["id"] == side["clip"]), None)
    take = active_take(clip) if clip else None
    if not item or not take:
        return False
    which, frame = side_cut(link, key)
    return take["file"] != side["file"] or item[which] != frame


def touches(project, clip, side):
    """Only a join that still exists can show, so a clip that moved away from its source is left alone."""
    source = next((i for i in project.get("timeline", []) if i["id"] == side["item"]), None)
    if not source:
        return False
    return any(i["clip"] == clip["id"] and (item_end(i) == source["start"] or item_end(source) == i["start"])
               for i in project.get("timeline", []))


def is_stale(project, clip):
    link = clip.get("link")
    return bool(clip.get("takes")) and any(side_changed(project, link, key) and touches(project, clip, link[key])
                                           for key in ("a", "b") if (link or {}).get(key))


def refresh_link(project, clip):
    """A linked clip always renders from what its sources show on the timeline right now.

    A source still cut where the link left it is held from the same frames, of the take it shows now. One that was
    trimmed since is joined again at its new cut, as a new linked clip would be.
    """
    link = clip.get("link")
    for key in ("a", "b"):
        side = (link or {}).get(key)
        item = next((i for i in project.get("timeline", []) if side and i["id"] == side["item"]), None)
        source = next((c for c in project.get("clips", []) if item and c["id"] == item["clip"]), None)
        take = active_take(source) if source else None
        if not take:
            continue
        which, frame = side_cut(link, key)
        if item[which] == frame:
            side.update({"file": take["file"], "subfolder": take.get("subfolder")})
        else:
            join_side(project, link, key, item)


def join_side(project, link, key, item, snap=True):
    """Join one side of a link to its source where the source is cut now. Sets the side, and returns it.

    The frames the linked clip will hold end at the source's cut, or start at it. Two things then move that cut:

    The grid. A take can hold another take's frames as the latents they were sampled in only when they start on a
    multiple of 17 frames (see holds.py). A clip trimmed anywhere else would be held from its video, through the
    decoder and the encoder again. So the end a clip is extended from moves back, and the start a clip is arrived
    at or extended before moves forward, by 16 frames at most. Imported footage has no latents and is left alone,
    and so is a clip too short to give the frames up, or a merge, which has to keep the scene's length.

    The handover. The source then stops a few frames inside the held frames, and the linked clip plays the rest of
    them itself (see HANDOVER). Both files show the same picture there, so nothing is seen twice.

    What follows on the timeline closes up.
    """
    tail, over = is_tail(link, key), link_handover(link)
    snap = snap and find_clip(project, item["clip"]).get("kind") != "footage"
    side = side_of(project, item)
    if key == "b" and arrives_on_pin(link):
        # A bridge that arrives on a pinned frame leaves the next clip as it is cut.
        link[key] = side
        return side
    if tail:
        side["last"] = holds.snap_out(item["out"], HOLD, item["in"]) if snap else item["out"]
        which, cut = "out", side["last"] - over
    else:
        side["first"] = holds.snap_in(item["in"], end_hold(link) if key == "b" else HOLD, item["out"]) if snap else item["in"]
        which, cut = "in", side["first"] + over
    if (cut <= item["in"]) if tail else (cut >= item["out"]):
        raise ValueError("%s is too short on the timeline to carry on from" % find_clip(project, item["clip"])["name"])
    if cut != item[which]:
        end = item_end(item)
        item[which] = cut
        shift_from(project, end, item_end(item) - end, skip=item)
    link[key] = side
    return side


def add_linked(project, item_id, mode, seconds=None, action="", arrive=None):
    """Extend a clip forward or back, bridge it to the next one, or merge the two with a generated crossing.

    arrive is how a bridge reaches the next clip: "short", the default, on a short run of its first frames (see
    end_hold), "pin" on its first frame alone (see arrives_on_pin), or "hold" on as many as of the first clip.
    """
    if mode not in LINK_MODES:
        raise ValueError("mode must be one of %s" % ", ".join(LINK_MODES))
    if arrive not in (None, "") and mode != "bridge":
        raise ValueError("arrive is for a bridge")
    if arrive not in (None, "") + ARRIVALS:
        raise ValueError("arrive must be one of %s" % ", ".join(ARRIVALS))
    item = find_item(project, item_id)
    source = find_clip(project, item["clip"])
    side_of(project, item)        # refuses a source that is not rendered
    if item_length(item) < HOLD:
        raise ValueError("the clip needs at least %s frames on the timeline" % HOLD)
    nxt = following_item(project, item) if mode in ("bridge", "merge") else None
    if mode in ("bridge", "merge"):
        if not nxt or not active_take(find_clip(project, nxt["clip"])):
            raise ValueError("%s needs a rendered clip after this one" % mode)
        need = HOLD if mode == "merge" or arrive == "hold" else 1 if arrive == "pin" else END_HOLD
        if item_length(nxt) < need:
            raise ValueError("the next clip needs at least %s frames on the timeline" % need)
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
        seconds = 2.625

    link = {"mode": "bridge" if mode == "merge" else mode, "hold": HOLD, "handover": HANDOVER}
    if mode == "bridge" and arrive == "pin":
        link["arrive"] = "pin"
    elif mode == "bridge" and arrive != "hold":
        link.update({"arrive": "short", "end_hold": END_HOLD})
    join_side(project, link, "a", item, snap=mode != "merge")
    if nxt:
        join_side(project, link, "b", nxt, snap=mode != "merge")
    seconds = float(seconds if seconds is not None else BRIDGE_SECONDS if link.get("arrive") else 4.25)
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
    # The new frames carry on the source's shot, so they are drawn by the same model with the same LoRAs. Left to
    # the project's model, an extension of a clip on another model changes its look where it joins.
    if source.get("profile"):
        clip["profile"] = source["profile"]
    if source.get("loras"):
        clip["loras"] = copy.deepcopy(source["loras"])
    track = continued_track(source, item, link["mode"], linked_length(link, seconds), link["a"])
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
    else:
        # A merge trimmed the two clips to keep the scene's length. Its handovers trim them a little more, and the
        # new clip is that much longer, so the next clip still ends where it did.
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
    first, last = linked_range(clip.get("link"), take["frames"])
    for item in project.get("timeline", []):
        if item["clip"] == clip_id:
            item["out"] = min(item["out"], last)
            item["in"] = max(first, min(item["in"], item["out"] - 1))
    return clip


def remove_takes(project, clip_id, ids, allow_empty=False):
    """Remove selected versions from this clip. Other clips can reference the same media."""
    clip = find_clip(project, clip_id)
    if not isinstance(ids, list) or not ids or not all(isinstance(i, str) and i for i in ids):
        raise ValueError("takes must be a non-empty list of take ids")
    selected = set(ids)
    takes = clip.get("takes") or []
    unknown = selected.difference(t["id"] for t in takes)
    if unknown:
        raise KeyError("no take %s on clip %s" % (sorted(unknown)[0], clip_id))
    kept = [take for take in takes if take["id"] not in selected]
    if not kept and clip.get("kind") == "footage":
        raise ValueError("keep the last take of imported footage")
    if not kept and allow_empty is not True:
        raise ValueError("set allow_empty to true to remove every take of this shot")
    active = active_take(clip)["id"]
    clip["takes"] = kept
    if not kept:
        clip["active"] = None
        # The next render replaces the removed media beneath the existing trims and splits.
        clip["awaiting_replacement"] = True
    elif active in selected:
        choose_take(project, clip_id, kept[-1]["id"])
    return {"clip": clip_id, "removed": [take["id"] for take in takes if take["id"] in selected],
            "active": clip["active"]}


def accept_take(project, clip_id, take):
    """A finished render becomes the clip's active take. A first render sets its timeline range."""
    clip = find_clip(project, clip_id)
    replacing = clip.pop("awaiting_replacement", False)
    fresh = not clip.get("takes") and not replacing
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
            item["in"] = max(first, min(item["in"], item["out"] - 1))
    settle(project)
    return clip


def clean_strength(value, limits, what, digits=2):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not limits[0] <= value <= limits[1]:
        raise ValueError("%s must be a number from %g to %g" % (what, limits[0], limits[1]))
    return round(float(value), digits)


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


def clean_patches(value):
    """Model patches as an agent or the editor writes them.

    Each is a node id, the values for that node's inputs, and whether it is on. Only the shape
    is checked here: which nodes are installed, and what their inputs allow, is known when a
    render is planned, so a project keeps a patch for a pack this machine does not have.
    """
    if not isinstance(value, list):
        raise ValueError("patches must be a list of {node, inputs}")
    if len(value) > PATCH_LIMIT:
        raise ValueError("a project takes up to %s model patches" % PATCH_LIMIT)
    rows = []
    for row in value:
        if not isinstance(row, dict) or not isinstance(row.get("node"), str) or not row["node"].strip():
            raise ValueError("each patch needs node, a node's id as patches in GET /projects lists it")
        unknown = sorted(set(row) - {"id", "node", "inputs", "on"})
        if unknown:
            raise ValueError("unknown patch fields: %s. Allowed: node, inputs, on" % ", ".join(unknown))
        inputs = {} if row.get("inputs") is None else row["inputs"]
        if not isinstance(inputs, dict) or any(not isinstance(name, str) or not isinstance(v, (str, int, float, bool))
                                               for name, v in inputs.items()):
            raise ValueError("a patch's inputs is {name: value}, each value a number, true or false, or text")
        if (len(row["node"]) > patching.NAME_MOST or len(inputs) > patching.INPUTS_MOST
                or any(len(name) > patching.NAME_MOST or (isinstance(v, str) and len(v) > patching.TEXT_MOST) for name, v in inputs.items())):
            raise ValueError("a patch takes up to %s inputs, names of up to %s characters and text of up to %s"
                             % (patching.INPUTS_MOST, patching.NAME_MOST, patching.TEXT_MOST))
        made = {"id": str(row.get("id") or new_id("p")), "node": row["node"].strip(), "inputs": dict(inputs),
                "on": row.get("on") is not False}
        if any(r["id"] == made["id"] for r in rows):
            raise ValueError("two patches share the id %s" % made["id"])
        rows.append(made)
    return rows


def rest_patches(project):
    """Switch off the model patches of a project that came from somewhere else. Returns the nodes switched off.

    A patch runs a node of another pack with the values the project gives it, and whoever made
    the project chose both, not the person opening it. They stay in the project, and render
    once they are ticked here. A list that is not one a project can hold is dropped.
    """
    if not project.get("patches"):
        return []
    try:
        rows = clean_patches(project["patches"])
    except ValueError:
        del project["patches"]
        return []
    rested = [row["node"] for row in rows if row["on"]]
    for row in rows:
        row["on"] = False
    project["patches"] = rows
    return rested


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


def clean_sizes(value, shape="16:9"):
    """The sizes a project asks for, without the ones that are the usual size anyway. None asks for none."""
    if value is None:
        return {}
    if not isinstance(value, dict) or any(key not in SIZE_CHOICES for key in value):
        raise ValueError("sizes takes draft and final, such as {\"draft\": \"medium\", \"final\": \"large\"}")
    chosen = {}
    for quality, name in value.items():
        if name is None:
            continue
        if isinstance(name, dict):
            chosen[quality] = "%sx%s" % fitted_size(shape, name)
            continue
        if isinstance(name, str) and name not in SIZE_CHOICES[quality]:
            # A size of the project's own, however it was typed: "1024 × 576", "1024X576".
            written = re.sub(r"\s+", "", name).lower().replace("×", "x").replace("*", "x")
            if own_size(written):
                chosen[quality] = written
                continue
        if name not in SIZE_CHOICES[quality]:
            raise ValueError("the %s size must be one of %s, or a size of its own such as \"1024x576\": "
                             "each side a multiple of %s from %s to %s"
                             % (quality, ", ".join(SIZE_CHOICES[quality]), SIDE_STEP, SIDE_MIN, SIDE_MAX))
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
        if key == "patches":
            project["patches"] = clean_patches(value)
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
            chosen = clean_sizes(value, clean_format(fields.get("format", project_format(project))))
            project.pop("sizes", None)
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
        if key == "format":
            value = clean_format(value)
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
    out = []
    for asset in value:
        cleaned = clean_asset(asset, "a picture")
        if isinstance(asset, dict) and asset.get("view") is not None:
            cleaned["view"] = sprite.normalize_view(asset["view"])
        out.append(cleaned)
    return out


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
    if fields.get("view") is not None:
        person["view"] = sprite.normalize_view(fields["view"])
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
        if key == "view":
            if value is None:
                person.pop("view", None)
            else:
                person["view"] = sprite.normalize_view(value)
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
            "text_encoder": graph["2"]["inputs"].get("clip_name") or graph["2"]["inputs"]["adapter"],
            "vae": graph["3"]["inputs"]["vae_name"]}
    if graph["2"]["class_type"] == CONDITIONER_NODE:
        used["text_conditioner"] = graph["2"]["inputs"]["adapter"]
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
        used["profile_lora_strength"] = graph["94"]["inputs"].get("strength", graph["94"]["inputs"].get("strength_model"))
    if "95" in graph:
        used["lora"] = graph["95"]["inputs"]["lora_name"]
        used["lora_strength"] = graph["95"]["inputs"]["strength"]
    extra = [graph[n]["inputs"] for n in sorted((n for n in graph if n.isdigit() and LORA_NODE <= int(n) < PATCH_NODE), key=int)
             if graph[n]["class_type"] == "H3Lora"]
    if extra:
        used["loras"] = [dict({"file": i["lora_name"], "strength": i["strength"]},
                              **({"start_sigma": i["start_sigma"]} if "start_sigma" in i else {})) for i in extra]
    wired = [graph[str(n)] for n in range(PATCH_NODE, PATCH_NODE + PATCH_LIMIT) if str(n) in graph]
    if wired:
        # Each patch with the values its node ran with, the defaults included. The model wire is not a value.
        used["patches"] = [{"node": node["class_type"], "inputs": {name: v for name, v in node["inputs"].items() if not isinstance(v, list)}}
                           for node in wired]
    return used


# Upscaling. A finished final can be enlarged two times by a video upscaler ComfyUI runs natively. The result is a
# new take of the clip, at twice the size, with the same sound. Which upscaler is the user's choice, among the ones
# whose files are installed: they differ a great deal in what they do to a picture and in what they cost.
UPSCALERS = {
    "relayvsr": {
        "title": "RelayVSR",
        "about": "Fast: seconds a clip, about 16 GB of GPU memory. Sharpens the edges that are there; it does not add new texture.",
        "files": {"model": RELAYVSR_MODEL, "vae": RELAYVSR_VAE},
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
# A final made from the clip's active draft: the draft's own latents enlarged and redrawn at the refine size, so the
# shot keeps the take that was chosen, and only the refine pass runs.
FINISH_QUALITY = "finish"
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
    return [*rows, vosr2.status(available)]


def pick_upscaler(name, available=None):
    """The upscaler to use: the one asked for, or the first that is installed. Its files come back with it."""
    if name in (None, ""):
        name = next((row["id"] for row in upscaler_status(available) if row["installed"] and not row.get("experimental")), next(iter(UPSCALERS)))
    if name == "vosr2":
        integration = vosr2.status(available)
        if not integration["installed"]:
            raise ValueError(integration["setup"] + " " + vosr2.SOURCE)
        return name, {"model": "VOSR2"}
    if name not in UPSCALERS:
        raise ValueError("upscaler must be one of %s" % ", ".join((*UPSCALERS, "vosr2")))
    files = upscaler_files(name, available)
    missing = [UPSCALERS[name]["files"][kind] for kind in ("model", "vae") if not files[kind]]
    if missing:
        raise ValueError("Upscaling with %s needs %s in ComfyUI's models folder (diffusion_models and vae), from %s on Hugging Face"
                         % (UPSCALERS[name]["title"], " and ".join(missing), UPSCALERS[name]["source"]))
    return name, files


def upscale_graph(folder, clip, take, source, name, files, detail=UPSCALE_DETAIL):
    """The graph that enlarges one take of a clip two times with the named upscaler and saves it as another take."""
    file = "%s/%s" % (source["subfolder"], source["file"]) if source.get("subfolder") else source["file"]
    if name == "vosr2":
        return {"31": {"class_type": "VOSR2ModelLoader", "inputs": {"model": files["model"], "dtype": "default"}},
                "19": {"class_type": "H3VOSR2Video", "inputs": {"model": ["31", 0], "file": file,
                       "filename_prefix": "%s/%s/clips/%s_%s" % (ROOT, folder, clip["id"], take)}}}
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
    if upscaler != "vosr2" and int(source.get("frames") or 0) > UPSCALE_MOST:
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
    if name != "vosr2" and UPSCALERS[name]["detail"]:
        job["detail"] = detail
    if name == "vosr2":
        job["models"]["video_upscaler_seed"] = vosr2.SETTINGS["seed"]
    if clip.get("kind") == "footage" and source.get("source"):
        job["source"] = source["source"]
    return job, upscale_graph(folder, clip, take, source, name, files, detail)


def prompt_edit(clip, edit, available=None, detect_style=None):
    if not isinstance(edit, dict):
        raise ValueError("send an edit with a source take, in and out frames, and a prompt")
    source = next((take for take in clip.get("takes", []) if take["id"] == edit.get("take")), None)
    if not source:
        raise ValueError("choose an existing rendered take to edit")
    first, last = edit.get("in"), edit.get("out")
    if type(first) is not int or type(last) is not int or not 0 <= first < last <= source["frames"]:
        raise ValueError("the edit range must fall inside the source take")
    motion = edit.get("motion")
    max_seconds = MOTION_MAX_SECONDS if motion is not None else 15
    if not 2 * FPS <= last - first <= max_seconds * FPS:
        raise ValueError("select between 2 and %s seconds of the source take" % max_seconds)
    prompt = edit.get("prompt", "")
    if not isinstance(prompt, str) or (motion is None and not prompt.strip()) or len(prompt) > 12000:
        raise ValueError("write an edit prompt of up to 12,000 characters")
    picture = clean_asset(edit.get("first"), "the replacement first frame")
    info = {"take": source["id"], "in": first, "out": last, "prompt": prompt.strip(),
            "file": source["file"], "subfolder": source.get("subfolder", ""), "frames": source["frames"]}
    if picture:
        info["first"] = picture
    if motion is not None:
        if not isinstance(motion, dict) or picture:
            raise ValueError("motion transfer takes a character image and a motion strength")
        info["motion"] = resolve_motion_runtime(motion, available, detect_style=detect_style)
    return source, info


def _edit_latent_frames(edit):
    return 5 + holds.CYCLE * math.ceil((int(edit["out"]) - int(edit["in"]) - 5) / holds.CYCLE)


def _edit_slice_window(rel_in, rel_out, edit_len, draft_latent_frames):
    """On-grid latent window covering [rel_in, rel_out) inside a prompt-edit draft latent,
    keeping at least 2*FPS real source frames inside [0, edit_len) when edit_len >= 2*FPS."""
    min_ref = min(2 * FPS, edit_len)
    grid_in, grid_len = holds.slice_window(rel_in, rel_out, draft_latent_frames, min_frames=min_ref)
    if min(edit_len, grid_in + grid_len) - grid_in < min_ref:
        grid_in = max(0, ((edit_len - min_ref) // holds.CYCLE) * holds.CYCLE)
        need = max(rel_out - grid_in, min_ref)
        grid_len = 5 + holds.CYCLE * max(0, math.ceil((need - 5) / holds.CYCLE))
        if grid_in + grid_len > draft_latent_frames:
            grid_len = draft_latent_frames - grid_in
    return grid_in, grid_len


def _fallback_edit_window(target_in, target_out, edit_in, edit_out):
    """Window inside [edit_in, edit_out) covering [target_in, target_out) with at least 2*FPS frames."""
    min_ref = min(2 * FPS, edit_out - edit_in)
    if target_out - target_in >= min_ref:
        return target_in, target_out
    extra = min_ref - (target_out - target_in)
    win_in = max(edit_in, target_in - extra // 2)
    win_out = min(edit_out, win_in + min_ref)
    win_in = max(edit_in, win_out - min_ref)
    return win_in, win_out


def _slice_style_markers(markers, start, length, default_style=""):
    """Shift a clip's style markers to cover a sliced sub-window [start, start + length)."""
    rows = style_timeline.schedule(markers, start + length, default_style)
    if not rows:
        return []
    active = rows[0]
    for row in rows:
        if row["frame"] <= start:
            active = row
    kept = [{"frame": 0, "style": active["style"], "name": active["name"]}]
    for row in rows:
        if start < row["frame"] < start + length:
            kept.append({"frame": row["frame"] - start, "style": row["style"], "name": row["name"]})
    if len(kept) == 1 and kept[0]["style"] == default_style:
        return []
    return kept


def _clean_part(part, total):
    if part is None:
        return 0, total, False
    if not isinstance(part, dict):
        raise ValueError("part must be an object with integer in and out frame numbers")
    first, last = part.get("in"), part.get("out")
    if type(first) is not int or type(last) is not int or not 0 <= first < last <= total:
        raise ValueError("the part to finish must fall inside the take (0..%s)" % total)
    return first, last, not (first == 0 and last == total)


def _finish_fallback_reason(project, clip, base_draft, available, has_latents, check_grid=False):
    """Reason draft latents cannot be reused for a partial/edit finish, or None when they can."""
    if base_draft is None:
        return "The original draft take is no longer in this clip, so this part is rendering fresh at Final quality."
    if size_name(project, "final") == SHARP:
        return "Sharp final size uses a video upscaler instead of refining draft latents, so this part is rendering fresh."
    models = project_models(project, available, clip)
    if not (models.get("latent_upscaler") and installed(models["latent_upscaler"], available)):
        return "The latent upscaler is not installed, so this part is rendering fresh at Final quality."
    if has_latents is not None and not has_latents(base_draft):
        return "This draft has no saved latents, so this part is rendering fresh at Final quality."
    if check_grid and not holds.legal_length(int(base_draft.get("frames") or 0)):
        return "This draft's length is not on the H3 latent grid, so this part is rendering fresh at Final quality."
    return None


def _plan_edit_finish(project, folder, clip, base_draft, composite_source, existing_final_ranges,
                      req_in, req_out, seed, available, patches, has_latents):
    """Finish all or part of a prompt-edit / motion-transfer draft, reusing its latents when possible."""
    clip_id = clip["id"]
    total = int(composite_source.get("frames") or 0)
    edit_meta = copy.deepcopy((base_draft or composite_source)["edit"])
    edit_in, edit_out = int(edit_meta["in"]), int(edit_meta["out"])
    target_in, target_out = max(req_in, edit_in), min(req_out, edit_out)
    if target_in >= target_out:
        raise ValueError("the selected part is outside the edited draft range")
    if existing_final_ranges and range_covered(existing_final_ranges, target_in, target_out):
        raise ValueError("that part is already at final quality")
    orig_source = next((t for t in clip.get("takes", []) if t["id"] == edit_meta.get("take")), None)
    if not orig_source:
        raise ValueError("the source take for this edit is no longer in the clip")

    work_project = project
    work_clip = {"id": clip_id, "name": clip["name"], "profile": clip.get("profile"), "loras": clip.get("loras"),
                 "action": edit_meta.get("prompt", ""), "first": edit_meta.get("first")}
    motion_info = None
    if edit_meta.get("motion"):
        settings = motion_status(available)
        if not settings["available"]:
            raise ValueError("motion transfer needs " + ", ".join(settings["missing"]))
        motion_info = {k: settings[k] for k in ("pose", "detector", "control", "profile")}
        motion_info.update(character=copy.deepcopy(edit_meta["motion"]["character"]),
                           strength=clean_strength(edit_meta["motion"].get("strength", 1), (0.1, 2.0), "motion strength"))
        work_clip["profile"] = motion_info["profile"]
        work_project = dict(project, models={k: v for k, v in (project.get("models") or {}).items()
                                             if k not in ("unet", "unet_text")})
        work_clip["action"] = work_clip["action"] or "A full-body performance."

    if seed in (None, ""):
        seed = (base_draft or composite_source).get("seed")
    if seed in (None, ""):
        seed = random.randrange(2 ** 31)

    fallback = _finish_fallback_reason(work_project, work_clip, base_draft, available, has_latents)
    take = new_id("t")
    if fallback is None:
        draft_latent_frames = _edit_latent_frames(edit_meta)
        grid_in, grid_len = _edit_slice_window(target_in - edit_in, target_out - edit_in,
                                               edit_out - edit_in, draft_latent_frames)
        ref_first = edit_in + grid_in
        ref_last = min(edit_out, edit_in + grid_in + grid_len)
        render_edit = {"take": orig_source["id"], "in": ref_first, "out": ref_last,
                       "prompt": edit_meta.get("prompt", ""), "file": orig_source["file"],
                       "subfolder": orig_source.get("subfolder", ""), "frames": orig_source["frames"]}
        if grid_in == 0 and edit_meta.get("first"):
            render_edit["first"] = copy.deepcopy(edit_meta["first"])
        else:
            work_clip["first"] = None
        if motion_info:
            render_edit["motion"] = motion_info
        graph = render_graph(work_project, folder, work_clip, take, "final", int(seed), available, patches,
                             finish=base_draft, edit=render_edit, finish_window=(grid_in, grid_len))
        trim_start = target_in - ref_first
    else:
        win_in, win_out = _fallback_edit_window(target_in, target_out, edit_in, edit_out)
        render_edit = {"take": orig_source["id"], "in": win_in, "out": win_out,
                       "prompt": edit_meta.get("prompt", ""), "file": orig_source["file"],
                       "subfolder": orig_source.get("subfolder", ""), "frames": orig_source["frames"]}
        if win_in == edit_in and edit_meta.get("first"):
            render_edit["first"] = copy.deepcopy(edit_meta["first"])
        else:
            work_clip["first"] = None
        if motion_info:
            render_edit["motion"] = motion_info
        graph = render_graph(work_project, folder, work_clip, take, "final", int(seed), available, patches,
                             finish=None, edit=render_edit)
        trim_start = target_in - win_in

    outer = []
    if edit_in > 0:
        outer.append([0, edit_in])
    if edit_out < total:
        outer.append([edit_out, total])
    merged = merge_ranges([*outer, *existing_final_ranges, [target_in, target_out]])
    job = {"clip": clip_id, "take": take, "quality": "final", "seed": int(seed),
           "saved": {k: copy.deepcopy(clip.get(k)) for k in SAVED_FIELDS},
           "models": dict(render_models(graph), profile=render_profile(work_project, available, work_clip, "final")[0],
                          profile_title=render_profile(work_project, available, work_clip, "final")[1]["title"],
                          steps=sigma_steps(graph["12"]), sampler=graph["11"]["inputs"]["sampler_name"]),
           "finished": (base_draft or composite_source)["id"],
           "edit": edit_meta,
           "size": list(take_size(project, orig_source)),
           "part": {"in": target_in, "out": target_out},
           "splice": {"file": composite_source["file"], "subfolder": composite_source.get("subfolder", ""),
                      "frames": total, "in": target_in, "out": target_out, "trim_start": trim_start, "is_edit": True}}
    if "36" in graph:
        job["upscaler"], job["detail"] = SHARP_UPSCALER, SHARP_DETAIL
        job["models"]["video_upscaler"] = graph["31"]["inputs"]["unet_name"]
    if merged != [[0, total]]:
        job["final_ranges"] = merged
    if fallback:
        job["finish_fallback"] = fallback
    if orig_source.get("source"):
        job["source"] = orig_source["source"]
    return job, graph


def _plan_partial_finish(project, folder, clip, base_draft, composite_source, existing_final_ranges,
                         req_in, req_out, seed, available, patches, has_latents):
    """Finish part of a normal generated draft, slicing and refining its saved latents when possible."""
    clip_id = clip["id"]
    total = int(composite_source.get("frames") or 0)
    if (base_draft or composite_source).get("sprite"):
        raise ValueError("Sprite Draft takes are finished whole")
    if seed in (None, ""):
        seed = (base_draft or composite_source).get("seed")
    if seed in (None, ""):
        seed = clip.get("seed")
    if seed in (None, ""):
        seed = random.randrange(2 ** 31)

    fallback = _finish_fallback_reason(project, clip, base_draft, available, has_latents, check_grid=True)
    take = new_id("t")
    if fallback is None:
        draft_frames = int(base_draft["frames"])
        grid_in, grid_len = holds.slice_window(req_in, req_out, draft_frames, min_frames=min(39, draft_frames))
        slice_clip = copy.deepcopy(clip)
        slice_clip.pop("link", None)
        if grid_in > 0:
            slice_clip.pop("first", None)
        if grid_in + grid_len < draft_frames:
            slice_clip.pop("last", None)
        if slice_clip.get("mid_guides"):
            slice_clip["mid_guides"] = [dict(g, frame=g["frame"] - grid_in) for g in slice_clip["mid_guides"]
                                        if grid_in < g["frame"] < grid_in + grid_len - 1]
        if slice_clip.get("style_markers"):
            slice_clip["style_markers"] = _slice_style_markers(slice_clip["style_markers"], grid_in, grid_len,
                                                               project.get("style", ""))
        if slice_clip.get("track"):
            slice_clip["track"] = dict(slice_clip["track"], **{"from": float(slice_clip["track"].get("from") or 0) + grid_in / float(FPS)})
        graph = render_graph(project, folder, slice_clip, take, "final", int(seed), available, patches,
                             finish=base_draft, finish_window=(grid_in, grid_len))
        trim_start = req_in - grid_in
        profile_clip = slice_clip
    else:
        work_clip = copy.deepcopy(clip)
        refresh_link(project, work_clip)
        if not work_clip.get("link") and snap_frames(work_clip.get("seconds", 5)) < req_out:
            work_clip["seconds"] = total / float(FPS)
        graph = render_graph(project, folder, work_clip, take, "final", int(seed), available, patches)
        trim_start = req_in
        profile_clip = work_clip

    merged = merge_ranges([*existing_final_ranges, [req_in, req_out]])
    job = {"clip": clip_id, "take": take, "quality": "final", "seed": int(seed),
           "saved": {k: copy.deepcopy(clip.get(k)) for k in SAVED_FIELDS},
           "models": dict(render_models(graph), profile=render_profile(project, available, profile_clip, "final")[0],
                          profile_title=render_profile(project, available, profile_clip, "final")[1]["title"],
                          steps=sigma_steps(graph["12"]), sampler=graph["11"]["inputs"]["sampler_name"]),
           "finished": (base_draft or composite_source)["id"],
           "part": {"in": req_in, "out": req_out},
           "splice": {"file": composite_source["file"], "subfolder": composite_source.get("subfolder", ""),
                      "frames": total, "in": req_in, "out": req_out, "trim_start": trim_start, "is_edit": False}}
    if "91" in graph:
        job["size"] = [graph["80"]["inputs"]["width"], graph["80"]["inputs"]["height"]]
    elif "36" in graph:
        job["size"] = [graph["36"]["inputs"]["width"], graph["36"]["inputs"]["height"]]
        job["upscaler"], job["detail"] = SHARP_UPSCALER, SHARP_DETAIL
        job["models"]["video_upscaler"] = graph["31"]["inputs"]["unet_name"]
    elif tuple(render_size(project, "final")[:2]) != tuple(frame_size(project, "final")):
        job["size"] = [graph["10"]["inputs"]["width"], graph["10"]["inputs"]["height"]]
    if merged != [[0, total]]:
        job["final_ranges"] = merged
    if fallback:
        job["finish_fallback"] = fallback
    return job, graph


def plan_render(project, folder, clip_id, quality, seed=None, available=None, detail=None, upscaler=None,
                patches=None, edit=None, part=None, has_latents=None, detect_style=None, check_cache=None):
    """The graph for a new take, and the job that will turn its file into a take. detail and upscaler are for an upscale only.

    patches is the installed patch nodes by id, as project_patches takes them.
    part is an optional {in, out} frame range for finishing only part of a draft.
    has_latents is an optional callable(take) -> bool used to detect missing draft latents and fall back.
    """
    if part is not None and quality != FINISH_QUALITY:
        raise ValueError("part is only supported when finishing a draft")
    if edit is not None and quality not in QUALITIES:
        raise ValueError("choose draft or final for a prompt edit")
    if quality == UPSCALE_QUALITY:
        return plan_upscale(project, folder, clip_id, available, detail, upscaler)
    clip = find_clip(project, clip_id)
    original = clip
    motion_audio = None
    if edit is not None:
        source, edit = prompt_edit(clip, edit, available, detect_style=detect_style)
        clip = {"id": clip_id, "name": clip["name"], "profile": clip.get("profile"), "loras": clip.get("loras"),
                "action": edit["prompt"], "first": edit.get("first")}
        if edit.get("motion"):
            motion = edit["motion"]
            if (motion.get("mode") or "choreography") == "choreography":
                clip["profile"] = motion["profile"]
                # Keep text encoder and VAE choices while using the known-compatible standard H3 checkpoint.
                project = dict(project, models={key: value for key, value in (project.get("models") or {}).items()
                                                if key not in ("unet", "unet_text")})
                clip["action"] = clip["action"] or "A full-body performance."
            else:
                clip["action"] = clip["action"] or "Preserve the source performance and camera."
    elif clip.get("motion"):
        stored = clean_motion_field(clip["motion"])
        resolved = resolve_motion_runtime(stored, available, detect_style=detect_style)
        resolved["source"] = stored["source"]
        if stored.get("cast"):
            resolved["cast"] = stored["cast"]
        clip = dict(clip, motion=resolved)
        motion_audio = {
            "file": stored["source"]["file"],
            "subfolder": stored["source"].get("subfolder") or "",
            "in": stored["source"]["in"],
            "out": stored["source"]["out"],
        }
        if (resolved.get("mode") or "choreography") == "choreography" and not clip.get("profile"):
            clip["profile"] = resolved["profile"]
            project = dict(project, models={key: value for key, value in (project.get("models") or {}).items()
                                            if key not in ("unet", "unet_text")})
    finish = None
    if quality == FINISH_QUALITY:
        if clip.get("motion"):
            raise ValueError("use Render final to render a motion transfer clip at final quality")
        if any(j["clip"] == clip_id for j in project.get("pending", [])):
            raise ValueError("%s is already rendering" % clip["name"])
        current = active_take(clip)
        if current and current.get("quality") == "draft":
            base_draft, composite_source, existing_final_ranges = current, current, []
        elif current and current.get("quality") == "final" and current.get("final_ranges"):
            composite_source = current
            existing_final_ranges = merge_ranges(current.get("final_ranges"))
            base_draft = next((t for t in clip.get("takes", [])
                               if t["id"] == current.get("finished") and t.get("quality") == "draft"), None)
        else:
            raise ValueError("%s has no draft to finish. Render a draft, or choose one of its draft takes" % clip["name"])
        total = int(composite_source.get("frames") or 0)
        req_in, req_out, is_partial = _clean_part(part, total)
        if existing_final_ranges and range_covered(existing_final_ranges, req_in, req_out):
            raise ValueError("that part is already at final quality")
        if (base_draft or composite_source).get("edit"):
            return _plan_edit_finish(project, folder, clip, base_draft, composite_source, existing_final_ranges,
                                     req_in, req_out, seed, available, patches, has_latents)
        if is_partial or existing_final_ranges:
            if clip.get("kind") == "footage":
                raise ValueError("%s is imported footage. Extend or bridge it to generate from it" % clip["name"])
            return _plan_partial_finish(project, folder, clip, base_draft, composite_source, existing_final_ranges,
                                        req_in, req_out, seed, available, patches, has_latents)
        finish = base_draft
        if has_latents is not None and not has_latents(finish):
            raise ValueError("This draft has no saved latents. Render a new draft to finish, or render a new final.")
        # The final keeps the draft's seed: the refine pass draws its noise from it.
        quality, seed = "final", finish.get("seed")
        if finish.get("sprite"):
            clip = dict(clip, **(finish.get("saved") or {}))
            clip["profile"] = finish.get("models", {}).get("profile") or clip.get("profile")
        elif clip.get("sprite"):
            raise ValueError("Choose a Sprite Draft take to finish with sprite settings")
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
    graph = render_graph(project, folder, clip, take, quality, int(seed), available, patches, finish, edit,
                         check_cache=check_cache)
    saved = {k: copy.deepcopy(clip.get(k)) for k in SAVED_FIELDS}
    if original.get("motion"):
        saved["motion"] = copy.deepcopy(original["motion"])
    job = {"clip": clip_id, "take": take, "quality": quality, "seed": int(seed),
           "saved": saved,
           "models": dict(render_models(graph), profile=render_profile(project, available, clip, quality)[0],
                          profile_title=render_profile(project, available, clip, quality)[1]["title"],
                          steps=sigma_steps(graph["12"]),
                          sampler=(graph["11"]["inputs"].get("sampler_name")
                                   or graph["11"]["inputs"].get("sampler")
                                   or graph["11"]["class_type"]))}
    if "91" in graph:
        job["size"] = [graph["80"]["inputs"]["width"], graph["80"]["inputs"]["height"]]
    elif "36" in graph:
        # A sharp final: the take records the upscaler that brought it to this size.
        job["size"] = [graph["36"]["inputs"]["width"], graph["36"]["inputs"]["height"]]
        job["upscaler"] = SHARP_UPSCALER
        job["detail"] = graph["35"]["inputs"].get("detail", SHARP_DETAIL)
        job["models"]["video_upscaler"] = graph["31"]["inputs"]["unet_name"]
    elif clip.get("sprite") or clip.get("motion") or tuple(render_size(project, quality)[:2]) != tuple(frame_size(project, quality)):
        # A size the project chose. A take that records none is at the usual size for its format.
        job["size"] = [graph["10"]["inputs"]["width"], graph["10"]["inputs"]["height"]]
    if finish:
        # The draft this final was finished from.
        job["finished"] = finish["id"]
    if clip.get("track"):
        # Once rendered, the take's sound is replaced by this exact slice of the track.
        track_frames = (motion_audio["out"] - motion_audio["in"]) if motion_audio else graph["10"]["inputs"]["length"]
        job["track"] = dict(clip["track"], seconds=track_frames / float(FPS))
    if motion_audio:
        job["motion_audio"] = motion_audio
    if edit:
        job["edit"] = edit
        if not ("36" in graph and edit["in"] == 0 and edit["out"] == source["frames"]):
            job["size"] = list(take_size(project, source))
        job["saved"] = {k: copy.deepcopy(original.get(k)) for k in SAVED_FIELDS}
        if source.get("source"):
            job["source"] = source["source"]
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
    for key in ("models", "upscale", "size", "upscaled", "upscaler", "detail", "finished", "source", "final_ranges"):
        if job.get(key):
            take[key] = job[key]
    if info.get("size") and job.get("splice"):
        take["size"] = list(info["size"])
    if info.get("sprite"):
        take["sprite"] = info["sprite"]
    if info.get("strip"):
        take["strip"] = info["strip"]
    if job.get("edit"):
        take["edit"] = {key: value for key, value in job["edit"].items() if key not in ("file", "subfolder", "frames")}
        if not job.get("finished"):
            clip = find_clip(project, job["clip"])
            clip.setdefault("takes", []).append(take)
            clip.pop("error", None)
            return clip
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


def merge_editor_save(base, mine, current):
    """Only explicit operations change an existing clip's take list. A newly placed copy keeps its own take."""
    merged = merge(base, mine, current)
    if "audio_tracks" in merged:
        merged["audio_tracks"] = clean_audio_tracks(merged["audio_tracks"])
    existing = {clip["id"]: clip for clip in current.get("clips", [])}
    for clip in merged.get("clips", []):
        if clip.get("sprite") is not None:
            if clip.get("link") or clip.get("kind") == "footage":
                raise ValueError("Sprite Draft needs an independent generated shot")
            clip["sprite"] = sprite.settings(clip["sprite"])
        if clip.get("motion") is not None:
            if clip.get("kind") == "footage":
                raise ValueError("motion transfer belongs to a generated clip")
            clip["motion"] = clean_motion_field(clip["motion"])
        if "mid_guides" in clip:
            clip["mid_guides"] = clean_mid_guides(clip)
        stored = existing.get(clip["id"])
        if stored is None:
            continue
        ids = {take["id"] for take in stored.get("takes", [])}
        # Keep client-added display metadata such as filmstrips on surviving takes.
        clip["takes"] = [take for take in clip.get("takes", []) if take["id"] in ids]
        if stored.get("awaiting_replacement"):
            clip["awaiting_replacement"] = True
        else:
            clip.pop("awaiting_replacement", None)
        if not clip["takes"]:
            clip["active"] = None
        if clip["takes"] and not any(t["id"] == clip.get("active") for t in clip["takes"]):
            choose_take(merged, clip["id"], active_take(stored)["id"])
    return merged


def restore_revision(current, old):
    """Restore the historical order of takes while retaining any rendered since that revision."""
    restored = merge(current, old, current)
    earlier_clips = {clip["id"]: clip for clip in old.get("clips", [])}
    for clip in restored.get("clips", []):
        earlier = earlier_clips.get(clip["id"])
        if earlier and clip.get("takes"):
            clip.pop("awaiting_replacement", None)
            order = {take["id"]: i for i, take in enumerate(earlier.get("takes", []))}
            clip["takes"].sort(key=lambda take: order.get(take["id"], len(order)))
    return restored


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
        if clip.get("style_markers"):
            entry["style_markers"] = clip["style_markers"]
        if clip.get("sprite"):
            entry["sprite"] = clip["sprite"]
            entry["seconds"] = (clip["sprite"].get("frames", sprite.FRAMES) if isinstance(clip["sprite"], dict) else sprite.FRAMES) / FPS
        if clip.get("motion"):
            entry["motion"] = {"mode": clip["motion"].get("mode"), "style": clip["motion"].get("style"),
                               "source": clip["motion"].get("source", {}).get("clip")}
        if take and take.get("sprite"):
            entry["sprite_sheet"] = take["sprite"]
        if clip.get("mid_guides"):
            entry["mid_guides"] = clip["mid_guides"]
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
        if clean_gain(item.get("gain")) != 1:
            entry["gain"] = item["gain"]
        entry.update({key: item[key] for key in ("fade_in", "fade_out", "gain_points") if item.get(key)})
        if item.get("crop"):
            entry["crop"] = item["crop"]
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
               "offset": s.get("offset", 0.0), "gain": s.get("gain", 1.0),
               **{key: s[key] for key in ("fade_in", "fade_out", "gain_points") if s.get(key)}}
              for s in sorted(project.get("sounds") or [], key=lambda s: (s.get("lane", 0), s.get("start", 0)))]
    return {
        "sounds": sounds,
        "audio_tracks": project.get("audio_tracks") or {},
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
                                name=size_name(project, quality))
                  for quality in SIZE_CHOICES},
        "checkpoints": [{"id": name, "title": profile["title"], "unet": profile["unet"][0], "steps": profile["steps"],
                         "sampler": profile["sampler"], "scheduler": profile.get("scheduler") or "simple"}
                        for name, profile in profiles.items() if profile.get("custom")],
        "loras": [{"file": r["file"], "strength": r.get("strength", 1.0), "on": r.get("on") is not False}
                  for r in project.get("loras") or [] if isinstance(r, dict) and r.get("file")],
        "patches": [{"node": r["node"], "inputs": r.get("inputs") or {}, "on": r.get("on") is not False}
                    for r in project.get("patches") or [] if isinstance(r, dict) and isinstance(r.get("node"), str)],
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
        clip, item = add_linked(project, item_id, kind, op.get("seconds"), op.get("action") or "", op.get("arrive"))
        if op.get("fields"):
            update_clip(project, clip["id"], op["fields"])
        result = {"clip": clip["id"], "item": item["id"]}
    elif kind == "add_motion_transfer":
        motion_payload = dict(op.get("motion") or {})
        if motion_payload.get("cast"):
            motion_payload["cast"] = _resolve(motion_payload["cast"], names, "cast")
        clip, item = add_motion_clip(
            project, clip_id, take=op.get("take"), first=op.get("in", 0), last=op.get("out"),
            motion=motion_payload, action=op.get("action") or "", name=op.get("name"),
            item_id=item_id, start=op.get("start"), place=op.get("place", True),
            replace_item=bool(op.get("replace_item", False)),
        )
        if op.get("fields"):
            update_clip(project, clip["id"], op["fields"])
        result = {"clip": clip["id"], "item": item["id"] if item else None}
    elif kind == "choose_take":
        choose_take(project, clip_id, op["take"])
        result = {"clip": clip_id}
    elif kind == "remove_takes":
        result = remove_takes(project, clip_id, op.get("takes"), op.get("allow_empty"))
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
    elif kind == "set_audio_track":
        lane = sound_frame(op.get("lane"), SOUND_LANES, "audio track")
        key = str(lane)
        tracks = dict(project.get("audio_tracks") or {})
        updates = op.get("fields") or {}
        check_fields(updates, ("gain", "gain_points"))
        tracks[key] = {**tracks.get(key, {}), **updates}
        project["audio_tracks"] = clean_audio_tracks(tracks)
        result = {"lane": lane}
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


def plan_prop_render(project, folder, spec, seed=None, available=None):
    """Plan a 4+ MP MiniMax-H3-Image render for a static prop, paired state variant, or turnaround plate."""
    if not isinstance(spec, dict):
        raise ValueError("prop spec must be an object")
    kind = spec.get("kind", "prop")
    if kind not in ("prop", "state_variant", "turnaround"):
        raise ValueError("kind must be prop, state_variant, or turnaround")
    name = slug(spec.get("name") or kind)
    width, height = h3image.resolve_size(spec.get("size", "square"))
    seed = random.randint(1, 2**31 - 1) if seed is None else int(seed)
    style_ref = clean_asset(spec.get("style_ref"), "style_ref") if spec.get("style_ref") else None
    base_prop = clean_asset(spec.get("base_prop"), "base_prop") if spec.get("base_prop") else None
    # When a style_ref plate is supplied and no explicit style override is given, lock to the plate
    # rather than inheriting a video project's live-action look.
    style = spec["style"] if "style" in spec else ("" if style_ref else project.get("style", ""))
    camera = spec.get("camera", "rpg")
    bg_model = None if spec.get("cutout") == "edge" else sprite.matte_model(available)
    prefix = "%s/%s/props/%s" % (ROOT, folder, name)
    if kind == "turnaround":
        if not style_ref:
            raise ValueError("turnaround needs style_ref (the front reference plate)")
        graph = h3image.turnaround_graph(
            spec.get("view", "back"), front_ref=style_ref, side_ref=base_prop,
            description=spec.get("description", ""), style=style, camera=camera,
            proportions=spec.get("proportions", "adult"), holding=spec.get("holding", ""),
            width=width, height=height, seed=seed, filename_prefix=prefix, bg_model=bg_model,
            available=available,
        )
    elif kind == "state_variant":
        if not base_prop:
            raise ValueError("state_variant needs base_prop")
        prompt_text = h3image.state_variant_prompt(
            spec.get("description", ""), style=style, style_ref=style_ref is not None,
        )
        graph = h3image.prop_graph(
            prompt_text, width=width, height=height, seed=seed, filename_prefix=prefix,
            style_ref=style_ref, base_prop=base_prop, bg_model=bg_model, available=available,
        )
    else:
        prompt_text = h3image.prop_prompt(
            spec.get("description", ""), style=style, camera=camera,
            style_ref=style_ref is not None, top_canopy=bool(spec.get("top_canopy")),
        )
        graph = h3image.prop_graph(
            prompt_text, width=width, height=height, seed=seed, filename_prefix=prefix,
            style_ref=style_ref, bg_model=bg_model, available=available,
        )
    output_size = list(sprite.validate_tile_size(spec["output_size"])) if spec.get("output_size") else None
    pivot = spec.get("pivot")
    if pivot is not None:
        if not isinstance(pivot, (list, tuple)) or len(pivot) != 2 or any(type(v) is not int for v in pivot):
            raise ValueError("pivot must be [x, y] integers")
        pivot = [int(pivot[0]), int(pivot[1])]
    job = {
        "kind": kind, "name": name, "seed": seed, "native_size": [width, height],
        "output_size": output_size, "pivot": pivot, "prefix": prefix,
        "cutout": "foreground model" if bg_model else "background key",
    }
    return job, graph
