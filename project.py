"""An editing project: clips with rendered takes, and a timeline that only points at them.

Moving or trimming on the timeline never touches a render. No Comfy import.
"""

import re

try:
    from .prompting import compose_prompt
except ImportError:
    from prompting import compose_prompt

FPS = 24
ROOT = "h3take_projects"
QUALITIES = {
    "draft": {"width": 640, "height": 352, "steps": 4},
    "final": {"width": 1344, "height": 768, "steps": 8},
}
MODELS = {
    "unet": "fasth3_omniref_qad_nvfp4_comfy.safetensors",
    "clip": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "vae": "minimax_h3_video_vae_fp16.safetensors",
    "audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
}


def slug(name):
    """A safe folder name. Refuses anything that would leave the projects folder."""
    cleaned = re.sub(r"[^A-Za-z0-9 _-]", "", str(name or "")).strip().replace(" ", "_")
    if not cleaned:
        raise ValueError("a project needs a name with letters or numbers")
    return cleaned[:64]


def new_project(name):
    return {
        "name": str(name),
        "style": "Live-action, cinematic",
        "models": dict(MODELS),
        "sampler": "euler",
        "shift": 12.0,
        "clips": [],
        "timeline": [],
    }


def snap_frames(seconds):
    """Nearest legal H3 length, at least about a second."""
    frames = float(seconds) * FPS
    return 5 + 17 * max(1, int(round((frames - 5) / 17.0)))


def clip_prompt(project, clip):
    return compose_prompt(
        project.get("style", ""),
        clip.get("action", ""),
        camera=clip.get("camera", ""),
        speaker=clip.get("speaker", ""),
        dialogue=clip.get("dialogue", ""),
        sound=clip.get("sound", ""),
        music=clip.get("music", ""),
    )


def render_graph(project, folder, clip, take, quality, seed):
    """One clip as a ComfyUI graph, saved as its own file in the project."""
    if not str(clip.get("action", "")).strip():
        raise ValueError("describe what happens in this clip before rendering it")
    if quality not in QUALITIES:
        raise ValueError("quality must be one of %s" % (tuple(QUALITIES),))
    size = QUALITIES[quality]
    models = project.get("models") or MODELS
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": models["unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": models["clip"], "type": "minimax", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": models["vae"]}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": models["audio_vae"]}},
        "5": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["1", 0], "shift_video": project.get("shift", 12.0), "shift_audio": 3.0}},
        "10": {"class_type": "H3Take", "inputs": {
            "clip": ["2", 0], "vae": ["3", 0], "audio_vae": ["4", 0],
            "prompt": clip_prompt(project, clip), "width": size["width"], "height": size["height"],
            "length": snap_frames(clip.get("seconds", 5)), "checkpoint": "ref2va", "anchor": "none", "mid_frames": "",
            "video_mode": "off", "hold_frames": 39, "strength": "hold", "keep": 0.0, "align": "av", "video_feather": 0,
            "audio_hold": "off", "audio_feather": 8, "ref_image_size": "match"}},
        "11": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": project.get("sampler", "euler")}},
        "12": {"class_type": "BasicScheduler", "inputs": {"model": ["5", 0], "scheduler": "simple", "steps": size["steps"], "denoise": 1.0}},
        "13": {"class_type": "BasicGuider", "inputs": {"model": ["5", 0], "conditioning": ["10", 0]}},
        "14": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}},
        "15": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["14", 0], "guider": ["13", 0], "sampler": ["11", 0], "sigmas": ["12", 0], "latent_image": ["10", 1]}},
        "16": {"class_type": "VAEDecode", "inputs": {"samples": ["15", 0], "vae": ["3", 0]}},
        "17": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["15", 0], "vae": ["4", 0]}},
        "18": {"class_type": "CreateVideo", "inputs": {"images": ["16", 0], "fps": float(FPS), "audio": ["17", 0]}},
        "19": {"class_type": "SaveVideo", "inputs": {
            "video": ["18", 0], "filename_prefix": "%s/%s/clips/%s_%s" % (ROOT, folder, clip["id"], take),
            "format": "auto", "codec": "auto"}},
    }


def active_take(clip):
    takes = clip.get("takes") or []
    for take in takes:
        if take.get("id") == clip.get("active"):
            return take
    return takes[-1] if takes else None


def export_command(project, paths, out_path, width=None, height=None):
    """ffmpeg arguments that join the timeline's rendered clips. Gaps become black and silence.

    paths maps a clip id to the file of its active take. Clips with no render are skipped.
    """
    clips = {c["id"]: c for c in project.get("clips", [])}
    items = sorted(
        (i for i in project.get("timeline", []) if i.get("clip") in paths),
        key=lambda i: i["start"],
    )
    if not items:
        raise ValueError("there are no rendered clips on the timeline to export")
    if width is None or height is None:
        take = active_take(clips[items[0]["clip"]]) or {}
        size = QUALITIES.get(take.get("quality"), QUALITIES["final"])
        width, height = size["width"], size["height"]

    args = ["ffmpeg", "-y", "-v", "error"]
    chains = []
    labels = []
    cursor = 0
    count = 0
    for index, item in enumerate(items):
        args += ["-i", paths[item["clip"]]]
        gap = item["start"] - cursor
        if gap > 0:
            seconds = gap / float(FPS)
            chains.append("color=c=black:s=%sx%s:r=%s:d=%.5f,setsar=1[gv%s]" % (width, height, FPS, seconds, count))
            chains.append("anullsrc=r=48000:cl=stereo:d=%.5f[ga%s]" % (seconds, count))
            labels += ["[gv%s]" % count, "[ga%s]" % count]
            count += 1
        first, last = int(item["in"]), int(item["out"])
        chains.append(
            "[%s:v]trim=start_frame=%s:end_frame=%s,setpts=PTS-STARTPTS,fps=%s,"
            "scale=%s:%s:force_original_aspect_ratio=increase,crop=%s:%s,setsar=1[v%s]"
            % (index, first, last, FPS, width, height, width, height, index)
        )
        chains.append(
            "[%s:a]atrim=start=%.5f:end=%.5f,asetpts=PTS-STARTPTS,aresample=48000,aformat=channel_layouts=stereo[a%s]"
            % (index, first / float(FPS), last / float(FPS), index)
        )
        labels += ["[v%s]" % index, "[a%s]" % index]
        cursor = item["start"] + (last - first)
    pieces = len(labels) // 2
    chains.append("%sconcat=n=%s:v=1:a=1[outv][outa]" % ("".join(labels), pieces))
    args += ["-filter_complex", ";".join(chains), "-map", "[outv]", "-map", "[outa]",
             "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", out_path]
    return args
