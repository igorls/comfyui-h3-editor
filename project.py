"""An editing project: clips with rendered takes, and a timeline that only points at them.

Moving or trimming on the timeline never touches a render. No Comfy import.
"""

import copy
import math
import random
import re
import uuid

try:
    from .prompting import compose_prompt, compose_reference_prompt
except ImportError:
    from prompting import compose_prompt, compose_reference_prompt

FPS = 24
JOINT = 51
ROOT = "h3take_projects"
QUALITIES = {
    "draft": {"width": 640, "height": 352, "steps": 4},
    "final": {"width": 1344, "height": 768, "steps": 8},
}
# Picture sizes per project format. Draft is for trying things, final for the cut.
FORMATS = {
    "16:9": {"draft": (640, 352), "final": (1344, 768)},
    "9:16": {"draft": (352, 640), "final": (768, 1344)},
    "1:1": {"draft": (480, 480), "final": (1024, 1024)},
}
LANGUAGE = "English"


def frame_size(project, quality):
    """Width and height for a quality in this project's format. Imported footage counts as final."""
    sizes = FORMATS.get(project.get("format") or "16:9", FORMATS["16:9"])
    return sizes["draft" if quality == "draft" else "final"]
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


def clip_prompt(project, clip):
    fields = {
        "camera": clip.get("camera", ""),
        "speaker": speaker_text(project, clip),
        "dialogue": clip.get("dialogue", ""),
        "sound": clip.get("sound", ""),
        "music": clip.get("music", ""),
        "language": clip_language(project, clip),
    }
    cast = clip_cast(project, clip)
    first, last = frame_of(clip, "first"), frame_of(clip, "last")
    voice = clip_voice(project, clip)
    if cast or first or last or voice:
        owner = None
        if voice:
            number = next((n for n, c in enumerate(cast, 1) if c["id"] == voice["id"]), None)
            owner = {"subject": number} if number else {"description": voice.get("description") or voice["name"]}
        return compose_reference_prompt(
            project.get("style", ""), clip.get("action", ""), subjects=cast,
            first_frame=bool(first), last_frame=bool(last), voice=owner, **fields,
        )
    return compose_prompt(project.get("style", ""), clip.get("action", ""), **fields)


def still_node(asset):
    file = "%s/%s" % (asset["subfolder"], asset["file"]) if asset.get("subfolder") else asset["file"]
    return {"class_type": "H3LoadStill", "inputs": {"file": file}}


def render_graph(project, folder, clip, take, quality, seed):
    """One clip as a ComfyUI graph, saved as its own file in the project."""
    if clip.get("kind") == "footage":
        raise ValueError("%s is imported footage. Extend or bridge it to generate from it" % clip.get("name", "this clip"))
    if not str(clip.get("action", "")).strip():
        raise ValueError("describe what happens in this clip before rendering it")
    if quality not in QUALITIES:
        raise ValueError("quality must be one of %s" % (tuple(QUALITIES),))
    width, height = frame_size(project, quality)
    size = dict(QUALITIES[quality], width=width, height=height)
    models = project.get("models") or MODELS
    link = clip.get("link")
    length = linked_length(link, clip.get("seconds", 5)) if link else snap_frames(clip.get("seconds", 5))
    graph = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": models["unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": models["clip"], "type": "minimax", "device": "default"}},
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
        "11": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": project.get("sampler", "euler")}},
        "12": {"class_type": "BasicScheduler", "inputs": {"model": ["5", 0], "scheduler": "simple", "steps": size["steps"], "denoise": 1.0}},
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
    }
    take_inputs = graph["10"]["inputs"]
    first, last = frame_of(clip, "first"), frame_of(clip, "last")
    if first:
        graph["40"] = still_node(first)
        take_inputs["start_image"] = ["40", 0]
    if last:
        graph["41"] = still_node(last)
        take_inputs["end_image"] = ["41", 0]
    take_inputs["anchor"] = "both" if first and last else "start" if first else "end" if last else "none"
    for index, member in enumerate(clip_cast(project, clip)):
        node = str(50 + index)
        graph[node] = still_node(member)
        take_inputs["ref_images.ref_image_%s" % index] = [node, 0]
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
            "source_video": ["6", 0], "source_audio": ["6", 1],
        })
        if link["mode"] == "bridge":
            graph["7"] = load_node(link["b"], hold, False)
            # A hold alone does not make the take arrive. The guide tells H3 what it must land on.
            graph["8"] = {"class_type": "ImageFromBatch", "inputs": {"image": ["7", 0], "batch_index": 0, "length": 1}}
            take_inputs.update({
                "end_video": ["7", 0], "end_audio": ["7", 1],
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
    return graph


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


def export_command(project, paths, out_path, width=None, height=None):
    """ffmpeg arguments that join the timeline's rendered clips. Gaps become black and silence.

    paths maps a clip id to the file of its active take. Clips with no render
    are skipped. Speed and transitions live on the timeline and are applied here.
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
        width, height = frame_size(project, take.get("quality"))

    args = ["ffmpeg", "-y", "-v", "error"]
    chains = []
    finish = "fps=%s,scale=%s:%s:force_original_aspect_ratio=increase,crop=%s:%s,setsar=1,format=yuv420p,settb=1/%s" % (
        FPS, width, height, width, height, FPS)
    sound = "aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo"

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

    args += ["-filter_complex", ";".join(chains), "-map", "[%s]" % video, "-map", "[%s]" % audio,
             "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", out_path]
    return args


# Editing operations. The editor, the HTTP API and agents share these, so the rules live in one place.

HOLD = 39
MERGE_FRAMES = 63
CLIP_FIELDS = ("name", "action", "camera", "speaker", "dialogue", "sound", "music", "seconds", "cast", "seed",
               "first", "last", "language", "track")
TEXT_FIELDS = ("name", "action", "camera", "speaker", "dialogue", "sound", "music", "language")
PROJECT_FIELDS = ("style", "format", "language")
CAST_FIELDS = ("name", "description", "voice")
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
    return out


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
    before = planned_frames(clip)
    for key, value in fields.items():
        if value is None and key in ("first", "last", "track"):
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
    find_clip(project, clip_id)
    project["clips"] = [c for c in project["clips"] if c["id"] != clip_id]
    project["timeline"] = [i for i in project.get("timeline", []) if i["clip"] != clip_id]


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


def set_project(project, fields):
    check_fields(fields, PROJECT_FIELDS)
    for key, value in fields.items():
        if not isinstance(value, str):
            raise ValueError("%s must be text" % key)
        if key == "format" and value not in FORMATS:
            raise ValueError("format must be one of %s" % ", ".join(FORMATS))
        if key == "language" and not value.strip():
            raise ValueError("language names the language people speak, such as English or Portuguese")
        project[key] = value.strip() if key == "language" else value


def add_cast(project, fields):
    """A person or place from a picture, a voice with no picture, or both."""
    check_fields({k: v for k, v in fields.items() if k not in ("file", "subfolder")}, CAST_FIELDS)
    voice = clean_asset(fields.get("voice"), "a voice")
    asset = clean_asset(fields, "a cast member") if fields.get("file") or not voice else None
    project.setdefault("cast", [])
    person = {"id": new_id("p"), "name": str(fields.get("name") or "Person %s" % (len(project["cast"]) + 1)),
              "description": str(fields.get("description") or "")}
    if asset:
        person.update(asset)
    if voice:
        person["voice"] = voice
    project["cast"].append(person)
    return person


def update_cast(project, cast_id, fields):
    check_fields(fields, CAST_FIELDS)
    person = find_cast(project, cast_id)
    for key, value in fields.items():
        if key == "voice":
            if value is None and not person.get("file"):
                raise ValueError("%s is only a voice. Remove the cast member instead" % person["name"])
            if value is None:
                person.pop("voice", None)
            else:
                person["voice"] = clean_asset(value, "a voice")
            continue
        if not isinstance(value, str):
            raise ValueError("%s must be text" % key)
        person[key] = value.strip() or person[key] if key == "name" else value
    return person


def add_footage(project, name, take, start=None, place=True):
    """Real video or a picture, already made into a take. It edits like any clip but is never rendered."""
    clip = {"id": take.pop("clip", None) or new_id("c"), "kind": "footage", "name": str(name or "Footage"),
            "action": "", "camera": "", "speaker": "", "dialogue": "", "sound": "", "music": "",
            "seconds": round(take["frames"] / float(FPS), 3), "takes": [take], "active": take["id"]}
    project.setdefault("clips", []).append(clip)
    item = place_clip(project, clip["id"], start) if place else None
    return clip, item


def remove_cast(project, cast_id):
    find_cast(project, cast_id)
    project["cast"] = [c for c in project["cast"] if c["id"] != cast_id]
    for clip in project.get("clips", []):
        if cast_id in (clip.get("cast") or []):
            clip["cast"] = [c for c in clip["cast"] if c != cast_id]


# Renders. The server queues the graph; these keep the project's record of it.

def plan_render(project, folder, clip_id, quality, seed=None):
    """The graph for a new take, and the job that will turn its file into a take."""
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
    graph = render_graph(project, folder, clip, take, quality, int(seed))
    job = {"clip": clip_id, "take": take, "quality": quality, "seed": int(seed),
           "saved": {k: copy.deepcopy(clip.get(k)) for k in SAVED_FIELDS}}
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
    return {
        "name": project.get("name"), "revision": project.get("revision", 0), "style": project.get("style", ""),
        "format": project.get("format") or "16:9", "language": project.get("language") or LANGUAGE,
        "frames": scene_end(project), "length": timecode(scene_end(project)), "fps": FPS,
        "cast": [{"id": p["id"], "name": p["name"], "description": p.get("description", ""),
                  "picture": bool(p.get("file")), "voice": bool(p.get("voice"))} for p in project.get("cast", [])],
        "clips": clips, "timeline": timeline, "gaps": gaps,
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
    else:
        raise ValueError("unknown op %s" % kind)
    if op.get("as"):
        names[str(op["as"])] = result
    return result
