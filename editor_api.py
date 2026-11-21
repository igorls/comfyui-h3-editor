"""HTTP routes for the editor. Projects live in ComfyUI's output folder, so renders stay with them."""

import asyncio
import copy
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
import uuid

from aiohttp import web

import folder_paths

from . import archive
from . import hold_cache
from . import fetch
from . import cuts as cutscore
from . import listen as soundscore
from . import lips as lipscore
from . import scene as scenescore
from . import detail as finescore
from . import media
from . import vosr2
from . import looks
from . import patches as patching
from . import project as store
from .policy import editor_id, group_by_model

PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "editor")


def root():
    path = os.path.join(folder_paths.get_output_directory(), store.ROOT)
    os.makedirs(path, exist_ok=True)
    return path


def project_file(name):
    return os.path.join(root(), store.slug(name), "project.json")


def read_project(name):
    path = project_file(name)
    if not os.path.exists(path):
        raise web.HTTPNotFound(text="no project named %s" % name)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data, indent=1):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)
    os.replace(temporary, path)


def write_project(name, data):
    write_json(project_file(name), data)


# Revisions. Every change is numbered and kept for a while, so two writers can be merged
# against what they both started from, and an agent can step back to an earlier state.

HISTORY = 100
LOCKS = {}


def lock(folder):
    return LOCKS.setdefault(folder, asyncio.Lock())


def history_folder(folder):
    return os.path.join(root(), folder, "history")


def read_revision(folder, revision):
    path = os.path.join(history_folder(folder), "%06d.json" % int(revision))
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)["project"]


def broadcast(event, data):
    from server import PromptServer

    PromptServer.instance.send_sync(event, data, None)


def commit(folder, data, source, client=None):
    """Write the next revision of a project. Hold the folder's lock while calling this."""
    path = project_file(folder)
    current = 0
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            current = json.load(f).get("revision", 0)
    data["revision"] = current + 1
    write_project(folder, data)
    kept = history_folder(folder)
    write_json(os.path.join(kept, "%06d.json" % data["revision"]),
               {"time": time.time(), "source": source, "project": data}, indent=None)
    for old in sorted(n for n in os.listdir(kept) if n.endswith(".json"))[:-HISTORY]:
        os.remove(os.path.join(kept, old))
    broadcast("h3take_project", {"folder": folder, "revision": data["revision"], "source": source, "client": client})
    if data.get("pending"):
        ensure_watcher()
    return data


def refused(message, status=400, **extra):
    """An error an agent can read: JSON with the reason, never an HTML page."""
    body = {"error": str(message).strip("'\"")}
    body.update(extra)
    return web.json_response(body, status=status)


def output_path(subfolder, filename):
    base = os.path.realpath(folder_paths.get_output_directory())
    path = os.path.realpath(os.path.join(base, subfolder or "", os.path.basename(filename)))
    if not path.startswith(base + os.sep):
        raise web.HTTPBadRequest(text="that file is outside the output folder")
    return path


async def run_bytes(args):
    """Run a command, ffmpeg found as media.ffmpeg says, and return what it printed."""
    try:
        args = media.command(args)
    except RuntimeError as error:
        raise web.HTTPInternalServerError(text=str(error))
    process = await asyncio.create_subprocess_exec(*args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = await process.communicate()
    if process.returncode != 0:
        raise web.HTTPInternalServerError(text=err.decode(errors="replace")[-600:])
    return out


async def run(args):
    return (await run_bytes(args)).decode(errors="replace")


async def video_info(path, count=True):
    """What a video holds, read in a thread so a long file never stalls the server."""
    try:
        return await asyncio.to_thread(media.video_info, path, count)
    except Exception as error:  # PyAV raises its own errors for files it cannot read
        raise web.HTTPInternalServerError(text="could not read %s: %s" % (os.path.basename(path), error))


STRIP_HEIGHT = 90
STRIP_TILES = 48


async def make_peaks(path, peaks):
    """How loud a file is at each frame of the timeline, 0 to 100, written once beside it. Returns the values."""
    if os.path.exists(peaks):
        with open(peaks, encoding="utf-8") as f:
            return json.load(f)
    values = []
    try:
        import numpy as np

        rate = 4800
        pcm = await run_bytes(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(rate), "-f", "s16le", "-"])
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        per = rate // store.FPS
        usable = (len(samples) // per) * per
        if usable:
            loud = np.abs(samples[:usable].reshape(-1, per)).max(axis=1)
            values = [int(round(v)) for v in np.sqrt(np.clip(loud, 0, 1)) * 100]
    except web.HTTPException:
        values = []
    with open(peaks, "w", encoding="utf-8") as f:
        json.dump(values, f)
    return values


async def make_strip(path, frames):
    """A row of frames from across a clip and its loudness per frame, made once and kept beside it."""
    base = os.path.splitext(path)[0]
    sprite, peaks = base + ".strip.jpg", base + ".peaks.json"
    info = await video_info(path, count=False)
    width, height = info["width"], info["height"]
    tile_w = max(2, int(round(STRIP_HEIGHT * width / float(height) / 2.0)) * 2)
    step = max(1, int(math.ceil(frames / float(STRIP_TILES))))
    count = max(1, int(math.ceil(frames / float(step))))
    if not os.path.exists(sprite):
        await run(["ffmpeg", "-y", "-v", "error", "-i", path, "-vf",
                   "select=not(mod(n\\,%d)),scale=%d:%d,tile=%dx1" % (step, tile_w, STRIP_HEIGHT, count),
                   "-frames:v", "1", "-q:v", "4", sprite])
    await make_peaks(path, peaks)
    return {"sprite": os.path.basename(sprite), "peaks": os.path.basename(peaks), "w": tile_w, "h": STRIP_HEIGHT,
            "count": count, "step": step}


async def strip(request):
    body = await request.json()
    path = output_path(body.get("subfolder"), body["filename"])
    if not os.path.exists(path):
        raise web.HTTPNotFound(text="that clip is gone")
    return web.json_response(await make_strip(path, int(body.get("frames") or 1)))


SPECTROGRAM_GATE = asyncio.Semaphore(2)


async def spectrogram(request):
    """Generate only on request, shared by all timelines that use this file."""
    body = await request.json()
    path = output_path(body.get("subfolder"), body["filename"])
    if not os.path.isfile(path):
        raise web.HTTPNotFound(text="that sound is gone")
    try:
        async with lock("spectrogram:" + path):
            async with SPECTROGRAM_GATE:
                made = await asyncio.to_thread(media.spectrogram, path)
    except (ValueError, RuntimeError, subprocess.SubprocessError) as error:
        return refused("could not draw this spectrogram: %s" % error)
    return web.json_response({"spectrogram": made})


async def sheet(request):
    """Frames from a clip in one grid, large enough for someone, or an agent, to judge it.

    By default 12 frames spread across the whole clip, 384 pixels wide each. first and last narrow
    the range, count sets how many frames (1 to 24), and tile_width how wide each one is.
    """
    body = await request.json()
    path = output_path(body.get("subfolder"), body["filename"])
    if not os.path.exists(path):
        raise web.HTTPNotFound(text="that clip is gone")
    frames = int(body.get("frames") or 0)
    if frames <= 0:
        frames = (await video_info(path))["frames"]
    try:
        first = max(0, min(frames - 1, int(body.get("first") or 0)))
        last = max(first + 1, min(frames, int(body.get("last") or frames)))
        count = max(1, min(24, int(body.get("count") or 12), last - first))
        width = max(160, min(1344, int(body.get("tile_width") or 384)))
    except (TypeError, ValueError):
        return refused("first, last, count and tile_width are whole numbers")
    picked = sorted({first + (i * (last - first)) // count for i in range(count)})
    columns = 4 if len(picked) > 6 else 2 if len(picked) > 1 else 1
    rows = -(-len(picked) // columns)
    target = os.path.splitext(path)[0] + ".sheet-%d-%d-%d-%d.jpg" % (first, last, len(picked), width)
    if not os.path.exists(target):
        chosen = "+".join("eq(n\\,%d)" % n for n in picked)
        await run(["ffmpeg", "-y", "-v", "error", "-i", path, "-vf",
                   "select=%s,scale=%d:-2,tile=%dx%d:padding=6:color=black" % (chosen, width, columns, rows),
                   "-frames:v", "1", "-q:v", "3", target])
    return web.json_response({"sheet": os.path.basename(target), "picked": picked, "frames": frames})


def pack_version():
    """The pack's release, as pyproject.toml names it: shown in the editor, so a screenshot says which one it is."""
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pyproject.toml"), encoding="utf-8") as f:
            for line in f:
                key, _, value = line.partition("=")
                if key.strip() == "version":
                    return value.strip().strip("\"'") or "unknown"
    except OSError:
        pass
    return "unknown"


async def page(request):
    """The page always comes fresh and names its files by version, so a browser never mixes old and new."""
    version = int(max(os.path.getmtime(os.path.join(PAGE, n)) for n in os.listdir(PAGE)))
    with open(os.path.join(PAGE, "index.html"), encoding="utf-8") as f:
        html = f.read().replace("__VERSION__", str(version)).replace("__RELEASE__", pack_version())
    return web.Response(text=html, content_type="text/html", headers={"Cache-Control": "no-cache"})


def refmod_files():
    """The RefMods the ComfyUI-MiniMaxH3Mod pack's loader offers, or None when that pack is not installed."""
    import nodes as comfy_nodes

    loader = comfy_nodes.NODE_CLASS_MAPPINGS.get("MiniMaxH3RefModsLoader")
    if loader is None or "MiniMaxH3RefModApply" not in comfy_nodes.NODE_CLASS_MAPPINGS:
        return None
    try:
        options = loader.INPUT_TYPES()["required"]["mod_1"][0]
    except Exception:
        logging.exception("H3 Editor could not read the RefMod list")
        return []
    return [name for name in options if isinstance(name, str) and name not in ("", "None", "(none)")]


def encoder_choices(available):
    """The text encoders and video VAEs to offer, with the one a render takes by itself and what is missing."""
    import folder_paths

    def listed(*kinds):
        found = set()
        for kind in kinds:
            if kind in folder_paths.folder_names_and_paths:
                found.update(folder_paths.get_filename_list(kind))
        # The ones made for MiniMax H3 when there are any, so the menu is not every model on the disk.
        made = sorted((f for f in found if "minimax_h3" in f.lower().replace("-", "_")), key=str.lower)
        return made or sorted(found, key=str.lower)

    def text_encoders():
        # The 32B ones, then the 4B and 8B ones a projection is installed for (clipproj.py).
        small = set()
        for kind in ("text_encoders", "clip"):
            if kind in folder_paths.folder_names_and_paths:
                small.update(f for f in folder_paths.get_filename_list(kind) if store.projection_for(f, available))
        return listed("text_encoders", "clip") + sorted(small, key=str.lower)

    status = store.encoder_status(available)
    encoders = text_encoders()
    return {"text_encoders": encoders,
            "projections": {f: store.projections_for(f, available) for f in encoders if store.projections_for(f, available)},
            "vaes": video_vaes(listed("vae")), "auto": status["auto"], "missing": status["missing"]}


def video_vaes(files):
    """The video VAEs among H3's VAE files, which also include its audio and image ones."""
    video = [f for f in files if "video" in f.lower()]
    return video or [f for f in files if "audio" not in f.lower()]


def refmods_status():
    files = refmod_files()
    return {"installed": files is not None, "files": files or []}


def available_models():
    """The model files ComfyUI lists, for finding a checkpoint profile's files, and the RefMods on offer."""
    import folder_paths
    import nodes as comfy_nodes

    names = set()
    if ("LoadBackgroundRemovalModel" in comfy_nodes.NODE_CLASS_MAPPINGS
            and "background_removal" in folder_paths.folder_names_and_paths):
        names.update(store.sprite.MATTE_PREFIX + name for name in folder_paths.get_filename_list("background_removal"))
    for kind in ("diffusion_models", "unet_gguf", "loras", "latent_upscale_models", "text_encoders", "clip", "vae"):
        if kind in folder_paths.folder_names_and_paths:
            names.update(folder_paths.get_filename_list(kind))
    if all(name in comfy_nodes.NODE_CLASS_MAPPINGS for name in store.SAM3_NODES):
        names.add("sam3-nodes")
    if all(name in comfy_nodes.NODE_CLASS_MAPPINGS for name in store.MOTION_NODES):
        names.add("motion-nodes")
    if store.H3_IMAGE_SAMPLER_NODE in comfy_nodes.NODE_CLASS_MAPPINGS:
        names.add("h3-image-sampler")
    for node_id in ("PH3ImageConditioning", "MiniMaxH3ImageConditioning", "MiniMaxH3SigmaShift"):
        if node_id in comfy_nodes.NODE_CLASS_MAPPINGS:
            names.add("node:" + node_id)
    if "checkpoints" in folder_paths.folder_names_and_paths:
        names.update("motion-checkpoints:" + name for name in folder_paths.get_filename_list("checkpoints"))
    if "model_patches" in folder_paths.folder_names_and_paths:
        names.update("motion-control:" + name for name in folder_paths.get_filename_list("model_patches"))
    names.update(store.REFMOD_PREFIX + name for name in refmod_files() or [])
    names.update(vosr2.capabilities(comfy_nodes.NODE_CLASS_MAPPINGS, folder_paths.models_dir))
    if store.PROJECTION_FOLDER in folder_paths.folder_names_and_paths:
        names.update(store.PROJECTION_PREFIX + name for name in folder_paths.get_filename_list(store.PROJECTION_FOLDER))
    return names


# Get the models: the files a fresh install needs for the default profile, downloaded into ComfyUI's
# models folders. The plan is built from store.SETUP_FILES alone; a request names no URL and no place.

def setup_folder(kind):
    """The folder a setup file of this kind goes in: ComfyUI's own folder of that name, else its first one."""
    paths = folder_paths.get_folder_paths(kind) if kind in folder_paths.folder_names_and_paths else []
    for path in paths:
        if os.path.basename(os.path.normpath(path)).lower() == kind.lower():
            return path
    return paths[0] if paths else os.path.join(folder_paths.models_dir, kind)


def setup_plan(status, catalog=None):
    """What a download fetches: each missing file of status, with its URL, size, hash and the path it lands at."""
    entries = {entry["id"]: entry for entry in (catalog if catalog is not None else store.SETUP_FILES)}
    plan = []
    for name in status["missing"]:
        entry = entries[name]
        base = os.path.realpath(setup_folder(entry["folder"]))
        dest = os.path.realpath(os.path.join(base, *entry["needs"][0].split("/")))
        if not dest.startswith(base + os.sep):
            raise ValueError("%s would land outside %s" % (entry["needs"][0], base))
        plan.append({"id": name, "url": store.setup_url(entry), "dest": dest, "bytes": entry["bytes"], "sha256": entry["sha256"],
                     "kind": entry["folder"]})
    return plan


def free_space(plan):
    """What is short on each disk the plan writes to: [{folder, needs, free}], empty when everything fits."""
    needs = {}
    for item in plan:
        folder = os.path.dirname(item["dest"])
        while not os.path.isdir(folder):
            folder = os.path.dirname(folder)
        part = item["dest"] + fetch.PART
        left = item["bytes"] - (os.path.getsize(part) if os.path.exists(part) else 0)
        device = os.stat(folder).st_dev
        row = needs.setdefault(device, {"folder": folder, "needs": 0})
        row["needs"] += left
    short = []
    for row in needs.values():
        free = shutil.disk_usage(row["folder"]).free
        if free < row["needs"] + 2 ** 30:
            short.append(dict(row, free=free))
    return short


def _annotate_setup_rows(status, plan):
    home = os.path.dirname(os.path.realpath(folder_paths.models_dir))
    for row in status["files"]:
        row["folder_path"] = setup_folder(row["folder"])
        inside = os.path.realpath(row["folder_path"])
        row["folder_shown"] = os.path.relpath(inside, home).replace(os.sep, "/") if inside.startswith(home + os.sep) else row["folder_path"]
        part = next((item["dest"] + fetch.PART for item in plan if item["id"] == row["id"]), None)
        row["partial"] = os.path.getsize(part) if part and os.path.exists(part) else 0
    status["short"] = free_space(plan)
    status["job"] = fetch.status()
    return status


def setup_report():
    available = available_models()
    status = store.setup_status(available)
    plan = setup_plan(status, store.SETUP_FILES)
    return _annotate_setup_rows(status, plan)


def motion_setup_report(groups=None):
    available = available_models()
    status = store.motion_setup_status(available, groups)
    plan = setup_plan(status, store.MOTION_SETUP_FILES)
    report = _annotate_setup_rows(status, plan)
    report["motion"] = store.motion_status(available)
    return report


def setup_placed(item):
    """ComfyUI lists a downloaded file at once, without waiting for its folder's cache to notice."""
    cache = getattr(folder_paths, "filename_list_cache", None)
    if isinstance(cache, dict):
        cache.pop(item["kind"], None)
        if item["kind"] == "text_encoders":
            cache.pop("clip", None)


async def model_setup(request):
    """What the default profile is missing, with the total size, and the download in progress if any."""
    return web.json_response(setup_report())


async def model_setup_download(request):
    """Download the missing setup files into their folders. Takes no body: the server's own list says what and where."""
    status = store.setup_status(available_models())
    plan = setup_plan(status, store.SETUP_FILES)
    if not plan:
        return web.json_response(setup_report())
    short = free_space(plan)
    if short:
        row = short[0]
        return refused("there is not enough room in %s: the download needs %.1f GB and %.1f GB is free"
                       % (row["folder"], row["needs"] / 1e9, row["free"] / 1e9), 507, short=short)
    try:
        fetch.start(plan, setup_placed)
    except fetch.Refused as error:
        return refused(error, 409)
    return web.json_response(setup_report())


async def model_setup_cancel(request):
    """Stop the download. What was written stays, and the next download carries on from it."""
    fetch.cancel()
    return web.json_response(setup_report())


async def motion_model_setup(request):
    """Optional motion transfer models (replace, pose, sharp): installed status, sizes and active download."""
    raw = request.query.get("groups")
    groups = [g.strip() for g in raw.split(",") if g.strip()] if raw else None
    try:
        return web.json_response(motion_setup_report(groups))
    except ValueError as error:
        return refused(plain(error))


async def motion_model_download(request):
    """Download missing optional motion models for the chosen group(s) from Hugging Face."""
    try:
        body = await request.json() if request.can_read_body else {}
    except ValueError:
        body = {}
    groups = body.get("groups") if isinstance(body, dict) else None
    if groups is None:
        groups = ["replace"]
    if not isinstance(groups, list) or not all(isinstance(g, str) for g in groups):
        return refused("groups must be a list of motion model groups")
    try:
        status = store.motion_setup_status(available_models(), groups)
    except ValueError as error:
        return refused(plain(error))
    plan = setup_plan(status, store.MOTION_SETUP_FILES)
    if not plan:
        return web.json_response(motion_setup_report(groups))
    short = free_space(plan)
    if short:
        row = short[0]
        return refused("there is not enough room in %s: the download needs %.1f GB and %.1f GB is free"
                       % (row["folder"], row["needs"] / 1e9, row["free"] / 1e9), 507, short=short)
    try:
        fetch.start(plan, setup_placed)
    except fetch.Refused as error:
        return refused(error, 409)
    return web.json_response(motion_setup_report(groups))


async def motion_model_cancel(request):
    """Stop an active motion model download; partial files remain so it can resume."""
    fetch.cancel()
    return web.json_response(motion_setup_report())


def node_definitions():
    """The installed nodes that return one model, each as ComfyUI's GET /object_info describes it.

    Read from the node classes, as that route does, so nothing is imported from another pack.
    A node whose definition cannot be read is left out: it could not be drawn or wired anyway.
    """
    import nodes as comfy_nodes

    found = {}
    for node_id, cls in list(comfy_nodes.NODE_CLASS_MAPPINGS.items()):
        try:
            # Most nodes return something else, and are passed over before their inputs are asked for.
            if tuple(getattr(cls, "RETURN_TYPES", None) or ()) != ("MODEL",):
                continue
            if hasattr(cls, "GET_NODE_INFO_V1"):
                found[node_id] = cls.GET_NODE_INFO_V1()
                continue
            found[node_id] = {
                "input": cls.INPUT_TYPES(), "output": list(cls.RETURN_TYPES),
                "output_is_list": list(getattr(cls, "OUTPUT_IS_LIST", None) or ()),
                "display_name": comfy_nodes.NODE_DISPLAY_NAME_MAPPINGS.get(node_id, node_id),
                "description": getattr(cls, "DESCRIPTION", ""), "category": getattr(cls, "CATEGORY", ""),
                "python_module": getattr(cls, "RELATIVE_PYTHON_MODULE", "nodes"),
            }
        except Exception:
            logging.debug("H3 Editor could not read the definition of %s", node_id, exc_info=True)
    return found


def patch_list():
    """The installed nodes a project can put on the model wire, without this pack's own."""
    from . import nodes, preview

    # Every render already ends the model wire with the shift, set from the project: a second one would do nothing.
    own = set(nodes.NODE_CLASS_MAPPINGS) | set(preview.NODE_CLASS_MAPPINGS) | {"MiniMaxH3SigmaShift"}
    return patching.listing(node_definitions(), own)


def installed_patches():
    """The same nodes by id, as a render graph takes them."""
    return {row["node"]: row for row in patch_list()}


def lora_files():
    """Every file in models/loras, as ComfyUI lists it."""
    import folder_paths

    return sorted(folder_paths.get_filename_list("loras"), key=str.lower)


def presets_file():
    return os.path.join(root(), "checkpoints.json")


def read_presets():
    """The checkpoints saved for every project to use. A file that cannot be read counts as none."""
    try:
        with open(presets_file(), encoding="utf-8") as f:
            rows = json.load(f)["checkpoints"]
    except (OSError, ValueError, KeyError, TypeError):
        return []
    return [r for r in rows if isinstance(r, dict) and isinstance(r.get("id"), str) and isinstance(r.get("unet"), str) and r["unet"]]


async def presets(request):
    return web.json_response({"checkpoints": read_presets()})


async def save_presets(request):
    """Replace the shared presets with the list sent."""
    body = await request.json()
    try:
        rows = store.clean_checkpoints(body.get("checkpoints") if isinstance(body, dict) else None, store.PRESET_LIMIT)
    except ValueError as error:
        raise web.HTTPBadRequest(text=str(error))
    async with lock("checkpoints.json"):
        path = presets_file()
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump({"checkpoints": rows}, f, indent=1)
        os.replace(path + ".tmp", path)
    return web.json_response({"checkpoints": rows})


# The cast catalog: cast members kept once per install, beside the projects in a folder of its own that
# holds its own copies of their pictures and voices, so deleting a project never breaks it.

CATALOG = "_cast"


def catalog_folder():
    return os.path.join(root(), CATALOG)


def catalog_file():
    return os.path.join(catalog_folder(), "catalog.json")


def read_catalog():
    """The catalog's entries. A file that cannot be read counts as none."""
    try:
        with open(catalog_file(), encoding="utf-8") as f:
            rows = json.load(f)["cast"]
    except (OSError, ValueError, KeyError, TypeError):
        return []
    return [r for r in rows if isinstance(r, dict) and isinstance(r.get("id"), str)]


def write_catalog(rows):
    """Write the catalog, then delete its files that no entry uses any more, such as an earlier version's
    pictures. Projects keep their own copies, so nothing outside the catalog points at them."""
    write_json(catalog_file(), {"cast": rows})
    used = {os.path.basename(name) for entry in rows for _, name in store.cast_assets(entry)}
    files, _ = catalog_files_folder()
    try:
        names = os.listdir(files)
    except OSError:
        return
    for name in names:
        path = os.path.join(files, name)
        if name not in used:
            try:
                # A file copied in the last few minutes may belong to a batch that has not written the catalog yet.
                if time.time() - os.path.getmtime(path) < 600:
                    continue
                os.remove(path)
            except OSError:
                pass


def copy_assets(found, folder, subfolder):
    """Copy each file, given as (subfolder, file) under the output folder, into folder under a new name.
    Returns {(subfolder, file): {"file", "subfolder"}} for the copies."""
    os.makedirs(folder, exist_ok=True)
    moved = {}
    for source_folder, name in found:
        if (source_folder, name) in moved:
            continue
        source = output_path(source_folder, name)
        if not os.path.isfile(source):
            raise ValueError("the file %s of this cast member is missing" % name)
        new = uuid.uuid4().hex[:12] + os.path.splitext(name)[1].lower()
        shutil.copyfile(source, os.path.join(folder, new))
        moved[(source_folder, name)] = {"file": new, "subfolder": subfolder}
    return moved


def catalog_files_folder():
    return os.path.join(catalog_folder(), "files"), "%s/%s/files" % (store.ROOT, CATALOG)


def project_assets_folder(folder):
    return os.path.join(root(), folder, "assets"), "%s/%s/assets" % (store.ROOT, folder)


def catalog_op(work, folder, op, names, catalog):
    """The catalog's project operations: they copy files, so they run here rather than in project.py."""
    kind = op["op"]
    if kind == "save_cast_to_catalog":
        cast_id = store._resolve(op.get("cast"), names, "cast")
        person = store.find_cast(work, cast_id)
        if store.catalog_state(person, catalog) == "current":
            # Nothing changed since it matched the catalog: no new version for other projects to take.
            entry = store.find_catalog(catalog, person["catalog"]["id"])
            return {"cast": cast_id, "catalog": entry["id"], "version": entry["version"]}
        target, sub = catalog_files_folder()
        entry = store.save_to_catalog(catalog, person, copy_assets(store.cast_assets(person), target, sub))
        return {"cast": cast_id, "catalog": entry["id"], "version": entry["version"]}
    if kind == "add_cast_from_catalog":
        entry = store.find_catalog(catalog, op.get("catalog"))
        target, sub = project_assets_folder(folder)
        person = store.cast_from_catalog(work, entry, copy_assets(store.cast_assets(entry), target, sub))
        result = {"cast": person["id"], "catalog": entry["id"], "version": entry["version"]}
        if op.get("as"):
            names[op["as"]] = result
        return result
    if kind == "update_cast_from_catalog":
        cast_id = store._resolve(op.get("cast"), names, "cast")
        link = store.find_cast(work, cast_id).get("catalog") or {}
        entry = store.find_catalog(catalog, link.get("id"))
        target, sub = project_assets_folder(folder)
        store.cast_from_catalog(work, entry, copy_assets(store.cast_assets(entry), target, sub), cast_id)
        return {"cast": cast_id, "catalog": entry["id"], "version": entry["version"]}
    raise ValueError("unknown operation %s" % kind)


CATALOG_OPS = ("save_cast_to_catalog", "add_cast_from_catalog", "update_cast_from_catalog")


async def cast_catalog(request):
    return web.json_response({"cast": read_catalog()})


async def remove_from_catalog(request):
    """Take an entry out of the catalog, with its files. Projects keep their own copies."""
    body = await request.json()
    async with lock(CATALOG):
        rows = read_catalog()
        try:
            entry = store.find_catalog(rows, body.get("id"))
        except KeyError as error:
            return refused(plain(error), 404)
        rows = [r for r in rows if r is not entry]
        write_catalog(rows)
    return web.json_response({"cast": rows})


async def spread_catalog(request):
    """Bring every project's copy of a catalog entry up to date, except copies changed in their project."""
    body = await request.json()
    catalog = read_catalog()
    try:
        entry = store.find_catalog(catalog, body.get("id"))
    except KeyError as error:
        return refused(plain(error), 404)
    updated, kept = [], []
    for folder in project_folders():
        async with lock(folder):
            data = read_project(folder)
            changed = False
            for person in data.get("cast", []):
                if (person.get("catalog") or {}).get("id") != entry["id"]:
                    continue
                state = store.catalog_state(person, catalog)
                if state in ("changed", "newer") and store.cast_edited(person):
                    kept.append({"project": folder, "cast": person["id"], "name": person["name"]})
                elif state == "newer":
                    target, sub = project_assets_folder(folder)
                    store.cast_from_catalog(data, entry, copy_assets(store.cast_assets(entry), target, sub), person["id"])
                    updated.append({"project": folder, "cast": person["id"]})
                    changed = True
            if changed:
                commit(folder, data, "catalog", body.get("client"))
    return web.json_response({"updated": updated, "kept": kept})


def sampling_choices():
    """The checkpoint files, samplers and schedulers ComfyUI offers, for a project's own checkpoints."""
    import folder_paths

    files = set()
    for kind in ("diffusion_models", "unet_gguf"):
        if kind in folder_paths.folder_names_and_paths:
            files.update(folder_paths.get_filename_list(kind))
    try:
        import comfy.samplers

        samplers, schedulers = list(comfy.samplers.KSampler.SAMPLERS), list(comfy.samplers.SCHEDULER_NAMES)
    except Exception:
        samplers, schedulers = ["euler", "res_multistep"], ["simple"]
    if store.BETA57 not in schedulers:
        schedulers.append(store.BETA57)      # the editor makes this one itself
    return {"checkpoints": sorted(files, key=str.lower), "samplers": samplers, "schedulers": schedulers}


# Deleted projects are moved here, inside the projects folder, and never erased by the editor.
TRASH = ".trash"


def project_folders():
    return sorted(n for n in os.listdir(root()) if os.path.exists(os.path.join(root(), n, "project.json")))


CARDS = {}    # folder -> (when its project.json last changed, its card), so listing many projects reads each file once


def project_cards(names):
    """What the project picker shows for each project, and when the project last changed.

    A project file is read again only after it changed. One that cannot be read still gets a card, with its folder's name.
    """
    cards = {}
    for name in names:
        try:
            changed = os.path.getmtime(os.path.join(root(), name, "project.json"))
            kept = CARDS.get(name)
            if not kept or kept[0] != changed:
                with open(os.path.join(root(), name, "project.json"), encoding="utf-8") as f:
                    kept = CARDS[name] = (changed, {**store.project_card(json.load(f)), "changed": int(changed)})
            cards[name] = kept[1]
        except (OSError, ValueError, AttributeError):
            cards[name] = {"name": "", "format": "16:9", "clips": 0, "frames": 0, "picture": None, "rendering": 0, "changed": 0}
    for gone in set(CARDS) - set(names):
        CARDS.pop(gone, None)
    return cards


def project_titles(names, cards=None):
    """The name each project shows, which a rename changes while its folder stays."""
    cards = project_cards(names) if cards is None else cards
    return {name: cards[name]["name"] or name.replace("_", " ") for name in names}


async def rename(request):
    """Give a project another name. Its folder, and so every path to its files, stays as it is."""
    folder = store.slug(request.match_info["name"])
    body = await request.json()
    try:
        title = store.clean_title(body.get("name"))
    except ValueError as error:
        return refused(plain(error))
    async with lock(folder):
        data = read_project(folder)
        if data.get("name") != title:
            data["name"] = title
            commit(folder, data, "rename", body.get("client_id"))
    return web.json_response({"folder": folder, "project": data})


async def duplicate(request):
    """A copy of a project with its clips, takes and pictures, in a folder of its own."""
    folder = store.slug(request.match_info["name"])
    body = await request.json() if request.can_read_body else {}
    data = read_project(folder)
    names = project_folders()
    try:
        title = store.clean_title(body["name"]) if body.get("name") else store.copy_name(data.get("name") or folder, names)
        made = store.slug(title)
    except ValueError as error:
        return refused(plain(error))
    if made.lower() in {n.lower() for n in names} or os.path.exists(os.path.join(root(), made)):
        return refused("there is already a project called %s" % title, 409)
    async with lock(folder):
        data = read_project(folder)
        source, target = os.path.join(root(), folder), os.path.join(root(), made)
        # Renders can be large, so the copy runs off the server's loop. Its history starts afresh.
        await asyncio.get_running_loop().run_in_executor(
            None, lambda: shutil.copytree(source, target, ignore=shutil.ignore_patterns("history", "*.tmp", "project.json")))
    copied = store.retarget(data, folder, made)
    copied["name"] = title
    copied["pending"] = []
    copied.pop("revision", None)
    async with lock(made):
        commit(made, copied, "duplicate", body.get("client_id"))
    return web.json_response({"folder": made, "project": read_project(made)})


PACKING = ".packing"      # zips on their way in or out, beside the projects; never a project of its own
KEEP_FREE = 2 * 1024 ** 3  # an opened project never leaves the disk with less than this free


def spool(suffix):
    folder = os.path.join(root(), PACKING)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, uuid.uuid4().hex + suffix)


def zip_name(title):
    """A file name for a project's zip that any browser and any system takes: letters, digits, spaces and - _ ."""
    safe = "".join(ch if ch.isascii() and (ch.isalnum() or ch in " -_.") else "_" for ch in str(title)).strip(" ._")
    return (safe or "project") + ".h3project.zip"


async def archive_project(request):
    """A project as one zip, to open on another machine with Open a project zip.

    Without ?keep= it answers how big the zip is at each level (all, cut, text), for the editor to show before
    anything is made. With ?keep= it makes the zip and sends it. See archive.py for what each level holds.
    """
    folder = store.slug(request.match_info["name"])
    data = read_project(folder)
    base = os.path.join(root(), folder)
    keep = request.query.get("keep")
    if not keep:
        levels = await asyncio.to_thread(archive.measure, base, store.ROOT, folder, data)
        return web.json_response({"name": data.get("name") or folder, "levels": levels, "release": pack_version()})
    if keep not in archive.LEVELS:
        return refused("keep is one of %s" % ", ".join(archive.LEVELS))
    target = spool(".zip")
    try:
        # A project can be gigabytes: the zip is made on disk, off the server's loop, and sent in pieces.
        await asyncio.to_thread(archive.pack, base, store.ROOT, folder, data, keep, target, pack_version())
        response = web.StreamResponse(headers={
            "Content-Type": "application/zip",
            "Content-Disposition": 'attachment; filename="%s"' % zip_name(data.get("name") or folder),
            "Content-Length": str(os.path.getsize(target)),
        })
        await response.prepare(request)
        with open(target, "rb") as f:
            while True:
                chunk = await asyncio.to_thread(f.read, 1 << 20)
                if not chunk:
                    break
                await response.write(chunk)
        await response.write_eof()
        return response
    finally:
        try:
            os.remove(target)
        except OSError:
            pass


async def open_project(request):
    """Open a zip made by Download as zip as a new project. The zip is the request's body, sent as it is.

    It is read from the stream onto the disk, so ComfyUI's limit on the size of an upload does not apply. The new
    project takes the name it had, or that name with a number when a project has it already.
    """
    # The zip and the project it opens into both need room: an upload stops before the disk would fill.
    free = shutil.disk_usage(root()).free
    most = max(0, (free - KEEP_FREE) // 2)
    declared = request.content_length
    if declared is not None and declared > most:
        return refused("the zip is %.1f GB, and this disk has room to open %.1f GB (%.1f GB free)"
                       % (declared / 1e9, most / 1e9, free / 1e9), 413)
    upload = spool(".upload.zip")
    try:
        size = 0
        with open(upload, "wb") as f:
            async for chunk in request.content.iter_chunked(1 << 20):
                size += len(chunk)
                if size > most:
                    return refused("the zip is larger than this disk has room to open (%.1f GB free)" % (free / 1e9), 413)
                f.write(chunk)
        if not size:
            return refused("send the project's zip as the body of the request")
        try:
            manifest, _ = await asyncio.to_thread(archive.read, upload)
            title = archive.free_name(store.clean_title(manifest.get("name") or "Opened project"), project_folders(), store.slug)
            made = store.slug(title)
            manifest, data = await asyncio.to_thread(archive.unpack, upload, os.path.join(root(), made))
        except (archive.ArchiveError, ValueError) as error:
            return refused(plain(error))
        opened = store.retarget(data, store.slug(manifest.get("folder") or made), made)
        opened["name"] = title
        opened["pending"] = []
        opened.pop("revision", None)
        # A zip can come from anyone. Its model patches wait, switched off, for whoever opened it to tick them.
        rested = store.rest_patches(opened)
        async with lock(made):
            commit(made, opened, "open", request.query.get("client"))
        have = {name.lower() for name in lora_files()}
        missing = [name for name in manifest.get("models", {}).get("loras", []) if name.lower() not in have]
        return web.json_response({"folder": made, "name": title, "release": manifest.get("release"), "keep": manifest.get("keep"),
                                  "missing_loras": missing, "patches_off": rested, "outside": manifest.get("outside", []), "files": manifest.get("files", 0)})
    finally:
        try:
            os.remove(upload)
        except OSError:
            pass


async def remove(request):
    """Take a project out of the editor. Its folder moves to the trash folder beside the projects."""
    folder = store.slug(request.match_info["name"])
    async with lock(folder):
        data = read_project(folder)
        if data.get("pending"):
            return refused("%s is rendering. Cancel its renders, then delete it" % (data.get("name") or folder), 409)
        kept = "%s_%s" % (folder, time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(os.path.join(root(), TRASH), exist_ok=True)
        os.replace(os.path.join(root(), folder), os.path.join(root(), TRASH, kept))
    LOCKS.pop(folder, None)
    broadcast("h3take_project", {"folder": folder, "deleted": True})
    return web.json_response({"deleted": folder, "trash": "%s/%s/%s" % (store.ROOT, TRASH, kept)})


async def project_storage(request):
    folder = request.match_info["name"]
    try:
        if folder != store.slug(folder):
            return refused("Use the project folder name from the projects list.")
    except ValueError as error:
        return refused(str(error))
    scope = version = None
    if request.method == "POST":
        body = await request.json()
        if not isinstance(body, dict) or body.get("scope") not in ("unused", "all") or not isinstance(body.get("version"), str):
            return refused("Send a cleanup scope and its version from the storage report.")
        scope, version = body["scope"], body["version"]
    async with lock(folder):
        data = read_project(folder)
        try:
            report = await asyncio.to_thread(hold_cache.project_cache, root(), folder, data, scope, version)
        except hold_cache.CacheConflict as error:
            return refused(str(error), 409)
        except (OSError, ValueError) as error:
            return refused(str(error))
    return web.json_response(report, headers={"Cache-Control": "no-store"})


async def sizes(request):
    """Describe the presets for any project ratio, optionally resolving a custom size on the same grid."""
    try:
        shape = store.clean_format(request.query.get("format", "16:9"))
        result = {"format": shape, "choices": store.size_choices([shape])[shape]}
        target = {key: float(request.query[key]) for key in ("longest_side", "megapixels") if key in request.query}
        if target:
            result["size"] = store.fitted_size(shape, target)
    except ValueError as error:
        return refused(plain(error))
    return web.json_response(result)


async def projects(request):
    names = project_folders()
    available = available_models()
    cards = project_cards(names)
    return web.json_response({"projects": names, "titles": project_titles(names, cards), "cards": cards, "profiles": store.profile_status(available),
                              "motion": store.motion_status(available),
                              "finals": store.finals_status(available),
                              "loras": lora_files(), "patches": patch_list(), "realism": store.lora_status(available), "presets": read_presets(), "catalog": read_catalog(), "refmods": refmods_status(), "encoders": encoder_choices(available), "sizes": store.size_choices(), "upscalers": store.upscaler_status(available), "release": pack_version(), **sampling_choices()})


async def create(request):
    body = await request.json()
    name = store.slug(body.get("name"))
    async with lock(name):
        if not os.path.exists(project_file(name)):
            data = store.new_project(body.get("name"))
            available = available_models()
            if any(row["available"] for row in store.profile_status(available)):
                # A new project keeps the model it starts with, even if another is installed later.
                data["profile"] = store.installed_profile(available)
            commit(name, data, "create")
    return web.json_response({"folder": name, "project": read_project(name)})


async def load(request):
    return web.json_response(read_project(request.match_info["name"]))


async def save(request):
    """Save the editor's copy. A copy made from an older revision is merged, never written over newer work."""
    folder = store.slug(request.match_info["name"])
    mine = await request.json()
    client = request.query.get("client")
    async with lock(folder):
        current = read_project(folder)
        based_on = mine.get("revision", 0)
        if based_on == current.get("revision", 0):
            base = current
        else:
            base = read_revision(folder, based_on) or store.UNKNOWN
        try:
            merged = store.merge_editor_save(base, mine, current)
        except ValueError as error:
            return refused(plain(error))
        if merged != current:
            commit(folder, merged, "edit", client)
        else:
            merged = current
    return web.json_response({"revision": merged.get("revision", 0), "project": merged,
                              "merged": based_on != current.get("revision", 0)})


async def summary(request):
    return web.json_response(store.describe(read_project(request.match_info["name"])))


CUT_SCORES = {}


async def cuts(request):
    """A score for every cut of the timeline: each place where one rendered clip ends on the frame the next begins.

    The numbers come from the pictures alone (see cuts.py), so the same takes and the same trims give the same scores.
    """
    data = read_project(request.match_info["name"])
    found, edges, inside, heard, arrivals, lips, places = await asyncio.gather(
        score_cuts(data), score_effect_cuts(data), score_inside(data), score_sound_inside(data), score_arrivals(data),
        score_lips(data), score_scene(data))
    scored = [c["score"] for c in found + edges + inside + heard + arrivals + lips + places if c["score"] is not None]
    scored += [c["sound"]["score"] for c in found if c.get("sound")]
    scored += [c["movement"]["score"] for c in found if c.get("movement") and c["movement"]["score"] is not None]
    scored += [c["parts"]["score"] for c in found if c.get("parts") and c["parts"]["score"] is not None]
    return web.json_response({"cuts": found, "effects": edges, "inside": inside, "sound_inside": heard, "arrivals": arrivals,
                              "lips": lips, "lips_missing": lipscore.missing(),
                              "scene": places, "scene_missing": scenescore.missing(), "scene_off": scenescore.broken(),
                              "worst": min(scored) if scored else None, "revision": data.get("revision", 0)})


SOUND_SCORES = {}


async def cached(store_in, key, work, *args):
    """A score worked out once per key, off the event loop."""
    if key not in store_in:
        store_in[key] = await asyncio.to_thread(work, *args)
        while len(store_in) > 2000:
            store_in.pop(next(iter(store_in)))
    return store_in[key]


async def score_sound_inside(data):
    """Where the sound of a generated take on the timeline jumps, drops out, or ends far louder or quieter than it
    began (see listen.py), within the part its clip shows."""
    gate = asyncio.Semaphore(3)

    async def one(item):
        clip = store.find_clip(data, item["clip"]) or {}
        take = store.active_take(clip)
        if not take or clip.get("kind") == "footage":
            return []
        try:
            path = output_path(take.get("subfolder"), take["file"])
            async with gate:
                rows = await cached(SOUND_SCORES, ("inside", path, os.path.getmtime(path)), soundscore.inside, path, media.command)
        except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError, web.HTTPException):
            return []
        speed = store.clean_speed(item.get("speed"))
        return [{"item": item["id"], "take": take["id"], "frame": row["at"], "at": item["start"] + int(round((max(row["at"], item["in"]) - item["in"]) / speed)),
                 "kind": row["kind"], "score": row["score"], "notes": row["notes"]}
                for row in rows if row["kind"] in ("swell", "fade") or item["in"] <= row["at"] < item["out"]]

    return [row for rows in await asyncio.gather(*[one(item) for item in store.ordered(data)]) for row in rows]


LIP_GATE = None


async def score_lips(data):
    """Whether the lips of each generated clip with words or a track follow its sound, over the part the timeline shows
    (see lips.py). Nothing when SyncNet's files are not installed. One clip at a time: it runs on the CPU."""
    global LIP_GATE
    if lipscore.missing():
        return []
    LIP_GATE = LIP_GATE or asyncio.Semaphore(1)

    async def one(item):
        clip = store.find_clip(data, item["clip"]) or {}
        take = store.active_take(clip)
        if not take or clip.get("kind") == "footage" or not (str(clip.get("dialogue") or "").strip() or clip.get("track")):
            return None
        try:
            path = output_path(take.get("subfolder"), take["file"])
            key = ("lips", path, os.path.getmtime(path), item["in"], item["out"])
            async with LIP_GATE:
                row = await cached(SOUND_SCORES, key, lipscore.score, path, item["in"], item["out"], media.command)
        except FileNotFoundError:
            return None                 # a take whose file is gone has nothing to check
        except Exception as error:      # one check failing never takes the cuts report down with it
            logging.warning("H3 Editor: the lip check could not read %s: %s", take.get("file"), error)
            return None
        if not row:
            return None
        return dict(row, item=item["id"], take=take["id"], at=item["start"])

    return [row for row in await asyncio.gather(*[one(item) for item in store.ordered(data)]) if row]


SCENE_GATE = None


async def score_scene(data):
    """Whether the place behind the people holds in each generated clip on the timeline (see scene.py), over the part
    the timeline shows. An extension is also held against the first frame its shot shows, so a chain that turns the
    room into another is caught where it happens. Nothing when MODNet's file is not installed. One clip at a time."""
    global SCENE_GATE
    if scenescore.missing() or scenescore.broken():
        return []
    SCENE_GATE = SCENE_GATE or asyncio.Semaphore(1)
    shown = {}
    for item in store.ordered(data):
        clip = store.find_clip(data, item["clip"]) or {}
        take = store.active_take(clip)
        if not take or clip.get("kind") == "footage":
            continue
        try:
            shown[item["id"]] = (item, output_path(take.get("subfolder"), take["file"]), take)
        except (ValueError, web.HTTPException):
            continue
    first_item = {}
    for item, _, _ in shown.values():
        first_item.setdefault(item["clip"], item["id"])

    async def one(item, path, take):
        root = first_item.get(shot_root(data, item["clip"]))
        anchor = None
        if root and root != item["id"] and root in shown:
            head, head_path, _ = shown[root]
            anchor = (head_path, head["in"])
        try:
            key = ("scene", path, os.path.getmtime(path), item["in"], item["out"],
                   anchor and (anchor[0], os.path.getmtime(anchor[0]), anchor[1]))
            async with SCENE_GATE:
                row = await cached(SOUND_SCORES, key, scenescore.score, path, item["in"], item["out"], media.command, anchor)
        except FileNotFoundError:
            return None
        except Exception as error:      # one check failing never takes the cuts report down with it
            if scenescore.broken():
                logging.warning("H3 Editor: the place check is off: %s", scenescore.broken())
            else:
                logging.warning("H3 Editor: the place check could not read %s: %s", take.get("file"), error)
            return None
        if not row:
            return None
        speed = store.clean_speed(item.get("speed"))
        row = {k: v for k, v in row.items() if k != "seconds"}
        return dict(row, item=item["id"], take=take["id"], since_shot=anchor is not None,
                    at=item["start"] + int(round((max(row["frame"], item["in"]) - item["in"]) / speed)))

    return [row for row in await asyncio.gather(*[one(*entry) for entry in shown.values()]) if row]


async def score_arrivals(data):
    """How each bridge that ends on a pinned frame of the next clip gets there (see cuts.judge_arrival)."""
    async def one(item):
        clip = store.find_clip(data, item["clip"]) or {}
        take, link = store.active_take(clip), clip.get("link")
        if not take or not store.arrives_on_pin(link) or not link.get("b"):
            return None
        side = link["b"]
        try:
            path, target = output_path(take.get("subfolder"), take["file"]), output_path(side.get("subfolder"), side["file"])
            key = ("arrival", path, os.path.getmtime(path), target, os.path.getmtime(target), side["first"])
            row = await cached(SOUND_SCORES, key, cutscore.score_arrival, path, int(take["frames"]), target, int(side["first"]), media.command)
        except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError, web.HTTPException):
            return None
        return dict(row, item=item["id"], take=take["id"], at=store.item_end(item))

    return [row for row in await asyncio.gather(*[one(item) for item in store.ordered(data)]) if row]


async def score_cuts(data):
    """The score of every cut of a project's timeline, in order."""
    order = store.ordered(data)
    gate = asyncio.Semaphore(4)

    async def one(before, after):
        entry = {"before": before["id"], "after": after["id"], "at": after["start"]}
        takes = [store.active_take(store.find_clip(data, item["clip"]) or {}) for item in (before, after)]
        if not all(takes):
            return dict(entry, kind="not rendered", score=None, notes=[])
        try:
            paths = [output_path(take.get("subfolder"), take["file"]) for take in takes]
            key = (paths[0], os.path.getmtime(paths[0]), before["out"], paths[1], os.path.getmtime(paths[1]), after["in"])
            if key not in CUT_SCORES:
                async with gate:
                    CUT_SCORES[key] = await asyncio.to_thread(cutscore.score_cut, paths[0], before["out"], paths[1], after["in"], media.command)
                while len(CUT_SCORES) > 2000:
                    CUT_SCORES.pop(next(iter(CUT_SCORES)))
            entry = dict(entry, **CUT_SCORES[key])
            try:
                # What the cut sounds like, beside what it looks like.
                async with gate:
                    entry["sound"] = await cached(SOUND_SCORES, ("cut",) + key, soundscore.score_cut, paths[0], before["out"], paths[1], after["in"], media.command)
            except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError):
                entry["sound"] = None
            try:
                # Whether the camera's move carries across it. Scored inside one shot; described at a new one.
                async with gate:
                    entry["movement"] = await cached(SOUND_SCORES, ("move",) + key, cutscore.score_movement, paths[0], before["out"], paths[1],
                                                     after["in"], entry["kind"] != "new shot", media.command)
            except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError):
                entry["movement"] = None
            try:
                # Whether what moves inside the frame carries across it, with the camera's move taken out.
                async with gate:
                    entry["parts"] = await cached(SOUND_SCORES, ("parts",) + key, cutscore.score_parts, paths[0], before["out"], paths[1],
                                                  after["in"], entry["kind"] != "new shot", media.command)
            except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError):
                entry["parts"] = None
            return entry
        except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError, web.HTTPException) as error:
            return dict(entry, kind="unreadable", score=None, notes=[str(getattr(error, "text", None) or error)[:200]])

    pairs = [(a, b) for a, b in zip(order[:-1], order[1:]) if store.item_end(a) == b["start"]]
    return await asyncio.gather(*[one(a, b) for a, b in pairs])


def shown_at(data, frame, skip=None):
    """What the cut shows at a timeline frame: (file, frame of that file), or None for black.

    A VFX clip's take covers the picture under it, so it wins. skip names a VFX clip to look under.
    """
    for effect in data.get("effects") or []:
        if effect["id"] != skip and effect["start"] <= frame < effect["start"] + effect["frames"]:
            take = store.active_take(effect)
            if take:
                return output_path(take.get("subfolder"), take["file"]), frame - effect["start"]
    for item in store.ordered(data):
        if item["start"] <= frame < store.item_end(item):
            take = store.active_take(store.find_clip(data, item["clip"]) or {})
            if not take:
                return None
            speed = store.clean_speed(item.get("speed"))
            return output_path(take.get("subfolder"), take["file"]), item["in"] + int(round((frame - item["start"]) * speed))
    return None


async def score_effect_cuts(data):
    """A score for the cut into and out of every rendered VFX clip: what the cut shows just before against the clip's
    first frame, and its last frame against what comes after. Black on the other side is not a cut to score."""
    found = []
    for effect in data.get("effects") or []:
        take = store.active_take(effect)
        if not take:
            continue
        first, last = effect["start"], effect["start"] + effect["frames"]
        try:
            own = output_path(take.get("subfolder"), take["file"])
            sides = (("start", first, shown_at(data, first - 1, effect["id"]) if first > 0 else None, (own, 0)),
                     ("end", last, (own, effect["frames"] - 1), shown_at(data, last, effect["id"])))
        except (ValueError, web.HTTPException):
            continue
        for edge, at, before, after in sides:
            if not before or not after:
                continue
            entry = {"effect": effect["id"], "take": take["id"], "edge": edge, "at": at}
            try:
                key = (before[0], os.path.getmtime(before[0]), before[1] + 1, after[0], os.path.getmtime(after[0]), after[1])
                if key not in CUT_SCORES:
                    CUT_SCORES[key] = await asyncio.to_thread(cutscore.score_cut, before[0], before[1] + 1, after[0], after[1], media.command)
                found.append(dict(entry, **CUT_SCORES[key]))
            except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
                found.append(dict(entry, kind="unreadable", score=None, notes=[str(getattr(error, "text", None) or error)[:200]]))
    return found


INSIDE_SCORES = {}


async def score_inside(data):
    """Where a generated take on the timeline changes picture within the part its clip shows: a cut inside it, or a
    fade to another picture. And where it stops moving: a frozen stretch (a held frame) or a still one, with frames and
    length (on the timeline) saying how long. Imported footage is left alone: it may hold cuts and stills of its own."""
    gate = asyncio.Semaphore(3)

    async def one(item):
        clip = store.find_clip(data, item["clip"]) or {}
        take = store.active_take(clip)
        if not take or clip.get("kind") == "footage":
            return []
        try:
            path = output_path(take.get("subfolder"), take["file"])
            key = (path, os.path.getmtime(path))
            if key not in INSIDE_SCORES:
                async with gate:
                    INSIDE_SCORES[key] = await asyncio.to_thread(cutscore.inside, path, media.command)
                while len(INSIDE_SCORES) > 2000:
                    INSIDE_SCORES.pop(next(iter(INSIDE_SCORES)))
        except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError, web.HTTPException):
            return []
        speed = store.clean_speed(item.get("speed"))
        found = []
        for row in INSIDE_SCORES[key]:
            if "frames" in row:
                # A frozen or still stretch: the part of it the clip shows, if that part is still long enough to report.
                first, last = max(row["at"], item["in"]), min(row["at"] + row["frames"], item["out"])
                least = cutscore.FROZEN_RUN if row["kind"] == "frozen" else cutscore.STILL_RUN
                if last - first < least:
                    continue
                found.append({"item": item["id"], "take": take["id"], "frame": first, "frames": last - first,
                              "at": item["start"] + int(round((first - item["in"]) / speed)), "length": max(1, int(round((last - first) / speed))),
                              "kind": row["kind"], "score": row["score"], "notes": row["notes"]})
            elif item["in"] <= row["at"] < item["out"]:
                found.append({"item": item["id"], "take": take["id"], "frame": row["at"], "at": item["start"] + int(round((row["at"] - item["in"]) / speed)),
                              "kind": row["kind"], "score": row["score"], "notes": row["notes"]})
        return found

    return [row for rows in await asyncio.gather(*[one(item) for item in store.ordered(data)]) for row in rows]


FINE_SCORES = {}


async def detail(request):
    """How sharp each clip of the timeline is at the size of the cut, and which are soft next to the rest.

    A clip is soft when its fine detail is under half the level of the cut: a draft among finals, or small footage.
    The numbers come from the pictures alone (see detail.py).
    """
    data = read_project(request.match_info["name"])
    found = await score_detail(data)
    return web.json_response(dict(found, revision=data.get("revision", 0)))


async def score_detail(data):
    """The fine detail of every rendered clip on a project's timeline, in order."""
    gate = asyncio.Semaphore(3)
    shown = []
    for item in store.ordered(data):
        take = store.active_take(store.find_clip(data, item["clip"]) or {})
        if not take:
            continue
        try:
            shown.append((item, output_path(take.get("subfolder"), take["file"]), store.take_size(data, take)))
        except (ValueError, web.HTTPException):
            continue
    if not shown:
        return {"level": None, "size": None, "clips": []}
    width, height = max((size for _, _, size in shown), key=lambda size: size[0])      # the cut takes its largest picture

    async def one(item, path):
        try:
            key = (path, os.path.getmtime(path), item["in"], item["out"], width, height)
            if key not in FINE_SCORES:
                async with gate:
                    FINE_SCORES[key] = await asyncio.to_thread(finescore.clip_fine, path, item["in"], item["out"], width, height, media.command)
                while len(FINE_SCORES) > 2000:
                    FINE_SCORES.pop(next(iter(FINE_SCORES)))
            return FINE_SCORES[key]
        except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError):
            return None

    async def harsh(item, path):
        try:
            key = (path, os.path.getmtime(path), item["in"], item["out"], "texture")
            if key not in FINE_SCORES:
                async with gate:
                    FINE_SCORES[key] = await asyncio.to_thread(finescore.clip_texture, path, item["in"], item["out"], media.command)
            return FINE_SCORES[key]
        except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError):
            return None

    # A shot is a take and the clips that extend it, one after another. Each extension is held against the first.
    first_item = {}
    for item, _, _ in shown:
        first_item.setdefault(item["clip"], item["id"])
    roots = {item["id"]: first_item.get(shot_root(data, item["clip"])) for item, _, _ in shown}
    chained = {key for key, root in roots.items() if root and root != key} | {root for key, root in roots.items() if root and root != key}
    values = await asyncio.gather(*[one(item, path) for item, path, _ in shown])
    textures = await asyncio.gather(*[harsh(item, path) if item["id"] in chained else asyncio.sleep(0) for item, path, _ in shown])
    found = finescore.soft_clips([(item["id"], value, item["out"] - item["in"]) for (item, _, _), value in zip(shown, values)])
    moved = finescore.drifting([(item["id"], roots[item["id"]] if roots[item["id"]] != item["id"] else None, value)
                                for (item, _, _), value in zip(shown, textures)])
    clips = []
    for row, (item, _, _) in zip(found["clips"], shown):
        row = dict(row, item=row.pop("id"), at=item["start"])
        if row["item"] in moved:
            row["drift"] = dict(moved[row["item"]], limit=finescore.DRIFT)
        clips.append(row)
    return {"level": found["level"], "size": [width, height], "clips": clips}


def shot_root(data, clip_id):
    """The clip a chain of extensions started from: the clip itself when it extends nothing."""
    seen = set()
    while clip_id not in seen:
        seen.add(clip_id)
        try:
            link = store.find_clip(data, clip_id).get("link") or {}
        except KeyError:
            break
        source = (link.get("a") or {}).get("clip")
        if link.get("mode") not in ("extend_after", "extend_before") or not source:
            break
        clip_id = source
    return clip_id


EFFECT_FRAME_LIMIT = 8 * 1024 * 1024     # one frame of a VFX render, as a JPEG


def effect_job_folder(name, job):
    """Where the frames of a VFX render collect until it is made into a take."""
    if not job or not all(ch.isalnum() for ch in job) or len(job) > 40:
        raise web.HTTPBadRequest(text="job is the render's id: letters and numbers")
    return os.path.join(root(), name, "clips", ".fx_" + job)


async def effect_frames(request):
    """Take a batch of rendered frames of a VFX clip. The body is JPEGs, each after its length as 4 bytes, big end first."""
    name = store.slug(request.match_info["name"])
    data = read_project(name)
    try:
        store.find_effect(data, request.match_info["effect"])
    except KeyError as error:
        raise web.HTTPNotFound(text=str(error))
    folder = effect_job_folder(name, request.query.get("job"))
    os.makedirs(folder, exist_ok=True)
    body = await request.read()
    count = len([f for f in os.listdir(folder) if f.endswith(".jpg")])
    at = 0
    while at < len(body):
        if at + 4 > len(body):
            raise web.HTTPBadRequest(text="the frames end in the middle of a length")
        size = int.from_bytes(body[at:at + 4], "big")
        at += 4
        if size < 4 or size > EFFECT_FRAME_LIMIT or at + size > len(body) or body[at:at + 2] != b"\xff\xd8":
            raise web.HTTPBadRequest(text="frame %s is not a JPEG of a sensible size" % count)
        with open(os.path.join(folder, "f_%05d.jpg" % count), "wb") as out:
            out.write(body[at:at + size])
        at += size
        count += 1
    return web.json_response({"frames": count})


async def effect_take(request):
    """Make the collected frames of a VFX render into the clip's take. {"job"} names the render, {"error"} records a failure."""
    name = store.slug(request.match_info["name"])
    effect_id = request.match_info["effect"]
    try:
        body = await request.json()
    except ValueError:
        body = {}
    body = body if isinstance(body, dict) else {}
    data = read_project(name)
    try:
        effect = store.find_effect(data, effect_id)
    except KeyError as error:
        raise web.HTTPNotFound(text=str(error))
    if body.get("claim"):
        async with lock(name):
            data = read_project(name)
            claimed = store.claim_effect_render(data, effect_id, body["claim"])
            if claimed:
                commit(name, data, "render")
        return web.json_response({"claimed": claimed})
    if body.get("error"):
        async with lock(name):
            data = read_project(name)
            failed = store.find_effect(data, effect_id)
            failed["error"] = str(body["error"])[:600]
            failed.pop("queued", None)
            commit(name, data, "render")
        return web.json_response({"revision": data.get("revision", 0), "project": data})
    folder = effect_job_folder(name, body.get("job"))
    frames = sorted(f for f in os.listdir(folder) if f.endswith(".jpg")) if os.path.isdir(folder) else []
    if len(frames) != effect["frames"]:
        shutil.rmtree(folder, ignore_errors=True)
        raise web.HTTPBadRequest(text="the render has %s frames and the clip %s" % (len(frames), effect["frames"]))
    take_id = "t" + uuid.uuid4().hex[:8]
    # Named from an id made here, never from the clip's own id, which a saved project could have set to anything.
    file = "fx_%s.mp4" % take_id
    clips_folder = os.path.realpath(os.path.join(root(), name, "clips"))
    target = os.path.realpath(os.path.join(clips_folder, file))
    if os.path.dirname(target) != clips_folder:
        raise web.HTTPBadRequest(text="the take would be saved outside the project")
    try:
        await run(["ffmpeg", "-y", "-v", "error", "-framerate", str(store.FPS), "-i", os.path.join(folder, "f_%05d.jpg"),
                   "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2:out_color_matrix=bt709,format=yuv420p", "-c:v", "libx264", "-crf", "14",
                   "-preset", "medium", "-movflags", "+faststart", target])
        info = await probe_file(target)
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    take = {"id": take_id, "file": file, "subfolder": "%s/%s/clips" % (store.ROOT, name), "frames": info["frames"],
            "poster": info["poster"], "strip": info["strip"]}
    async with lock(name):
        data = read_project(name)
        store.accept_effect_take(data, effect_id, take)
        commit(name, data, "render")
    return web.json_response({"revision": data.get("revision", 0), "project": data, "take": take_id})


def settings_file():
    """Where the editor keeps what belongs to this installation and to no project: for now, the user's Gemini key."""
    return os.path.join(folder_paths.get_user_directory(), "h3take", "settings.json")


def looks_file():
    return os.path.join(folder_paths.get_user_directory(), "h3take", "looks.json")


async def look_library(request):
    try:
        return web.json_response({"looks": looks.library(looks_file())})
    except (OSError, ValueError) as error:
        return refused("could not read the look library: %s" % error)


async def save_look(request):
    body = await request.json()
    if not isinstance(body, dict):
        return refused("send a look name and style")
    async with lock("_looks"):
        try:
            entry = looks.save(looks_file(), body.get("name"), body.get("style"), body.get("id"))
            return web.json_response({"look": entry, "looks": looks.library(looks_file())})
        except (OSError, ValueError) as error:
            return refused(str(error))


async def remove_look(request):
    async with lock("_looks"):
        try:
            looks.remove(looks_file(), request.match_info["look"])
            return web.json_response({"looks": looks.library(looks_file())})
        except (OSError, ValueError) as error:
            return refused(str(error))


def read_settings():
    try:
        with open(settings_file(), encoding="utf-8") as file:
            found = json.load(file)
        return found if isinstance(found, dict) else {}
    except (OSError, ValueError):
        return {}


def gemini_key():
    """The key ratings are made with: the one saved in the editor, else the GEMINI_API_KEY of the environment."""
    return str(read_settings().get("gemini_key") or os.environ.get("GEMINI_API_KEY") or "").strip()


def settings_reply():
    saved = bool(str(read_settings().get("gemini_key") or "").strip())
    return {"gemini": {"set": bool(gemini_key()), "source": "saved" if saved else "environment" if gemini_key() else None}}


async def settings(request):
    """Whether a Gemini key is set. The key itself is never sent back."""
    return web.json_response(settings_reply())


async def save_settings(request):
    """Save or clear the Gemini key: {"gemini_key": "..."}, empty to clear."""
    body = await request.json()
    if not isinstance(body, dict) or not isinstance(body.get("gemini_key"), str):
        return refused("send gemini_key, the key as text, or empty text to remove it")
    key = body["gemini_key"].strip()
    if len(key) > 400 or any(ch.isspace() for ch in key):
        return refused("that does not look like an API key")
    found = read_settings()
    if key:
        found["gemini_key"] = key
    else:
        found.pop("gemini_key", None)
    path = settings_file()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_json(path, found)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return web.json_response(settings_reply())


async def rate(request):
    """Rate a range of the cut against a rubric with Gemini Decision. {"from", "to"} are timeline frames.

    The range is exported small, sent to Google with the user's own key, rated, and deleted there. Optional: "rubric"
    (a list of {key, title?, instructions, levels}, worst level first), "resolution" (low, medium, high) and "note".
    """
    from . import judge

    name = store.slug(request.match_info["name"])
    data = read_project(name)
    try:
        body = await request.json()
    except ValueError:
        body = {}
    body = body if isinstance(body, dict) else {}
    end = store.scene_end(data)
    try:
        first = int(body.get("from", 0))
        last = min(end, int(body.get("to", end)))
        if first < 0 or last - first < store.FPS // 2:
            return refused("the range must hold at least half a second of the cut; the cut ends at frame %s" % end)
        if last - first > store.RATING_FRAMES:
            return refused("a rating looks at up to %s seconds at a time; this range is %.1f" % (
                store.RATING_FRAMES // store.FPS, (last - first) / float(store.FPS)))
        rubric = judge.clean_rubric(body.get("rubric"))
        part = store.slice_project(data, first, last)
    except (TypeError, ValueError) as error:
        return refused(plain(error))
    key = gemini_key()
    if not key:
        return refused("No Gemini API key is set. Add one in the editor (Rate, then Key), or set GEMINI_API_KEY for ComfyUI.", 409)

    def path_of(holder):
        try:
            found = output_path(holder.get("subfolder"), holder["file"])
        except web.HTTPException:
            return None
        return found if os.path.exists(found) else None

    paths = {}
    for clip in data.get("clips", []):
        take = store.active_take(clip)
        if take and path_of(take):
            paths[clip["id"]] = path_of(take)
    sound_paths = {s["id"]: path_of(s) for s in part.get("sounds") or [] if path_of(s)}
    effect_paths = {}
    for effect in part.get("effects") or []:
        take = store.active_take(effect)
        if take and path_of(take):
            effect_paths[effect["id"]] = path_of(take)
    folder = os.path.join(root(), name, "ratings")
    os.makedirs(folder, exist_ok=True)
    target = os.path.join(folder, "range_%s.mp4" % uuid.uuid4().hex[:10])
    try:
        try:
            args = store.export_command(part, paths, target, scale=0.5, quality="small", sound_paths=sound_paths, effect_paths=effect_paths)
        except ValueError as error:
            return refused(plain(error))
        await run(args)
        if os.path.getsize(target) > 300 * 1024 * 1024:
            return refused("that range makes too large a file to send; rate a shorter one")
        try:
            pictures = []
            if body.get("reference", True):
                # The first picture of each cast member seen in the range, so the model can compare against it.
                seen = {i["clip"] for i in part["timeline"]}
                wanted = {m for c in data.get("clips", []) if c["id"] in seen for m in (c.get("cast") or [])}
                for person in data.get("cast", []):
                    held = store.member_pictures(person)
                    if person["id"] in wanted and held and len(pictures) < 3 and path_of(held[0]):
                        pictures.append((person.get("name") or "the person", path_of(held[0])))
            result = await judge.rate(key, target, rubric, body.get("resolution") or "low", body.get("note") or "", pictures)
        except judge.JudgeError as error:
            return refused(str(error), 502)
        except (ValueError, OSError, asyncio.TimeoutError) as error:
            return refused("The rating could not be made: %s" % (plain(error) or type(error).__name__), 502)
    finally:
        if os.path.exists(target):
            os.remove(target)
    rows = result["rows"]
    if body.get("rubric") is None:
        # The model sees about one frame a second, too few to notice a jump cut. The editor measures every cut from
        # the frames on both sides, so that row comes from the measurement, not from the model.
        inside = [c for c in await score_cuts(data) if first < c["at"] < last and c["score"] is not None]
        if inside:
            worst = min(inside, key=lambda c: c["score"])
            why = "; ".join(worst.get("notes") or []) or worst["kind"]
            rows = rows + [{"key": "cuts", "title": "Smooth cuts", "measured": True, "score": round(worst["score"] / 25.0, 1), "of": 4,
                            "part": round(worst["score"] / 100.0, 4), "confidence": None,
                            "level": "Measured by the editor: the worst of %s cut%s scores %s of 100 (%s)" % (
                                len(inside), "" if len(inside) == 1 else "s", worst["score"], why)}]
    rating = {"from": first, "to": last, "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "resolution": body.get("resolution") or "low",
              "rows": rows, "tokens": result["tokens"]}
    async with lock(name):
        data = read_project(name)
        kept = store.add_rating(data, rating)
        commit(name, data, "rating")
    return web.json_response({"rating": kept, "revision": data.get("revision", 0), "project": data})


async def history(request):
    folder = store.slug(request.match_info["name"])
    kept = history_folder(folder)
    names = sorted((n for n in os.listdir(kept) if n.endswith(".json")), reverse=True) if os.path.isdir(kept) else []
    entries = []
    for name in names:
        with open(os.path.join(kept, name), encoding="utf-8") as f:
            entry = json.load(f)
        entries.append({"revision": entry["project"].get("revision"), "time": entry["time"], "source": entry["source"],
                        "clips": len(entry["project"].get("clips", [])), "frames": store.scene_end(entry["project"])})
    return web.json_response({"revisions": entries})


async def restore(request):
    """Go back to an earlier revision as a new one. Takes rendered since then are kept."""
    folder = store.slug(request.match_info["name"])
    body = await request.json()
    async with lock(folder):
        old = read_revision(folder, body.get("revision", -1))
        if old is None:
            return refused("revision %s is not kept any more" % body.get("revision"), 404)
        current = read_project(folder)
        if "if_revision" in body and body["if_revision"] != current.get("revision", 0):
            return refused("the project has changed since that edit", 409)
        restored = store.restore_revision(current, old)
        commit(folder, restored, "restore", body.get("client"))
    return web.json_response({"revision": restored["revision"], "summary": store.describe(restored)})


async def graph(request):
    name = request.match_info["name"]
    body = await request.json()
    try:
        built = store.render_graph(body["project"], store.slug(name), body["clip"], body["take"], body["quality"], body["seed"],
                                   available_models(), installed_patches())
    except ValueError as error:
        raise web.HTTPBadRequest(text=str(error))
    import nodes as comfy_nodes
    if store.CONTEXT_WINDOWS not in comfy_nodes.NODE_CLASS_MAPPINGS:
        # A ComfyUI from before context windows refines a long take whole, which needs far more memory.
        built = store.without_nodes(built, {store.CONTEXT_WINDOWS})
    return web.json_response({"graph": own_nodes(built)})


async def probe_file(path):
    """Frame count, a poster and a filmstrip for a finished render."""
    meta = await video_info(path)
    frames = meta["frames"]
    poster = os.path.splitext(path)[0] + ".jpg"
    frame = max(0, min(12, frames - 1))
    await run(["ffmpeg", "-y", "-v", "error", "-i", path, "-vf", "select=eq(n\\,%d),scale=480:-2" % frame, "-frames:v", "1", poster])
    try:
        made = await make_strip(path, frames)
    except web.HTTPException:
        made = None
    out = {"frames": frames, "poster": os.path.basename(poster), "strip": made}
    if meta.get("width") and meta.get("height"):
        out["size"] = [int(meta["width"]), int(meta["height"])]
    return out


async def probe(request):
    body = await request.json()
    return web.json_response(await probe_file(output_path(body.get("subfolder"), body["filename"])))


# Rendering. Renders are queued here, not in the browser, so a finished take reaches the project
# even when no editor is open, and an agent can render without a browser at all.

class Refused(Exception):
    pass


def plain(error):
    return str(error).strip("'\"")


def comfy_base():
    from comfy.cli_args import args

    host = (args.listen or "127.0.0.1").split(",")[0].strip()
    if host in ("", "0.0.0.0", "::"):
        host = "127.0.0.1"
    if ":" in host:
        host = "[%s]" % host
    scheme = "https" if getattr(args, "tls_keyfile", None) and getattr(args, "tls_certfile", None) else "http"
    return "%s://%s:%s" % (scheme, host, args.port)


async def comfy(method, path, body=None):
    import aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.request(method, comfy_base() + path, json=body, ssl=False) as response:
            text = await response.text()
            try:
                data = json.loads(text) if text else None
            except ValueError:
                data = {"error": {"message": text[:300]}}
            return response.status, data


def own_nodes(graph):
    """The graph with the pack's nodes named by their twins, ids no other pack registers.

    Another pack that also registers a node called H3LivePreview or H3Lora replaces ours under
    that id. The twins keep the editor's renders on this pack's own nodes whatever else is installed.
    """
    import nodes as comfy_nodes

    out = {}
    for key, node in graph.items():
        twin = editor_id(str(node.get("class_type", "")))
        out[key] = dict(node, class_type=twin) if twin in comfy_nodes.NODE_CLASS_MAPPINGS else node
    return out


async def queue_prompt(graph, client_id=None):
    from . import nodes

    graph = own_nodes(graph)

    # Only a .gguf needs the gguf package. The INT8 safetensors load without it.
    if nodes.HYPERFLOW_MISSING and any(
        n.get("class_type") in ("H3HyperFlowLoader", "H3EditorHyperFlowLoader") and str(n.get("inputs", {}).get("unet_name", "")).endswith(".gguf")
        for n in graph.values()
    ):
        raise Refused(nodes.HYPERFLOW_MISSING)
    payload = {"prompt": graph}
    if client_id:
        payload["client_id"] = client_id
    status, data = await comfy("POST", "/prompt", payload)
    data = data or {}
    if status != 200 or not data.get("prompt_id"):
        error = data.get("error") or {}
        message = (error.get("message") if isinstance(error, dict) else str(error)) or "ComfyUI refused the render"
        for node, found in (data.get("node_errors") or {}).items():
            for problem in found.get("errors", [])[:1]:
                message = "%s. %s: %s %s" % (message, found.get("class_type", node), problem.get("message", ""),
                                            problem.get("details", ""))
            break
        raise Refused(message.strip())
    return data["prompt_id"]


def comfy_queue():
    from server import PromptServer

    running, queued = PromptServer.instance.prompt_queue.get_current_queue()
    return {entry[1] for entry in running}, {entry[1] for entry in queued}


SPRITE_EXPORTS = {}
PROP_JOBS = {}
QUEUE_HISTORY = []
MAX_QUEUE_HISTORY = 25


def _record_queue_history(entry):
    """Record a finished, failed, or cancelled job for the editor's managed render queue UI."""
    item = dict(entry)
    item.setdefault("at", time.time())
    job_id = item.get("id")
    if job_id:
        QUEUE_HISTORY[:] = [h for h in QUEUE_HISTORY if h.get("id") != job_id]
    QUEUE_HISTORY.insert(0, item)
    del QUEUE_HISTORY[MAX_QUEUE_HISTORY:]


def _project_pending_lookup():
    """Map prompt_id -> enriched job metadata across all projects on disk."""
    found = {}
    try:
        names = os.listdir(root())
    except OSError:
        return found
    for folder in names:
        path = os.path.join(root(), folder, "project.json")
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            continue
        pending = data.get("pending") or []
        if not pending:
            continue
        clips = {c.get("id"): c for c in data.get("clips") or [] if isinstance(c, dict)}
        project_name = data.get("name") or folder
        for job in pending:
            pid = job.get("id")
            if not pid:
                continue
            clip = clips.get(job.get("clip")) or {}
            sprite_cfg = (job.get("saved") or {}).get("sprite") or clip.get("sprite")
            kind = "sprite" if sprite_cfg else "clip"
            found[pid] = {
                "id": pid,
                "kind": kind,
                "folder": folder,
                "project": project_name,
                "clip": job.get("clip"),
                "name": clip.get("name") or job.get("clip") or "Clip",
                "quality": job.get("quality") or "draft",
                "seed": job.get("seed"),
                "size": job.get("size"),
                "frames": job.get("frames"),
                "part": job.get("part"),
                "edit": bool(job.get("edit")),
                "upscaler": job.get("upscaler"),
                "sprite": {
                    "direction": sprite_cfg.get("direction"),
                    "camera": sprite_cfg.get("camera"),
                    "motion": sprite_cfg.get("animation") or sprite_cfg.get("motion") or "run",
                } if isinstance(sprite_cfg, dict) else None,
            }
    return found


def _reconcile_prop_jobs(active_ids):
    """Move completed or failed H3-Image prop jobs out of PROP_JOBS into QUEUE_HISTORY."""
    from server import PromptServer

    if not PROP_JOBS:
        return
    queue = PromptServer.instance.prompt_queue
    for pid in list(PROP_JOBS):
        if pid in active_ids:
            continue
        meta = PROP_JOBS.pop(pid, None)
        if not meta:
            continue
        record = queue.get_history(prompt_id=pid).get(pid)
        err = None
        state = "done"
        if record:
            status = record.get("status") or {}
            if status.get("status_str") == "error":
                state = "failed"
                err = failure(record)
            elif any(msg[0] == "execution_interrupted" for msg in (status.get("messages") or [])):
                state = "cancelled"
        _record_queue_history({**meta, "state": state, "error": err})


def build_queue_snapshot():
    """Return ComfyUI's live running + pending queue enriched with H3 Editor project/prop metadata."""
    from server import PromptServer

    raw_running, raw_pending = PromptServer.instance.prompt_queue.get_current_queue()
    pending_sorted = sorted(raw_pending, key=lambda entry: entry[0])
    running_sorted = sorted(raw_running, key=lambda entry: entry[0])
    active_ids = {entry[1] for entry in running_sorted} | {entry[1] for entry in pending_sorted}
    _reconcile_prop_jobs(active_ids)
    lookup = _project_pending_lookup()

    def describe(entry, state, position):
        pid = entry[1]
        if pid in lookup:
            return {**lookup[pid], "state": state, "position": position}
        if pid in PROP_JOBS:
            return {**PROP_JOBS[pid], "state": state, "position": position}
        return {
            "id": pid,
            "kind": "comfyui",
            "folder": None,
            "project": None,
            "clip": None,
            "name": "ComfyUI workflow",
            "quality": None,
            "state": state,
            "position": position,
        }

    return {
        "running": [describe(e, "running", idx) for idx, e in enumerate(running_sorted)],
        "pending": [describe(e, "queued", idx + 1) for idx, e in enumerate(pending_sorted)],
        "history": list(QUEUE_HISTORY),
    }


def sprite_export_fps(body):
    fps = body.get("fps")
    if fps is not None and (type(fps) is not int or fps != 24):
        raise ValueError("Choose the original frame rate or 24 fps.")
    return fps


async def make_sprite_export(take, variant, folder, stem, names, export_24fps):
    """Repack a sprite take. Reuses lossless cached RGBA frames when present; falls back to ComfyUI otherwise."""
    from server import PromptServer

    sheet = take["sprite"]
    poses = sheet.get("source_poses", sheet["poses"])
    tile_w, tile_h = store.sprite.validate_tile_size(variant.get("tile_size") or variant["tile"])
    total_frames = int(take.get("frames") or sheet.get("frames") or store.sprite.FRAMES)
    canvas = sheet.get("canvas") or "portrait"
    idle = sheet["poses"] == 2 * poses - 2
    cache_name = sheet.get("frame_cache")
    if cache_name:
        try:
            cache_path = output_path(take.get("subfolder"), cache_name)
        except web.HTTPException:
            cache_path = None
        if cache_path and os.path.isdir(cache_path):
            rendered = await asyncio.to_thread(
                store.sprite.export_from_cache,
                cache_path, sheet["layout"], poses, (tile_w, tile_h),
                idle=idle, export_24fps=export_24fps, total_frames=total_frames, canvas=canvas,
            )
            if rendered is not None:
                full, indexed = rendered
                for style, img in (("illustrated", full), ("indexed", indexed)):
                    dest = output_path(folder, names[style])
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    tmp = dest + ".tmp"
                    await asyncio.to_thread(img.save, tmp, format="PNG")
                    os.replace(tmp, dest)
                return

    prefix = stem + ".partial-" + uuid.uuid4().hex
    sheet_inputs = {
        "images": ["1", 0], "poses": poses, "layout": json.dumps(sheet["layout"]),
        "cell_size": str(tile_w if tile_w == tile_h and tile_w in store.sprite.EXPORT_SIZES else store.sprite.TILE),
        "tile_width": tile_w, "tile_height": tile_h, "canvas": canvas,
        "export_24fps": export_24fps, "save_cache": False,
        "idle": idle, "filename_prefix": folder + "/" + prefix,
    }
    graph = {"1": {"class_type": "H3LoadClip", "inputs": {
        "file": "/".join(filter(None, (take.get("subfolder"), take["file"]))), "first": 0,
        "last": total_frames, "frames_only": True}},
        "2": {"class_type": "H3SpriteSheet", "inputs": sheet_inputs}}
    if sheet.get("cutout") == "foreground model":
        model = store.sprite.matte_model(available_models())
        if not model:
            raise Refused("This take used BiRefNet. Install its background removal model to export this variant.")
        graph["3"] = {"class_type": "LoadBackgroundRemovalModel", "inputs": {"bg_removal_name": model}}
        graph["2"]["inputs"]["bg_model"] = ["3", 0]
    temporary = {style: output_path(folder, prefix + "." + style + ".png") for style in names}
    try:
        job = await queue_prompt(graph)
        queue = PromptServer.instance.prompt_queue
        while True:
            running, waiting = comfy_queue()
            record = queue.get_history(prompt_id=job).get(job)
            if record:
                break
            if job not in running and job not in waiting:
                raise Refused("The sprite export left the queue before saving. Retry the export.")
            await asyncio.sleep(.25)
        result = record.get("outputs", {}).get("2", {}).get("sprite")
        if not result:
            raise Refused(failure(record))
        if (any(result[0].get(k) != variant[k] for k in ("poses", "fps", "tile"))
                or not all(os.path.isfile(p) for p in temporary.values())):
            raise Refused("The export did not produce both PNGs with the requested timing and size.")
        for style, path in temporary.items():
            os.replace(path, output_path(folder, names[style]))
    finally:
        for path in temporary.values():
            if os.path.isfile(path):
                os.remove(path)


async def sprite_export(request):
    name = store.slug(request.match_info["name"])
    body = await request.json()
    raw_tile = body.get("tile_size") if isinstance(body, dict) and body.get("tile_size") is not None else (
        body.get("tile") if isinstance(body, dict) else None
    )
    try:
        tile = store.sprite.validate_tile_size(raw_tile)
    except ValueError as error:
        return refused(error)
    data = read_project(name)
    take = next((t for c in data["clips"] for t in c.get("takes", [])
                 if t["id"] == body.get("take") and c["id"] == body.get("clip")), None)
    if not take or not take.get("sprite"):
        return refused("Choose an existing Sprite Draft take.", 404)
    try:
        return web.json_response(await sized_sprite(name, take, tile, sprite_export_fps(body)))
    except FileNotFoundError as error:
        return refused(error, 404)
    except (ValueError, Refused) as error:
        return refused(error)


async def sized_sprite(name, take, tile, fps=None):
    sheet = take["sprite"]
    tile_w, tile_h = store.sprite.validate_tile_size(tile)
    base_size = tuple(sheet.get("tile_size") or (sheet["tile"], sheet["tile"]))
    if (tile_w, tile_h) == base_size and fps is None:
        return dict(sheet, subfolder=take["subfolder"])
    folder = "%s/%s/clips" % (store.ROOT, name)
    size_tag = "%dpx" % tile_w if tile_w == tile_h else "%dx%dpx" % (tile_w, tile_h)
    stem = os.path.splitext(os.path.basename(take["file"]))[0] + "." + size_tag
    if fps is not None:
        stem += ".24fps"
    names = {style: stem + "." + style + ".png" for style in ("illustrated", "indexed")}
    variant = dict(sheet, **names, tile=tile_w, tile_size=[tile_w, tile_h], subfolder=folder,
                   pivot=[sheet["pivot"][0] * tile_w // base_size[0], sheet["pivot"][1] * tile_h // base_size[1]])
    if fps is not None:
        variant.update(fps=fps, poses=round(sheet["poses"] * fps / sheet["fps"]))
    if not all(os.path.isfile(output_path(folder, file)) for file in names.values()):
        has_cache = bool(sheet.get("frame_cache")) and os.path.isdir(output_path(take.get("subfolder"), sheet["frame_cache"]))
        if not has_cache and not os.path.isfile(output_path(take.get("subfolder"), take["file"])):
            raise FileNotFoundError("The source movie for this take is missing.")
        key = (folder, stem)
        if key not in SPRITE_EXPORTS:
            task = asyncio.create_task(make_sprite_export(take, variant, folder, stem, names, fps is not None))
            SPRITE_EXPORTS[key] = task
            task.add_done_callback(lambda _: SPRITE_EXPORTS.pop(key, None))
        await asyncio.shield(SPRITE_EXPORTS[key])
    return variant


async def sprite_atlas(request):
    name = store.slug(request.match_info["name"])
    body = await request.json()
    if not isinstance(body, dict):
        return refused("Choose sprite takes to export.")
    raw_tile = body.get("tile_size") if body.get("tile_size") is not None else body.get("tile")
    style, picks = body.get("style"), body.get("picks")
    try:
        tile_w, tile_h = store.sprite.validate_tile_size(raw_tile)
    except ValueError as error:
        return refused(error)
    if style not in ("illustrated", "indexed"):
        return refused("Choose an export size and style.")
    if not isinstance(picks, list) or not picks:
        return refused("Choose at least one animation slot.")
    if len(picks) * tile_w * tile_h * 4 > store.sprite.MAX_ATLAS_PIXELS:
        return refused("This atlas is too large. Choose a smaller export size.")
    include_webp = bool(body.get("webp", True))
    project = read_project(name)
    rows = []
    seen = set()
    try:
        fps = sprite_export_fps(body)
        for pick in picks:
            if not isinstance(pick, dict):
                raise ValueError("Invalid sprite selection.")
            key = (pick.get("motion"), pick.get("direction"))
            if not isinstance(key[0], str) or key[1] not in store.sprite.DIRECTIONS or key in seen:
                raise ValueError("Choose each motion and direction once.")
            seen.add(key)
            clip = store.find_clip(project, pick.get("clip"))
            take = next((t for t in clip["takes"] if t["id"] == pick.get("take") and t.get("sprite")), None)
            if not take:
                raise ValueError("A chosen sprite take is missing.")
            config = (take.get("saved") or {}).get("sprite") or clip.get("sprite") or {}
            facing = config.get("direction", "S")
            mirror = pick.get("mirror", False)
            motion = config.get("animation") or (
                "idle" if config.get("motion") == "idle" else
                "custom" if config.get("guide") or config.get("motion") == "prompt" else "run")
            if key[0] != motion or type(mirror) is not bool:
                raise ValueError("The selection does not match the motion of the saved take.")
            expected = facing.translate(str.maketrans("EW", "WE")) if mirror else facing
            if expected != key[1]:
                raise ValueError("The selected take faces a different direction.")
            sheet = await sized_sprite(name, take, (tile_w, tile_h), fps)
            rows.append(("_".join(key), output_path(sheet["subfolder"], sheet[style]), sheet,
                         dict(clip=clip["id"], take=take["id"], seed=take.get("seed"), source_direction=facing, mirrored=mirror)))
        archive_bytes = await asyncio.to_thread(store.sprite.pack_atlas, rows, (tile_w, tile_h), style, include_webp=include_webp)
    except (Refused, OSError, ValueError, KeyError, TypeError) as error:
        return refused(error)
    return web.Response(body=archive_bytes, content_type="application/zip",
                        headers={"Content-Disposition": 'attachment; filename="sprites.zip"'})


async def sprite_self_pin(request):
    """Extract a clean non-edge-touching stride-phase frame from a sprite take for loop self-pinning."""
    import tempfile
    from PIL import Image

    name = store.slug(request.match_info["name"])
    body = await request.json()
    if not isinstance(body, dict):
        return refused("Choose a Sprite Draft take to self-pin.")
    data = read_project(name)
    take = next((t for c in data["clips"] for t in c.get("takes", [])
                 if t["id"] == body.get("take") and c["id"] == body.get("clip")), None)
    if not take or not take.get("sprite"):
        return refused("Choose an existing Sprite Draft take.", 404)
    sheet = take["sprite"]
    default_stride = min(36, max(4, 2 * int(sheet.get("source_poses") or sheet.get("poses") or 7)))
    stride = body.get("stride", default_stride)
    if type(stride) is not int or not 4 <= stride <= 36:
        return refused("Self-pin stride must be an integer between 4 and 36.")
    preferred = body.get("frame")
    if preferred is not None and type(preferred) is not int:
        return refused("Preferred self-pin frame must be an integer.")

    frames = None
    total_frames = int(take.get("frames") or sheet.get("frames") or store.sprite.FRAMES)
    cache_name = sheet.get("frame_cache")
    if cache_name:
        try:
            cache_path = output_path(take.get("subfolder"), cache_name)
        except web.HTTPException:
            cache_path = None
        if cache_path and os.path.isdir(cache_path):
            frames = await asyncio.to_thread(store.sprite.load_frame_cache, cache_path)

    if not frames or len(frames) < stride + 1:
        source = output_path(take.get("subfolder"), take["file"])
        if os.path.isfile(source):
            with tempfile.TemporaryDirectory() as tmp:
                pattern = os.path.join(tmp, "f_%04d.png")
                await run(["ffmpeg", "-y", "-v", "error", "-i", source, pattern])
                files = sorted(f for f in os.listdir(tmp) if f.endswith(".png"))
                frames = []
                for fname in files:
                    with Image.open(os.path.join(tmp, fname)) as im:
                        frames.append(im.convert("RGB"))
        elif not frames:
            return refused("The source movie for this take is missing.", 404)

    try:
        if len(frames) >= stride + 1:
            chosen, rgb, pins = await asyncio.to_thread(store.sprite.pick_self_pin_frame, frames, stride, preferred)
        else:
            # Sampled pose cache (e.g. 7 poses): pick a clean non-edge-touching pose and preserve the clip's stride pins.
            pick_stride = max(1, len(frames) // 2)
            chosen, rgb, _ = await asyncio.to_thread(store.sprite.pick_self_pin_frame, frames, pick_stride, preferred)
            pins = [i for i in (0, stride, 2 * stride) if i < total_frames]
    except ValueError as error:
        return refused(error)
    target, asset = asset_target(name, ".png")
    await asyncio.to_thread(rgb.save, target, format="PNG")
    return web.json_response({
        "file": asset["file"],
        "subfolder": asset["subfolder"],
        "stride": stride,
        "frame": chosen,
        "pins": pins,
    })


async def list_props(request):
    """List rendered 4 MP prop and turnaround PNGs in a project's props/ folder, newest first."""
    name = store.slug(request.match_info["name"])
    read_project(name)
    subfolder = "%s/%s/props" % (store.ROOT, name)
    folder = output_path(subfolder, "")
    items = []
    if os.path.isdir(folder):
        for fname in os.listdir(folder):
            if not fname.lower().endswith((".png", ".webp")):
                continue
            full = os.path.join(folder, fname)
            if not os.path.isfile(full):
                continue
            try:
                st = os.stat(full)
            except OSError:
                continue
            items.append({
                "file": fname,
                "subfolder": subfolder,
                "bytes": st.st_size,
                "modified": st.st_mtime,
            })
    items.sort(key=lambda item: (-item["modified"], item["file"]))
    return web.json_response({"props": items})


async def sprite_prop(request):
    """Plan or queue a 4+ MP MiniMax-H3-Image static prop, paired state variant, or turnaround plate."""
    name = store.slug(request.match_info["name"])
    body = await request.json()
    if not isinstance(body, dict):
        return refused("Send a prop render specification object.")
    project = read_project(name)
    try:
        job, graph = store.plan_prop_render(project, name, body, seed=body.get("seed"), available=available_models())
    except ValueError as error:
        return refused(error)
    if body.get("dry_run"):
        return web.json_response({"job": job, "graph": graph})
    try:
        prompt_id = await queue_prompt(graph, body.get("client_id"))
    except Refused as error:
        return refused(plain(error), 502)
    PROP_JOBS[prompt_id] = {
        "id": prompt_id,
        "kind": "prop",
        "folder": name,
        "project": project.get("name") or name,
        "clip": None,
        "name": job["name"],
        "quality": job["kind"].replace("_", " "),
        "seed": job.get("seed"),
        "size": job.get("native_size"),
        "frames": 1,
    }
    return web.json_response({"prompt_id": prompt_id, "job": job})


async def render_queue(request):
    """Live ComfyUI render queue enriched with H3 Editor project clips, sprites, props and recent history."""
    return web.json_response(build_queue_snapshot())


async def cancel_queue(request):
    """Cancel one job, all waiting jobs, or all running + waiting jobs from H3 Editor's Render Queue UI."""
    try:
        body = await request.json()
    except ValueError:
        body = {}
    if not isinstance(body, dict):
        return refused("Send a cancellation target.")
    snapshot = build_queue_snapshot()
    job_id = body.get("id")
    scope = body.get("scope")
    if job_id:
        targets = [item for item in (*snapshot["running"], *snapshot["pending"]) if item["id"] == job_id]
        if not targets:
            return refused("That render is no longer in the queue.", 404)
    elif scope == "pending":
        targets = list(snapshot["pending"])
    elif scope == "all":
        targets = [*snapshot["pending"], *snapshot["running"]]
    elif scope == "project" and body.get("folder"):
        folder = store.slug(body["folder"])
        targets = [item for item in (*snapshot["pending"], *snapshot["running"]) if item.get("folder") == folder]
    else:
        return refused("Specify id, or scope: pending, all, or project.")

    ids = [item["id"] for item in targets]
    if ids:
        try:
            status, _ = await comfy("POST", "/queue", {"delete": ids})
            if status != 200:
                raise Refused("ComfyUI could not remove queued renders; try again")
            running, _ = comfy_queue()
            for item in targets:
                if item["id"] in running:
                    status, _ = await comfy("POST", "/interrupt", {"prompt_id": item["id"]})
                    if status != 200:
                        raise Refused("ComfyUI could not interrupt the active render; try again")
        except Refused as error:
            return refused(plain(error), 502)

    by_folder = {}
    for item in targets:
        PROP_JOBS.pop(item["id"], None)
        _record_queue_history({**item, "state": "cancelled", "error": None})
        if item.get("folder") and item.get("kind") in ("clip", "sprite"):
            by_folder.setdefault(item["folder"], []).append(item["id"])

    for folder, folder_ids in by_folder.items():
        try:
            async with lock(folder):
                data = read_project(folder)
                changed = False
                for pid in folder_ids:
                    if store.pop_pending(data, pid):
                        changed = True
                if changed:
                    commit(folder, data, "render", body.get("client_id"))
        except web.HTTPException:
            continue

    return web.json_response({"cancelled": ids, "queue": build_queue_snapshot()})


async def clear_queue_history(request):
    """Clear recent finished/failed/cancelled entries from H3 Editor's Render Queue history."""
    QUEUE_HISTORY.clear()
    return web.json_response({"ok": True, "queue": build_queue_snapshot()})


def take_has_latents(take):
    """Whether a draft take has a whole-take latent file beside its video on disk."""
    if not take or not take.get("file"):
        return False
    try:
        path = output_path(take.get("subfolder"), take["file"])
    except web.HTTPException:
        return False
    return store.holds.has_take(store.holds.hold_path(path))


def detect_motion_style(character):
    """Classify a character reference asset on disk as 'preserve' (2D/anime) or 'photoreal'."""
    if not isinstance(character, dict) or not character.get("file"):
        return "photoreal"
    try:
        path = output_path(character.get("subfolder"), character["file"])
        if not os.path.exists(path):
            return "photoreal"
        return store.motion_replace.classify_character_style(path)
    except Exception:
        return "photoreal"


def _output_rel_path(rel_file):
    base = os.path.realpath(folder_paths.get_output_directory())
    rel = str(rel_file or "").replace("\\", "/").lstrip("/")
    path = os.path.realpath(os.path.join(base, *rel.split("/")))
    if not path.startswith(base + os.sep):
        raise web.HTTPBadRequest(text="that file is outside the output folder")
    return path


def check_motion_cache(source_info, character, motion):
    """Check which preprocessing caches (SAM3 masks, SDPose keypoints, character cutout) already hit on disk."""
    hits = {"mask": False, "pose": False, "cutout": False}
    try:
        if source_info and source_info.get("file"):
            src_path = _output_rel_path(source_info["file"])
            first, last = int(source_info["first"]), int(source_info["last"])
            hits["mask"] = store.motion_replace.has_mask_cache(src_path, first, last, subject="person")
            hits["pose"] = store.motion_replace.has_pose_cache(src_path, first, last)
    except Exception:
        pass
    try:
        if isinstance(character, dict) and character.get("file"):
            char_path = output_path(character.get("subfolder"), character["file"])
            hits["cutout"] = store.motion_replace.has_cutout_cache(char_path, subject="person", backdrop="white")
    except Exception:
        pass
    return hits


async def start_render(data, folder, clip_id, quality, seed=None, client_id=None, detail=None, upscaler=None,
                       patches=None, edit=None, part=None):
    """Queue one render. patches is the installed patch nodes, for a batch that has read them already."""
    job, graph = store.plan_render(data, folder, clip_id, quality or "draft", seed, available_models(), detail, upscaler,
                                   installed_patches() if patches is None else patches, edit, part=part,
                                   has_latents=take_has_latents, detect_style=detect_motion_style,
                                   check_cache=check_motion_cache)
    job["id"] = await queue_prompt(graph, client_id)
    store.add_pending(data, job)
    return job


async def stop_render(data, job):
    running, _ = comfy_queue()
    if job["id"] in running:
        await comfy("POST", "/interrupt", {"prompt_id": job["id"]})
    else:
        await comfy("POST", "/queue", {"delete": [job["id"]]})
    store.pop_pending(data, job["id"])


async def stop_renders(data, quality=None):
    """Delete this project's selected queued jobs together before interrupting a selected running job."""
    if quality not in (None, "draft", "final"):
        raise ValueError("quality must be draft or final")
    jobs = [job for job in data.get("pending", []) if quality is None or job.get("quality") == quality]
    ids = [job["id"] for job in jobs]
    if not ids:
        return []
    status, _ = await comfy("POST", "/queue", {"delete": ids})
    if status != 200:
        raise Refused("ComfyUI could not remove the queued renders; try again")
    # The running job can have changed during the delete request. Never interrupt a different project's job.
    running, _ = comfy_queue()
    for job_id in ids:
        if job_id in running:
            status, _ = await comfy("POST", "/interrupt", {"prompt_id": job_id})
            if status != 200:
                raise Refused("ComfyUI could not interrupt the render; try again")
    for job_id in ids:
        store.pop_pending(data, job_id)
    return ids


def find_job(data, body):
    for job in data.get("pending") or []:
        if job.get("id") == body.get("job") or (body.get("clip") and job["clip"] == body.get("clip")):
            return job
    raise KeyError("nothing is rendering for %s" % (body.get("job") or body.get("clip")))


async def render(request):
    folder = store.slug(request.match_info["name"])
    body = await request.json()
    async with lock(folder):
        data = read_project(folder)
        try:
            job = await start_render(data, folder, body.get("clip"), body.get("quality"), body.get("seed"),
                                     body.get("client_id"), body.get("detail"), body.get("upscaler"),
                                     edit=body.get("edit"), part=body.get("part"))
        except (ValueError, KeyError) as error:
            return refused(plain(error))
        except Refused as error:
            return refused(plain(error), 502)
        commit(folder, data, "render", body.get("client_id"))
    return web.json_response({"job": job, "revision": data["revision"], "project": data})


async def cancel(request):
    folder = store.slug(request.match_info["name"])
    body = await request.json()
    async with lock(folder):
        data = read_project(folder)
        try:
            if body.get("all") is True:
                cancelled = await stop_renders(data, body.get("quality"))
                if cancelled:
                    commit(folder, data, "render", body.get("client_id"))
                return web.json_response({"cancelled": cancelled, "revision": data["revision"], "project": data})
            job = find_job(data, body)
        except KeyError as error:
            return refused(plain(error), 404)
        except (ValueError, Refused) as error:
            return refused(plain(error))
        await stop_render(data, job)
        commit(folder, data, "render", body.get("client_id"))
    return web.json_response({"cancelled": job["id"], "revision": data["revision"], "project": data})


async def operations(request):
    """A batch of edits and renders. The edits apply together or not at all."""
    folder = store.slug(request.match_info["name"])
    body = await request.json()
    ops = body.get("ops")
    if not isinstance(ops, list) or not ops:
        return refused("send ops: a list of operations")
    async with lock(folder):
        data = read_project(folder)
        if "revision" in body and body["revision"] != data.get("revision", 0):
            return refused("the project changed since revision %s" % body["revision"], 409, revision=data.get("revision", 0))
        work = copy.deepcopy(data)
        names, results, later = {}, [], []
        catalog = read_catalog() if any(isinstance(op, dict) and op.get("op") in CATALOG_OPS for op in ops) else None
        catalog_before = copy.deepcopy(catalog)
        # The installed patch nodes are read once for the whole batch, however many renders it holds.
        patches = installed_patches() if any(isinstance(op, dict) and op.get("op") == "render" for op in ops) else None
        for index, op in enumerate(ops):
            kind = op.get("op") if isinstance(op, dict) else None
            try:
                if kind in CATALOG_OPS:
                    results.append(catalog_op(work, folder, op, names, catalog))
                elif kind == "render":
                    clip_id = store._resolve(op.get("clip"), names, "clip")
                    planned, _ = store.plan_render(copy.deepcopy(work), folder, clip_id, op.get("quality") or "draft",
                                                   op.get("seed"), available_models(), op.get("detail"), op.get("upscaler"),
                                                   patches, op.get("edit"), part=op.get("part"),
                                                   has_latents=take_has_latents, detect_style=detect_motion_style,
                                                   check_cache=check_motion_cache)
                    later.append((index, op, clip_id, planned.get("loads") or planned["models"]["unet"]))
                    results.append(None)
                elif kind == "cancel":
                    later.append((index, op, store._resolve(op.get("clip"), names, "clip"), None))
                    results.append(None)
                elif kind == "make_refmod":
                    taken = refmod_files()
                    if taken is None or store.REFMOD_EXTRACT not in __import__("nodes").NODE_CLASS_MAPPINGS:
                        raise ValueError("making a RefMod needs the ComfyUI-MiniMaxH3Mod pack, which is not installed")
                    cast_id = store._resolve(op.get("cast"), names, "cast")
                    name, graph = store.refmod_graph(work, cast_id, op.get("name"), op.get("mode") or "compressed",
                                                     available_models())
                    if name in taken and (store.find_cast(work, cast_id).get("refmod") or {}).get("file") != name and not op.get("replace"):
                        raise ValueError("a RefMod named %s already exists. Send another name, or replace: true" % name)
                    later.append((index, op, (cast_id, name, graph), None))
                    results.append(None)
                else:
                    results.append(store.apply_op(work, op, names))
            except (ValueError, KeyError, TypeError) as error:
                return refused("operation %s (%s): %s" % (index + 1, kind, plain(error)), index=index)
        # The catalog changes only once every edit of the batch has gone through.
        if catalog is not None and catalog != catalog_before:
            async with lock(CATALOG):
                fresh = {r["id"]: r for r in read_catalog()}
                for row in catalog:
                    fresh[row["id"]] = row
                write_catalog(list(fresh.values()))
        # Renders that share a model run one after another, so a mixed batch loads each model once.
        # They trade places only with each other: a cancel stays where it was asked for.
        slots = [n for n, entry in enumerate(later) if entry[1]["op"] == "render"]
        queued = work.get("pending") or []
        ends_on = (queued[-1].get("loads") or (queued[-1].get("models") or {}).get("unet")) if queued else None
        order = group_by_model([later[n][3] for n in slots], ends_on)
        arranged = list(later)
        for slot, pick in zip(slots, order):
            arranged[slot] = later[slots[pick]]
        problem = None
        for index, op, clip_id, _ in arranged:
            try:
                if op["op"] == "render":
                    job = await start_render(work, folder, clip_id, op.get("quality"), op.get("seed"), op.get("client_id"),
                                             op.get("detail"), op.get("upscaler"), patches, op.get("edit"), part=op.get("part"))
                    results[index] = {"clip": clip_id, "job": job["id"], "take": job["take"]}
                elif op["op"] == "make_refmod":
                    cast_id, name, graph = clip_id
                    prompt = await queue_prompt(graph, op.get("client_id"))
                    # The member carries it from now on. Renders leave it out until the file is there.
                    person = store.find_cast(work, cast_id)
                    person["refmod"] = {"file": name, "strength": (person.get("refmod") or {}).get("strength", 1.0)}
                    results[index] = {"cast": cast_id, "refmod": name, "job": prompt}
                else:
                    job = find_job(work, {"clip": clip_id, "job": op.get("job")})
                    await stop_render(work, job)
                    results[index] = {"cancelled": job["id"]}
            except (ValueError, KeyError, Refused) as error:
                problem = "operation %s (%s): %s" % (index + 1, op["op"], plain(error))
                results[index] = {"error": problem}
                break
        if work != data:
            commit(folder, work, "ops", body.get("client"))
    for index, op in enumerate(ops):
        if isinstance(results[index], dict):
            label = {"op": op.get("op")}
            if op.get("as"):
                label["as"] = op["as"]
            results[index] = dict(label, **results[index])
    reply = {"results": results, "revision": work.get("revision", 0), "summary": store.describe(work)}
    if problem:
        reply["error"] = problem
    return web.json_response(reply, status=502 if problem else 200)


async def wait(request):
    """Block until nothing is rendering, or until the project moves past a revision. For agents."""
    folder = store.slug(request.match_info["name"])
    try:
        timeout = min(1800.0, max(1.0, float(request.query.get("timeout", 600))))
        after = int(request.query["after"]) if "after" in request.query else None
    except ValueError:
        return refused("timeout and after are numbers")
    deadline = time.monotonic() + timeout
    while True:
        data = read_project(folder)
        # A VFX render waits for an open editor, so it counts as rendering until one has done it.
        done = data.get("revision", 0) > after if after is not None else not (data.get("pending") or store.effects_waiting(data))
        if done or time.monotonic() >= deadline:
            reply = store.describe(data)
            reply["timed_out"] = not done
            return web.json_response(reply)
        await asyncio.sleep(1.0)


async def lay_track(path, track):
    """Give a take the exact slice of the track it was generated to, in place of the model's decoded copy.

    The picture is kept as it is. The sound is trimmed from the decoded track, so it is sample aligned,
    and its level is never changed.
    """
    source = output_path(track.get("subfolder"), track["file"])
    temporary = os.path.splitext(path)[0] + ".track.mp4"
    try:
        await run(["ffmpeg", "-y", "-v", "error", "-i", path, "-i", source, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                   "-af", "atrim=start=%.6f:duration=%.6f,asetpts=PTS-STARTPTS,apad" % (float(track["from"]), float(track["seconds"])),
                   "-t", "%.6f" % float(track["seconds"]), "-c:a", "aac", "-b:a", "256k", "-ar", "48000",
                   "-movflags", "+faststart", temporary])
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


async def assemble_splice(rendered, splice):
    """Replace frames splice['in']..splice['out'] of the base take with the newly finalized slice, keeping its audio."""
    source = output_path(splice.get("subfolder"), splice["file"])
    first, last, total = int(splice["in"]), int(splice["out"]), int(splice["frames"])
    trim_start = int(splice.get("trim_start") or 0)
    count = last - first
    base_info = await video_info(source, count=False)
    rendered_info = await video_info(rendered)
    if rendered_info["frames"] < trim_start + count:
        raise web.HTTPInternalServerError(text="the finalized video is shorter than the selected part")
    bw, bh = int(base_info["width"]), int(base_info["height"])
    rw, rh = int(rendered_info["width"]), int(rendered_info["height"])
    if splice.get("is_edit") and bw * bh >= rw * rh * 0.75:
        width, height = bw, bh
    else:
        width, height = rw, rh
    fit = (f"scale={width}:{height}:force_original_aspect_ratio=decrease:"
           f"force_divisible_by=2,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,setpts=PTS-STARTPTS")
    filters, segments = [], []
    if first:
        filters.append(f"[1:v]trim=end_frame={first},{fit}[prefix]")
        segments.append("[prefix]")
    filters.append(f"[0:v]trim=start_frame={trim_start}:end_frame={trim_start + count},{fit}[part]")
    segments.append("[part]")
    if last < total:
        filters.append(f"[1:v]trim=start_frame={last},{fit}[suffix]")
        segments.append("[suffix]")
    filters.append("".join(segments) + f"concat=n={len(segments)}:v=1:a=0,setpts=N/({store.FPS}*TB)[picture]")
    target = os.path.splitext(rendered)[0] + "_part.mp4"
    temporary = target + ".tmp.mp4"
    try:
        await run(["ffmpeg", "-y", "-v", "error", "-i", rendered, "-i", source,
                   "-filter_complex", ";".join(filters), "-map", "[picture]", "-map", "1:a:0?",
                   "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "copy",
                   "-movflags", "+faststart", temporary])
        if (await video_info(temporary))["frames"] != total:
            raise web.HTTPInternalServerError(text="the finalized part did not match the take's frame count")
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return target


async def mux_motion_audio(rendered, motion_audio):
    """Trim a motion-transfer clip to its exact source frame span and lay the exact source audio slice,
    preserving the rendered video's full resolution (e.g. 1088x1920 Sharp RelayVSR)."""
    source = output_path(motion_audio["subfolder"], motion_audio["file"])
    first, last = int(motion_audio["in"]), int(motion_audio["out"])
    expected = last - first
    info = await video_info(rendered)
    if info["frames"] < expected:
        raise web.HTTPInternalServerError(text="the motion transfer render is shorter than the selected range")
    start_sec = first / float(store.FPS)
    dur_sec = expected / float(store.FPS)
    samples = expected * 48000 // store.FPS
    sound = await asyncio.to_thread(media.has_sound, source)
    temporary = rendered + ".motion.mp4"
    args = ["ffmpeg", "-y", "-v", "error", "-i", rendered]
    if sound:
        args.extend(["-i", source, "-map", "0:v:0", "-map", "1:a:0",
                     "-af", "atrim=start=%.6f:duration=%.6f,asetpts=PTS-STARTPTS,aresample=48000,apad=whole_len=%d,atrim=end_sample=%d"
                     % (start_sec, dur_sec, samples, samples),
                     "-c:a", "aac", "-b:a", "192k", "-ar", "48000"])
    else:
        args.extend(["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-map", "0:v:0", "-map", "1:a:0",
                     "-af", "atrim=end_sample=%d" % samples,
                     "-c:a", "aac", "-b:a", "192k", "-ar", "48000"])
    if info["frames"] == expected:
        args.extend(["-c:v", "copy"])
    else:
        args.extend(["-vf", f"trim=end_frame={expected},setpts=PTS-STARTPTS",
                     "-c:v", "libx264", "-crf", "15", "-pix_fmt", "yuv420p"])
    args.extend(["-movflags", "+faststart", temporary])
    try:
        await run(args)
        if (await video_info(temporary))["frames"] != expected:
            raise web.HTTPInternalServerError(text="the motion clip did not match the selected frame count")
        os.replace(temporary, rendered)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return rendered


async def assemble_prompt_edit(rendered, edit):
    """Replace exactly the requested source frames, retaining the original audio stream."""
    source = output_path(edit["subfolder"], edit["file"])
    first, last, total = edit["in"], edit["out"], edit["frames"]
    original = await video_info(source, count=False)
    rendered_info = await video_info(rendered)
    if rendered_info["frames"] < last - first:
        raise web.HTTPInternalServerError(text="the generated video is shorter than the selected edit range")
    # Preserve higher rendered resolution (e.g. Sharp RelayVSR) when replacing the entire clip.
    if first == 0 and last == total and rendered_info["width"] > original["width"]:
        width, height = rendered_info["width"], rendered_info["height"]
    else:
        width, height = original["width"], original["height"]
    filters, segments = [], []
    if first:
        filters.append(f"[1:v]trim=end_frame={first},setsar=1,setpts=PTS-STARTPTS[prefix]")
        segments.append("[prefix]")
    filters.append(
        f"[0:v]trim=end_frame={last - first},scale={width}:{height}:force_original_aspect_ratio=decrease:"
        f"force_divisible_by=2,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,setpts=PTS-STARTPTS[edit]"
    )
    segments.append("[edit]")
    if last < total:
        filters.append(f"[1:v]trim=start_frame={last},setsar=1,setpts=PTS-STARTPTS[suffix]")
        segments.append("[suffix]")
    filters.append("".join(segments) + f"concat=n={len(segments)}:v=1:a=0,setpts=N/({store.FPS}*TB)[picture]")
    target = os.path.splitext(rendered)[0] + "_edit.mp4"
    temporary = target + ".tmp.mp4"
    try:
        await run(["ffmpeg", "-y", "-v", "error", "-i", rendered, "-i", source,
                   "-filter_complex", ";".join(filters), "-map", "[picture]", "-map", "1:a:0?",
                   "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "copy",
                   "-movflags", "+faststart", temporary])
        if (await video_info(temporary))["frames"] != total:
            raise web.HTTPInternalServerError(text="the edit did not match the source frame count")
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return target


def saved_file(entry):
    for value in ((entry or {}).get("outputs") or {}).get("19", {}).values():
        if isinstance(value, list):
            for found in value:
                if isinstance(found, dict) and found.get("filename"):
                    return found
    return None


def failure(entry):
    if not entry:
        return "ComfyUI lost this render, probably in a restart. Render it again."
    for name, detail in (entry.get("status") or {}).get("messages") or []:
        if name == "execution_error":
            return "%s failed: %s" % (detail.get("node_type", "a node"), str(detail.get("exception_message", "")).strip())
        if name == "execution_interrupted":
            return "The render was interrupted."
    return "The render finished without saving a video."


SCAN_EVERY = 10.0
WATCHING = {"folders": set(), "scanned": 0.0}


def pending_folders():
    found = set()
    for name in os.listdir(root()):
        path = os.path.join(root(), name, "project.json")
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                if json.load(f).get("pending"):
                    found.add(name)
        except (OSError, ValueError):
            continue
    return found


async def watch_once():
    """Attach finished renders to their projects, and record the ones that failed."""
    from server import PromptServer

    if time.monotonic() - WATCHING["scanned"] > SCAN_EVERY:
        WATCHING["folders"] |= pending_folders()
        WATCHING["scanned"] = time.monotonic()
    if not WATCHING["folders"]:
        return
    running, queued = comfy_queue()
    queue = PromptServer.instance.prompt_queue
    for folder in list(WATCHING["folders"]):
        try:
            data = read_project(folder)
        except web.HTTPException:
            WATCHING["folders"].discard(folder)
            continue
        if not data.get("pending"):
            WATCHING["folders"].discard(folder)
            continue
        finished = []
        for job in data["pending"]:
            if job.get("id") in running or job.get("id") in queued:
                continue
            entry = queue.get_history(prompt_id=job["id"]).get(job["id"])
            file = saved_file(entry)
            sheets = ((entry or {}).get("outputs") or {}).get("211", {}).get("sprite")
            expects_sheet = not job.get("edit") and (job.get("saved") or {}).get("sprite") is not None
            if not file or (expects_sheet and not sheets):
                finished.append((job, None, None, failure(entry)))
                continue
            try:
                path = output_path(file.get("subfolder"), file["filename"])
                if job.get("splice"):
                    path = await assemble_splice(path, job["splice"])
                    file = dict(file, filename=os.path.basename(path))
                elif job.get("edit"):
                    path = await assemble_prompt_edit(path, job["edit"])
                    file = dict(file, filename=os.path.basename(path))
                elif job.get("motion_audio"):
                    path = await mux_motion_audio(path, job["motion_audio"])
                if job.get("track") and not job.get("splice"):
                    await lay_track(path, job["track"])
                info = await probe_file(path)
                if sheets:
                    info["sprite"] = sheets[0]
                finished.append((job, file, info, None))
            except web.HTTPException as error:
                finished.append((job, None, None, "The render saved a file that cannot be read: %s" % error.text))
        if not finished:
            continue
        async with lock(folder):
            data = read_project(folder)
            clips_by_id = {c.get("id"): c for c in data.get("clips") or [] if isinstance(c, dict)}
            project_name = data.get("name") or folder
            for job, file, info, message in finished:
                clip = clips_by_id.get(job.get("clip")) or {}
                sprite_cfg = (job.get("saved") or {}).get("sprite") or clip.get("sprite")
                if file:
                    store.finish_render(data, job["id"], file, info)
                    _record_queue_history({
                        "id": job["id"],
                        "kind": "sprite" if sprite_cfg else "clip",
                        "folder": folder,
                        "project": project_name,
                        "clip": job.get("clip"),
                        "name": clip.get("name") or job.get("clip") or "Clip",
                        "quality": job.get("quality") or "draft",
                        "size": job.get("size"),
                        "frames": (info or {}).get("frames") or job.get("frames"),
                        "state": "done",
                        "error": None,
                    })
                else:
                    store.fail_render(data, job["id"], message)
                    _record_queue_history({
                        "id": job["id"],
                        "kind": "sprite" if sprite_cfg else "clip",
                        "folder": folder,
                        "project": project_name,
                        "clip": job.get("clip"),
                        "name": clip.get("name") or job.get("clip") or "Clip",
                        "quality": job.get("quality") or "draft",
                        "size": job.get("size"),
                        "frames": job.get("frames"),
                        "state": "failed",
                        "error": message,
                    })
            commit(folder, data, "render")


async def watch_forever():
    while True:
        try:
            await sys.modules[__name__].watch_once()
        except asyncio.CancelledError:
            raise
        except Exception:
            logging.exception("H3 editor could not check its renders")
        await asyncio.sleep(1.5)


def ensure_watcher():
    """One watcher for the whole server. It looks its code up each round, so a hot reload takes effect."""
    from server import PromptServer

    server = PromptServer.instance
    task = getattr(server, "h3take_watcher", None)
    if task is None or task.done():
        server.h3take_watcher = server.loop.create_task(watch_forever())
    WATCHING["scanned"] = 0.0


def asset_target(name, extension):
    folder = store.slug(name)
    directory = os.path.join(root(), folder, "assets")
    os.makedirs(directory, exist_ok=True)
    file = uuid.uuid4().hex[:12] + extension
    return os.path.join(directory, file), {"file": file, "subfolder": "%s/%s/assets" % (store.ROOT, folder)}


PICTURES = store.PICTURES
SOUNDS = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".aac")
VIDEOS = (".mp4", ".mov", ".webm", ".mkv", ".m4v", ".avi")
LONGEST_IMPORT = 900.0
LONGEST_VOICE = 30.0


async def upload(request):
    """Keep an uploaded picture or sound with the project."""
    form = await request.post()
    part = form.get("file") or form.get("image")
    if part is None or not hasattr(part, "file"):
        return refused("choose a picture or a sound to upload")
    extension = os.path.splitext(part.filename or "")[1].lower()
    if extension not in PICTURES + SOUNDS:
        return refused("use a PNG, JPEG or WebP picture, or a WAV, MP3, FLAC, M4A or OGG sound")
    target, asset = asset_target(request.match_info["name"], extension)
    with open(target, "wb") as f:
        f.write(part.file.read())
    if extension in SOUNDS and request.query.get("sound"):
        # A sound for the timeline also needs its length and its loudness, to place it and to draw it.
        values = await make_peaks(target, os.path.splitext(target)[0] + ".peaks.json")
        if not values:
            os.remove(target)
            return refused("that file has no sound the editor can read")
        asset = dict(asset, seconds=round(len(values) / float(store.FPS), 4),
                     peaks=os.path.basename(os.path.splitext(target)[0] + ".peaks.json"))
    return web.json_response(asset)


async def grab_sound(request):
    """Save the sound of part of a clip, at most 30 seconds, as a voice a cast member can speak with."""
    body = await request.json()
    source = output_path(body.get("subfolder"), body["file"])
    first = max(0, int(body.get("first", 0)))
    last = int(body.get("last") or first + LONGEST_VOICE * store.FPS)
    seconds = min(LONGEST_VOICE, max(0.5, (last - first) / float(store.FPS)))
    target, asset = asset_target(request.match_info["name"], ".wav")
    await run(["ffmpeg", "-y", "-v", "error", "-ss", "%.3f" % (first / float(store.FPS)), "-t", "%.3f" % seconds,
               "-i", source, "-vn", "-ac", "2", "-ar", "48000", target])
    if not os.path.exists(target) or os.path.getsize(target) < 1000:
        return refused("that part of the clip has no sound")
    return web.json_response(asset)


def source_path(where, subfolder, filename):
    """A file an agent already put in ComfyUI's input or output folder."""
    if where not in ("input", "output"):
        raise ValueError("source folder is input or output")
    if not filename:
        raise ValueError("name the file to import")
    base = os.path.realpath(folder_paths.get_input_directory() if where == "input" else folder_paths.get_output_directory())
    path = os.path.realpath(os.path.join(base, subfolder or "", filename))
    if not path.startswith(base + os.sep) or not os.path.isfile(path):
        raise ValueError("there is no %s in the %s folder" % (os.path.join(subfolder or "", filename), where))
    return path


LARGEST_UPLOAD = 4 << 30
LARGEST_FIELD = 4096


async def receive_upload(request, allowed, directory):
    """Stream one uploaded file to disk, so a long recording never sits in memory.

    One file per upload and at most 4 GiB. Whatever was written is removed if the upload fails.
    """
    reader = await request.multipart()
    fields, saved, original = {}, None, None
    try:
        while True:
            part = await reader.next()
            if part is None:
                break
            if not part.filename:
                value = await part.read_chunk(LARGEST_FIELD + 1)
                if len(value) > LARGEST_FIELD:
                    raise ValueError("the %s field is too long" % part.name)
                fields[part.name] = value.decode("utf-8", errors="replace")
                continue
            if saved:
                raise ValueError("import one file at a time")
            extension = os.path.splitext(part.filename)[1].lower()
            if extension not in allowed:
                raise ValueError("import a video (%s) or a picture (%s)" % (", ".join(VIDEOS), ", ".join(PICTURES)))
            os.makedirs(directory, exist_ok=True)
            saved = os.path.join(directory, uuid.uuid4().hex[:12] + extension)
            original = part.filename
            written = 0
            with open(saved, "wb") as f:
                while True:
                    chunk = await part.read_chunk(1 << 20)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > LARGEST_UPLOAD:
                        raise ValueError("that file is larger than 4 GiB. Put it in ComfyUI's input folder and import it from there")
                    f.write(chunk)
    except BaseException:
        if saved and os.path.exists(saved):
            os.remove(saved)
        raise
    return saved, original, fields


async def normalize(source, target, width, height, fit, still, seconds, start=None, end=None):
    """Any video or picture as a clip like the renders: 24 fps, the project's size, H.264, a sound track."""
    if fit == "cover":
        frame = "scale=%d:%d:force_original_aspect_ratio=increase,crop=%d:%d" % (width, height, width, height)
    else:
        frame = "scale=%d:%d:force_original_aspect_ratio=decrease,pad=%d:%d:(ow-iw)/2:(oh-ih)/2:color=black" % (
            width, height, width, height)
    args = ["ffmpeg", "-y", "-v", "error"]
    if still:
        args += ["-loop", "1", "-framerate", str(store.FPS), "-t", "%.3f" % seconds, "-i", source]
        sound = False
    else:
        if start:
            args += ["-ss", "%.3f" % start]
        if end:
            args += ["-t", "%.3f" % max(0.1, end - (start or 0))]
        args += ["-i", source]
        try:
            sound = await asyncio.to_thread(media.has_sound, source)
        except Exception as error:  # PyAV raises its own errors for files it cannot read
            raise ValueError("could not read %s: %s" % (os.path.basename(source), error))
    if not sound:
        args += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
    args += ["-map", "0:v:0", "-map", "0:a:0" if sound else "1:a:0",
             "-vf", frame + ",setsar=1,fps=%d,format=yuv420p" % store.FPS, "-af", "aresample=48000,apad",
             "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "192k", "-ac", "2", "-shortest", "-t", "%.3f" % LONGEST_IMPORT,
             "-movflags", "+faststart", target]
    await run(args)


async def import_media(request):
    """Bring in real footage or a picture as a clip. It sits on the timeline like a render and can be extended."""
    folder = store.slug(request.match_info["name"])
    data = read_project(folder)
    cleanup, saved = False, None
    try:
        if request.content_type.startswith("multipart/"):
            saved, original, fields = await receive_upload(request, PICTURES + VIDEOS, os.path.join(root(), folder, "incoming"))
            cleanup = bool(saved)
        else:
            fields = await request.json()
            found = fields.get("source") or {}
            saved = source_path(found.get("folder", "input"), found.get("subfolder"), found.get("file"))
            original = found.get("file")
        if not saved:
            return refused("choose a video or a picture to import")
        extension = os.path.splitext(saved)[1].lower()
        if extension not in PICTURES + VIDEOS:
            return refused("import a video (%s) or a picture (%s)" % (", ".join(VIDEOS), ", ".join(PICTURES)))
        fit = fields.get("fit") or "contain"
        if fit not in ("contain", "cover"):
            return refused("fit is contain, which keeps the whole picture, or cover, which fills the frame")
        seconds = float(fields.get("seconds") or 4)
        if not 0.5 <= seconds <= 60:
            return refused("a picture can last 0.5 to 60 seconds")
        if fields.get("bin") and not any(b["id"] == fields["bin"] for b in data.get("bins", [])):
            return refused("no bin %s. GET the project for its bins" % fields["bin"])
        start = float(fields["start_seconds"]) if fields.get("start_seconds") not in (None, "") else None
        end = float(fields["end_seconds"]) if fields.get("end_seconds") not in (None, "") else None
        clip_id, take_id = store.new_id("c"), store.new_id("t")
        target = os.path.join(root(), folder, "clips", "%s_%s.mp4" % (clip_id, take_id))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        # Footage is kept at the size this project's finals come out at, so it is never smaller than the shots beside it.
        width, height, refined = store.render_size(data, "final")
        if refined:
            width, height = store.refine_size(data)
        await normalize(saved, target, width, height, fit, extension in PICTURES, seconds, start, end)
    except ValueError as error:
        return refused(plain(error))
    finally:
        if cleanup and saved and os.path.exists(saved):
            os.remove(saved)
    info = await probe_file(target)
    take = {"id": take_id, "clip": clip_id, "file": os.path.basename(target), "subfolder": "%s/%s/clips" % (store.ROOT, folder),
            "poster": info["poster"], "frames": info["frames"], "quality": "footage", "seed": None, "source": original,
            "size": [width, height]}
    if info.get("strip"):
        take["strip"] = info["strip"]
    place = str(fields.get("place", "true")).lower() not in ("false", "0", "no")
    at = int(fields["start"]) if fields.get("start") not in (None, "") else None
    async with lock(folder):
        data = read_project(folder)
        clip, item = store.add_footage(data, fields.get("name") or os.path.splitext(original or "Footage")[0], take, at, place,
                                       fields.get("bin") or None)
        commit(folder, data, "import", fields.get("client"))
    return web.json_response({"clip": clip["id"], "item": item["id"] if item else None, "frames": info["frames"],
                              "revision": data["revision"], "project": data})


async def grab(request):
    """Save one frame of a rendered clip as a picture."""
    body = await request.json()
    source = output_path(body.get("subfolder"), body["file"])
    target, asset = asset_target(request.match_info["name"], ".png")
    frame = max(0, int(body.get("frame", 0)))
    await run(["ffmpeg", "-y", "-v", "error", "-i", source, "-vf", "select=eq(n\\,%s)" % frame,
               "-frames:v", "1", target])
    if not os.path.exists(target):
        raise web.HTTPBadRequest(text="that clip has no frame %s" % frame)
    return web.json_response(asset)


async def export(request):
    """Join the timeline into one video. With "plan" in the body, only say what the export would hold."""
    name = store.slug(request.match_info["name"])
    data = read_project(name)
    try:
        body = await request.json()
    except ValueError:
        body = {}
    body = body if isinstance(body, dict) else {}
    scale = body.get("scale", 1.0)
    quality = body.get("quality") or "high"
    if scale not in store.EXPORT_SCALES or quality not in store.EXPORT_QUALITY:
        raise web.HTTPBadRequest(text="scale is one of %s and quality one of %s" % (
            ", ".join("%g" % s for s in store.EXPORT_SCALES), ", ".join(store.EXPORT_QUALITY)))
    try:
        plan = store.export_plan(data, scale, span=body.get("range"))
    except ValueError as error:
        raise web.HTTPBadRequest(text=str(error))
    if body.get("plan"):
        return web.json_response({"plan": plan, "name": name})
    paths = {}
    for clip in data.get("clips", []):
        take = store.active_take(clip)
        if take:
            paths[clip["id"]] = output_path(take.get("subfolder"), take["file"])
    folder = os.path.join(root(), name, "exports")
    os.makedirs(folder, exist_ok=True)
    # The file is named after the project, or after what the user typed, and numbered so nothing is overwritten.
    stem = store.slug(body["name"]) if str(body.get("name") or "").strip() else name
    temporary = os.path.join(folder, ".export_%s.mp4" % uuid.uuid4().hex)
    sound_paths = {}
    for sound in data.get("sounds") or []:
        try:
            found = output_path(sound.get("subfolder"), sound["file"])
        except web.HTTPException:
            continue
        if os.path.exists(found):
            sound_paths[sound["id"]] = found
    effect_paths = {}
    for effect in data.get("effects") or []:
        take = store.active_take(effect)
        if not take:
            continue
        try:
            found = output_path(take.get("subfolder"), take["file"])
        except web.HTTPException:
            continue
        if os.path.exists(found):
            effect_paths[effect["id"]] = found
    try:
        args = store.export_command(data, paths, temporary, scale=scale, quality=quality, sound_paths=sound_paths,
                                    effect_paths=effect_paths, span=body.get("range"))
    except ValueError as error:
        raise web.HTTPBadRequest(text=str(error))
    try:
        await run(args)
        async with lock(name):
            number = 1
            target = os.path.join(folder, "%s_%03d.mp4" % (stem, number))
            while os.path.exists(target):
                number += 1
                target = os.path.join(folder, "%s_%03d.mp4" % (stem, number))
            os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return web.json_response({
        "filename": os.path.basename(target),
        "subfolder": "%s/%s/exports" % (store.ROOT, name),
        "bytes": os.path.getsize(target), "plan": plan,
    })


def _extract_rgb_frames(video_path, indices):
    """Decode only the requested 0-based frame indices from video_path as uint8 RGB numpy arrays."""
    import numpy as np

    wanted = sorted({int(i) for i in indices if int(i) >= 0})
    if not wanted:
        return {}
    wanted_set = set(wanted)
    stop = wanted[-1] + 1
    found = {}
    try:
        import av

        with av.open(video_path) as container:
            stream = container.streams.video[0]
            for idx, frame in enumerate(container.decode(stream)):
                if idx in wanted_set:
                    found[idx] = frame.to_ndarray(format="rgb24")
                if idx + 1 >= stop:
                    break
        return found
    except Exception:
        pass
    # Fallback via ffmpeg rawvideo when PyAV is unavailable
    import subprocess
    from PIL import Image

    info = media.video_info(video_path, count=False)
    w, h = int(info["width"]), int(info["height"])
    for idx in wanted:
        ts = idx / float(store.FPS)
        out = subprocess.check_output(
            media.command(["ffmpeg", "-v", "error", "-ss", f"{ts:.6f}", "-i", video_path,
                           "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        )
        if len(out) == w * h * 3:
            found[idx] = np.frombuffer(out, dtype=np.uint8).reshape((h, w, 3))
    return found


def _project_folder_path(folder):
    return os.path.join(root(), store.slug(folder))


def _project_rel_path(folder, abs_path):
    base = os.path.realpath(folder_paths.get_output_directory())
    real = os.path.realpath(abs_path)
    proj = os.path.realpath(_project_folder_path(folder))
    if not (real == proj or real.startswith(proj + os.sep)):
        raise web.HTTPForbidden(text="path outside project")
    return os.path.relpath(real, base).replace("\\", "/")


def _safe_project_file(folder, rel_file):
    real = _output_rel_path(rel_file)
    proj = os.path.realpath(_project_folder_path(folder))
    if not real.startswith(proj + os.sep):
        raise web.HTTPForbidden(text="path outside project")
    return real


def _view_output_url(filename, subfolder=""):
    from urllib.parse import urlencode

    return "/view?" + urlencode({"type": "output", "filename": filename, "subfolder": subfolder or ""})


def _preview_url(folder, kind, rel_file, first=0, last=0, expand_px=32, bust=0):
    from urllib.parse import urlencode

    params = {"kind": kind, "file": rel_file}
    if kind != "cutout":
        params.update({"first": int(first), "last": int(last)})
    if kind == "mask":
        params["expand_px"] = int(expand_px)
    if bust:
        params["v"] = int(bust)
    return "/h3take/api/project/%s/motion/preview?%s" % (folder, urlencode(params))


def _motion_pipeline_report(data, folder, clip_id=None, available=None):
    """Build complete Motion Transfer pipeline status, stage breakdown, and project cache inventory."""
    available = available_models() if available is None else available
    ms = store.motion_status(available)
    setup = motion_setup_report()
    models = []
    for f in setup.get("files") or []:
        models.append({
            "id": f["id"], "title": f["title"], "group": f.get("group") or "replace",
            "installed": bool(f.get("installed")), "bytes": int(f.get("bytes") or 0),
        })
    # Also surface RelayVSR 2x (Sharp) and base/flash H3 status
    relay_ok = any(row["id"] == "relayvsr" and row["installed"] for row in store.upscaler_status(available))
    models.append({
        "id": "relayvsr", "title": "RelayVSR 2x (Sharp upscale)", "group": "upscale",
        "installed": bool(relay_ok), "bytes": 0,
    })

    clips = data.get("clips") or []
    clip_map = {c["id"]: c for c in clips}
    motion_clips = []
    source_ids = set()
    for c in clips:
        if c.get("motion"):
            m = c["motion"]
            src = m.get("source") or {}
            if src.get("clip"):
                source_ids.add(src["clip"])
            take = store.active_take(c)
            motion_clips.append({
                "id": c["id"], "name": c.get("name") or "Motion",
                "mode": m.get("mode") or "replace",
                "preset": m.get("preset") or "standard",
                "style": m.get("style") or "auto",
                "frames": int(src.get("out", 0)) - int(src.get("in", 0)),
                "rendered": bool(take),
                "quality": take.get("quality") if take else None,
            })

    # Inventory all cached masks, poses, and cutouts across the project
    caches = []
    seen_sources = set()
    seen_chars = set()

    def scan_source(rel_file, label, clip_ref_id=None):
        if not rel_file or rel_file in seen_sources:
            return
        seen_sources.add(rel_file)
        try:
            src_path = _safe_project_file(folder, rel_file)
        except web.HTTPException:
            return
        if not os.path.exists(src_path):
            return
        m_sum = store.motion_replace.mask_cache_summary(src_path, subject="person")
        if m_sum:
            bust = int(os.path.getmtime(m_sum["path"]))
            first_span = m_sum["spans"][0] if m_sum["spans"] else [0, m_sum["frames"]]
            caches.append({
                "kind": "mask", "label": label, "clip": clip_ref_id, "file": rel_file,
                "cache_file": m_sum["file"], "size_bytes": m_sum["size_bytes"],
                "cached_frames": m_sum["cached_frames"], "total_frames": m_sum["frames"],
                "spans": m_sum["spans"],
                "preview_url": _preview_url(folder, "mask", rel_file, first_span[0], first_span[1], 32, bust),
            })
        p_sum = store.motion_replace.pose_cache_summary(src_path)
        if p_sum:
            bust = int(os.path.getmtime(p_sum["path"]))
            first_span = p_sum["spans"][0] if p_sum["spans"] else [0, p_sum["frames"]]
            caches.append({
                "kind": "pose", "label": label, "clip": clip_ref_id, "file": rel_file,
                "cache_file": p_sum["file"], "size_bytes": p_sum["size_bytes"],
                "cached_frames": p_sum["cached_frames"], "total_frames": p_sum["frames"],
                "detected_frames": p_sum["detected_frames"], "spans": p_sum["spans"],
                "preview_url": _preview_url(folder, "pose", rel_file, first_span[0], first_span[1], 0, bust),
            })

    def scan_character(asset, label):
        if not isinstance(asset, dict) or not asset.get("file"):
            return
        rel_file = "/".join(filter(None, (asset.get("subfolder"), asset.get("file"))))
        if not rel_file or rel_file in seen_chars:
            return
        seen_chars.add(rel_file)
        try:
            char_path = _safe_project_file(folder, rel_file)
        except web.HTTPException:
            return
        c_sum = store.motion_replace.cutout_cache_summary(char_path, subject="person", backdrop="white")
        if c_sum:
            bust = int(os.path.getmtime(c_sum["path"]))
            caches.append({
                "kind": "cutout", "label": label, "file": rel_file,
                "cache_file": c_sum["file"], "size_bytes": c_sum["size_bytes"],
                "preview_url": _preview_url(folder, "cutout", rel_file, bust=bust),
            })

    for c in clips:
        if c.get("motion"):
            src = c["motion"].get("source") or {}
            src_rel = "/".join(filter(None, (src.get("subfolder"), src.get("file"))))
            src_clip = clip_map.get(src.get("clip"))
            scan_source(src_rel, (src_clip.get("name") if src_clip else src.get("file")) or c.get("name"), src.get("clip"))
            char = c["motion"].get("character")
            cast_member = next((p for p in (data.get("cast") or []) if p["id"] == c["motion"].get("cast")), None)
            scan_character(char, (cast_member.get("name") if cast_member else None) or (char or {}).get("file") or "Character")
        elif c.get("kind") == "footage" or c["id"] in source_ids:
            t = store.active_take(c)
            if t and t.get("file"):
                scan_source("/".join(filter(None, (t.get("subfolder"), t.get("file")))), c.get("name") or t["file"], c["id"])

    # Pick inspected clip
    target = clip_map.get(clip_id) if clip_id else None
    if target is None and motion_clips:
        target = clip_map.get(motion_clips[0]["id"])
    if target is None:
        target = next((c for c in clips if c.get("kind") == "footage"), None)

    clip_report = None
    if target is not None:
        if target.get("motion"):
            m = target["motion"]
            src = m.get("source") or {}
            mode = m.get("mode") or "replace"
            preset = m.get("preset") or "standard"
            style_cfg = m.get("style") or "auto"
            char = m.get("character") or {}
            resolved_style = detect_motion_style(char) if style_cfg == "auto" else style_cfg
            swap_default = ((0.85 if preset != "standard" else 1.0) if resolved_style == "photoreal" else 0.5) if (mode == "replace" and ms.get("swap_lora")) else 0.0
            swap_eff = float(m["swap_lora"]) if (mode == "replace" and m.get("swap_lora") is not None) else swap_default
            pose_strength = float(m.get("strength", 0.0 if mode in ("replace", "rescene") else 1.0))
            in_f, out_f = int(src.get("in") or 0), int(src.get("out") or 0)
            exact_f = max(1, out_f - in_f)
            src_total = max(out_f, int(src.get("frames") or out_f))
            if preset in ("realism_fast", "realism"):
                windows = store.motion_replace.plan_relay_windows(
                    exact_f,
                    single_max=store.motion_replace.REALISM_WINDOW_FRAMES,
                    window_frames=store.motion_replace.REALISM_WINDOW_FRAMES,
                    overlap_frames=store.motion_replace.OVERLAP_FRAMES,
                )
            else:
                windows = store.motion_replace.plan_relay_windows(exact_f)
            needed_f = windows[-1]["start"] + windows[-1]["length"]
            load_last = min(src_total, in_f + needed_f)
            src_rel = "/".join(filter(None, (src.get("subfolder"), src.get("file"))))
            char_rel = "/".join(filter(None, (char.get("subfolder"), char.get("file"))))
            expand_px = int(m.get("expand_px", 32))
            draft_w, draft_h, _ = store.render_size(data, "draft")
            if mode in ("replace", "rescene"):
                med_w, med_h = store.format_sizes(store.project_format(data))["medium"]
                if draft_w * draft_h < med_w * med_h:
                    draft_w, draft_h = med_w, med_h
            profile_id, profile = store.project_profile(data, available, target)
            draft_steps = max(6, profile["steps"]["draft"]) if mode in ("replace", "rescene") and profile_id == "hyperflow" else profile["steps"]["draft"]

            src_path = None
            char_path = None
            try:
                src_path = _safe_project_file(folder, src_rel) if src_rel else None
            except web.HTTPException:
                pass
            try:
                char_path = _safe_project_file(folder, char_rel) if char_rel else None
            except web.HTTPException:
                pass

            m_sum = store.motion_replace.mask_cache_summary(src_path, subject="person") if src_path else None
            m_hit = bool(src_path and store.motion_replace.has_mask_cache(src_path, in_f, load_last, subject="person"))
            p_sum = store.motion_replace.pose_cache_summary(src_path) if src_path else None
            p_hit = bool(src_path and store.motion_replace.has_pose_cache(src_path, in_f, load_last))
            c_sum = store.motion_replace.cutout_cache_summary(char_path, subject="person", backdrop="white") if char_path else None
            c_hit = bool(c_sum)

            isolate_on = bool(m.get("isolate", mode != "rescene"))
            plate_on = mode == "replace" and bool(m.get("plate", True))
            sharp_on = bool(preset == "fast_sharp" or (preset not in ("realism_fast", "realism") and (m.get("sharp") or store.size_name(data, "final") == store.SHARP)))
            active_t = store.active_take(target)

            preset_label = {
                "realism_fast": "F1-Fast (1088p 2-step H3Sampler)",
                "realism": "F1 Max (1088p 3-step H3Sampler)",
                "fast_sharp": "F2 Fast Sharp (768p + RelayVSR 2x 0.85)",
                "standard": "Classic",
            }.get(preset, preset)

            stages = [
                {
                    "id": "source", "title": "Source frames", "component": "H3LoadClip",
                    "nodes": ["213"],
                    "status": "ready" if (src_path and os.path.exists(src_path)) else "missing",
                    "badge": f"{exact_f}f ({exact_f / store.FPS:.2f}s)",
                    "detail": (f"Frames {in_f}–{out_f - 1} of {src_total}"
                               + (f" (+{load_last - out_f}f real look-ahead to {load_last})" if load_last > out_f
                                  else f" (reflect-padded +{needed_f - exact_f}f to grid)" if needed_f > exact_f else " (exact grid)")),
                },
                {
                    "id": "cutout", "title": "Character cutout & style", "component": "H3IsolateCharacter",
                    "nodes": ["214", "239"],
                    "status": ("off" if not isolate_on else "cached" if c_hit else "ready" if ms.get("sam3") else "missing"),
                    "badge": ("Bypassed" if not isolate_on else "Cached" if c_hit else "Uncached"),
                    "detail": (f"Style: {style_cfg}" + (f" -> {resolved_style}" if style_cfg == "auto" else "")
                               + (f" · Swap LoRA {swap_eff:.2f}" + (" (sigma 1.0→0.45)" if preset != "standard" else "") if mode == "replace" else " · No Swap LoRA (pure motion)")
                               + (f" · {c_sum['size_bytes'] // 1024} KB" if c_sum else "")),
                    "original_url": _view_output_url(char.get("file", ""), char.get("subfolder", "")) if char.get("file") else None,
                    "preview_url": _preview_url(folder, "cutout", char_rel, bust=int(os.path.getmtime(c_sum["path"]))) if c_sum else None,
                    "cache": {"kind": "cutout", "file": char_rel} if c_sum else None,
                },
                {
                    "id": "segmentation", "title": "Subject segmentation", "component": "SAM 3.1 · H3TrackSubjectMask",
                    "nodes": ["238", "240", "245"],
                    "status": ("off" if mode != "replace" else "cached" if m_hit else "partial" if m_sum else "ready" if ms.get("replace") else "missing"),
                    "badge": ("Off (new-scene mode)" if mode == "rescene" else "Off (pose mode)" if mode != "replace" else "Cached" if m_hit else f"Partial ({m_sum['cached_frames']}/{m_sum['frames']}f)" if m_sum else "Not cached"),
                    "detail": ("Bypassed for pure new-scene motion transfer (unmasked RGB <Video 1>)" if mode == "rescene"
                               else f"Expand {expand_px}px · "
                               + (f"Reused on next render · {m_sum['cached_frames']}/{m_sum['frames']}f ({m_sum['size_bytes'] / 1048576:.2f} MB)"
                                  if m_hit and m_sum else
                                  f"Cached spans: {m_sum['spans']} ({m_sum['size_bytes'] / 1048576:.2f} MB)" if m_sum
                                  else "Will run SAM 3.1 once and cache beside source video")),
                    "coverage": {"cached": m_sum["cached_frames"], "total": m_sum["frames"], "spans": m_sum["spans"]} if m_sum else None,
                    "preview_url": (_preview_url(folder, "mask", src_rel, in_f, load_last, expand_px, int(os.path.getmtime(m_sum["path"])))
                                    if m_sum and mode == "replace" else None),
                    "cache": {"kind": "mask", "file": src_rel} if m_sum else None,
                },
                {
                    "id": "pose", "title": "Pose keypoints", "component": "RT-DETR + SDPose · H3GatedSDPose",
                    "nodes": ["215", "216", "217", "218", "219"],
                    "status": ("off" if pose_strength <= 0 else "cached" if p_hit else "partial" if p_sum else "ready" if ms.get("pose_control") else "missing"),
                    "badge": ("Off (0.0)" if pose_strength <= 0 else "Cached · loaders skipped" if p_hit else f"Partial ({p_sum['cached_frames']}/{p_sum['frames']}f)" if p_sum else "Not cached"),
                    "detail": (f"ControlNet strength {pose_strength:.2f}"
                               + (f" · {p_sum['detected_frames']}/{p_sum['cached_frames']}f with pose ({p_sum['size_bytes'] // 1024} KB)" if p_sum else
                                  " · Pure M5 video-motion (no extra pose models)" if (mode == "rescene" and pose_strength <= 0) else
                                  " · No pose reinforcement (SAM3 mask + video ref only)" if pose_strength <= 0 else
                                  " · M7 pose lock + face landmarks (Pass-1 only)" if mode == "rescene" else " · Will run RT-DETR + SDPose once and cache")),
                    "coverage": {"cached": p_sum["cached_frames"], "total": p_sum["frames"], "spans": p_sum["spans"]} if p_sum else None,
                    "preview_url": (_preview_url(folder, "pose", src_rel, in_f, load_last, 0, int(os.path.getmtime(p_sum["path"])))
                                    if p_sum else None),
                    "cache": {"kind": "pose", "file": src_rel} if p_sum else None,
                },
                {
                    "id": "sampling", "title": "Diffusion & Refine", "component": "H3Take" + (" + H3ImageSampler" if preset != "standard" else "") + (" + H3SwapGuard" if mode == "replace" else ""),
                    "nodes": ["10", "15", "248", "16", "80", "82", "83", "86", "90", "91"],
                    "status": "ready",
                    "badge": f"{len(windows)} window{'s' if len(windows) != 1 else ''} · {preset_label}",
                    "detail": (f"Preset: {preset_label} · Draft floor {draft_w}×{draft_h}, {draft_steps} steps · "
                               + ("Swap Guard auto-retry on Window 0" if mode == "replace"
                                  else ("Unmasked RGB + Fun ControlNet 2.0 (M7)" if pose_strength > 0 else "Unmasked RGB pure motion (M5)") if mode == "rescene"
                                  else "Pose ControlNet choreography")
                               + (f" · {len(windows)} relay windows (22f double latent carry)" if (len(windows) > 1 and preset in ("realism_fast", "realism"))
                                  else f" · {len(windows)} relay windows (22f overlap)" if len(windows) > 1 else "")),
                    "windows": [
                        {"index": idx + 1, "start": in_f + w["start"], "end": min(out_f, in_f + w["start"] + w["length"]),
                         "length": w["length"], "overlap": w["overlap"]}
                        for idx, w in enumerate(windows)
                    ],
                },
                {
                    "id": "plate", "title": "Plate composite & output", "component": ("H3CompositeSourcePlate" if plate_on else "Direct 1088p output") + (" + RelayVSR 2x" if sharp_on else ""),
                    "nodes": ["250", "33", "34", "35", "36", "18", "19"],
                    "status": "cached" if active_t else ("ready" if (not sharp_on or relay_ok) else "missing"),
                    "badge": (f"{active_t['quality'].title()} · {active_t['frames']}f" if active_t else ("Fast Sharp F2 ready" if preset == "fast_sharp" else "Sharp 2x ready" if sharp_on else "Pending render")),
                    "detail": (("Zero-halo plate on (src +18px / gen +4px)" if plate_on else "Source plate off (full new scene)")
                               + (" · RelayVSR 2x (0.85) before 1080p plate" if preset == "fast_sharp" else " · RelayVSR 2x final enabled" if sharp_on else "")
                               + (f" · Active take {active_t.get('size', [0, 0])[0]}×{active_t.get('size', [0, 0])[1]}" if active_t and active_t.get("size") else "")),
                    "preview_url": _view_output_url(active_t["strip"]["sprite"], active_t.get("subfolder", ""))
                                   if active_t and active_t.get("strip", {}).get("sprite") else None,
                },
            ]
            clip_report = {
                "id": target["id"], "kind": "motion", "name": target.get("name") or "Motion",
                "mode": mode, "preset": preset, "style": style_cfg, "resolved_style": resolved_style,
                "swap_lora": swap_eff, "exact_frames": exact_f, "grid_frames": needed_f,
                "stages": stages,
            }
        else:
            t = store.active_take(target)
            src_rel = "/".join(filter(None, (t.get("subfolder"), t.get("file")))) if t and t.get("file") else ""
            src_path = None
            try:
                src_path = _safe_project_file(folder, src_rel) if src_rel else None
            except web.HTTPException:
                pass
            m_sum = store.motion_replace.mask_cache_summary(src_path, subject="person") if src_path else None
            p_sum = store.motion_replace.pose_cache_summary(src_path) if src_path else None
            stages = [
                {
                    "id": "segmentation", "title": "Cached SAM 3.1 masks", "component": "H3TrackSubjectMask",
                    "nodes": ["240"],
                    "status": "cached" if m_sum else "ready",
                    "badge": f"{m_sum['cached_frames']}/{m_sum['frames']}f" if m_sum else "Not segmented yet",
                    "detail": (f"Spans {m_sum['spans']} · {m_sum['size_bytes'] / 1048576:.2f} MB — reused automatically by all splits"
                               if m_sum else "Run Motion transfer on this footage to populate reusable SAM 3.1 masks."),
                    "coverage": {"cached": m_sum["cached_frames"], "total": m_sum["frames"], "spans": m_sum["spans"]} if m_sum else None,
                    "preview_url": (_preview_url(folder, "mask", src_rel, m_sum["spans"][0][0], m_sum["spans"][0][1], 32, int(os.path.getmtime(m_sum["path"])))
                                    if m_sum and m_sum["spans"] else None),
                    "cache": {"kind": "mask", "file": src_rel} if m_sum else None,
                },
                {
                    "id": "pose", "title": "Cached pose keypoints", "component": "RT-DETR + SDPose",
                    "nodes": ["218"],
                    "status": "cached" if p_sum else "ready",
                    "badge": f"{p_sum['cached_frames']}/{p_sum['frames']}f" if p_sum else "Not extracted yet",
                    "detail": (f"{p_sum['detected_frames']}/{p_sum['cached_frames']}f with person · {p_sum['size_bytes'] // 1024} KB"
                               if p_sum else "Cached automatically whenever pose reinforcement or choreography runs."),
                    "coverage": {"cached": p_sum["cached_frames"], "total": p_sum["frames"], "spans": p_sum["spans"]} if p_sum else None,
                    "preview_url": (_preview_url(folder, "pose", src_rel, p_sum["spans"][0][0], p_sum["spans"][0][1], 0, int(os.path.getmtime(p_sum["path"])))
                                    if p_sum and p_sum["spans"] else None),
                    "cache": {"kind": "pose", "file": src_rel} if p_sum else None,
                },
            ]
            clip_report = {
                "id": target["id"], "kind": "source", "name": target.get("name") or "Footage",
                "stages": stages,
            }

    return {
        "capabilities": {"replace": bool(ms.get("replace")), "pose_control": bool(ms.get("pose_control")),
                         "swap_lora": bool(ms.get("swap_lora")), "relayvsr": bool(relay_ok)},
        "models": models,
        "clip": clip_report,
        "motion_clips": motion_clips,
        "caches": caches,
    }


async def motion_pipeline_status(request):
    folder = store.slug(request.match_info["name"])
    data = read_project(folder)
    clip_id = request.query.get("clip") or None
    report = await asyncio.to_thread(_motion_pipeline_report, data, folder, clip_id)
    return web.json_response(report)


async def motion_preview(request):
    folder = store.slug(request.match_info["name"])
    kind = request.query.get("kind") or ""
    rel_file = request.query.get("file") or ""
    if kind not in ("mask", "pose", "cutout") or not rel_file:
        return refused("send kind (mask, pose, or cutout) and file")
    target = _safe_project_file(folder, rel_file)
    if not os.path.exists(target):
        return refused("source file not found", 404)
    if kind == "cutout":
        summary = store.motion_replace.cutout_cache_summary(target, subject="person", backdrop="white")
        if not summary:
            return refused("character cutout is not cached yet", 404)
        return web.FileResponse(summary["path"], headers={"Cache-Control": "no-cache"})

    first = max(0, int(request.query.get("first") or 0))
    last = max(0, int(request.query.get("last") or 0))
    expand_px = max(0, min(128, int(request.query.get("expand_px") or 32)))

    def build_sheet():
        if kind == "mask":
            summary = store.motion_replace.mask_cache_summary(target, subject="person")
            if not summary or not summary["spans"]:
                raise FileNotFoundError("segmentation mask is not cached yet")
            lo = first if last > first else summary["spans"][0][0]
            hi = min(last, summary["frames"]) if last > first else summary["spans"][0][1]
            # Pick frames that are actually cached inside [lo, hi)
            cached_indices = []
            for a, b in summary["spans"]:
                ca, cb = max(lo, a), min(hi, b)
                if cb > ca:
                    cached_indices.extend(range(ca, cb))
            if not cached_indices:
                a, b = summary["spans"][0]
                cached_indices = list(range(a, b))
            picked = [cached_indices[i] for i in store.motion_replace.sample_frame_indices(0, len(cached_indices), 6)]
            frames_map = _extract_rgb_frames(target, picked)
            rgbs, indices, masks = [], [], []
            for fidx in picked:
                if fidx not in frames_map:
                    continue
                m = store.motion_replace.load_mask_cache(target, fidx, fidx + 1, subject="person")
                if m is None:
                    continue
                rgbs.append(frames_map[fidx])
                indices.append(fidx)
                masks.append(m[0])
            return store.motion_replace.render_mask_sheet(rgbs, indices, masks, expand_px=expand_px)

        summary = store.motion_replace.pose_cache_summary(target)
        if not summary or not summary["spans"]:
            raise FileNotFoundError("pose keypoints are not cached yet")
        lo = first if last > first else summary["spans"][0][0]
        hi = min(last, summary["frames"]) if last > first else summary["spans"][0][1]
        cached_indices = []
        for a, b in summary["spans"]:
            ca, cb = max(lo, a), min(hi, b)
            if cb > ca:
                cached_indices.extend(range(ca, cb))
        if not cached_indices:
            a, b = summary["spans"][0]
            cached_indices = list(range(a, b))
        picked = [cached_indices[i] for i in store.motion_replace.sample_frame_indices(0, len(cached_indices), 6)]
        frames_map = _extract_rgb_frames(target, picked)
        rgbs, indices, kps = [], [], []
        for fidx in picked:
            if fidx not in frames_map:
                continue
            kp = store.motion_replace.load_pose_cache(target, fidx, fidx + 1)
            if kp is None:
                continue
            rgbs.append(frames_map[fidx])
            indices.append(fidx)
            kps.append(kp[0])
        return store.motion_replace.render_pose_sheet(rgbs, indices, kps)

    try:
        jpeg = await asyncio.to_thread(build_sheet)
    except FileNotFoundError as error:
        return refused(str(error), 404)
    except (ValueError, OSError) as error:
        return refused(plain(error))
    return web.Response(body=jpeg, content_type="image/jpeg", headers={"Cache-Control": "no-cache"})


async def motion_cache_clear(request):
    folder = store.slug(request.match_info["name"])
    body = await request.json()
    kind = body.get("kind") or "all"
    rel_file = body.get("file") or ""
    if kind not in ("mask", "pose", "cutout", "all"):
        return refused("kind must be mask, pose, cutout, or all")
    removed = []
    if kind == "all" and not rel_file:
        proj = _project_folder_path(folder)
        for sub in ("clips", "assets"):
            d = os.path.join(proj, sub)
            if not os.path.isdir(d):
                continue
            for entry in os.listdir(d):
                if (".mask." in entry and entry.endswith(".npz")) or entry.endswith(".pose.json.gz") or (".iso." in entry and entry.endswith(".png")):
                    full = os.path.join(d, entry)
                    try:
                        os.remove(full)
                        removed.append(entry)
                    except OSError:
                        pass
    else:
        if not rel_file:
            return refused("send file to clear a specific cache")
        target = _safe_project_file(folder, rel_file)
        candidates = []
        if kind in ("mask", "all"):
            candidates.append(store.motion_replace.mask_cache_path(target, "person"))
        if kind in ("pose", "all"):
            candidates.append(store.motion_replace.pose_cache_path(target))
        if kind in ("cutout", "all"):
            candidates.append(store.motion_replace.cutout_cache_path(target, "person", "white"))
        for path in candidates:
            if os.path.exists(path):
                try:
                    os.remove(path)
                    removed.append(os.path.basename(path))
                except OSError:
                    pass
    return web.json_response({"removed": len(removed), "files": removed})


ROUTES = (
    ("GET", "/h3take/api/sizes", "sizes"),
    ("GET", "/h3take/api/queue", "render_queue"),
    ("POST", "/h3take/api/queue/cancel", "cancel_queue"),
    ("POST", "/h3take/api/queue/clear-history", "clear_queue_history"),
    ("GET", "/h3take/editor", "page"),
    ("GET", "/h3take/api/projects", "projects"),
    ("POST", "/h3take/api/projects", "create"),
    ("POST", "/h3take/api/projects/open", "open_project"),
    ("GET", "/h3take/api/models/setup", "model_setup"),
    ("POST", "/h3take/api/models/setup/download", "model_setup_download"),
    ("POST", "/h3take/api/models/setup/cancel", "model_setup_cancel"),
    ("GET", "/h3take/api/models/motion", "motion_model_setup"),
    ("POST", "/h3take/api/models/motion/download", "motion_model_download"),
    ("POST", "/h3take/api/models/motion/cancel", "motion_model_cancel"),
    ("GET", "/h3take/api/presets", "presets"),
    ("PUT", "/h3take/api/presets", "save_presets"),
    ("GET", "/h3take/api/cast-catalog", "cast_catalog"),
    ("POST", "/h3take/api/cast-catalog/remove", "remove_from_catalog"),
    ("POST", "/h3take/api/cast-catalog/spread", "spread_catalog"),
    ("GET", "/h3take/api/project/{name}", "load"),
    ("PUT", "/h3take/api/project/{name}", "save"),
    ("DELETE", "/h3take/api/project/{name}", "remove"),
    ("POST", "/h3take/api/project/{name}/rename", "rename"),
    ("POST", "/h3take/api/project/{name}/duplicate", "duplicate"),
    ("GET", "/h3take/api/project/{name}/archive", "archive_project"),
    ("GET", "/h3take/api/project/{name}/storage", "project_storage"),
    ("POST", "/h3take/api/project/{name}/storage", "project_storage"),
    ("GET", "/h3take/api/project/{name}/summary", "summary"),
    ("GET", "/h3take/api/project/{name}/motion", "motion_pipeline_status"),
    ("GET", "/h3take/api/project/{name}/motion/preview", "motion_preview"),
    ("POST", "/h3take/api/project/{name}/motion/cache/clear", "motion_cache_clear"),
    ("GET", "/h3take/api/project/{name}/cuts", "cuts"),
    ("GET", "/h3take/api/project/{name}/detail", "detail"),
    ("GET", "/h3take/api/project/{name}/history", "history"),
    ("POST", "/h3take/api/project/{name}/restore", "restore"),
    ("POST", "/h3take/api/project/{name}/ops", "operations"),
    ("POST", "/h3take/api/project/{name}/render", "render"),
    ("POST", "/h3take/api/project/{name}/sprite-export", "sprite_export"),
    ("POST", "/h3take/api/project/{name}/sprite-atlas", "sprite_atlas"),
    ("POST", "/h3take/api/project/{name}/sprite-self-pin", "sprite_self_pin"),
    ("GET", "/h3take/api/project/{name}/props", "list_props"),
    ("POST", "/h3take/api/project/{name}/props", "sprite_prop"),
    ("POST", "/h3take/api/project/{name}/cancel", "cancel"),
    ("GET", "/h3take/api/project/{name}/wait", "wait"),
    ("POST", "/h3take/api/project/{name}/graph", "graph"),
    ("POST", "/h3take/api/project/{name}/export", "export"),
    ("POST", "/h3take/api/project/{name}/asset", "upload"),
    ("POST", "/h3take/api/project/{name}/rate", "rate"),
    ("GET", "/h3take/api/settings", "settings"),
    ("GET", "/h3take/api/looks", "look_library"),
    ("POST", "/h3take/api/looks", "save_look"),
    ("DELETE", "/h3take/api/looks/{look}", "remove_look"),
    ("PUT", "/h3take/api/settings", "save_settings"),
    ("POST", "/h3take/api/project/{name}/effect/{effect}/frames", "effect_frames"),
    ("POST", "/h3take/api/project/{name}/effect/{effect}/take", "effect_take"),
    ("POST", "/h3take/api/project/{name}/grab", "grab"),
    ("POST", "/h3take/api/project/{name}/grab_sound", "grab_sound"),
    ("POST", "/h3take/api/project/{name}/import", "import_media"),
    ("POST", "/h3take/api/probe", "probe"),
    ("POST", "/h3take/api/strip", "strip"),
    ("POST", "/h3take/api/spectrogram", "spectrogram"),
    ("POST", "/h3take/api/sheet", "sheet"),
)


def install():
    """Routes look handlers up by name, so a hot reload swaps them without re-registering."""
    from server import PromptServer

    routes = PromptServer.instance.routes
    module = sys.modules[__name__].__name__

    def bound(handler):
        async def call(request):
            return await getattr(sys.modules[module], handler)(request)

        return call

    for method, path, handler in ROUTES:
        routes.route(method, path)(bound(handler))
    routes.static("/h3take/editor-files/", PAGE)
    PromptServer.instance.loop.call_soon_threadsafe(lambda: getattr(sys.modules[module], "ensure_watcher")())
