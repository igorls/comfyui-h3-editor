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
from . import cuts as cutscore
from . import detail as finescore
from . import media
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

    status = store.encoder_status(available)
    return {"text_encoders": listed("text_encoders", "clip"),
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

    names = set()
    for kind in ("diffusion_models", "unet_gguf", "loras", "latent_upscale_models", "text_encoders", "clip", "vae"):
        if kind in folder_paths.folder_names_and_paths:
            names.update(folder_paths.get_filename_list(kind))
    names.update(store.REFMOD_PREFIX + name for name in refmod_files() or [])
    return names


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


def project_titles(names):
    """The name each project shows, which a rename changes while its folder stays."""
    titles = {}
    for name in names:
        try:
            with open(os.path.join(root(), name, "project.json"), encoding="utf-8") as f:
                titles[name] = str(json.load(f).get("name") or name.replace("_", " "))
        except (OSError, ValueError):
            titles[name] = name.replace("_", " ")
    return titles


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
        async with lock(made):
            commit(made, opened, "open", request.query.get("client"))
        have = {name.lower() for name in lora_files()}
        missing = [name for name in manifest.get("models", {}).get("loras", []) if name.lower() not in have]
        return web.json_response({"folder": made, "name": title, "release": manifest.get("release"), "keep": manifest.get("keep"),
                                  "missing_loras": missing, "outside": manifest.get("outside", []), "files": manifest.get("files", 0)})
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


async def projects(request):
    names = project_folders()
    available = available_models()
    return web.json_response({"projects": names, "titles": project_titles(names), "profiles": store.profile_status(available),
                              "finals": store.finals_status(available),
                              "loras": lora_files(), "realism": store.lora_status(available), "presets": read_presets(), "refmods": refmods_status(), "encoders": encoder_choices(available), "sizes": store.size_choices(), "upscalers": store.upscaler_status(available), "release": pack_version(), **sampling_choices()})


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
        merged = store.merge(base, mine, current)
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
    found, edges, inside = await asyncio.gather(score_cuts(data), score_effect_cuts(data), score_inside(data))
    scored = [c["score"] for c in found + edges + inside if c["score"] is not None]
    return web.json_response({"cuts": found, "effects": edges, "inside": inside, "worst": min(scored) if scored else None,
                              "revision": data.get("revision", 0)})


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
            return dict(entry, **CUT_SCORES[key])
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
        restored = store.merge(current, old, current)
        commit(folder, restored, "restore", body.get("client"))
    return web.json_response({"revision": restored["revision"], "summary": store.describe(restored)})


async def graph(request):
    name = request.match_info["name"]
    body = await request.json()
    try:
        built = store.render_graph(body["project"], store.slug(name), body["clip"], body["take"], body["quality"], body["seed"],
                                   available_models())
    except ValueError as error:
        raise web.HTTPBadRequest(text=str(error))
    return web.json_response({"graph": own_nodes(built)})


async def probe_file(path):
    """Frame count, a poster and a filmstrip for a finished render."""
    frames = (await video_info(path))["frames"]
    poster = os.path.splitext(path)[0] + ".jpg"
    await run(["ffmpeg", "-y", "-v", "error", "-i", path, "-vf", "select=eq(n\\,12),scale=480:-2", "-frames:v", "1", poster])
    try:
        made = await make_strip(path, frames)
    except web.HTTPException:
        made = None
    return {"frames": frames, "poster": os.path.basename(poster), "strip": made}


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


async def start_render(data, folder, clip_id, quality, seed=None, client_id=None, detail=None, upscaler=None):
    job, graph = store.plan_render(data, folder, clip_id, quality or "draft", seed, available_models(), detail, upscaler)
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
                                     body.get("client_id"), body.get("detail"), body.get("upscaler"))
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
            job = find_job(data, body)
        except KeyError as error:
            return refused(plain(error), 404)
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
        for index, op in enumerate(ops):
            kind = op.get("op") if isinstance(op, dict) else None
            try:
                if kind == "render":
                    clip_id = store._resolve(op.get("clip"), names, "clip")
                    planned, _ = store.plan_render(copy.deepcopy(work), folder, clip_id, op.get("quality") or "draft",
                                                   op.get("seed"), available_models(), op.get("detail"), op.get("upscaler"))
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
                    job = await start_render(work, folder, clip_id, op.get("quality"), op.get("seed"), op.get("client_id"), op.get("detail"), op.get("upscaler"))
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
            if not file:
                finished.append((job, None, None, failure(entry)))
                continue
            try:
                path = output_path(file.get("subfolder"), file["filename"])
                if job.get("track"):
                    await lay_track(path, job["track"])
                info = await probe_file(path)
                finished.append((job, file, info, None))
            except web.HTTPException as error:
                finished.append((job, None, None, "The render saved a file that cannot be read: %s" % error.text))
        if not finished:
            continue
        async with lock(folder):
            data = read_project(folder)
            for job, file, info, message in finished:
                if file:
                    store.finish_render(data, job["id"], file, info)
                else:
                    store.fail_render(data, job["id"], message)
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


PICTURES = (".png", ".jpg", ".jpeg", ".webp")
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
        plan = store.export_plan(data, scale)
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
    number = 1
    while os.path.exists(os.path.join(folder, "%s_%03d.mp4" % (stem, number))):
        number += 1
    target = os.path.join(folder, "%s_%03d.mp4" % (stem, number))
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
        args = store.export_command(data, paths, target, scale=scale, quality=quality, sound_paths=sound_paths,
                                    effect_paths=effect_paths)
    except ValueError as error:
        raise web.HTTPBadRequest(text=str(error))
    await run(args)
    return web.json_response({
        "filename": os.path.basename(target),
        "subfolder": "%s/%s/exports" % (store.ROOT, name),
        "bytes": os.path.getsize(target), "plan": plan,
    })


ROUTES = (
    ("GET", "/h3take/editor", "page"),
    ("GET", "/h3take/api/projects", "projects"),
    ("POST", "/h3take/api/projects", "create"),
    ("POST", "/h3take/api/projects/open", "open_project"),
    ("GET", "/h3take/api/presets", "presets"),
    ("PUT", "/h3take/api/presets", "save_presets"),
    ("GET", "/h3take/api/project/{name}", "load"),
    ("PUT", "/h3take/api/project/{name}", "save"),
    ("DELETE", "/h3take/api/project/{name}", "remove"),
    ("POST", "/h3take/api/project/{name}/rename", "rename"),
    ("POST", "/h3take/api/project/{name}/duplicate", "duplicate"),
    ("GET", "/h3take/api/project/{name}/archive", "archive_project"),
    ("GET", "/h3take/api/project/{name}/summary", "summary"),
    ("GET", "/h3take/api/project/{name}/cuts", "cuts"),
    ("GET", "/h3take/api/project/{name}/detail", "detail"),
    ("GET", "/h3take/api/project/{name}/history", "history"),
    ("POST", "/h3take/api/project/{name}/restore", "restore"),
    ("POST", "/h3take/api/project/{name}/ops", "operations"),
    ("POST", "/h3take/api/project/{name}/render", "render"),
    ("POST", "/h3take/api/project/{name}/cancel", "cancel"),
    ("GET", "/h3take/api/project/{name}/wait", "wait"),
    ("POST", "/h3take/api/project/{name}/graph", "graph"),
    ("POST", "/h3take/api/project/{name}/export", "export"),
    ("POST", "/h3take/api/project/{name}/asset", "upload"),
    ("POST", "/h3take/api/project/{name}/rate", "rate"),
    ("GET", "/h3take/api/settings", "settings"),
    ("PUT", "/h3take/api/settings", "save_settings"),
    ("POST", "/h3take/api/project/{name}/effect/{effect}/frames", "effect_frames"),
    ("POST", "/h3take/api/project/{name}/effect/{effect}/take", "effect_take"),
    ("POST", "/h3take/api/project/{name}/grab", "grab"),
    ("POST", "/h3take/api/project/{name}/grab_sound", "grab_sound"),
    ("POST", "/h3take/api/project/{name}/import", "import_media"),
    ("POST", "/h3take/api/probe", "probe"),
    ("POST", "/h3take/api/strip", "strip"),
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
