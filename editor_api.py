"""HTTP routes for the editor. Projects live in ComfyUI's output folder, so renders stay with them."""

import asyncio
import copy
import json
import logging
import math
import os
import subprocess
import sys
import time
import uuid

from aiohttp import web

import folder_paths

from . import project as store

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


async def run(args):
    process = await asyncio.create_subprocess_exec(*args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = await process.communicate()
    if process.returncode != 0:
        raise web.HTTPInternalServerError(text=err.decode(errors="replace")[-600:])
    return out.decode(errors="replace")


async def run_bytes(args):
    process = await asyncio.create_subprocess_exec(*args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = await process.communicate()
    if process.returncode != 0:
        raise web.HTTPInternalServerError(text=err.decode(errors="replace")[-600:])
    return out


STRIP_HEIGHT = 90
STRIP_TILES = 48


async def make_strip(path, frames):
    """A row of frames from across a clip and its loudness per frame, made once and kept beside it."""
    base = os.path.splitext(path)[0]
    sprite, peaks = base + ".strip.jpg", base + ".peaks.json"
    size = (await run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                       "-of", "csv=p=0", path])).strip().split(",")
    width, height = int(size[0]), int(size[1])
    tile_w = max(2, int(round(STRIP_HEIGHT * width / float(height) / 2.0)) * 2)
    step = max(1, int(math.ceil(frames / float(STRIP_TILES))))
    count = max(1, int(math.ceil(frames / float(step))))
    if not os.path.exists(sprite):
        await run(["ffmpeg", "-y", "-v", "error", "-i", path, "-vf",
                   "select=not(mod(n\\,%d)),scale=%d:%d,tile=%dx1" % (step, tile_w, STRIP_HEIGHT, count),
                   "-vsync", "0", "-frames:v", "1", "-q:v", "4", sprite])
    if not os.path.exists(peaks):
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
    return {"sprite": os.path.basename(sprite), "peaks": os.path.basename(peaks), "w": tile_w, "h": STRIP_HEIGHT,
            "count": count, "step": step}


async def strip(request):
    body = await request.json()
    path = output_path(body.get("subfolder"), body["filename"])
    if not os.path.exists(path):
        raise web.HTTPNotFound(text="that clip is gone")
    return web.json_response(await make_strip(path, int(body.get("frames") or 1)))


async def page(request):
    """The page always comes fresh and names its files by version, so a browser never mixes old and new."""
    version = int(max(os.path.getmtime(os.path.join(PAGE, n)) for n in os.listdir(PAGE)))
    with open(os.path.join(PAGE, "index.html"), encoding="utf-8") as f:
        html = f.read().replace("__VERSION__", str(version))
    return web.Response(text=html, content_type="text/html", headers={"Cache-Control": "no-cache"})


async def projects(request):
    names = sorted(n for n in os.listdir(root()) if os.path.exists(os.path.join(root(), n, "project.json")))
    return web.json_response({"projects": names})


async def create(request):
    body = await request.json()
    name = store.slug(body.get("name"))
    async with lock(name):
        if not os.path.exists(project_file(name)):
            commit(name, store.new_project(body.get("name")), "create")
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
        built = store.render_graph(body["project"], store.slug(name), body["clip"], body["take"], body["quality"], body["seed"])
    except ValueError as error:
        raise web.HTTPBadRequest(text=str(error))
    return web.json_response({"graph": built})


async def probe_file(path):
    """Frame count, a poster and a filmstrip for a finished render."""
    out = await run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                     "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", path])
    poster = os.path.splitext(path)[0] + ".jpg"
    await run(["ffmpeg", "-y", "-v", "error", "-i", path, "-vf", "select=eq(n\\,12),scale=480:-2", "-vsync", "0", "-frames:v", "1", poster])
    frames = int(out.strip().strip(","))
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


async def queue_prompt(graph, client_id=None):
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


async def start_render(data, folder, clip_id, quality, seed=None, client_id=None):
    job, graph = store.plan_render(data, folder, clip_id, quality or "draft", seed)
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
                                     body.get("client_id"))
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
                    store.plan_render(copy.deepcopy(work), folder, clip_id, op.get("quality") or "draft", op.get("seed"))
                    later.append((index, op, clip_id))
                    results.append(None)
                elif kind == "cancel":
                    later.append((index, op, store._resolve(op.get("clip"), names, "clip")))
                    results.append(None)
                else:
                    results.append(store.apply_op(work, op, names))
            except (ValueError, KeyError, TypeError) as error:
                return refused("operation %s (%s): %s" % (index + 1, kind, plain(error)), index=index)
        problem = None
        for index, op, clip_id in later:
            try:
                if op["op"] == "render":
                    job = await start_render(work, folder, clip_id, op.get("quality"), op.get("seed"), op.get("client_id"))
                    results[index] = {"clip": clip_id, "job": job["id"], "take": job["take"]}
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
        done = data.get("revision", 0) > after if after is not None else not data.get("pending")
        if done or time.monotonic() >= deadline:
            reply = store.describe(data)
            reply["timed_out"] = not done
            return web.json_response(reply)
        await asyncio.sleep(1.0)


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
                info = await probe_file(output_path(file.get("subfolder"), file["filename"]))
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


async def receive_upload(request, allowed, directory):
    """Stream a multipart upload to disk, so a long recording never sits in memory."""
    reader = await request.multipart()
    fields, saved, original = {}, None, None
    while True:
        part = await reader.next()
        if part is None:
            break
        if part.filename:
            extension = os.path.splitext(part.filename)[1].lower()
            if extension not in allowed:
                raise ValueError("import a video (%s) or a picture (%s)" % (", ".join(VIDEOS), ", ".join(PICTURES)))
            os.makedirs(directory, exist_ok=True)
            saved = os.path.join(directory, uuid.uuid4().hex[:12] + extension)
            with open(saved, "wb") as f:
                while True:
                    chunk = await part.read_chunk(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            original = part.filename
        else:
            fields[part.name] = await part.text()
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
        sound = bool((await run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                                 "-of", "csv=p=0", source])).strip())
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
        start = float(fields["start_seconds"]) if fields.get("start_seconds") not in (None, "") else None
        end = float(fields["end_seconds"]) if fields.get("end_seconds") not in (None, "") else None
        clip_id, take_id = store.new_id("c"), store.new_id("t")
        target = os.path.join(root(), folder, "clips", "%s_%s.mp4" % (clip_id, take_id))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        width, height = store.frame_size(data, "final")
        await normalize(saved, target, width, height, fit, extension in PICTURES, seconds, start, end)
    except ValueError as error:
        return refused(plain(error))
    finally:
        if cleanup and saved and os.path.exists(saved):
            os.remove(saved)
    info = await probe_file(target)
    take = {"id": take_id, "clip": clip_id, "file": os.path.basename(target), "subfolder": "%s/%s/clips" % (store.ROOT, folder),
            "poster": info["poster"], "frames": info["frames"], "quality": "footage", "seed": None, "source": original}
    if info.get("strip"):
        take["strip"] = info["strip"]
    place = str(fields.get("place", "true")).lower() not in ("false", "0", "no")
    at = int(fields["start"]) if fields.get("start") not in (None, "") else None
    async with lock(folder):
        data = read_project(folder)
        clip, item = store.add_footage(data, fields.get("name") or os.path.splitext(original or "Footage")[0], take, at, place)
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
               "-vsync", "0", "-frames:v", "1", target])
    if not os.path.exists(target):
        raise web.HTTPBadRequest(text="that clip has no frame %s" % frame)
    return web.json_response(asset)


async def export(request):
    name = store.slug(request.match_info["name"])
    data = read_project(name)
    paths = {}
    for clip in data.get("clips", []):
        take = store.active_take(clip)
        if take:
            paths[clip["id"]] = output_path(take.get("subfolder"), take["file"])
    folder = os.path.join(root(), name, "exports")
    os.makedirs(folder, exist_ok=True)
    number = 1
    while os.path.exists(os.path.join(folder, "%s_%03d.mp4" % (name, number))):
        number += 1
    target = os.path.join(folder, "%s_%03d.mp4" % (name, number))
    try:
        args = store.export_command(data, paths, target)
    except ValueError as error:
        raise web.HTTPBadRequest(text=str(error))
    await run(args)
    return web.json_response({
        "filename": os.path.basename(target),
        "subfolder": "%s/%s/exports" % (store.ROOT, name),
    })


ROUTES = (
    ("GET", "/h3take/editor", "page"),
    ("GET", "/h3take/api/projects", "projects"),
    ("POST", "/h3take/api/projects", "create"),
    ("GET", "/h3take/api/project/{name}", "load"),
    ("PUT", "/h3take/api/project/{name}", "save"),
    ("GET", "/h3take/api/project/{name}/summary", "summary"),
    ("GET", "/h3take/api/project/{name}/history", "history"),
    ("POST", "/h3take/api/project/{name}/restore", "restore"),
    ("POST", "/h3take/api/project/{name}/ops", "operations"),
    ("POST", "/h3take/api/project/{name}/render", "render"),
    ("POST", "/h3take/api/project/{name}/cancel", "cancel"),
    ("GET", "/h3take/api/project/{name}/wait", "wait"),
    ("POST", "/h3take/api/project/{name}/graph", "graph"),
    ("POST", "/h3take/api/project/{name}/export", "export"),
    ("POST", "/h3take/api/project/{name}/asset", "upload"),
    ("POST", "/h3take/api/project/{name}/grab", "grab"),
    ("POST", "/h3take/api/project/{name}/grab_sound", "grab_sound"),
    ("POST", "/h3take/api/project/{name}/import", "import_media"),
    ("POST", "/h3take/api/probe", "probe"),
    ("POST", "/h3take/api/strip", "strip"),
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
