"""HTTP routes for the editor. Projects live in ComfyUI's output folder, so renders stay with them."""

import asyncio
import json
import math
import os
import subprocess
import sys
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


def write_project(name, data):
    path = project_file(name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    os.replace(temporary, path)


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
    if not os.path.exists(project_file(name)):
        write_project(name, store.new_project(body.get("name")))
    return web.json_response({"folder": name, "project": read_project(name)})


async def load(request):
    return web.json_response(read_project(request.match_info["name"]))


async def save(request):
    write_project(request.match_info["name"], await request.json())
    return web.json_response({"saved": True})


async def graph(request):
    name = request.match_info["name"]
    body = await request.json()
    try:
        built = store.render_graph(body["project"], store.slug(name), body["clip"], body["take"], body["quality"], body["seed"])
    except ValueError as error:
        raise web.HTTPBadRequest(text=str(error))
    return web.json_response({"graph": built})


async def probe(request):
    """Frame count and a poster for a finished render."""
    body = await request.json()
    path = output_path(body.get("subfolder"), body["filename"])
    out = await run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                     "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", path])
    poster = os.path.splitext(path)[0] + ".jpg"
    await run(["ffmpeg", "-y", "-v", "error", "-i", path, "-vf", "select=eq(n\\,12),scale=480:-2", "-vsync", "0", "-frames:v", "1", poster])
    frames = int(out.strip().strip(","))
    try:
        made = await make_strip(path, frames)
    except web.HTTPException:
        made = None
    return web.json_response({"frames": frames, "poster": os.path.basename(poster), "strip": made})


def asset_target(name, extension):
    folder = store.slug(name)
    directory = os.path.join(root(), folder, "assets")
    os.makedirs(directory, exist_ok=True)
    file = uuid.uuid4().hex[:12] + extension
    return os.path.join(directory, file), {"file": file, "subfolder": "%s/%s/assets" % (store.ROOT, folder)}


async def upload(request):
    """Keep an uploaded picture with the project."""
    form = await request.post()
    part = form.get("image")
    if part is None or not hasattr(part, "file"):
        raise web.HTTPBadRequest(text="choose a picture to upload")
    extension = os.path.splitext(part.filename or "")[1].lower()
    if extension not in (".png", ".jpg", ".jpeg", ".webp"):
        raise web.HTTPBadRequest(text="use a PNG, JPEG, or WebP picture")
    target, asset = asset_target(request.match_info["name"], extension)
    with open(target, "wb") as f:
        f.write(part.file.read())
    return web.json_response(asset)


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
    ("POST", "/h3take/api/project/{name}/graph", "graph"),
    ("POST", "/h3take/api/project/{name}/export", "export"),
    ("POST", "/h3take/api/project/{name}/asset", "upload"),
    ("POST", "/h3take/api/project/{name}/grab", "grab"),
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
