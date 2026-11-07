"""HTTP routes for the editor. Projects live in ComfyUI's output folder, so renders stay with them."""

import asyncio
import json
import os
import subprocess
import sys

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


async def page(request):
    return web.FileResponse(os.path.join(PAGE, "index.html"))


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
    return web.json_response({"frames": int(out.strip().strip(",")), "poster": os.path.basename(poster)})


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
    number = len([n for n in os.listdir(folder) if n.endswith(".mp4")]) + 1
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
    ("POST", "/h3take/api/probe", "probe"),
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
