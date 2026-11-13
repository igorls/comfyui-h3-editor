"""A project as one zip, to take it to another machine and open it there: a cloud GPU, Colab, or a friend's ComfyUI.

The zip holds project.json, a manifest.json, and the project's files under files/, at the paths they have inside the
project's folder. Three levels:

  all   everything in the folder: every take, the exports, the pictures and sounds. Not the revision history or the
        live previews, which belong to this machine.
  cut   the takes the timeline shows (the active take of every clip on the timeline and of every VFX clip), footage,
        pictures and sounds. Other takes are left out of the project too, so nothing points at a missing file.
  text  no renders: the shots, the cast, its pictures and voices, sounds, footage and VFX code. Every rendered take
        is left out, so the clips open unrendered, with their text, seeds and settings, ready to render again.

Opening a zip makes a new project, under the name it had if that is free. Every member is checked first: nothing is
written outside the new project's folder, links are refused, and the size must fit on the disk.
"""
import json
import os
import shutil
import time
import zipfile

LEVELS = ("all", "cut", "text")
FORMAT = "h3-editor-project"
FORMAT_VERSION = 1
SKIP_DIRS = {"history", "preview"}           # this machine's own: the revisions and the live previews
STORED = {".mp4", ".webm", ".mov", ".jpg", ".jpeg", ".png", ".webp", ".mp3", ".m4a", ".ogg", ".flac", ".safetensors"}


class ArchiveError(ValueError):
    """A zip that cannot be opened as a project, with the reason in words."""


def _project_prefix(root_name, folder):
    return "%s/%s/" % (root_name, folder)


def _stem(name):
    """What a file's companions start with: a take's c..._t... id for its poster, sprite, peaks and hold, or the
    file's name without its extension for a sound's peaks."""
    base = os.path.basename(name)
    cut = base.find("_0000")
    return base[:cut] if cut > 0 else os.path.splitext(base)[0]


def _referenced(value, found):
    """Every {file, subfolder} pair anywhere in the project's data, as subfolder/file paths."""
    if isinstance(value, dict):
        if isinstance(value.get("file"), str) and isinstance(value.get("subfolder"), str):
            found.add((value["subfolder"].replace("\\", "/").strip("/"), os.path.basename(value["file"])))
        for inner in value.values():
            _referenced(inner, found)
    elif isinstance(value, list):
        for inner in value:
            _referenced(inner, found)
    return found


def trimmed(data, keep):
    """The project's data as the zip carries it at a level: the takes it leaves out are left out of the data too."""
    if keep not in LEVELS:
        raise ArchiveError("keep is one of %s" % ", ".join(LEVELS))
    out = json.loads(json.dumps(data))
    out["pending"] = []
    out.pop("revision", None)
    if keep == "all":
        return out
    placed = {item.get("clip") for item in out.get("timeline", [])}
    for clip in out.get("clips", []):
        active = next((take for take in clip.get("takes", []) if take.get("id") == clip.get("active")), None)
        footage = clip.get("kind") == "footage"
        kept = active if active and (footage or (keep == "cut" and clip.get("id") in placed)) else None
        clip["takes"] = [kept] if kept else []
        clip["active"] = kept["id"] if kept else None
    for effect in out.get("effects", []) or []:
        active = next((take for take in effect.get("takes", []) if take.get("id") == effect.get("active")), None)
        kept = active if keep == "cut" else None
        effect["takes"] = [kept] if kept else []
        effect["active"] = kept["id"] if kept else None
    return out


def files_for(project_dir, root_name, folder, data, keep):
    """(paths inside the project folder to pack, files the project points at outside its folder)."""
    every = []
    for here, dirs, names in os.walk(project_dir):
        rel_dir = os.path.relpath(here, project_dir).replace("\\", "/")
        top = rel_dir.split("/")[0]
        dirs[:] = [d for d in dirs if not (rel_dir == "." and d in SKIP_DIRS) and not d.startswith(".")]
        if top in SKIP_DIRS:
            continue
        for name in names:
            if name == "project.json" and rel_dir == ".":
                continue
            if name.endswith(".tmp"):
                continue
            every.append(name if rel_dir == "." else rel_dir + "/" + name)
    prefix = _project_prefix(root_name, folder)
    refs = _referenced(data, set())
    outside = sorted("%s/%s" % pair for pair in refs if not (pair[0] + "/").startswith(prefix))
    if keep == "all":
        return sorted(every), outside
    wanted = {pair[0][len(prefix):] + "/" + pair[1] if pair[0] + "/" != prefix else pair[1]
              for pair in refs if (pair[0] + "/").startswith(prefix)}
    stems = {}
    for path in wanted:
        stems.setdefault(os.path.dirname(path), set()).add(_stem(path))
    chosen = set()
    for path in every:
        directory, name = os.path.dirname(path), os.path.basename(path)
        if path in wanted or directory == "ratings":
            chosen.add(path)
            continue
        for stem in stems.get(directory, ()):
            if name == stem or (name.startswith(stem) and name[len(stem):len(stem) + 1] in ("_", ".")):
                chosen.add(path)
                break
    return sorted(chosen), outside


def measure(project_dir, root_name, folder, data):
    """How big the zip is at each level, before it is made: {level: {"bytes", "files"}}."""
    sizes = {}
    for keep in LEVELS:
        paths, outside = files_for(project_dir, root_name, folder, trimmed(data, keep), keep)
        total = sum(os.path.getsize(os.path.join(project_dir, p)) for p in paths if os.path.isfile(os.path.join(project_dir, p)))
        sizes[keep] = {"bytes": total, "files": len(paths), "outside": len(outside)}
    return sizes


def models_of(data):
    """What the project renders with, for whoever opens it to check before rendering."""
    loras = sorted({entry.get("file") for entry in (data.get("loras") or []) if isinstance(entry, dict) and entry.get("file")}
                   | {entry.get("file") for clip in data.get("clips", []) for entry in (clip.get("loras") or [])
                      if isinstance(entry, dict) and entry.get("file")})
    profiles = sorted({data.get("profile") or "hyperflow"} | {clip.get("profile") for clip in data.get("clips", []) if clip.get("profile")})
    return {"profiles": profiles, "loras": loras,
            "checkpoints": sorted({row.get("unet") for row in (data.get("checkpoints") or []) if isinstance(row, dict) and row.get("unet")})}


def pack(project_dir, root_name, folder, data, keep, target, release="unknown"):
    """Write the zip to target. Returns its manifest."""
    out = trimmed(data, keep)
    paths, outside = files_for(project_dir, root_name, folder, out, keep)
    manifest = {"format": FORMAT, "format_version": FORMAT_VERSION, "release": release, "name": data.get("name") or folder,
                "folder": folder, "keep": keep, "made": time.strftime("%Y-%m-%dT%H:%M:%S"), "files": len(paths),
                "models": models_of(out), "outside": outside}
    with zipfile.ZipFile(target, "w", allowZip64=True) as zipped:
        zipped.writestr("manifest.json", json.dumps(manifest, indent=1), compress_type=zipfile.ZIP_DEFLATED)
        zipped.writestr("project.json", json.dumps(out, indent=1), compress_type=zipfile.ZIP_DEFLATED)
        for path in paths:
            source = os.path.join(project_dir, path)
            if not os.path.isfile(source):
                continue
            method = zipfile.ZIP_STORED if os.path.splitext(path)[1].lower() in STORED else zipfile.ZIP_DEFLATED
            zipped.write(source, "files/" + path, compress_type=method)
    return manifest


def _member_path(name):
    """A member's path inside the project folder, or an ArchiveError if it would land anywhere else."""
    clean = name.replace("\\", "/")
    if clean.startswith("/") or ":" in clean.split("/")[0] or any(part in ("..", "") for part in clean[len("files/"):].split("/")):
        raise ArchiveError("the zip holds a file with an unsafe path: %s" % name[:120])
    return clean[len("files/"):]


def read(source):
    """The manifest and project of a zip, after checking every member. Raises ArchiveError."""
    try:
        zipped = zipfile.ZipFile(source)
    except (zipfile.BadZipFile, OSError) as error:
        raise ArchiveError("that is not a zip file") from error
    with zipped:
        names = zipped.namelist()
        if "manifest.json" not in names or "project.json" not in names:
            raise ArchiveError("that zip is not an H3 Editor project: it has no manifest.json and project.json. "
                               "Make one with Download as zip")
        try:
            manifest = json.loads(zipped.read("manifest.json"))
            data = json.loads(zipped.read("project.json"))
        except ValueError as error:
            raise ArchiveError("the zip's manifest or project cannot be read") from error
        if manifest.get("format") != FORMAT:
            raise ArchiveError("that zip is not an H3 Editor project")
        if int(manifest.get("format_version", 0)) > FORMAT_VERSION:
            raise ArchiveError("that zip was made by a newer H3 Editor (%s). Update the pack to open it" % manifest.get("release", "?"))
        if not isinstance(data, dict) or not isinstance(data.get("clips"), list):
            raise ArchiveError("the zip's project.json is not a project")
        total = 0
        for info in zipped.infolist():
            if info.filename in ("manifest.json", "project.json") or info.is_dir():
                continue
            if not info.filename.startswith("files/"):
                raise ArchiveError("the zip holds a file outside files/: %s" % info.filename[:120])
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ArchiveError("the zip holds a link, which is not opened: %s" % info.filename[:120])
            _member_path(info.filename)
            total += info.file_size
        manifest["unpacked_bytes"] = total
    return manifest, data


def unpack(source, target_dir):
    """Write the zip's files into target_dir, which must not exist yet. Returns (manifest, data)."""
    manifest, data = read(source)
    free = shutil.disk_usage(os.path.dirname(os.path.abspath(target_dir))).free
    if manifest["unpacked_bytes"] > free * 0.95:
        raise ArchiveError("the project needs %.1f GB and the disk has %.1f GB free" % (manifest["unpacked_bytes"] / 1e9, free / 1e9))
    os.makedirs(target_dir)
    base = os.path.realpath(target_dir)
    try:
        with zipfile.ZipFile(source) as zipped:
            for info in zipped.infolist():
                if info.filename in ("manifest.json", "project.json") or info.is_dir():
                    continue
                path = os.path.realpath(os.path.join(base, _member_path(info.filename)))
                if not path.startswith(base + os.sep):
                    raise ArchiveError("the zip holds a file with an unsafe path: %s" % info.filename[:120])
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with zipped.open(info) as inside, open(path, "wb") as outside:
                    shutil.copyfileobj(inside, outside, 1024 * 1024)
    except BaseException:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise
    return manifest, data


def free_name(title, taken, slug):
    """The project's own name if no project has it, else the name with a number: Title 2, Title 3. slug makes a
    name into the folder it would get."""
    used = {str(name).lower() for name in taken}
    if slug(title).lower() not in used:
        return title
    number = 2
    while slug("%s %d" % (title, number)).lower() in used:
        number += 1
    return "%s %d" % (title, number)
