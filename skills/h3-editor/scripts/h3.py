#!/usr/bin/env python3
"""Drive the H3 Editor over HTTP. Python standard library only.

    python h3.py check
    python h3.py create "Morning teaser"
    python h3.py ops Morning_teaser shots.json
    python h3.py wait Morning_teaser
    python h3.py summary Morning_teaser
    python h3.py look Morning_teaser <clip id>

The editor lives in the user's ComfyUI. Set H3_EDITOR_URL, or pass --url, when it is
not at http://127.0.0.1:8188. Run any command with -h for its options.
"""

import argparse
import json
import mimetypes
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

URL = os.environ.get("H3_EDITOR_URL", "http://127.0.0.1:8188").rstrip("/")


class Refused(Exception):
    pass


def request(method, path, body=None, timeout=120, raw=False, headers=None):
    data = None
    headers = dict(headers or {})
    if body is not None and not hasattr(body, "read"):
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    elif body is not None:
        data = body
    req = urllib.request.Request(URL + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = response.read()
    except urllib.error.HTTPError as error:
        text = error.read().decode("utf-8", errors="replace")
        try:
            reason = json.loads(text)
        except ValueError:
            reason = {"error": text.strip()[:600] or error.reason}
        if not isinstance(reason, dict):
            reason = {"error": str(reason)}
        raise Refused(reason.get("error") or json.dumps(reason, ensure_ascii=False))
    except urllib.error.URLError as error:
        raise Refused("cannot reach the H3 Editor at %s: %s. Is the user's ComfyUI running?" % (URL, error.reason))
    return payload if raw else json.loads(payload or b"null")


def api(method, path, body=None, timeout=120):
    return request(method, "/h3take/api" + path, body, timeout)


def project_path(name):
    return "/project/" + urllib.parse.quote(name, safe="")


def download(file, subfolder, target):
    query = urllib.parse.urlencode({"filename": file, "subfolder": subfolder or "", "type": "output"})
    data = request("GET", "/view?" + query, raw=True, timeout=600)
    os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
    with open(target, "wb") as f:
        f.write(data)
    return os.path.abspath(target)


def upload(path, file_path, fields):
    """Multipart upload streamed from a temporary file, so large recordings are fine."""
    boundary = uuid.uuid4().hex
    kind = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
    with tempfile.TemporaryFile() as body:
        for name, value in fields.items():
            if value is None:
                continue
            body.write(("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n" % (boundary, name, value)).encode("utf-8"))
        body.write(("--%s\r\nContent-Disposition: form-data; name=\"file\"; filename=\"%s\"\r\nContent-Type: %s\r\n\r\n"
                    % (boundary, os.path.basename(file_path), kind)).encode("utf-8"))
        with open(file_path, "rb") as source:
            shutil.copyfileobj(source, body, 1 << 20)
        body.write(("\r\n--%s--\r\n" % boundary).encode("utf-8"))
        length = body.tell()
        body.seek(0)
        return json.loads(request("POST", "/h3take/api" + path, body, timeout=1800, raw=True, headers={
            "Content-Type": "multipart/form-data; boundary=%s" % boundary, "Content-Length": str(length)}))


def show(value):
    print(json.dumps(value, indent=1, ensure_ascii=False))


def text_summary(s):
    lines = ["%s  revision %s  %s  %s  %s (%s frames at %s fps)" % (
        s["name"], s["revision"], s.get("format", "16:9"), s.get("language", "English"), s["length"], s["frames"], s["fps"])]
    lines.append("Style: " + (s.get("style") or "none"))
    if s["cast"]:
        lines.append("Cast:")
        for p in s["cast"]:
            has = [k for k in ("picture", "voice") if p.get(k)]
            lines.append("  %-10s %-18s %s%s" % (p["id"], p["name"], " and ".join(has) or "nothing",
                                                 (". " + p["description"]) if p.get("description") else ""))
    lines.append("Timeline:")
    if not s["timeline"]:
        lines.append("  empty")
    states = {c["id"]: c for c in s["clips"]}
    for t in s["timeline"]:
        clip = states.get(t["clip"], {})
        extra = []
        if t.get("speed"):
            extra.append("speed %s" % t["speed"])
        if t.get("transition"):
            extra.append("%s %s frames" % (t["transition"]["type"], t["transition"]["frames"]))
        lines.append("  %s  frames %4s-%-4s  item %-10s clip %-10s %-22s %-14s in %s out %s%s" % (
            t["at"], t["start"], t["end"], t["item"], t["clip"], t["name"][:22], clip.get("state", "?"),
            t["in"], t["out"], ("  " + ", ".join(extra)) if extra else ""))
    for g in s["gaps"]:
        lines.append("  gap at frames %s-%s, black in the export" % (g["start"], g["end"]))
    placed = {t["clip"] for t in s["timeline"]}
    loose = [c for c in s["clips"] if c["id"] not in placed]
    if loose:
        lines.append("Clips not on the timeline:")
        for c in loose:
            lines.append("  clip %-10s %-22s %s" % (c["id"], c["name"][:22], c["state"]))
    for c in s["clips"]:
        if c.get("error"):
            lines.append("Failed: %s %s: %s" % (c["id"], c["name"], c["error"]))
        if c["state"] == "source changed":
            lines.append("Source changed: %s %s joins a clip that moved or changed. Render it again." % (c["id"], c["name"]))
    if s["rendering"]:
        lines.append("Rendering: " + ", ".join("%s (%s)" % (r["clip"], r["quality"]) for r in s["rendering"]))
    return "\n".join(lines)


def read_ops(source):
    raw = sys.stdin.read() if source == "-" else open(source, encoding="utf-8").read()
    data = json.loads(raw)
    return data if isinstance(data, dict) else {"ops": data}


def find_clip(project, clip_id):
    for clip in project["clips"]:
        if clip["id"] == clip_id:
            return clip
    raise Refused("no clip %s in %s" % (clip_id, project.get("name")))


def active_take(clip):
    takes = clip.get("takes") or []
    return next((t for t in takes if t["id"] == clip.get("active")), takes[-1] if takes else None)


def main():
    global URL
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", help="ComfyUI address. Default %s" % URL)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="Is the editor reachable? Lists the projects.")
    sub.add_parser("projects", help="List projects by folder name.")
    p = sub.add_parser("create", help="Create a project, or open it if it exists. Prints its folder name.")
    p.add_argument("name")
    p = sub.add_parser("summary", help="The cut in brief: cast, timeline with timecodes and ids, gaps, failures.")
    p.add_argument("project")
    p.add_argument("--json", action="store_true", help="The raw summary as JSON.")
    p = sub.add_parser("ops", help="Apply a batch of operations from a JSON file, or - for stdin.")
    p.add_argument("project")
    p.add_argument("file")
    p = sub.add_parser("render", help="Render one clip. Drafts unless --final.")
    p.add_argument("project")
    p.add_argument("clips", nargs="+")
    p.add_argument("--final", action="store_true")
    p.add_argument("--seed", type=int)
    p = sub.add_parser("wait", help="Wait until nothing is rendering, then print the summary.")
    p.add_argument("project")
    p.add_argument("--timeout", type=float, default=1800, help="Seconds. Default 1800.")
    p = sub.add_parser("import", help="Import a video or a picture as footage on the timeline.")
    p.add_argument("project")
    p.add_argument("path")
    p.add_argument("--name")
    p.add_argument("--seconds", type=float, help="How long a picture lasts. Default 4.")
    p.add_argument("--fit", choices=("contain", "cover"), help="contain keeps the whole picture (default), cover fills the frame.")
    p.add_argument("--from", dest="start_seconds", type=float, help="Start this many seconds into the video.")
    p.add_argument("--to", dest="end_seconds", type=float, help="Stop at this many seconds into the video.")
    p.add_argument("--at", type=int, help="Timeline frame to place it at. Default: the end.")
    p.add_argument("--no-place", action="store_true", help="Keep it in the bin only.")
    p = sub.add_parser("asset", help="Upload a picture or a sound. Prints the asset to use in ops.")
    p.add_argument("project")
    p.add_argument("path")
    p = sub.add_parser("voice-from", help="Take up to 30 s of sound, as a voice asset, from a timeline item or a clip.")
    p.add_argument("project")
    p.add_argument("source", help="A timeline item id (the part on the timeline) or a clip id (its active take from the start).")
    p = sub.add_parser("grab-frame", help="Save one frame of a timeline item as a picture asset, e.g. for a first frame.")
    p.add_argument("project")
    p.add_argument("item")
    p.add_argument("--at", type=int, help="Timeline frame. Default: the item's last frame.")
    p.add_argument("--out", help="Also download it here.")
    p = sub.add_parser("look", help="Download a sheet of frames from a clip's active take, to look at. No clip: every clip on the timeline.")
    p.add_argument("project")
    p.add_argument("clip", nargs="?")
    p.add_argument("--detail", action="store_true", help="4 large frames instead of 12 small ones: for faces, lettering, hands.")
    p.add_argument("--first", type=int, help="First frame of the take to show.")
    p.add_argument("--last", type=int, help="Stop before this frame of the take.")
    p.add_argument("--count", type=int, help="How many frames, 1 to 24.")
    p.add_argument("--out", help="Folder. Default: h3-look/<project> in the system's temporary folder.")
    p = sub.add_parser("export", help="Join the timeline into one video in ComfyUI's output folder.")
    p.add_argument("project")
    p.add_argument("--out", help="Also download it here.")
    p = sub.add_parser("history", help="Revisions, newest first.")
    p.add_argument("project")
    p = sub.add_parser("restore", help="Go back to a revision. Takes rendered since are kept.")
    p.add_argument("project")
    p.add_argument("revision", type=int)
    args = parser.parse_args()
    if args.url:
        URL = args.url.rstrip("/")

    c = args.command
    if c in ("check", "projects"):
        projects = api("GET", "/projects")["projects"]
        if c == "check":
            print("The H3 Editor answers at %s." % URL)
        print("Projects: " + (", ".join(projects) if projects else "none yet"))
    elif c == "create":
        made = api("POST", "/projects", {"name": args.name})
        print(made["folder"])
    elif c == "summary":
        s = api("GET", project_path(args.project) + "/summary")
        print(json.dumps(s, indent=1, ensure_ascii=False) if args.json else text_summary(s))
    elif c == "ops":
        result = api("POST", project_path(args.project) + "/ops", read_ops(args.file), timeout=600)
        show(result["results"])
        print(text_summary(result["summary"]))
    elif c == "render":
        ops = [{"op": "render", "clip": clip, "quality": "final" if args.final else "draft"} for clip in args.clips]
        if args.seed is not None:
            for op in ops:
                op["seed"] = args.seed
        show(api("POST", project_path(args.project) + "/ops", {"ops": ops}, timeout=600)["results"])
    elif c == "wait":
        deadline = time.monotonic() + args.timeout
        while True:
            left = max(1, min(1800, deadline - time.monotonic()))
            s = api("GET", project_path(args.project) + "/wait?timeout=%d" % left, timeout=left + 60)
            if not s["timed_out"] or time.monotonic() >= deadline:
                break
        if s["timed_out"]:
            print("Still rendering after %d s." % args.timeout)
        print(text_summary(s))
    elif c == "import":
        fields = {"name": args.name, "seconds": args.seconds, "fit": args.fit, "start_seconds": args.start_seconds,
                  "end_seconds": args.end_seconds, "start": args.at, "place": "false" if args.no_place else None}
        made = upload(project_path(args.project) + "/import", args.path, fields)
        print("clip %s, item %s, %s frames" % (made["clip"], made["item"], made["frames"]))
    elif c == "asset":
        show(upload(project_path(args.project) + "/asset", args.path, {}))
    elif c == "voice-from":
        project = api("GET", project_path(args.project))
        item = next((i for i in project["timeline"] if i["id"] == args.source), None)
        clip = find_clip(project, item["clip"] if item else args.source)
        take = active_take(clip)
        if not take:
            raise Refused("%s has no render yet" % clip["name"])
        first, last = (item["in"], item["out"]) if item else (0, take["frames"])
        show(api("POST", project_path(args.project) + "/grab_sound",
                 {"file": take["file"], "subfolder": take["subfolder"], "first": first, "last": last}))
    elif c == "grab-frame":
        project = api("GET", project_path(args.project))
        item = next((i for i in project["timeline"] if i["id"] == args.item), None)
        if not item:
            raise Refused("no timeline item %s" % args.item)
        take = active_take(find_clip(project, item["clip"]))
        if not take:
            raise Refused("that clip has no render yet")
        speed = item.get("speed") or 1
        length = max(1, round((item["out"] - item["in"]) / speed))
        at = item["start"] + length - 1 if args.at is None else args.at
        frame = min(take["frames"] - 1, max(0, round(item["in"] + (at - item["start"]) * speed)))
        asset = api("POST", project_path(args.project) + "/grab", {"file": take["file"], "subfolder": take["subfolder"], "frame": frame})
        show(asset)
        if args.out:
            print(download(asset["file"], asset["subfolder"], args.out))
    elif c == "look":
        project = api("GET", project_path(args.project))
        out = args.out or os.path.join(tempfile.gettempdir(), "h3-look", args.project)
        if args.clip:
            clips = [find_clip(project, args.clip)]
        else:
            order = [i["clip"] for i in sorted(project["timeline"], key=lambda i: i["start"])]
            clips = [find_clip(project, c_id) for c_id in dict.fromkeys(order)]
        for clip in clips:
            take = active_take(clip)
            if not take:
                print("%s %s: no render yet" % (clip["id"], clip["name"]))
                continue
            request_body = {"filename": take["file"], "subfolder": take["subfolder"], "frames": take["frames"],
                            "first": args.first, "last": args.last,
                            "count": args.count or (4 if args.detail else 12), "tile_width": 960 if args.detail else 384}
            made = api("POST", "/sheet", request_body)
            target = os.path.join(out, made["sheet"].replace(".sheet", "_sheet"))
            print(download(made["sheet"], take["subfolder"], target))
            print("  %s %s, take %s of %s, %s, %s frames. Frames in reading order: %s" % (
                clip["id"], clip["name"], clip["takes"].index(take) + 1, len(clip["takes"]), take.get("quality"),
                take["frames"], ", ".join(str(n) for n in made["picked"])))
            if args.clip:
                print("  video: %s/view?%s" % (URL, urllib.parse.urlencode({"filename": take["file"], "subfolder": take["subfolder"], "type": "output"})))
    elif c == "export":
        made = api("POST", project_path(args.project) + "/export", {}, timeout=1800)
        print("Exported to ComfyUI's output folder: %s/%s" % (made["subfolder"], made["filename"]))
        if args.out:
            print(download(made["filename"], made["subfolder"], args.out))
    elif c == "history":
        for r in api("GET", project_path(args.project) + "/history")["revisions"]:
            print("%5s  %s  %-8s %s clips, %s frames" % (r["revision"], time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["time"])),
                                                       r["source"], r["clips"], r["frames"]))
    elif c == "restore":
        made = api("POST", project_path(args.project) + "/restore", {"revision": args.revision})
        print("Restored as revision %s." % made["revision"])
        print(text_summary(made["summary"]))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        main()
    except Refused as error:
        print("Refused: %s" % error, file=sys.stderr)
        sys.exit(1)
