"""Render same-seed A/B/blend controls in a new project on a running ComfyUI.

python tools/test_style_blend.py [--url http://127.0.0.1:8188] [--seconds 5] [--three-styles]
The resulting project and takes are retained for visual comparison.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8188")
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--three-styles", action="store_true", help="Render one continuous graphite, cut paper, and clay shot")
    args = parser.parse_args()

    def request(path, data=None):
        body = None if data is None else json.dumps(data).encode()
        req = urllib.request.Request(args.url + "/h3take/api" + path, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=45) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(error.read().decode()) from error

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from project import snap_frames
    end = snap_frames(args.seconds) - 1
    a = "Monochrome graphite pencil drawing on white paper, visible crosshatching, flat hand-drawn illustration, black and white only"
    b = "Vivid colorful three-dimensional clay animation, sculpted plasticine texture, saturated orange and blue, miniature studio lighting"
    action = "A teapot sits on a simple table. The camera slowly orbits the teapot. One continuous shot, no cut."
    examples = [
        ("Pencil control", [{"frame": 0, "style": a}]),
        ("Clay control", [{"frame": 0, "style": b}]),
        ("Pencil to clay", [{"frame": 0, "style": a}, {"frame": end, "style": b}]),
    ]
    if args.three_styles:
        paper = "Flat layered cut-paper illustration, hand-cut geometric shapes, visible paper fibers, teal and magenta on mustard yellow"
        examples = [("Graphite to cut paper to clay", [
            {"frame": 0, "style": a, "name": "Graphite"},
            {"frame": end // 2, "style": paper, "name": "Cut paper"},
            {"frame": end, "style": b, "name": "Clay"},
        ])]
        action = "A toy rowboat bobs beside a small lighthouse. The camera glides slowly to the right, keeping both in view. One continuous shot, no cut."
    name = "Three styles QA " if args.three_styles else "Style blend QA "
    folder = request("/projects", {"name": name + time.strftime("%Y%m%d %H%M%S")})["folder"]
    ops = [{"op": "set_project", "fields": {"sizes": {"draft": "small", "final": "large"}}}]
    for name, markers in examples:
        ops.append({"op": "add_clip", "as": "clip" + str(len(ops)), "fields": {
            "name": name, "action": action, "seconds": args.seconds, "seed": 424242, "style_markers": markers}})
    request("/project/" + folder + "/ops", {"ops": ops})
    project = request("/project/" + folder)
    queued = request("/project/" + folder + "/ops", {"ops": [
        {"op": "render", "clip": clip["id"], "quality": "draft"} for clip in project["clips"]]})
    print(json.dumps({"folder": folder, "jobs": queued["results"]}), flush=True)
    deadline = time.monotonic() + 1800
    previous = None
    while time.monotonic() < deadline:
        time.sleep(5)
        project = request("/project/" + folder)
        status = [(clip["name"], len(clip.get("takes", [])), clip.get("error")) for clip in project["clips"]]
        if status != previous:
            print(json.dumps(status), flush=True)
            previous = status
        if not project.get("pending"):
            if any(clip.get("error") or not clip.get("takes") for clip in project["clips"]):
                raise RuntimeError("a comparison render failed")
            print(json.dumps({"folder": folder, "takes": [clip["takes"][-1] for clip in project["clips"]]}), flush=True)
            return
    raise TimeoutError("style comparison did not finish in 30 minutes")


if __name__ == "__main__":
    main()
