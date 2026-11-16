"""Whether the place in a take stays the same place: the background, with the person taken out, against itself.

A generated take can morph: a keyboard behind the singer turns into a monitor, a wall into a window, while the face
holds. The eye catches it late, because it watches the face. This looks only at what is not the person.

Every second, frame t is compared with the frame two seconds later, and with the take's first frame:
  - The person (hair, clothes and what they hold included) is found by MODNet, a portrait matting network, and left
    out of both, with a margin.
  - The camera's move between the two is solved on the background alone (ECC, affine). The comparison is made both
    with the camera held and with that move, and the one that explains the background better counts, so a held
    camera never reads a false move and a moving one is followed.
  - Each cell of a 4 x 8 grid that shows background in both is compared as blurred colour (Lab): what is there, at
    the scale of objects, not the grain or the noise of a frame.
  share   of the background cells seen, how many changed (a cell changed when its median colour difference is over
          LIMIT). A held camera on an unchanged room reads 0; a room turning into another reads 0.3 and up.

MODNet (Ke et al., "MODNet: Trimap-Free Portrait Matting in Real Time", AAAI 2022; ZHKKKe/MODNet, Apache-2.0) is not
part of the pack. Put its ONNX file (Xenova/modnet on Hugging Face, onnx/model.onnx, Apache-2.0) in ComfyUI's
models/matting folder as modnet.onnx. OpenCV runs it. Without it nothing is measured.

Usage: python scene.py take.mp4 [first last]     prints, per second, how much of the background changed
"""
import json
import os
import subprocess
import sys

import numpy as np

try:
    from . import cuts
except ImportError:
    import cuts

FPS = 24
W, H = 216, 384         # the size the pictures are compared at, portrait; a landscape take is turned to fit
ROWS, COLS = 8, 4
GAP = 48                # two seconds
EVERY = 24              # a comparison each second
LIMIT = 12.0            # a cell's median Lab difference (summed over L, a, b) over which it changed
MODEL_FILE = "modnet.onnx"
MIN_CELLS = 3           # fewer background cells than this is too little to judge
MIN_FRAMES = GAP + 1
# Two frames whose framing differs by more than this much zoom are not compared: a push from a wide shot into a
# close-up leaves a sliver of blurred background that no single move lines up, and read as a changed room.
ZOOM_LIMIT = 0.08
# Share of the background that changed: under HOLDS it holds; at CHANGES and over it changed.
HOLDS = 0.15
CHANGES = 0.2


def plain(args):
    return args


def folder():
    """Where the matting file lives: ComfyUI's models/matting, or H3_MATTING_DIR."""
    if os.environ.get("H3_MATTING_DIR"):
        return os.environ["H3_MATTING_DIR"]
    try:
        import folder_paths
        return os.path.join(folder_paths.models_dir, "matting")
    except ImportError:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "matting")


def missing():
    """The files the measure needs and does not have, by name."""
    return [] if os.path.exists(os.path.join(folder(), MODEL_FILE)) else [MODEL_FILE]


_NET = {}


def network():
    """MODNet, loaded once, on the CPU: ("cv2", net) through OpenCV's DNN module, or ("ort", session) through
    onnxruntime where OpenCV cannot read it (OpenCV 4.x fails on its graph; 5.0 reads it). RuntimeError when
    neither can, and broken() then says why."""
    if "net" not in _NET:
        path = os.path.join(folder(), MODEL_FILE)
        try:
            import cv2
            _NET["net"] = ("cv2", cv2.dnn.readNet(path))
        except Exception as error:      # cv2.error is not a subclass of anything narrower
            try:
                import onnxruntime
                _NET["net"] = ("ort", onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"]))
            except Exception:
                _NET["net"] = None
                first = (str(error).strip().splitlines() or ["unknown error"])[0]
                _NET["why"] = "this OpenCV cannot read %s and onnxruntime is not installed (%s)" % (MODEL_FILE, first[:160])
    if _NET["net"] is None:
        raise RuntimeError(_NET["why"])
    return _NET["net"]


def broken():
    """Why the matting network cannot run here, once loading it has failed; None otherwise."""
    return _NET.get("why")


def person(rgb):
    """True where the person is, with a margin: MODNet's matte at 288 x 512, over a fifth."""
    import cv2

    h, w = rgb.shape[:2]
    x = cv2.resize(rgb, (288, 512) if h >= w else (512, 288), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    kind, net = network()
    blob = np.ascontiguousarray((x / 127.5 - 1.0).transpose(2, 0, 1)[None], dtype=np.float32)
    if kind == "cv2":
        net.setInput(blob)
        matte = net.forward()[0, 0]
    else:
        matte = net.run(None, {net.get_inputs()[0].name: blob})[0][0, 0]
    mask = cv2.resize((matte > 0.2).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return cv2.dilate(mask, np.ones((9, 9), np.uint8)) > 0


def pictures(path, frames, command=plain):
    """{frame: RGB picture at the compared size} for the frames asked for, read in one pass."""
    wanted = sorted(set(int(f) for f in frames))
    if not wanted:
        return {}
    size = _size(path, command)
    w, h = (W, H) if size is None or size[1] >= size[0] else (H, W)
    first, last = wanted[0], wanted[-1] + 1
    proc = subprocess.Popen(command(["ffmpeg", "-v", "error", "-ss", "%.4f" % (first / float(FPS)), "-i", str(path), "-an",
                                     "-frames:v", str(last - first), "-vf", "scale=%d:%d:flags=area" % (w, h),
                                     "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]), stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    out, need, at = {}, set(wanted), first
    try:
        while at < last:
            raw = proc.stdout.read(w * h * 3)
            if len(raw) < w * h * 3:
                break
            if at in need:
                out[at] = np.frombuffer(raw, np.uint8).reshape(h, w, 3)
            at += 1
    finally:
        proc.stdout.close()
        proc.wait()
    return out


def _size(path, command):
    out = subprocess.run(command(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                                  "-of", "csv=p=0", str(path)]), capture_output=True, text=True)
    if out.returncode == 0 and out.stdout.strip():
        w, h = (int(x) for x in out.stdout.strip().split(",")[:2])
        return w, h
    return None


def scales(path, first, last, command=plain):
    """{frame: how much larger the picture is than at frame first}, from the camera's zoom frame to frame (cuts.py's
    whole-frame motion, which weighs down a person moving through it), on a small grey copy."""
    w, h = 108, 192
    size = _size(path, command)
    if size is not None and size[0] > size[1]:
        w, h = h, w
    proc = subprocess.Popen(command(["ffmpeg", "-v", "error", "-ss", "%.4f" % (first / float(FPS)), "-i", str(path), "-an",
                                     "-frames:v", str(last - first), "-vf", "scale=%d:%d:flags=area,format=gray" % (w, h),
                                     "-f", "rawvideo", "-pix_fmt", "gray", "-"]), stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    out, before, scale, at = {}, None, 1.0, first
    try:
        while at < last:
            raw = proc.stdout.read(w * h)
            if len(raw) < w * h:
                break
            now = np.frombuffer(raw, np.uint8).reshape(h, w).astype(np.float64) / 255.0
            if before is not None:
                scale *= 1.0 + float(cuts._global_flow(before, now)[2])
            out[at] = scale
            before, at = now, at + 1
    finally:
        proc.stdout.close()
        proc.wait()
    return out


def _framed_alike(zoom, a, b):
    """Whether frames a and b are framed near enough alike to compare: True when the zoom is not known."""
    if a not in zoom or b not in zoom:
        return True
    return abs(np.log(zoom[b] / zoom[a])) <= np.log(1.0 + ZOOM_LIMIT)


def _count(path, command):
    """Frames in the file: PyAV's count where it has one, else ffprobe's."""
    try:
        import av
        with av.open(str(path)) as container:
            if container.streams.video[0].frames:
                return container.streams.video[0].frames
    except ImportError:
        pass
    out = subprocess.run(command(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                                  "stream=nb_read_packets", "-of", "csv=p=0", str(path)]), capture_output=True, text=True)
    return int(out.stdout.strip().split(",")[0]) if out.returncode == 0 and out.stdout.strip() else 0


def align(a, b, ground):
    """The affine move taking b onto a, solved on background pixels, coarse then finer; None when it does not settle."""
    import cv2

    warp = np.eye(2, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
    for blur in (7, 3):
        try:
            _, warp = cv2.findTransformECC(cv2.GaussianBlur(a, (0, 0), blur), cv2.GaussianBlur(b, (0, 0), blur), warp,
                                           cv2.MOTION_AFFINE, crit, ground.astype(np.uint8) * 255, 5)
        except cv2.error:
            return None
    return warp


def grid(a, pa, b, pb, warp):
    """ROWS x COLS: per cell, the median blurred Lab difference over background seen in both, NaN where too little."""
    import cv2

    h, w = a.shape[:2]
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    lab_a = cv2.cvtColor(a, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_b = cv2.warpAffine(cv2.cvtColor(b, cv2.COLOR_RGB2LAB).astype(np.float32), warp, (w, h), flags=flags)
    inside = cv2.warpAffine(np.ones((h, w), np.float32), warp, (w, h), flags=flags) > 0.99
    covered = cv2.warpAffine(pb.astype(np.float32), warp, (w, h), flags=flags) > 0.01
    ground = ~pa & ~covered & inside
    diff = np.abs(cv2.GaussianBlur(lab_a, (0, 0), 3) - cv2.GaussianBlur(lab_b, (0, 0), 3)).sum(-1)
    rows, cols = (ROWS, COLS) if h >= w else (COLS, ROWS)
    out = np.full((rows, cols), np.nan)
    ch, cw = h // rows, w // cols
    for r in range(rows):
        for c in range(cols):
            cell = (slice(r * ch, (r + 1) * ch), slice(c * cw, (c + 1) * cw))
            if ground[cell].mean() > 0.4:
                out[r, c] = float(np.median(diff[cell][ground[cell]]))
    return out


def change(a, pa, b, pb):
    """Share of the background that changed from picture a to picture b (masks pa, pb), or None when too little shows."""
    import cv2

    options = [grid(a, pa, b, pb, np.eye(2, 3, dtype=np.float32))]
    grey_a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    grey_b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    moved = align(grey_a, grey_b, ~pa)
    if moved is not None:
        options.append(grid(a, pa, b, pb, moved))
    options = [g for g in options if np.isfinite(g).sum() >= MIN_CELLS]
    if not options:
        return None
    best = min(options, key=lambda g: float(np.nanmedian(g)))
    seen = best[np.isfinite(best)]
    return float(np.mean(seen > LIMIT))


def measure(path, first=0, last=None, command=plain, anchor=None):
    """[{"at", "near", "start"}] for frames first..last of a file: at each second, the share of background changed over
    the two seconds before it (near) and since the start (start), None where too little background shows. The start is
    frame first, or anchor, (path, frame), when the shot began in another file: the take an extension continues. None
    when the measure is not installed or the part is too short."""
    if missing():
        return None
    if last is None:
        last = _count(path, command)
    first, last = int(first), int(last)
    if last - first < MIN_FRAMES:
        return None
    ends = list(range(first + GAP, last, EVERY))
    seen = pictures(path, [first] + [u - GAP for u in ends] + ends, command)
    if anchor is not None:
        found = pictures(anchor[0], [anchor[1]], command).get(int(anchor[1]))
        if found is None or found.shape != seen.get(first, found).shape:
            return None
        seen["start"] = found
    else:
        seen["start"] = seen.get(first)
    masks = {}

    def mask(frame):
        if frame not in masks:
            masks[frame] = person(seen[frame])
        return masks[frame]

    rows = []
    if anchor is not None and first in seen:
        # An extension's first frames continue its shot: they are compared with the shot's start too.
        ends = [first] + ends
    zoom = scales(path, first, last, command)
    for u in ends:
        t = u - GAP
        if u not in seen or seen["start"] is None:
            continue
        near = change(seen[t], mask(t), seen[u], mask(u)) if t in seen and _framed_alike(zoom, t, u) else None
        if _framed_alike(zoom, first, u):
            start = near if t == first and anchor is None else change(seen["start"], mask("start"), seen[u], mask(u))
        else:
            # The camera has moved in or out since the start: the earliest frame framed like this one stands in for it.
            alike = [f for f in sorted(k for k in seen if k != "start") if f <= t and _framed_alike(zoom, f, u)]
            start = None if not alike else near if alike[0] == t else change(seen[alike[0]], mask(alike[0]), seen[u], mask(u))
        rows.append({"at": u, "near": None if near is None else round(near, 3),
                     "start": None if start is None else round(start, 3)})
    return rows


def judge(rows):
    """{"kind", "score", "share", "frame", "notes"}: "holds" or "changes", from the worst second; None with nothing seen."""
    worst = None
    for row in rows or []:
        for value in (row["near"], row["start"]):
            if value is not None and (worst is None or value > worst[0]):
                worst = (value, row["at"])
    if worst is None:
        return None
    share, frame = worst
    if share < CHANGES:
        return {"kind": "holds", "score": int(round(100 - 150 * share)), "share": share, "frame": frame, "notes": []}
    return {"kind": "changes", "score": int(round(max(5, 55 - 100 * (share - CHANGES)))), "share": share, "frame": frame,
            "notes": ["%d%% of the background is not what it was: the place changes behind the person. Render the clip again on "
                      "another seed" % round(share * 100)]}


def score(path, first=0, last=None, command=plain, anchor=None):
    """The judgement of frames first..last of a file, with the seconds it is made from, or None when it cannot be made."""
    rows = measure(path, first, last, command, anchor)
    found = judge(rows)
    return None if found is None else dict(found, seconds=rows)


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    first = int(argv[1]) if len(argv) > 1 else 0
    last = int(argv[2]) if len(argv) > 2 else None
    if missing():
        print("not installed: put %s in %s" % (" and ".join(missing()), folder()))
        return 1
    found = score(argv[0], first, last)
    if not found:
        print("too short, or too little background to judge")
        return 0
    for row in found["seconds"]:
        print("%6.2f s  last two seconds %4s  since the start %4s" % (
            row["at"] / float(FPS), "-" if row["near"] is None else "%.0f%%" % (row["near"] * 100),
            "-" if row["start"] is None else "%.0f%%" % (row["start"] * 100)))
    print(json.dumps({k: v for k, v in found.items() if k != "seconds"}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
