"""Find where a video loses fine detail, from its pixels alone. The same file always gives the same numbers.

A picture that was made small and enlarged (a draft take left in a cut of finals, a low-resolution import) fills the
frame but has nothing in its finest pixels. This measures that, a few times a second, at the file's own size:

  fine    how sharp the sharpest edges of the picture are: how far a pixel sits from the middle of its neighbours
          two pixels away, against the same measure nine pixels away, for the pixels that stand out most. A smooth
          slope gives nothing at either distance, so only real edges count. In a 1080p film cut from generated
          takes, a full-size final gave about 0.45 and a draft enlarged three times about 0.18. Because it is a
          ratio, a dark or low-contrast shot does not read as soft.

A stretch is reported as a drop when its fine detail stays under DROP times the level of the whole file (the median of
its samples) for at least MIN seconds. That finds the part that is softer than the rest; a file that is soft from start
to end has no drops, only a low level. Samples with almost nothing in them (a plain card, a fade)
are skipped: there is no detail to lose.

The editor uses the same measure on a timeline: clip_fine() gives one number for the part of a take a clip shows, at
the size of the cut, and soft_clips() says which clips sit well under the rest of the cut.
GET /h3take/api/project/{name}/detail returns them.

Usage: python detail.py film.mp4            prints each drop with its time, and the level of the file
       python detail.py --series film.mp4   also prints every sample
       python detail.py --json film.mp4
Needs ffmpeg, ffprobe and numpy.
"""
import json
import subprocess
import sys

import numpy as np

RATE = 2            # samples a second
DROP = 0.5          # a drop is under this part of the file's level
MIN = 1.0           # for at least this long, in seconds
EMPTY = 0.02        # weaker edges than this (0..1 luma) is a picture with nothing to measure


def plain(args):
    return args


def bend(y, n):
    """How far the pixels that stand out most sit from the middle of their neighbours n pixels away, across and down.

    The 99th percentile, not the mean: a face is mostly smooth skin, and what tells a sharp one from a soft one is
    its few real edges (lashes, hair, the rim of an earring). Every second pixel is enough to find them.
    """
    across = np.abs(2 * y[:, n:-n] - y[:, :-2 * n] - y[:, 2 * n:])[::2, ::2]
    down = np.abs(2 * y[n:-n, :] - y[:-2 * n, :] - y[2 * n:, :])[::2, ::2]
    return float(np.percentile(across, 99) + np.percentile(down, 99))


def fine(y):
    """The fine-detail ratio of one luma frame (0..1 floats), or None when the frame holds nothing to measure."""
    wide = bend(y, 9)
    if wide < EMPTY:
        return None
    return bend(y, 2) / wide


def size(path, command=plain):
    out = subprocess.run(command(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                                  "-of", "csv=p=0", str(path)]), capture_output=True, text=True, check=True).stdout.strip()
    w, h = (int(x) for x in out.split(",")[:2])
    return w, h


def series(path, command=plain, rate=RATE):
    """[(seconds, fine or None)] for the whole file, rate samples a second."""
    w, h = size(path, command)
    proc = subprocess.Popen(command(["ffmpeg", "-v", "error", "-i", str(path), "-an", "-vf", "fps=%s,format=gray" % rate,
                                     "-f", "rawvideo", "-pix_fmt", "gray", "-"]), stdout=subprocess.PIPE)
    out, k = [], 0
    while True:
        raw = proc.stdout.read(w * h)
        if len(raw) < w * h:
            break
        y = np.frombuffer(raw, dtype=np.uint8).reshape(h, w).astype(np.float32) / 255.0
        out.append((k / float(rate), fine(y)))
        k += 1
    proc.wait()
    return out


def drops(samples, drop=DROP, least=MIN):
    """The level of the file and the stretches under it: {"level", "drops": [{"from", "to", "fine", "part"}]}."""
    values = sorted(v for _, v in samples if v is not None)
    if not values:
        return {"level": None, "drops": []}
    level = float(np.median(values))
    step = samples[1][0] - samples[0][0] if len(samples) > 1 else 1.0
    found, run = [], []

    def close():
        if run and run[-1][0] + step - run[0][0] >= least:
            mean = float(np.mean([v for _, v in run]))
            found.append({"from": round(run[0][0], 2), "to": round(run[-1][0] + step, 2), "fine": round(mean, 3), "part": round(mean / level, 3)})
        run.clear()

    for t, v in samples:
        if v is None:
            continue                      # an empty picture neither starts nor ends a drop
        if v < drop * level:
            run.append((t, v))
        else:
            close()
    close()
    return {"level": round(level, 3), "drops": found}


def clip_fine(path, first, last, width, height, command=plain, most=6, fps=24):
    """The fine detail of frames first..last of a file as it looks at width x height: the median of up to most samples.

    The picture is scaled to the size of the cut first, so a small take is measured as the viewer will see it, enlarged.
    None when there is nothing to measure.
    """
    first, last = max(0, int(first)), max(int(first) + 1, int(last))
    seconds = (last - first) / float(fps)
    count = max(1, min(int(most), int(seconds * 2)))
    width, height = int(width), int(height)
    raw = subprocess.run(command(["ffmpeg", "-v", "error", "-ss", "%.4f" % (first / float(fps)), "-t", "%.4f" % seconds, "-i", str(path), "-an",
                                  "-vf", "fps=%.5f,scale=%d:%d:flags=bicubic,format=gray" % (count / seconds, width, height),
                                  "-frames:v", str(count), "-f", "rawvideo", "-pix_fmt", "gray", "-"]), capture_output=True, check=True).stdout
    frames = np.frombuffer(raw[:len(raw) - len(raw) % (width * height)], dtype=np.uint8).reshape(-1, height, width)
    values = [v for v in (fine(frame.astype(np.float32) / 255.0) for frame in frames) if v is not None]
    return float(np.median(values)) if values else None


def soft_clips(rows, drop=DROP):
    """Which clips of a cut are soft. rows is [(id, fine or None, frames)]; the level is the median over the cut's frames.

    Returns {"level", "clips": [{"id", "fine", "part", "soft"}]}. A clip with nothing to measure is never soft.
    """
    known = sorted((value, max(1, int(frames))) for _, value, frames in rows if value is not None)
    level = None
    if known:
        half, seen = sum(frames for _, frames in known) / 2.0, 0
        for value, frames in known:
            seen += frames
            if seen >= half:
                level = value
                break
    clips = []
    for key, value, _ in rows:
        part = None if value is None or not level else value / level
        clips.append({"id": key, "fine": None if value is None else round(value, 3), "part": None if part is None else round(part, 3),
                      "soft": part is not None and part < drop})
    return {"level": None if level is None else round(level, 3), "clips": clips}


def clock(t):
    return "%d:%04.1f" % (int(t) // 60, t - 60 * (int(t) // 60))


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    if len(args) != 1:
        print(__doc__)
        return 2
    samples = series(args[0])
    found = drops(samples)
    if "--json" in argv:
        print(json.dumps(dict(found, series=[[t, None if v is None else round(v, 3)] for t, v in samples]) if "--series" in argv else found))
        return 0
    if "--series" in argv:
        for t, v in samples:
            print("%s  %s" % (clock(t), "-" if v is None else "%.3f" % v))
    if found["level"] is None:
        print("nothing to measure in this file")
        return 0
    print("level %.3f, %d samples" % (found["level"], len(samples)))
    for d in found["drops"]:
        print("drop  %s to %s  fine %.3f  (%d%% of the level)" % (clock(d["from"]), clock(d["to"]), d["fine"], round(100 * d["part"])))
    if not found["drops"]:
        print("no drops")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
