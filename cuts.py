"""Score the cut between two pieces of video, from their pixels alone. The same files always give the same numbers.

A cut reads well in two cases: the picture carries on as if nothing happened (a continuation), or it changes enough to
be a new shot. What reads badly is the middle: the same shot with the subject popped to another pose, a jump cut.

For the last frames of the outgoing piece and the first frames of the incoming one, at 96x54:
  step   how much the picture changes across the cut, where it changes most (95th percentile of the luma
         difference over 6x6 blocks, 0..1), so a pose that pops in a small part of the frame still counts
  motion the same measure from frame to frame just before and just after the cut (the median)
  alike  how alike the two pictures are in layout (correlation of the blurred luma, -1..1)
  shift  how far the picture moved, as a fraction of its width (phase correlation)
  colour the largest change of a colour channel's mean (0..1)

kind   continuous  step is no more than twice the motion around it, plus 0.02
       jump        the same shot (alike >= 0.5) but not continuous: the worse the more alike
       new shot    a different picture
score  0..100. continuous 90-100, new shot 70-100, jump 0-60 (up to 90 when it is only just over the line). A colour change costs up to 20 on a continuous cut.

The editor scores every cut of a timeline with score_cut(); GET /h3take/api/project/{name}/cuts returns them.

scan() does the same for a finished file, where nothing says where the cuts are: it finds every place the picture
changes more than its own motion explains, scores it as above, and reports a shot of STRAY frames or fewer between
two cuts as a stray frame (score 0): a frame of another picture left in by mistake. A fade or a dip to black changes
a little on every frame, so it is not a cut and is not reported.

inside() looks into one generated take, which should be one shot: a cut in it, or a fade to another picture (the
model sometimes ends a take by dissolving into a different room), is reported with the frame where it starts.

Usage: python cuts.py a.mp4 b.mp4 [c.mp4 ...]      scores each cut between neighbours, prints a table
       python cuts.py --json a.mp4 b.mp4
       python cuts.py --scan film.mp4 [--all]     every cut inside one file that needs a look (--all: every cut)
       python cuts.py --scan film.mp4 --at 113,234,355   only the cuts at these frames: where the edit put them.
                                                  A screen recording or an animation changes abruptly inside one
                                                  shot, and a scan without --at reports each of those as a cut
       python cuts.py --inside take.mp4           where a take that should be one shot changes picture
Needs ffmpeg and numpy.
"""
import json
import subprocess
import sys

import numpy as np

W, H, N, FPS = 96, 54, 5, 24
STRAY = 3           # a shot of this many frames or fewer is a stray frame, not a shot
SPAN = 12           # a fade is looked for across this many frames
POP = 0.12          # a step under this is a blink, a gesture or a ripple, not a change of picture


def plain(args):
    return args


def window(path, first, count, command=plain):
    """count frames of a file from frame first on, as float RGB arrays of H x W x 3 in 0..1. Fewer if the file ends."""
    first = max(0, int(first))
    raw = subprocess.run(command(["ffmpeg", "-v", "error", "-ss", "%.4f" % (first / float(FPS)), "-i", str(path), "-an",
                                  "-vf", "fps=%d,scale=%d:%d:flags=area" % (FPS, W, H), "-frames:v", str(count),
                                  "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]), capture_output=True, check=True).stdout
    got = np.frombuffer(raw, dtype=np.uint8).reshape(-1, H, W, 3).astype(np.float64) / 255.0
    if not len(got):
        raise ValueError("no picture in %s at frame %d" % (path, first))
    return got


def score_cut(a, a_out, b, b_in, command=plain):
    """Score a cut on a timeline: file a played up to its frame a_out (not included), then file b from its frame b_in."""
    count = min(N, int(a_out))
    m = measure(window(a, a_out - count, count, command), window(b, b_in, N, command))
    return dict(judge(m), **{k: round(v, 4) for k, v in m.items()})


def frames(path, tail):
    """The last (tail) or first N frames of a file, as float RGB arrays of H x W x 3 in 0..1."""
    seek = ["-sseof", "-1"] if tail else []
    limit = [] if tail else ["-frames:v", str(N)]
    raw = subprocess.run(["ffmpeg", "-v", "error"] + seek + ["-i", str(path), "-an", "-vf", "scale=%d:%d:flags=area" % (W, H)] + limit
                         + ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    got = np.frombuffer(raw, dtype=np.uint8).reshape(-1, H, W, 3).astype(np.float64) / 255.0
    if not len(got):
        raise ValueError("no picture in %s" % path)
    return got[-N:] if tail else got[:N]


def luma(rgb):
    return rgb @ np.array([0.2126, 0.7152, 0.0722])


def blurred(y):
    return y.reshape(H // 3, 3, W // 3, 3).mean(axis=(1, 3))


def local(a, b):
    """How much two pictures differ where they differ most: the 95th percentile of the difference over 6x6 blocks.
    A subject that pops to another pose changes a small part of the frame a lot, which a mean over the frame hides."""
    return float(np.percentile(np.abs(a - b).reshape(H // 6, 6, W // 6, 6).mean(axis=(1, 3)), 95))


def measure(before, after):
    """The numbers for one cut. before: the outgoing frames, last one last. after: the incoming frames, first one first."""
    a, b = luma(before[-1]), luma(after[0])
    step = local(a, b)
    around = [local(luma(x), luma(y)) for x, y in list(zip(before[:-1], before[1:])) + list(zip(after[:-1], after[1:]))]
    motion = float(np.median(around)) if around else 0.0
    sa, sb = blurred(a), blurred(b)
    da, db = sa - sa.mean(), sb - sb.mean()
    spread = float(np.sqrt((da * da).sum() * (db * db).sum()))
    alike = float((da * db).sum() / spread) if spread > 1e-9 else (1.0 if step < 1e-4 else 0.0)
    cross = np.fft.fft2(a - a.mean()) * np.conj(np.fft.fft2(b - b.mean()))
    peak = np.fft.ifft2(cross / np.maximum(np.abs(cross), 1e-9)).real
    y, x = np.unravel_index(int(peak.argmax()), peak.shape)
    shift = float(np.hypot(min(x, W - x) / W, min(y, H - y) / W))
    colour = float(np.abs(before[-1].mean(axis=(0, 1)) - after[0].mean(axis=(0, 1))).max())
    return {"step": step, "motion": motion, "alike": alike, "shift": shift, "colour": colour}


def judge(m):
    """kind, score and what is wrong, from the numbers of measure()."""
    notes = []
    # A join between a clip and its extension steps by about 0.02 even when nothing moves: the two were decoded apart.
    allowed = 2 * m["motion"] + 0.02
    if m["step"] <= allowed:
        kind, score = "continuous", 100 - 10 * min(1.0, m["step"] / allowed)
        if m["colour"] > 0.02:
            score -= min(20.0, 400 * (m["colour"] - 0.02))
            notes.append("colour changes")
    elif m["alike"] >= 0.5:
        # The worse the more alike the two pictures are, and eased in over the first 0.03 past what is allowed,
        # so a cut just over the line does not fall from 90 to 0.
        worst = 60 * (1 - (m["alike"] - 0.5) / 0.5)
        kind, score = "jump", 90 - min(1.0, (m["step"] - allowed) / 0.03) * (90 - worst)
        notes.append("same shot, the subject pops" if m["shift"] < 0.02 else "same shot, moved %d%% of the width" % round(100 * m["shift"]))
        if m["colour"] > 0.04:
            notes.append("colour changes")
    else:
        kind, score = "new shot", 70 + 30 * (1 - max(m["alike"], 0.0) / 0.5)
    if m["motion"] < 0.001:
        notes.append("still on both sides")
    return {"kind": kind, "score": round(max(0.0, min(100.0, score))), "notes": notes}


def score(a, b):
    """Score the cut from file a to file b."""
    m = measure(frames(a, True), frames(b, False))
    return dict(judge(m), **{k: round(v, 4) for k, v in m.items()})


def whole(path, command=plain):
    """Every frame of a file at W x H, as float RGB in 0..1."""
    raw = subprocess.run(command(["ffmpeg", "-v", "error", "-i", str(path), "-an", "-vf", "fps=%d,scale=%d:%d:flags=area" % (FPS, W, H),
                                  "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]), capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, H, W, 3).astype(np.float32) / 255.0


def find_cuts(pictures):
    """The cuts inside a run of frames: [{"at", kind, score, notes, ...}], at being the first frame after the cut.

    A shot of STRAY frames or fewer between two cuts is one finding, a stray frame, with "frames" saying how many.
    """
    y = [luma(p.astype(np.float64)) for p in pictures]
    steps = [local(a, b) for a, b in zip(y[:-1], y[1:])]
    found = []
    for k, step in enumerate(steps):
        if step <= 0.02:
            continue                                   # under what any cut is allowed: not worth measuring
        before = pictures[max(0, k - N + 1):k + 1].astype(np.float64)
        after = pictures[k + 1:k + 1 + N].astype(np.float64)
        m = measure(before, after)
        verdict = judge(m)
        if verdict["kind"] != "continuous":
            found.append(dict(verdict, at=k + 1, **{name: round(value, 4) for name, value in m.items()}))
    out, k = [], 0
    while k < len(found):
        cut = found[k]
        strong = lambda c: c["kind"] == "new shot" or c["step"] >= POP       # a real change of picture, not a ripple
        if k + 1 < len(found) and found[k + 1]["at"] - cut["at"] <= STRAY and strong(cut) and strong(found[k + 1]):
            count = found[k + 1]["at"] - cut["at"]
            out.append({"at": cut["at"], "kind": "stray frame", "score": 0, "frames": count,
                        "notes": ["%d frame%s of another picture between two cuts" % (count, "" if count == 1 else "s")]})
            k += 2
        else:
            out.append(cut)
            k += 1
    return out


def likeness(a, b):
    """How alike two luma pictures are in layout, -1..1."""
    sa, sb = blurred(a), blurred(b)
    da, db = sa - sa.mean(), sb - sb.mean()
    spread = float(np.sqrt((da * da).sum() * (db * db).sum()))
    return float((da * db).sum() / spread) if spread > 1e-9 else 1.0


def find_inside(pictures):
    """Where a run of frames that should be one shot changes picture: [{"at", "kind", "score", "notes"}].

    kind is "cut" (a hard cut, or a stray frame) or "fade" (the picture dims or dissolves into another within SPAN
    frames, with no cut). at is the first frame of the change.
    """
    found = []
    hard = find_cuts(pictures)
    hard = [cut for cut in hard if cut["kind"] != "jump" or cut["step"] >= POP]
    for cut in hard:
        stray = cut["kind"] == "stray frame"
        found.append({"at": cut["at"], "kind": "cut", "score": 0 if stray or cut["kind"] == "new shot" else min(cut["score"], 60),
                      "notes": cut["notes"] if stray else ["the take cuts to another picture" if cut["kind"] == "new shot"
                                                           else "the subject pops inside the take"]})
    y = [luma(p.astype(np.float64)) for p in pictures]
    light = [float(v.mean()) for v in y]
    k = SPAN
    while k < len(y):
        if any(k - SPAN < cut["at"] <= k for cut in hard):
            k += 1
            continue
        dims = light[k] < 0.6 * light[k - SPAN] and light[k - SPAN] > 0.05
        other = likeness(y[k], y[k - SPAN]) < 0.3
        if dims or other:
            start = k - SPAN
            if dims:                                   # where it starts to dim, not where it was last whole
                start = next(j for j in range(k - SPAN + 1, k + 1) if light[j] < 0.95 * light[k - SPAN]) - 1
            found.append({"at": start, "kind": "fade", "score": 0,
                          "notes": ["the picture fades out" if dims else "the picture dissolves into another one"]})
            k += 3 * SPAN                              # one finding for one fade
        else:
            k += 1
    return sorted(found, key=lambda row: row["at"])


def inside(path, command=plain):
    """Where one take changes picture, each with its time in seconds."""
    return [dict(row, time=round(row["at"] / float(FPS), 3)) for row in find_inside(whole(path, command))]


def scan(path, command=plain):
    """Every cut inside one file, in order, each with its time in seconds."""
    return [dict(cut, time=round(cut["at"] / float(FPS), 3)) for cut in find_cuts(whole(path, command))]


def main(argv):
    as_json = "--json" in argv
    files = [f for f in argv if not f.startswith("--")]
    if "--inside" in argv:
        if len(files) != 1:
            raise SystemExit(__doc__)
        rows = inside(files[0])
        if as_json:
            print(json.dumps(rows, indent=1))
            return
        for r in rows:
            print("%3d  %-5s %d:%05.2f  frame %-6d %s" % (r["score"], r["kind"], int(r["time"]) // 60, r["time"] % 60, r["at"], "; ".join(r["notes"])))
        print("one shot, nothing found" if not rows else "%d found" % len(rows))
        return
    if "--scan" in argv:
        marks = None
        if "--at" in argv:
            marks = [int(n) for n in argv[argv.index("--at") + 1].split(",") if n.strip()]
            files = [f for f in files if f != argv[argv.index("--at") + 1]]
        if len(files) != 1:
            raise SystemExit(__doc__)
        rows = scan(files[0])
        if marks is not None:
            rows = [r for r in rows if any(abs(r["at"] - m) <= STRAY for m in marks)]
        if as_json:
            print(json.dumps(rows, indent=1))
            return
        shown = rows if "--all" in argv else [r for r in rows if r["score"] < 70]
        for r in shown:
            print("%3d  %-11s %d:%05.2f  frame %-6d %s" % (r["score"], r["kind"], int(r["time"]) // 60, r["time"] % 60, r["at"], "; ".join(r["notes"])))
        print("%d cuts, %d need a look" % (len(rows), sum(1 for r in rows if r["score"] < 70)))
        return
    if len(files) < 2:
        raise SystemExit(__doc__)
    rows = [dict(score(a, b), cut="%s > %s" % (a, b)) for a, b in zip(files[:-1], files[1:])]
    if as_json:
        print(json.dumps(rows, indent=1))
        return
    for r in rows:
        print("%3d  %-10s step %.3f  motion %.3f  alike %+.2f  %s  %s" % (r["score"], r["kind"], r["step"], r["motion"], r["alike"], r["cut"], "; ".join(r["notes"])))


if __name__ == "__main__":
    main(sys.argv[1:])
