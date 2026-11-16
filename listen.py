"""Score the sound at a cut, and inside a take, from the samples alone. The same files always give the same numbers.

cuts.py looks at the picture. This listens. A cut can be clean to the eye and wrong to the ear: the level drops in
one frame, a swell is chopped off, a room's hum becomes open air, or the wave steps and clicks.

For the 0.4 seconds before the cut and the 0.4 seconds after it:
  level    how loud each side is, as loudness in LUFS: the mean square of the wave weighted by frequency the way
           ITU-R BS.1770 (EBU R 128) weights it, so a low rumble counts for less than a voice at the same power
  step     the change in level across the cut, in LU (1 LU is 1 dB of loudness). Negative is quieter after
  usual    how much the level moves from one 0.4 second stretch to the next on each side of the cut, where
           nothing was cut: a sound that swells and falls on its own may step that much at a cut too
  colour   how much the shape of the spectrum changes across the cut, in dB, with the level taken out: the
           root mean square difference of 24 bands from 50 Hz to 12 kHz. A kitchen to a sea is a change of colour
  click    the step in the wave at the cut against the steps around it (see joins.click)

kind   silent      both sides are under SILENCE
       continuous  the level steps by no more than ALLOWED, or twice what is usual around the cut
       new sound   the level holds and the colour changes: what a cut to another place sounds like
       level jump  the level steps by more than that: the ear hears the cut
       dropout     sound on one side and silence on the other
score  0..100. continuous 90-100 and new sound 80-100; a level jump loses 5 for every LU past what is allowed,
       and a dropout scores under 30. A click takes up to 20 more.

inside() listens along one take for what a sound does not do by itself. It may start at once, as a word or a
wave does, but it does not stop at once: a fall of CHOP or more within a twentieth of a second that stays down is
a chop, and a hole of no signal at all is a dropout. A take that ends SWELL or more louder or quieter than it began
is reported too: it will not sit against its neighbours on a timeline.

Usage: python listen.py a.mp4 b.mp4 [c.mp4 ...]     the cut between neighbours, whole files
       python listen.py --inside take.mp4           where the sound of one take jumps or drops out
       python listen.py --json ...
Needs ffmpeg and numpy.
"""

import json
import subprocess
import sys

import numpy as np

RATE = 48000
FPS = 24
SIDE = 0.4          # seconds listened to on each side of a cut
CONTEXT = 2.0       # seconds on each side used to learn what a usual step is
FRAME = 2048        # samples per spectrum
SILENCE = -65.0     # LUFS under which there is nothing to hear
ALLOWED = 3.0       # LU a level may step at a cut before the ear takes it for a cut
COLOUR = 6.0        # dB of change in the spectrum's shape that makes it another sound
SHORT_STRETCH = 0.05  # seconds of the stretches a chop is looked for between
CHOP = 20.0         # LU a sound falls by, from one such stretch to the next but one, when it is cut off
STAYS = 0.3         # seconds it then stays down: a gap between two words is shorter
GAP = -85.0         # LUFS under which a stretch inside a take is a hole, not a quiet moment
SWELL = 20.0        # LU between the start and the end of one take
BANDS = 24


def plain(args):
    return args


def wave(path, start, seconds, command=plain):
    """seconds of a file's sound from start on, as one channel at RATE. Empty for a silent file or past its end."""
    start = max(0.0, float(start))
    got = subprocess.run(command(["ffmpeg", "-v", "error", "-i", str(path), "-vn",
                                  "-af", "atrim=start=%.6f:duration=%.6f" % (start, max(0.0, float(seconds))),
                                  "-ac", "1", "-ar", str(RATE), "-f", "f32le", "-"]), capture_output=True, check=False)
    return np.frombuffer(got.stdout, dtype=np.float32).astype(np.float64)


def _weights():
    """The power gain of BS.1770's K weighting at each bin of a FRAME-long spectrum: its two filters at 48 kHz."""
    z = np.exp(-2j * np.pi * np.fft.rfftfreq(FRAME, 1.0 / RATE) / RATE)

    def gain(b, a):
        return np.abs((b[0] + b[1] * z + b[2] * z * z) / (a[0] + a[1] * z + a[2] * z * z)) ** 2

    shelf = gain((1.53512485958697, -2.69169618940638, 1.19839281085285), (1.0, -1.69065929318241, 0.73248077421585))
    highpass = gain((1.0, -2.0, 1.0), (1.0, -1.99004745483398, 0.99007225036621))
    return shelf * highpass


K = _weights()
FREQS = np.fft.rfftfreq(FRAME, 1.0 / RATE)
EDGES = np.geomspace(50.0, 12000.0, BANDS + 1)
WINDOW = np.hanning(FRAME)


def spectra(samples):
    """The power spectrum of each half-overlapping FRAME of a wave, scaled so a spectrum sums to its mean square."""
    if len(samples) < FRAME:
        samples = np.concatenate([samples, np.zeros(FRAME - len(samples))])
    count = 1 + (len(samples) - FRAME) // (FRAME // 2)
    frames = np.stack([samples[n * FRAME // 2:n * FRAME // 2 + FRAME] for n in range(count)]) * WINDOW
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    power[:, 1:-1] *= 2
    return power / (FRAME * float((WINDOW ** 2).sum()))


def loudness(samples):
    """The loudness of a wave in LUFS, after BS.1770: K-weighted mean square, -0.691 dB. -120 for no sound."""
    if not len(samples):
        return -120.0
    square = float((spectra(samples) * K).sum(axis=1).mean())
    return -120.0 if square <= 1e-12 else max(-120.0, -0.691 + 10 * np.log10(square))


def shape(samples):
    """The spectrum of a wave in BANDS bands, in dB, with its own level taken out: the colour of the sound."""
    power = spectra(samples).mean(axis=0)
    bands = np.array([power[(FREQS >= low) & (FREQS < high)].sum() for low, high in zip(EDGES[:-1], EDGES[1:])])
    level = 10 * np.log10(np.maximum(bands, 1e-14))
    return level - level.mean()


def stretches(samples):
    """A wave cut into stretches of SIDE seconds, whole ones only."""
    size = int(SIDE * RATE)
    return [samples[n:n + size] for n in range(0, len(samples) - size + 1, size)]


def click(before, after, around=0.02):
    """The step where two waves are joined end to start, against the usual step between samples near it."""
    reach = int(around * RATE)
    if len(before) < reach or len(after) < reach:
        return None
    near = np.concatenate([np.diff(before[-reach:]), np.diff(after[:reach])])
    usual = float(np.sqrt(np.mean(near ** 2)))
    if usual < 1e-7:
        return None
    return round(abs(float(after[0]) - float(before[-1])) / usual, 2)


def measure(before, after):
    """The numbers for one cut. before: the sound up to the cut, CONTEXT seconds or less. after: the sound from it on."""
    size = int(SIDE * RATE)
    a, b = before[-size:], after[:size]
    levels = (loudness(a), loudness(b))
    # What the level does by itself on each side: the steps between neighbouring stretches that no cut lies between.
    moves = []
    for side in (stretches(before), stretches(after)):
        heard = [loudness(s) for s in side]
        moves += [abs(y - x) for x, y in zip(heard[:-1], heard[1:]) if max(x, y) > SILENCE]
    usual = float(np.median(moves)) if moves else 0.0
    colour = float(np.sqrt(np.mean((shape(a) - shape(b)) ** 2))) if len(a) and len(b) and min(levels) > SILENCE else 0.0
    return {"before": levels[0], "after": levels[1], "step": levels[1] - levels[0], "usual": usual, "colour": colour,
            "click": click(a, b)}


def judge(m):
    """kind, score and what is wrong, from the numbers of measure()."""
    notes = []
    quiet = (m["before"] <= SILENCE, m["after"] <= SILENCE)
    allowed = max(ALLOWED, 2 * m["usual"])
    step = abs(m["step"])
    if all(quiet):
        return {"kind": "silent", "score": 100, "notes": ["no sound on either side"]}
    if any(quiet):
        kind, score = "dropout", 25.0
        notes.append("the sound stops at the cut" if quiet[1] else "the sound starts at the cut")
    elif step > allowed:
        kind, score = "level jump", 90 - 5 * (step - allowed)
        notes.append("%d dB %s after the cut" % (round(step), "quieter" if m["step"] < 0 else "louder"))
    elif m["colour"] > COLOUR:
        kind, score = "new sound", 100 - 20 * min(1.0, (m["colour"] - COLOUR) / COLOUR)
        notes.append("another sound after the cut")
    else:
        kind, score = "continuous", 100 - 10 * step / allowed
    if m["click"] is not None and m["click"] > 5:
        score -= min(20.0, 2 * (m["click"] - 5))
        notes.append("the wave steps at the cut: a click")
    return {"kind": kind, "score": round(max(0.0, min(100.0, score))), "notes": notes}


def rounded(m):
    return {k: (None if v is None else round(float(v), 2)) for k, v in m.items()}


def score_cut(a, a_out, b, b_in, command=plain):
    """Score the sound of a cut on a timeline: file a played up to its frame a_out (not included), then b from b_in."""
    stop = a_out / float(FPS)
    before = wave(a, stop - CONTEXT, min(CONTEXT, stop), command)
    after = wave(b, b_in / float(FPS), CONTEXT, command)
    m = measure(before, after)
    return dict(judge(m), **rounded(m))


# A take's last stretch of digital silence this long or longer is said: its sound ran out before its picture.
ENDS_SILENT = 0.5
# Silence that only a file can hold, far below any room: what is left where a track or a sound has run out. A sound
# that dies away by itself stays above it for longer than ENDS_SILENT.
DIGITAL = -110.0


def find_inside(samples):
    """Where one take's sound is chopped off or drops out: [{"at", "kind", "score", "notes", ...}], at being a frame.

    A sound that starts at once is ordinary: a word, a door, a wave. What does not happen by itself is a sound that
    stops at once: things ring, echo and die away. So only falls are looked for, and only fast ones that stay down.
    """
    size, hop = int(SHORT_STRETCH * RATE), int(SHORT_STRETCH * RATE) // 2
    if len(samples) < int(2 * SIDE * RATE):
        return []
    heard = np.array([loudness(samples[n:n + size]) for n in range(0, len(samples) - size + 1, hop)])
    if heard.max() <= SILENCE:
        return []
    after = int(round(STAYS / SHORT_STRETCH * 2))
    found = []
    n = 0
    while n + 2 + after < len(heard):
        fall = heard[n] - heard[n + 2]
        rest = float(heard[n + 2:n + 2 + after].max())
        if fall >= CHOP and heard[n] > SILENCE + 15 and heard[n] - rest >= CHOP - 5:
            silent = rest <= GAP
            at = (n + 2) * hop / float(RATE)
            found.append({"at": int(round(at * FPS)), "kind": "dropout" if silent else "chop", "score": 25 if silent else round(max(0.0, 90 - 5 * (fall - ALLOWED))),
                          "step": round(-float(fall), 1),
                          "notes": ["the sound drops out" if silent else "the sound is cut off: %d dB quieter within a twentieth of a second" % round(fall)]})
            n += after
        n += 1
    # A take whose sound runs out before its picture does: the end of a song or a voice it was made to, or a sound
    # that dies away and never comes back. Said as that, not as a fade, since the cure is a shorter clip.
    tail = 0
    while tail < len(heard) and heard[len(heard) - 1 - tail] <= DIGITAL:
        tail += 1
    seconds = tail * hop / float(RATE)
    if seconds >= ENDS_SILENT and heard.max() > SILENCE + 15 and tail < len(heard):
        at = (len(heard) - tail) * hop / float(RATE)
        # A drop into that silence is the same thing: nothing comes after it, so it is not a hole.
        found = [row for row in found if not (row["kind"] == "dropout" and abs(row["at"] - at * FPS) <= FPS // 2)]
        if not any(abs(row["at"] - at * FPS) <= FPS // 2 for row in found):
            found.append({"at": int(round(at * FPS)), "kind": "ends silent", "score": 50, "step": None,
                          "notes": ["the sound stops %.1f s before the take ends" % seconds]})
        return found
    # A take that ends far louder or quieter than it began: a swell, or a fade. It will not sit against its neighbours.
    wide = int(SIDE * RATE)
    first, last = loudness(samples[:wide]), loudness(samples[-wide:])
    if abs(last - first) >= SWELL and max(first, last) > SILENCE:
        found.append({"at": 0, "kind": "swell" if last > first else "fade", "score": 60, "step": round(last - first, 1),
                      "notes": ["the take ends %d dB %s than it begins" % (round(abs(last - first)), "louder" if last > first else "quieter")]})
    return found


def inside(path, command=plain):
    """Where the sound of one take jumps or drops out, each with its frame and its time in seconds."""
    rows = find_inside(wave(path, 0, 3600, command))
    return [dict(row, time=round(row["at"] / float(FPS), 3)) for row in rows]


def score(a, b, command=plain):
    """Score the sound of the cut from the end of file a to the start of file b."""
    whole = wave(a, 0, 3600, command)
    m = measure(whole[-int(CONTEXT * RATE):], wave(b, 0, CONTEXT, command))
    return dict(judge(m), **rounded(m))


def main(argv):
    as_json = "--json" in argv
    files = [a for a in argv if not a.startswith("--")]
    if "--inside" in argv:
        rows = inside(files[0])
        if as_json:
            print(json.dumps(rows, indent=1))
        else:
            for row in rows:
                print("%3d  %-8s %s  frame %-5d %s" % (row["score"], row["kind"], "%d:%05.2f" % divmod(row["time"], 60), row["at"], "; ".join(row["notes"])))
            print("nothing found" if not rows else "%d found" % len(rows))
        return 0
    if len(files) < 2:
        print(__doc__)
        return 2
    rows = [dict(score(a, b), a=a, b=b) for a, b in zip(files[:-1], files[1:])]
    if as_json:
        print(json.dumps(rows, indent=1))
    else:
        for row in rows:
            print("%3d  %-11s step %+6.1f LU (usual %.1f)  colour %4.1f dB  click %s  %s" % (
                row["score"], row["kind"], row["step"], row["usual"], row["colour"], row["click"], "; ".join(row["notes"])))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
