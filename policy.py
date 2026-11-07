"""Rules for one H3 take. No Comfy import. The node calls these before it touches a latent."""

import math

ANCHORS = ("none", "start", "end", "both")
VIDEO_MODES = ("off", "v2v", "extend_after", "extend_before", "extend_both")
CHECKPOINTS = ("ref2va", "fl2va")
AUDIO_HOLDS = ("off", "span", "full")
FPS = 24
AUDIO_HZ = 40
MAX_PICTURES = 9
MAX_REF_VIDEOS = 3
MAX_REF_AUDIOS = 3
# Mask values measured on FastH3 Omni NVFP4, 8 steps. Below 0.95 the source wins
# outright and above 0.996 the composition is lost, so a linear slider would waste most of its travel.
STRENGTHS = {"hold": 0.0, "light": 0.95, "medium": 0.99, "heavy": 0.996, "regenerate": 1.0}


def legal_length(length):
    length = int(length)
    return length >= 5 and (length - 5) % 17 == 0


def av_hold(length):
    """24 fps video and 40 Hz audio land together on 39, 90, 141, ..."""
    length = int(length)
    return length >= 39 and legal_length(length) and (length - 39) % 51 == 0


def resolve_keep(strength, keep=0.0):
    if strength == "custom":
        return float(keep)
    if strength not in STRENGTHS:
        raise ValueError("strength must be custom or one of %s" % (tuple(STRENGTHS),))
    return STRENGTHS[strength]


def video_steps(frames):
    """Latent steps for a legal frame count. Each 17 frames is 5 steps of 1, 4, 4, 4, 4 frames."""
    frames = int(frames)
    return 2 if frames <= 5 else ((frames - 5) // 17) * 5 + 2


def frame_to_step(frame):
    group, rest = divmod(int(frame), 17)
    return group * 5 + (0 if rest == 0 else 1 + (rest - 1) // 4)


def audio_ticks(frames):
    """40 Hz audio latent ticks across a span of 24 fps frames. Matches the native rounding."""
    return int(round(int(frames) / FPS * AUDIO_HZ))


def check_length(length):
    if not legal_length(length):
        raise ValueError(
            "length %s is not an H3 frame count (5 + 17k). "
            "124 is about 5 seconds and is the short trained length." % length
        )
    if int(length) < 124:
        return "short"
    return "trained"


def snap_hold(requested, available, target, align):
    cap = min(int(requested), int(available), int(target))
    if align == "av":
        ok = av_hold
        need = "at least 39 source frames so video and audio share a boundary"
    elif align == "video":
        ok = legal_length
        need = "at least 5 source frames on the 17k+5 grid"
    else:
        raise ValueError("align must be av or video")
    best = None
    for n in range(5, cap + 1):
        if ok(n):
            best = n
    if best is None:
        raise ValueError(
            "hold does not fit: %s (asked %s, source %s, target %s)"
            % (need, requested, available, target)
        )
    return best


def frame_count(image, what):
    if image is None:
        return 0
    shape = getattr(image, "shape", None)
    if shape is None or len(shape) < 1:
        raise ValueError("%s is not an image batch" % what)
    return int(shape[0])


def require_still(image, what):
    n = frame_count(image, what)
    if n != 1:
        raise ValueError(
            "%s must be one still, got %s frames. A start or end pin is one "
            "frame. A source video is a held span, not a pin." % (what, n)
        )


def video_span(mode, source_frames, length, hold_frames=39, align="av", keep=0.0):
    """Where the source sits in the new latent, and which side is generated.

    keep 0 freezes the held frames. keep 1 regenerates them. The mask uses
    that value on the held span and 1 on the generated span. AddGuide is not
    this. AddGuide has no keep.
    """
    if mode not in VIDEO_MODES:
        raise ValueError("video mode must be one of %s" % (VIDEO_MODES,))
    if mode == "off":
        if source_frames:
            raise ValueError("source video is connected but video mode is off")
        return None
    source_frames = int(source_frames)
    length = int(length)
    if source_frames < 1:
        raise ValueError("video mode %s needs a source video" % mode)
    check_length(length)
    if not 0.0 <= float(keep) <= 1.0:
        raise ValueError("keep must be from 0 (freeze the held frames) to 1 (regenerate them)")

    if mode == "v2v":
        if source_frames != length:
            raise ValueError(
                "v2v needs the source and the target to be the same length "
                "(%s source frames, target %s). To add time, use extend_before, "
                "extend_after, or extend_both." % (source_frames, length)
            )
        hold = length
        insert = 0
        slice_from = "all"
        source_start = 0
    else:
        hold = snap_hold(hold_frames, source_frames, length - 5, align)
        if mode == "extend_after":
            insert = 0
            slice_from = "tail"
            source_start = source_frames - hold
        elif mode == "extend_before":
            insert = length - hold
            slice_from = "head"
            source_start = 0
        else:
            insert = ((length - hold) // 2 // 17) * 17
            slice_from = "middle"
            source_start = (source_frames - hold) // 2
        if length - hold < 5:
            raise ValueError("no room to generate: hold %s inside target %s" % (hold, length))
        if mode == "extend_both" and (insert < 5 or length - insert - hold < 5):
            raise ValueError(
                "extend_both needs at least 5 new frames on each side "
                "(hold %s, target %s)" % (hold, length)
            )

    return {
        "mode": mode,
        "insert": insert,
        "hold": hold,
        "keep": float(keep),
        "generate_before": insert,
        "generate_after": length - insert - hold,
        "slice_from": slice_from,
        "source_start": source_start,
        "audio_exact": insert % 51 == 0,
        "mask_hold": float(keep),
        "mask_generate": 1.0,
    }


def latent_frames(steps):
    """Frames covered by a latent of this many steps. The inverse of video_steps."""
    return ((int(steps) - 2) // 5) * 17 + 5


def audio_window(mode, span, length, audio_seconds):
    """Which audio ticks of the take are held, and where they come from in the source.

    span holds the source audio across the held video. full lays the source
    along the whole take, lined up with the held video when there is one, so
    the picture can be generated to an existing track. Ticks outside the
    source audio stay generated.
    """
    if mode not in AUDIO_HOLDS:
        raise ValueError("audio hold must be one of %s" % (AUDIO_HOLDS,))
    if mode == "off":
        return None
    if audio_seconds is None or float(audio_seconds) <= 0:
        raise ValueError("audio hold %s needs source audio" % mode)
    total = audio_ticks(length)
    if mode == "span":
        if span is None:
            raise ValueError("audio hold span follows the held video. Use full when there is no source video.")
        first = audio_ticks(span["insert"])
        ticks = audio_ticks(span["hold"])
        source_second = span["source_start"] / float(FPS)
    else:
        first = 0
        ticks = total
        offset = (span["source_start"] - span["insert"]) if span is not None else 0
        source_second = offset / float(FPS)
    if source_second < 0:
        skip = int(math.ceil(-source_second * AUDIO_HZ))
        first += skip
        ticks -= skip
        source_second += skip / float(AUDIO_HZ)
    available = int(math.floor((float(audio_seconds) - source_second) * AUDIO_HZ + 1e-6))
    ticks = min(ticks, available, total - first)
    if ticks < 1:
        raise ValueError("source audio does not reach the held span")
    return {"mode": mode, "first": first, "ticks": ticks, "total": total, "source_second": source_second}


def hold_curve(total, first, count, base=0.0, feather=0):
    """Mask values along one axis. 1 generates, `base` holds.

    An edge that meets generated material ramps from `base` up toward 1 over
    `feather` entries inside the hold, on a half cosine. An edge at the start
    or end of the take has nothing to blend into and stays hard.
    """
    total, first, count = int(total), int(first), int(count)
    if first < 0 or count < 1 or first + count > total:
        raise ValueError("hold %s+%s does not fit in %s" % (first, count, total))
    values = [1.0] * total
    for i in range(first, first + count):
        values[i] = float(base)
    width = max(0, min(int(feather), count // 2))
    for d in range(width):
        ramp = 0.5 + 0.5 * math.cos(math.pi * (d + 1) / (width + 1))
        value = float(base) + (1.0 - float(base)) * ramp
        if first > 0:
            values[first + d] = value
        if first + count < total:
            values[first + count - 1 - d] = value
    return values


def parse_frames(text):
    """Frame numbers from '30, 60 90'. Empty is no frames."""
    parts = [p for p in str(text or "").replace(",", " ").split() if p]
    try:
        return [int(p) for p in parts]
    except ValueError:
        raise ValueError("mid frames must be whole frame numbers, got %r" % (text,))


def check_mid_pins(frames, count, length):
    if len(frames) != count:
        raise ValueError("%s mid frame(s) for %s mid image(s). Give one frame per image." % (len(frames), count))
    if len(set(frames)) != len(frames):
        raise ValueError("mid frames repeat: %s" % (frames,))
    for frame in frames:
        if frame < 0 or frame >= int(length):
            raise ValueError("mid frame %s is outside the %s-frame take" % (frame, length))


def plan(anchor, start=None, end=None, soft=(), length=124, video=None, checkpoint="ref2va",
         mid_frames=(), mids=(), ref_videos=(), ref_audios=0, audio=None):
    """Return the prompt labels and the pins. A source video is not a pin, and neither is a soft reference.

    ref_videos is one bool per reference video: does it carry a soundtrack.
    Native order is images, then videos with each soundtrack labelled just
    before its video, then standalone audio, so <Audio j> counts soundtracks first.
    Mid pins are guides only, as in the official multiframe template. They take no picture number.
    """
    if anchor not in ANCHORS:
        raise ValueError("anchor must be one of %s" % (ANCHORS,))
    if checkpoint not in CHECKPOINTS:
        raise ValueError("checkpoint must be one of %s" % (CHECKPOINTS,))
    note = check_length(length)
    if anchor in ("start", "both"):
        require_still(start, "start image")
    elif start is not None:
        raise ValueError("start image is connected but anchor is %s. Disconnect it or pin it." % anchor)
    if anchor in ("end", "both"):
        require_still(end, "end image")
    elif end is not None:
        raise ValueError("end image is connected but anchor is %s. Disconnect it or pin it." % anchor)

    soft = [img for img in soft if img is not None]
    for i, img in enumerate(soft, 1):
        require_still(img, "soft reference %s" % i)
    mids = [img for img in mids if img is not None]
    for i, img in enumerate(mids, 1):
        require_still(img, "mid image %s" % i)
    mid_frames = list(mid_frames)
    check_mid_pins(mid_frames, len(mids), length)

    ref_videos = list(ref_videos)
    ref_audios = int(ref_audios)
    if checkpoint == "fl2va" and (soft or ref_videos or ref_audios):
        raise ValueError(
            "the fl2va checkpoint has no reference inputs. It takes a prompt and pinned frames. "
            "Use a ref2va checkpoint and set checkpoint to ref2va."
        )
    if len(ref_videos) > MAX_REF_VIDEOS:
        raise ValueError("H3 takes at most %s reference videos" % MAX_REF_VIDEOS)
    if ref_audios > MAX_REF_AUDIOS:
        raise ValueError("H3 takes at most %s reference audio clips" % MAX_REF_AUDIOS)

    pictures = []
    pins = []
    if anchor in ("start", "both"):
        pictures.append(("start", start))
        pins.append(("start", 0))
    if anchor in ("end", "both"):
        pictures.append(("end", end))
        pins.append(("end", int(length) - 1))
    for img in soft:
        pictures.append(("soft", img))
    if len(pictures) > MAX_PICTURES:
        raise ValueError("H3 takes at most %s pictures, start and end pins included" % MAX_PICTURES)
    for i, frame in enumerate(mid_frames):
        pins.append(("mid %s" % (i + 1), frame))

    labels = ["checkpoint %s" % checkpoint]
    for i, (role, _) in enumerate(pictures, 1):
        labels.append("<Picture %s> %s" % (i, role))
    audio_number = 0
    for i, has_track in enumerate(ref_videos, 1):
        if has_track:
            audio_number += 1
            labels.append("<Audio %s> soundtrack of <Video %s>" % (audio_number, i))
        labels.append("<Video %s> reference" % i)
    for _ in range(ref_audios):
        audio_number += 1
        labels.append("<Audio %s> reference" % audio_number)
    for i, frame in enumerate(mid_frames, 1):
        labels.append("mid pin %s at frame %s" % (i, frame))
    if video is not None:
        labels.append(
            "video %s hold %s at frame %s, generate %s before and %s after, keep %s"
            % (
                video["mode"],
                video["hold"],
                video["insert"],
                video["generate_before"],
                video["generate_after"],
                video["keep"],
            )
        )
        if not video["audio_exact"]:
            labels.append("audio clock is rounded at this insert; a multiple of 51 is exact")
    if audio is not None:
        labels.append(
            "audio %s holds %s ticks from tick %s of %s"
            % (audio["mode"], audio["ticks"], audio["first"], audio["total"])
        )
    return {
        "pictures": pictures,
        "pins": pins,
        "video": video,
        "audio": audio,
        "labels": labels,
        "length_note": note,
    }
