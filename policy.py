"""Rules for one H3 take. No Comfy import. The node calls these before it touches a latent."""

ANCHORS = ("none", "start", "end", "both")
VIDEO_MODES = ("off", "v2v", "extend_after", "extend_before", "extend_both")


def legal_length(length):
    length = int(length)
    return length >= 5 and (length - 5) % 17 == 0


def av_hold(length):
    """24 fps video and 40 Hz audio land together on 39, 90, 141, ..."""
    length = int(length)
    return length >= 39 and legal_length(length) and (length - 39) % 51 == 0


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
    else:
        hold = snap_hold(hold_frames, source_frames, length - 5, align)
        if mode == "extend_after":
            insert = 0
            slice_from = "tail"
        elif mode == "extend_before":
            insert = length - hold
            slice_from = "head"
        else:
            insert = ((length - hold) // 2 // 17) * 17
            slice_from = "middle"
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
        "audio_exact": insert % 51 == 0,
        "mask_hold": float(keep),
        "mask_generate": 1.0,
    }


def snap_down_legal(length):
    length = int(length)
    if length < 5:
        return None
    k = (length - 5) // 17
    return 5 + 17 * k


def max_block_for_vram(gib):
    """Trained H3 spans are 124, 243, and 362 frames. VRAM picks the longest one that fits."""
    gib = float(gib)
    if gib < 24:
        return 124
    if gib < 48:
        return 243
    return 362


def split_span(total, max_block=243, overlap=39):
    """Break a long generate into sequential blocks. Later blocks hold `overlap` frames of the previous one."""
    total = int(total)
    max_block = snap_down_legal(max_block)
    if max_block is None:
        raise ValueError("max block is below one H3 frame count")
    if not av_hold(overlap) and overlap != 0:
        raise ValueError("overlap %s is not an audio-aligned hold (39, 90, 141, ...)" % overlap)
    if total <= max_block:
        length = snap_down_legal(total)
        if length is None or length < 5:
            raise ValueError("span %s is too short to sample" % total)
        return [{"start": 0, "length": length, "hold": 0, "pass": 1}]
    blocks = []
    covered = 0
    while covered < total:
        if not blocks:
            length = min(max_block, snap_down_legal(total))
            blocks.append({"start": 0, "length": length, "hold": 0, "pass": 1})
            covered = length
            continue
        remain = total - covered
        length = snap_down_legal(min(max_block, remain + overlap))
        if length is None or length - overlap < 5:
            raise ValueError("cannot fit the remaining %s frames under a %s-frame cap" % (remain, max_block))
        start = covered - overlap
        blocks.append({
            "start": start,
            "length": length,
            "hold": overlap,
            "pass": len(blocks) + 1,
        })
        covered = start + length
    return blocks


def plan(anchor, start=None, end=None, soft=(), audio=None, length=124, video=None):
    """Return the picture order and the pins. A source video is not a pin."""
    if anchor not in ANCHORS:
        raise ValueError("anchor must be one of %s" % (ANCHORS,))
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

    if not pictures and audio is None and video is None:
        raise ValueError("nothing to condition: no anchor, no reference, no voice, no video")

    labels = []
    for i, (role, _) in enumerate(pictures, 1):
        labels.append("<Picture %s> %s" % (i, role))
    if audio is not None:
        labels.append("<Audio 1> timbre")
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
    return {
        "pictures": pictures,
        "pins": pins,
        "audio": audio is not None,
        "video": video,
        "labels": labels,
        "length_note": note,
    }
