"""A take's latents, kept beside its video, and the part of them a later take holds.

A take that extends or bridges another one holds some of its frames. Read back from the video file, those frames go
through the decoder and the encoder once more and come out a little darker and softer, and the take that carries on
from them inherits that. So every take saves the latents it was sampled in, picture and sound, in
<take>.hold.safetensors, and the next take holds those as they are.

The file holds
  video   the whole picture latent, [1, C, steps, h, w]
  audio   the whole sound latent, [1, C, bands, ticks]
  final   for a final that was refined at a larger size, the picture latent after that pass, at that size
all as float16. A file written before whole takes were saved holds `head` and `tail` instead, the picture's first and
last 39 frames, and is still read.

A final is sampled twice: once at H3's own size, then enlarged and refined. A take that extends it is sampled twice
too, and each of its passes has to hold the source at that pass's size: `video` in the first, `final` in the second.
The sound is the first pass's in both. A draft has one pass and one size, so no `final`.

Which frames can be held from the file:
  picture  H3's picture latent repeats every 17 frames: one step of 1 frame, then four steps of 4. A latent has to
           begin where that pattern begins, so the part must start on a multiple of 17 frames and be 5 + 17k frames
           long. on_grid() says whether a part is; snap_out() and snap_in() move a cut to the nearest place that is.
  sound    The sound latent has 40 steps a second, 5 for every 3 frames. A part that starts on a multiple of 3 frames
           and is a multiple of 3 frames long takes whole steps, and its sound is held from the file too. For any
           other part only the picture is, and the sound is encoded from the video file, cut to the sample.

No Comfy import: a latent comes in as the tensors of its streams.
"""
import os

try:
    from .policy import audio_ticks, frame_to_step, latent_frames, legal_length, video_steps
except ImportError:
    from policy import audio_ticks, frame_to_step, latent_frames, legal_length, video_steps

CYCLE = 17      # frames in one round of the picture latent's pattern
HOLD_STEPS = 12   # 39 frames: what a file from before whole takes were saved holds of each end


def hold_path(file):
    """Where a take keeps its latents: beside its video, named after the same take."""
    stem = os.path.splitext(file)[0]
    head, _, counter = stem.rstrip("_").rpartition("_")
    return (head if counter.isdigit() and head else stem) + ".hold.safetensors"


def on_grid(first, count):
    """Can frames first..first+count of a take be held as the picture latent they were sampled in."""
    return int(first) >= 0 and int(first) % CYCLE == 0 and legal_length(count)


def whole_ticks(first, count):
    """Does that part also start and end on a step of the sound latent."""
    return int(first) % 3 == 0 and int(count) % 3 == 0


def snap_out(out, hold, least=0):
    """The latest frame at or before out where a part of hold frames can end on the picture's grid.

    least is the earliest frame the part may start at. Returns out itself when nothing earlier fits.
    """
    out, hold = int(out), int(hold)
    snapped = hold + ((out - hold) // CYCLE) * CYCLE
    return snapped if snapped - hold >= max(0, int(least)) else out


def snap_in(start, hold, most):
    """The earliest frame at or after start where a part of hold frames can begin on the picture's grid.

    most is the frame the part has to end by. Returns start itself when nothing later fits.
    """
    start, hold = int(start), int(hold)
    snapped = -(-start // CYCLE) * CYCLE
    return snapped if snapped + hold <= int(most) else start


def streams(samples):
    """The picture and sound tensors of an H3 latent's samples. Sound is None for a picture alone."""
    if getattr(samples, "is_nested", False):
        parts = list(samples.tensors if hasattr(samples, "tensors") else samples.unbind())
        return parts[0], (parts[1] if len(parts) > 1 else None)
    return samples, None


def save(path, samples, final=None):
    """Write a sampled take's latents to path. final is the samples of its refine pass, when it had one."""
    import safetensors.torch

    video, audio = streams(samples)
    if video.dim() != 5:
        raise ValueError("this is not an H3 video latent")
    kept = {"video": video.detach().cpu().half().contiguous()}
    if audio is not None:
        kept["audio"] = audio.detach().cpu().half().contiguous()
    if final is not None:
        refined, _ = streams(final)
        if refined.dim() != 5 or refined.shape[2] != video.shape[2]:
            raise ValueError("the refined latent is not the same take: %s steps against %s" % (refined.shape[2], video.shape[2]))
        kept["final"] = refined.detach().cpu().half().contiguous()
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    safetensors.torch.save_file(kept, path, metadata={"frames": str(latent_frames(video.shape[2]))})


def load_take(path):
    """A take's latents as sampled, picture and sound, from the file beside its video: for a pass that starts from them,
    such as finishing a draft. Raises when the file is missing or keeps less than a whole take."""
    if not os.path.exists(path):
        raise FileNotFoundError("%s has no latents saved beside it. Render it again to finish it" % os.path.basename(path))
    from safetensors import safe_open

    with safe_open(path, framework="pt") as saved:
        keys = set(saved.keys())
        if "video" not in keys or "audio" not in keys:
            raise ValueError("%s keeps no whole take, picture and sound. Render it again to finish it" % os.path.basename(path))
        return saved.get_tensor("video").float(), saved.get_tensor("audio").float()


def part(path, first, stop, total):
    """The saved latents for frames first..stop of a take of total frames, or None when the file has none for them.

    Returns {"samples": picture}, with "audio" when the part takes whole steps of the sound, and "final" when the
    take was refined at a larger size: the same frames after that pass.
    """
    first, stop, total = int(first), int(stop), int(total)
    count = stop - first
    if not os.path.exists(path) or first < 0 or stop > total or not legal_length(count):
        return None
    steps = video_steps(count)
    try:
        from safetensors import safe_open

        with safe_open(path, framework="pt") as saved:
            keys = set(saved.keys())
            if "video" in keys:
                if not on_grid(first, count):
                    return None
                video = saved.get_tensor("video")
                at = frame_to_step(first)
                if latent_frames(video.shape[2]) != total or at + steps > video.shape[2]:
                    return None       # the file beside this video is of another take
                held = {"samples": video[:, :, at:at + steps].float()}
                if "audio" in keys and whole_ticks(first, count):
                    audio = saved.get_tensor("audio")
                    tick, ticks = first * 5 // 3, audio_ticks(count)
                    if tick + ticks <= audio.shape[-1]:
                        held["audio"] = audio[..., tick:tick + ticks].float()
                if "final" in keys:
                    refined = saved.get_tensor("final")
                    if refined.shape[2] == video.shape[2]:
                        held["final"] = refined[:, :, at:at + steps].float()
                return held
            if stop == total and "tail" in keys:
                tail = saved.get_tensor("tail")
                if tail.shape[2] >= steps:
                    return {"samples": tail[:, :, -steps:].float()}
            if first == 0 and "head" in keys:
                head = saved.get_tensor("head")
                if head.shape[2] >= steps:
                    return {"samples": head[:, :, :steps].float()}
    except Exception:  # an unreadable file: the frames are encoded as before
        return None
    return None


def held_picture(held, steps, height, width):
    """The picture latent of a hold, if it is for exactly these frames at this size. None means encode the frames.

    A hold of a refined final carries the frames at two sizes, and the one that fits the take is used: the first
    pass's for a take at H3's own size, the refined one for the take that conditions the refine pass.
    """
    if not isinstance(held, dict):
        return None
    for key in ("samples", "final"):
        if held.get(key) is None:
            continue
        video, _ = streams(held[key])
        if video.dim() == 5 and video.shape[2] == steps and tuple(video.shape[3:]) == (height // 16, width // 16):
            return video.float()
    return None


def held_sound(held, window, like):
    """The sound latent of a hold, if it covers exactly the ticks the take holds. None means encode the sound.

    window is the take's audio window. like is the take's own sound latent, for its shape.
    """
    if not isinstance(held, dict) or held.get("audio") is None or window is None:
        return None
    audio = held["audio"]
    if window.get("mode") != "span" or window.get("source_second") != 0:
        return None
    if audio.dim() != like.dim() or audio.shape[-1] != window["ticks"] or tuple(audio.shape[1:-1]) != tuple(like.shape[1:-1]):
        return None
    return audio.float()
