"""ffmpeg for the editor, and what a video file holds.

The editor converts, trims, joins and samples video with ffmpeg. It runs the build the
imageio-ffmpeg package carries, which this pack's requirements install, so a plain ComfyUI
needs nothing more. A system ffmpeg on the PATH is used when that package is missing, and the
H3TAKE_FFMPEG environment variable names any other build. Reading a file's size, frames and
streams needs no ffprobe: PyAV, which ComfyUI itself requires, opens the file directly.
"""

import os
import shutil

MISSING = ("ffmpeg is missing. Install this pack's requirements, which bring it "
           "(pip install -r requirements.txt in the pack's folder), or put ffmpeg on the PATH")
_found = []


def ffmpeg():
    """The ffmpeg program to run. Raises RuntimeError saying what to install when there is none."""
    if not _found:
        exe = os.environ.get("H3TAKE_FFMPEG") or None
        if not exe:
            try:
                import imageio_ffmpeg

                exe = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:  # not installed, or no build for this platform
                exe = None
        exe = exe or shutil.which("ffmpeg")
        if not exe:
            raise RuntimeError(MISSING)
        _found.append(exe)
    return _found[0]


def command(args):
    """An ffmpeg command with the program found. Other commands pass through unchanged."""
    args = list(args)
    if args and args[0] == "ffmpeg":
        args[0] = ffmpeg()
    return args


def video_info(path, count=True):
    """Width, height, whether there is sound, and with count the frames, counted rather than estimated."""
    import av

    with av.open(path) as container:
        if not container.streams.video:
            raise ValueError("%s has no picture" % os.path.basename(path))
        stream = container.streams.video[0]
        info = {"width": stream.codec_context.width, "height": stream.codec_context.height,
                "sound": bool(container.streams.audio)}
        if count:
            # Each video packet is one frame. The demuxer ends with an empty packet, which is not one.
            info["frames"] = sum(1 for packet in container.demux(stream) if packet.size)
    return info


def has_sound(path):
    """Whether a file has a sound track. A picture has none."""
    import av

    with av.open(path) as container:
        return bool(container.streams.audio)
