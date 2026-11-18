"""Optional VOSR 2.0 wrapper integration. The editor streams one source frame per inference."""

import inspect
import os
from pathlib import Path
import subprocess
import tempfile

try:
    from . import media
except ImportError:
    import media


SOURCE = "https://github.com/ylchen333/ComfyUI-VOSR2"
REGISTERED = "@vosr2/registered"
INCOMPATIBLE = "@vosr2/incompatible"
CACHED = "@vosr2/bundle"
SETTINGS = {"upscale": 2, "seed": 42, "color_alignment": "wavelet", "tile_size": 512, "tile_overlap": 32,
            "vae_tile_size": 1024, "vae_tile_overlap": 32}
SETUP = "Install ylchen333/ComfyUI-VOSR2 from GitHub, install its requirements in ComfyUI's Python, and restart ComfyUI."
FIRST_RUN = "The wrapper downloads missing parts of its roughly 7 GB bundle from CSWRY/VOSR on first use."


def capabilities(registry, models_dir):
    loader = registry.get("VOSR2ModelLoader")
    upscale = registry.get("VOSR2Upscale")
    if loader is None and upscale is None:
        return set()
    try:
        params = set(inspect.signature(upscale.execute).parameters)
        fields = {field.id: field for field in loader.define_schema().inputs}
        if not {"image", "model", *SETTINGS} <= params or "VOSR2" not in fields["model"].options or "default" not in fields["dtype"].options:
            return {INCOMPATIBLE}
    except (AttributeError, KeyError, TypeError, ValueError, OSError):
        return {INCOMPATIBLE}
    bundle = Path(models_dir) / "vosr2" / "VOSR2"
    components = ["args.json", "dinov2_vitl14.safetensors", "Qwen-Image-vae-2d/config.json",
                  "Qwen-Image-vae-2d/diffusion_pytorch_model.safetensors"]
    weights = ["clean_weights/ema_model.safetensors", "checkpoints/ema_model.safetensors", "ema_model.safetensors"]
    cached = all((bundle / file).is_file() for file in components) and any((bundle / file).is_file() for file in weights)
    return {REGISTERED, CACHED} if cached else {REGISTERED}


def status(available):
    available = available or set()
    ready = REGISTERED in available
    reason = ("Update ComfyUI-VOSR2 to its current GitHub version and restart ComfyUI." if INCOMPATIBLE in available else SETUP)
    return {"id": "vosr2", "title": "VOSR2 (experimental)", "experimental": True, "detail": False,
            "installed": ready, "weights_cached": CACHED in available,
            "missing": [] if ready else [reason], "source": SOURCE,
            "about": "Upscales individual frames with a fixed seed. Fine details can flicker in motion.",
            "setup": ("Model files are present." if CACHED in available else FIRST_RUN) if ready else reason}


def stream_video(source, target, transform, check, progress=None):
    """Encode transformed RGB frames as they arrive, copying the source audio packets into the output."""
    import av

    with av.open(source) as container:
        video = container.streams.video[0]
        width, height = 2 * video.codec_context.width, 2 * video.codec_context.height
        rate = video.average_rate or 24
        with tempfile.TemporaryDirectory(dir=os.path.dirname(target), prefix=".h3-vosr2-") as work:
            partial = os.path.join(work, "upscale.mp4")
            command = media.command(["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                     "-video_size", "%sx%s" % (width, height), "-framerate", str(rate), "-i", "pipe:0",
                                     "-i", source, "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264", "-preset", "fast",
                                     "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", partial])
            with tempfile.TemporaryFile() as errors:
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors)
                try:
                    for index, frame in enumerate(container.decode(video), 1):
                        check()
                        rgb = transform(frame.to_ndarray(format="rgb24"))
                        if rgb.shape != (height, width, 3) or rgb.dtype.name != "uint8":
                            raise ValueError("VOSR2 returned an unexpected picture size or pixel format")
                        process.stdin.write(memoryview(rgb).cast("B"))
                        if progress and video.frames:
                            progress(index, video.frames)
                    process.stdin.close()
                    if process.wait():
                        raise BrokenPipeError
                except BrokenPipeError as error:
                    process.wait()
                    errors.seek(0)
                    raise RuntimeError("VOSR2 video encoding failed: " + errors.read().decode(errors="replace")) from error
                finally:
                    if process.poll() is None:
                        process.kill()
                    process.wait()
                    if not process.stdin.closed:
                        try:
                            process.stdin.close()
                        except BrokenPipeError:
                            pass
            check()
            os.replace(partial, target)
