"""Moving live preview. Each sampling step becomes a short clip the editor can loop.

ComfyUI's own preview decodes one still frame. Here the whole clip is decoded with
H3's tiny autoencoder, sound included, so motion and timing show while sampling.
"""

import logging
import os
import re
import subprocess
import tempfile
import threading
import wave
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

import comfy.model_management
import comfy.model_prefetch
import comfy.patcher_extension
import comfy.sd
import comfy.taesd.taesd as taesd
import comfy.utils
import folder_paths
from comfy_api.latest import io
from server import PromptServer

from . import media

FPS = 24
TAE_PREFIX = "taeh3"
MAX_WIDTH = 640
KEY = "h3take_live_preview"
# Only clips this node wrote look like this, so cleanup can never touch a render.
OWN_CLIP = re.compile(r"^[0-9a-f-]{36}_[0-9]+\.mp4$")
_tae = {}


def preview_folder(subfolder):
    """The preview folder, which must sit inside the output folder."""
    base = os.path.realpath(folder_paths.get_output_directory())
    folder = os.path.realpath(os.path.join(base, subfolder))
    if not folder.startswith(base + os.sep):
        raise ValueError("the preview folder must be inside the output folder")
    return folder


def downscale(images):
    """[N, 3, H, W] no wider than MAX_WIDTH."""
    width = images.shape[-1]
    if width <= MAX_WIDTH:
        return images
    height = int(round(images.shape[-2] * MAX_WIDTH / width / 2)) * 2
    return torch.nn.functional.interpolate(images, size=(height, MAX_WIDTH), mode="area")


def to_bytes(images):
    return (images.clamp(0, 1) * 255).to(torch.uint8).movedim(1, -1).cpu().numpy()


class FlatDecoder:
    """A 2D tiny decoder, one picture per latent step. The layers come from the file itself."""

    def __init__(self, sd):
        layers = []
        for index in range(max(int(key.split(".")[0]) for key in sd) + 1):
            weight = sd.get("%s.weight" % index)
            block = sd.get("%s.conv.0.weight" % index)
            if block is not None:
                layers.append(taesd.Block(block.shape[1], sd["%s.conv.4.weight" % index].shape[0]))
            elif weight is not None:
                layers.append(taesd.conv(weight.shape[1], weight.shape[0], bias="%s.bias" % index in sd))
            elif index == 0:
                layers.append(taesd.Clamp())
            elif index == 2:
                layers.append(torch.nn.ReLU())
            else:
                layers.append(torch.nn.Upsample(scale_factor=2))
        self.net = torch.nn.Sequential(*layers)
        self.net.load_state_dict(sd)
        self.device = comfy.model_management.get_torch_device()
        self.net = self.net.eval().to(self.device, dtype=torch.float16)

    @torch.no_grad()
    def frames(self, video):
        pictures = []
        for step in range(video.shape[2]):
            latent = video[:1, :, step].to(self.device, dtype=torch.float16)
            pictures.append(to_bytes(downscale(self.net(latent).float())))
        return np.concatenate(pictures)


class TemporalDecoder:
    """madebyollin's temporal decoder, which ComfyUI loads as a VAE. Every frame comes back."""

    def __init__(self, sd):
        self.vae = comfy.sd.VAE(sd=sd)
        self.vae.throw_exception_if_invalid()
        self.vae.first_stage_model.show_progress_bar = False

    def frames(self, video):
        images = self.vae.decode(video[:1])
        images = images.reshape((-1,) + tuple(images.shape[-3:]))[..., :3]
        return to_bytes(downscale(images.movedim(-1, 1).float()))


def tiny_decoder():
    """H3's tiny decoder, or None when it is not installed."""
    if "decoder" not in _tae:
        name = next((n for n in folder_paths.get_filename_list("vae_approx") if n.startswith(TAE_PREFIX)), None)
        _tae["decoder"] = None
        if name:
            sd = comfy.utils.load_torch_file(folder_paths.get_full_path("vae_approx", name))
            temporal = any(key.startswith("decoder.") for key in sd)
            _tae["decoder"] = TemporalDecoder(sd) if temporal else FlatDecoder(sd)
    return _tae["decoder"]


def to_frames(video):
    """Decode a video latent inside the sampler's allocation scope, so it costs no extra compile work."""
    graphed = comfy.model_prefetch.malloc_graph_enabled(video.device)
    if graphed:
        comfy.model_prefetch.malloc_graph_begin(video.device)
    try:
        return tiny_decoder().frames(video)
    finally:
        if graphed:
            comfy.model_prefetch.malloc_graph_end()


def to_wave(audio_vae, audio):
    samples = audio_vae.decode(audio[:1]).movedim(-1, 1)[0].float().clamp(-1, 1).cpu().numpy()
    if samples.ndim == 1:
        samples = samples[None]
    return (samples.T * 32767).astype(np.int16), getattr(audio_vae, "audio_sample_rate", 32000)


def encode(frames, sound, path, seconds):
    """Write a clip that starts playing as soon as the browser has a little of it."""
    height, width = frames.shape[1:3]
    args = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%sx%s" % (width, height),
            "-r", "%.5f" % (frames.shape[0] / seconds), "-i", "pipe:0"]
    temporary = None
    if sound is not None:
        pcm, rate = sound
        handle, temporary = tempfile.mkstemp(suffix=".wav")
        os.close(handle)
        with wave.open(temporary, "wb") as target:
            target.setnchannels(pcm.shape[1])
            target.setsampwidth(2)
            target.setframerate(rate)
            target.writeframes(pcm.tobytes())
        args += ["-i", temporary, "-c:a", "aac", "-b:a", "96k", "-shortest"]
    args += ["-r", str(FPS), "-c:v", "libx264", "-preset", "ultrafast", "-crf", "26", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", path]
    try:
        subprocess.run(media.command(args), input=np.ascontiguousarray(frames).tobytes(), check=True, capture_output=True)
    finally:
        if temporary:
            os.remove(temporary)


class LivePreview:
    def __init__(self, audio_vae, subfolder):
        self.audio_vae = audio_vae
        self.subfolder = subfolder

    def __call__(self, executor, noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=None):
        model = executor.class_obj.model_patcher.model
        server = PromptServer.instance
        prompt_id = server.last_prompt_id
        folder = preview_folder(self.subfolder)
        os.makedirs(folder, exist_ok=True)
        for old in os.listdir(folder):
            if OWN_CLIP.match(old):
                os.remove(os.path.join(folder, old))
        worker = ThreadPoolExecutor(max_workers=1)
        busy = threading.Event()
        written = []

        def publish(frames, sound, seconds, step, total):
            try:
                name = "%s_%s.mp4" % (prompt_id, step)
                encode(frames, sound, os.path.join(folder, name), seconds)
                written.append(name)
                server.send_sync("h3take_preview", {
                    "prompt_id": prompt_id, "filename": name, "subfolder": self.subfolder, "step": step + 1, "total": total,
                }, server.client_id)
            except Exception as error:
                logging.warning("H3 live preview could not encode: %s", error)
            finally:
                busy.clear()

        def on_step(step, x0, x, total):
            if callback is not None:
                callback(step, x0, x, total)
            if busy.is_set():
                return
            try:
                clean = model.process_latent_out(x0)
                # The sampler works on one flat pack, so this hook can see it before it is unpacked.
                streams = clean.unbind() if clean.is_nested else comfy.utils.unpack_latents(clean, latent_shapes)
                frames = to_frames(streams[0])
                sound = to_wave(self.audio_vae, streams[1]) if self.audio_vae is not None and len(streams) > 1 else None
                seconds = (((streams[0].shape[2] - 2) // 5) * 17 + 5) / float(FPS)
                busy.set()
                worker.submit(publish, frames, sound, seconds, step, total)
            except Exception as error:
                logging.warning("H3 live preview could not decode: %s", error, exc_info=True)

        try:
            return executor(noise, latent_image, sampler, sigmas, denoise_mask, on_step, disable_pbar, seed, latent_shapes=latent_shapes)
        finally:
            worker.shutdown(wait=True)
            for name in written[:-1]:
                try:
                    os.remove(os.path.join(folder, name))
                except OSError:
                    pass


class H3LivePreview(io.ComfyNode):
    """Attach the moving preview to a model. Does nothing when taeh3 is not installed."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3LivePreview",
            display_name="H3 Live Preview",
            category="minimax/h3",
            description="Show the whole clip moving, with sound, at every sampling step. The H3 Editor adds this for you.",
            inputs=[
                io.Model.Input("model"),
                io.Vae.Input("audio_vae", optional=True, tooltip="H3 audio VAE. With it the preview has sound."),
                io.String.Input("folder", default="h3take_preview", tooltip="Where preview clips go, inside the output folder."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, folder, audio_vae=None) -> io.NodeOutput:
        preview_folder(folder)
        if tiny_decoder() is None:
            logging.info("H3 live preview is off: no %s file in models/vae_approx", TAE_PREFIX)
            return io.NodeOutput(model)
        patched = model.clone()
        patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, KEY, LivePreview(audio_vae, folder))
        return io.NodeOutput(patched)


NODE_CLASS_MAPPINGS = {"H3LivePreview": H3LivePreview}
NODE_DISPLAY_NAME_MAPPINGS = {"H3LivePreview": "H3 Live Preview"}
