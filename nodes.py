"""One H3 take. Calls ComfyUI's own MiniMax nodes. Does not import any other pack."""

import logging
import os
import subprocess

import folder_paths
import numpy as np
import torch
from comfy_api.latest import InputImpl, io
from PIL import Image, ImageOps

from .latent_writer import carry_tail, encode_audio, encode_video, write_holds
from . import h3lora
from .twins import with_twins
from . import hyperflow
from . import clipproj
from . import media

# Why a HyperFlow .gguf cannot load here (the gguf package, which this pack's requirements install),
# or None. The INT8 safetensors files load either way.
clipproj.register_folder()
HYPERFLOW_MISSING = hyperflow.gguf_missing()
if HYPERFLOW_MISSING:
    logging.warning("H3 Editor: %s", HYPERFLOW_MISSING)
from .policy import (
    AUDIO_ALIGNS,
    ANCHORS,
    AUDIO_HOLDS,
    CHECKPOINTS,
    FPS,
    LORA_MODES,
    MAX_PICTURES,
    MAX_REF_AUDIOS,
    MAX_REF_VIDEOS,
    STRENGTHS,
    VIDEO_MODES,
    audio_ticks,
    audio_window,
    bridge_spans,
    frame_count,
    frame_to_step,
    hold_curve,
    latent_frames,
    merge_curves,
    parse_frames,
    plan,
    resolve_keep,
    video_span,
    video_steps,
)


def ordered(group):
    """Autogrow inputs as a list, in socket order. Sockets are named prefix_N."""
    if not group:
        return []
    keys = sorted((k for k in group if group[k] is not None), key=lambda k: int(k.rsplit("_", 1)[-1]))
    return [(k, group[k]) for k in keys]


def audio_seconds(audio):
    if audio is None:
        return 0.0
    return audio["waveform"].shape[-1] / float(audio["sample_rate"])


class H3Take(io.ComfyNode):
    """One H3 shot: text, pinned frames, references, a held source video, or a held soundtrack.

    Sampling, the checkpoint, and the sigma shift stay on the graph.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3Take",
            display_name="H3 Take",
            category="minimax/h3",
            description=(
                "One H3 shot on ComfyUI's own nodes. Pin stills at any frame, add references, "
                "hold source video or audio, and generate everything else."
            ),
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae", optional=True, tooltip="H3 audio VAE. Needed for reference audio and held audio."),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=768, min=32, max=4096, step=32),
                io.Int.Input("height", default=1152, min=32, max=4096, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17, tooltip="Frames at 24 fps on the 5 + 17k grid. 124 is about 5 seconds."),
                io.Combo.Input("checkpoint", options=list(CHECKPOINTS), default="ref2va", tooltip="Which H3 checkpoint the graph loads. fl2va and ref2va read the prompt differently."),
                io.Combo.Input("anchor", options=list(ANCHORS), default="none"),
                io.Image.Input("start_image", optional=True),
                io.Image.Input("end_image", optional=True),
                io.String.Input("mid_frames", default="", tooltip="Frame numbers for the mid images, in order. Example: 36, 72"),
                io.Autogrow.Input("mid_images", optional=True, template=io.Autogrow.TemplatePrefix(
                    input=io.Image.Input("mid_image", tooltip="A still to guide the take at the matching mid frame."),
                    prefix="mid_image_", min=0, max=8)),
                io.Combo.Input("video_mode", options=list(VIDEO_MODES), default="off"),
                io.Image.Input("source_video", optional=True, tooltip="24 fps frames to hold."),
                io.Image.Input("end_video", optional=True, tooltip="With video_mode bridge, the clip the take must arrive at."),
                io.Audio.Input("end_audio", optional=True, tooltip="Soundtrack of end_video."),
                io.Latent.Input("source_hold", optional=True, tooltip="The held frames of source_video as the latent they were sampled in, from H3 Load Clip. Used in place of encoding the frames again, so a chain of extensions does not lose detail or brightness at each join."),
                io.Latent.Input("end_hold", optional=True, tooltip="The same for end_video."),
                io.Latent.Input("carry", optional=True, tooltip="A finished H3 latent. Its last hold_frames open this take, with no re-encode. Leave video_mode off."),
                io.Int.Input("hold_frames", default=39, min=5, max=3600, step=1),
                io.Combo.Input("strength", options=list(STRENGTHS) + ["custom"], default="hold", tooltip="How much the held video may change. light tints it, medium and heavy keep the shot but follow the prompt, regenerate keeps nothing."),
                io.Float.Input("keep", default=0.0, min=0.0, max=1.0, step=0.0001, tooltip="Raw mask value, read only when strength is custom. The useful range is about 0.95 to 0.996."),
                io.Combo.Input("align", options=["av", "video"], default="av", tooltip="av snaps holds to 39, 90, 141, so picture and sound share a boundary."),
                io.Mask.Input("region", optional=True, tooltip="White areas of the held video regenerate. One mask or one per frame."),
                io.Int.Input("video_feather", default=0, min=0, max=20, tooltip="Latent steps of soft edge where held video meets generated video."),
                io.Combo.Input("audio_hold", options=list(AUDIO_HOLDS), default="off", tooltip="span holds the source audio across the held video. full lays it along the whole take."),
                io.Audio.Input("source_audio", optional=True),
                io.Int.Input("audio_feather", default=8, min=0, max=80, tooltip="Audio ticks of soft edge. 40 ticks is one second."),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match"),
                io.Autogrow.Input("ref_images", optional=True, template=io.Autogrow.TemplatePrefix(
                    input=io.Image.Input("ref_image", tooltip="A picture the model may use, not a frame it must hit."),
                    prefix="ref_image_", min=0, max=MAX_PICTURES)),
                io.Autogrow.Input("ref_videos", optional=True, template=io.Autogrow.TemplatePrefix(
                    input=io.Image.Input("ref_video", tooltip="Reference video frames at 24 fps, 2 to 15 seconds."),
                    prefix="ref_video_", min=0, max=MAX_REF_VIDEOS)),
                io.Autogrow.Input("ref_video_audios", optional=True, template=io.Autogrow.TemplatePrefix(
                    input=io.Audio.Input("ref_video_audio", tooltip="Soundtrack of the same-numbered reference video."),
                    prefix="ref_video_audio_", min=0, max=MAX_REF_VIDEOS)),
                io.Autogrow.Input("ref_audios", optional=True, template=io.Autogrow.TemplatePrefix(
                    input=io.Audio.Input("ref_audio", tooltip="A voice or sound the model may use."),
                    prefix="ref_audio_", min=0, max=MAX_REF_AUDIOS)),
                io.Combo.Input("audio_align", options=list(AUDIO_ALIGNS), default="video", optional=True,
                               tooltip="With audio_hold full: video lines the source audio up with the held video, "
                                       "take starts it at the take's first frame."),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Latent.Output(display_name="latent"),
                io.String.Output(display_name="labels"),
            ],
        )

    @classmethod
    def execute(cls, clip, vae, prompt, width, height, length, checkpoint, anchor, mid_frames, video_mode,
                hold_frames, strength, keep, align, video_feather, audio_hold, audio_feather, ref_image_size,
                audio_vae=None, start_image=None, end_image=None, mid_images=None, source_video=None, end_video=None,
                end_audio=None, carry=None, region=None, source_audio=None, ref_images=None, ref_videos=None, ref_video_audios=None,
                ref_audios=None, audio_align="video", source_hold=None, end_hold=None) -> io.NodeOutput:
        keep = resolve_keep(strength, keep)
        second = None
        if video_mode == "bridge":
            if source_video is None or end_video is None:
                raise ValueError("a bridge needs source_video and end_video")
            span, end_span = bridge_spans(
                frame_count(source_video, "source video"), frame_count(end_video, "end video"),
                length, hold_frames=hold_frames, align=align, keep=keep,
            )
            window = audio_window(audio_hold, span, length, audio_seconds(source_audio), audio_align)
            end_window = audio_window(audio_hold, end_span, length, audio_seconds(end_audio), audio_align)
            second = (end_span, end_video, end_audio, end_window, end_hold)
        elif carry is not None:
            if source_video is not None or video_mode != "off":
                raise ValueError("carry replaces the source video. Disconnect it and leave video_mode off.")
            frames = latent_frames(carry["samples"].unbind()[0].shape[2])
            span = video_span("extend_after", frames, length, hold_frames=hold_frames, align=align, keep=keep)
            if audio_hold == "full":
                # The picture carries on from the latent, and the sound is a recorded track laid along the whole take.
                if source_audio is None:
                    raise ValueError("carry with audio_hold full needs source_audio, the track to lay along the take")
                window = audio_window(audio_hold, span, length, audio_seconds(source_audio), audio_align)
            else:
                window = audio_window(audio_hold, span, length, frames / float(FPS))
        else:
            source_frames = frame_count(source_video, "source video") if source_video is not None else 0
            span = video_span(video_mode, source_frames, length, hold_frames=hold_frames, align=align, keep=keep)
            window = audio_window(audio_hold, span, length, audio_seconds(source_audio), audio_align)
        soft = [image for _, image in ordered(ref_images)]
        mids = [image for _, image in ordered(mid_images)]
        videos = ordered(ref_videos)
        tracks = ref_video_audios or {}
        has_track = [tracks.get("ref_video_audio_" + key.rsplit("_", 1)[-1]) is not None for key, _ in videos]
        audios = ordered(ref_audios)
        decided = plan(
            anchor,
            start=start_image,
            end=end_image,
            soft=soft,
            length=length,
            video=span,
            checkpoint=checkpoint,
            mid_frames=parse_frames(mid_frames),
            mids=mids,
            ref_videos=has_track,
            ref_audios=len(audios),
            audio=window,
        )
        if (audios or any(has_track) or (window is not None and (carry is None or audio_hold == "full"))) and audio_vae is None:
            raise ValueError("audio inputs need the H3 audio VAE")
        if region is not None and span is None:
            raise ValueError("region needs a held source video")

        from comfy_extras.nodes_minimax_h3 import (
            MiniMaxH3AddGuide,
            MiniMaxH3ImageToVideo,
            MiniMaxH3ReferenceToVideo,
        )

        if checkpoint == "fl2va":
            out = MiniMaxH3ImageToVideo.execute(
                clip, vae, prompt, width, height, length, first_frame=start_image, last_frame=end_image
            )
            pinned = [pin for pin in decided["pins"] if pin[0].startswith("mid")]
        else:
            pictures = {"ref_image_%s" % i: image for i, (_, image) in enumerate(decided["pictures"], 1)}
            out = MiniMaxH3ReferenceToVideo.execute(
                clip,
                prompt,
                width,
                height,
                length,
                ref_image_size=ref_image_size,
                vae=vae,
                audio_vae=audio_vae,
                ref_images=pictures or None,
                ref_videos=dict(videos) or None,
                ref_video_audios=tracks or None,
                ref_audios=dict(audios) or None,
            )
            pinned = decided["pins"]
        positive, latent = out[0], out[1]

        images = {"start": start_image, "end": end_image}
        for i, image in enumerate(mids, 1):
            images["mid %s" % i] = image
        for role, frame in pinned:
            positive = MiniMaxH3AddGuide.execute(positive, latent, frame, vae=vae, image=images[role])[0]

        if span is not None or window is not None:
            # A carried take brings its own sound for the held span. A track given with it replaces that.
            track = source_audio if carry is None or audio_hold == "full" else None
            pieces = [(span, source_video, track, window, source_hold)]
            if second is not None:
                pieces.append(second)
            latent = cls.hold(latent, vae, audio_vae, pieces, length, width, height,
                              video_feather, audio_feather, region, carry)

        card = "; ".join(decided["labels"])
        if decided["length_note"] == "short":
            card += "; length is below the trained 124-frame range"
        return io.NodeOutput(positive, latent, card)

    @staticmethod
    def hold(latent, vae, audio_vae, pieces, length, width, height, video_feather, audio_feather, region, carry=None):
        total_steps = video_steps(length)
        video_curve = None
        audio_curve = None
        writes = []
        for span, frames, audio, window, held in pieces:
            video_latent = None
            video_first = 0
            audio_latent = None
            audio_first = 0
            audio_tail = None
            if span is not None:
                video_first = frame_to_step(span["insert"])
                steps = video_steps(span["hold"])
                if carry is not None:
                    video_latent, audio_tail = carry_tail(carry, steps, audio_ticks(span["hold"]))
                    if tuple(video_latent.shape[3:]) != (height // 16, width // 16):
                        raise ValueError("carry is a different size from this take")
                else:
                    video_latent = held_latent(held, steps, height, width)
                    if video_latent is None:
                        start = span["source_start"]
                        video_latent = encode_video(vae, frames[start:start + span["hold"]], width, height, steps)
                video_curve = merge_curves(
                    video_curve, hold_curve(total_steps, video_first, steps, base=span["keep"], feather=video_feather)
                )
            if window is not None:
                audio_first = window["first"]
                if carry is not None and audio is None:
                    audio_latent = audio_tail
                else:
                    audio_latent = encode_audio(audio_vae, audio, window["source_second"], window["ticks"])
                audio_curve = merge_curves(
                    audio_curve,
                    hold_curve(audio_ticks(length), audio_first, audio_latent.shape[-1], feather=audio_feather),
                )
            writes.append((video_latent, video_first, audio_latent, audio_first))
        for video_latent, video_first, audio_latent, audio_first in writes:
            latent = write_holds(latent, video_latent, video_first, audio_latent, audio_first,
                                 video_curve, audio_curve, region)
        return latent


def held_latent(held, steps, height, width):
    """The picture latent of a hold, if one was saved for exactly these frames at this size. None means encode the frames."""
    if not isinstance(held, dict) or held.get("samples") is None:
        return None
    video = held["samples"]
    if getattr(video, "is_nested", False):
        video = video.tensors[0]
    if video.dim() != 5 or video.shape[2] != steps or tuple(video.shape[3:]) != (height // 16, width // 16):
        return None
    return video.float()


def hold_file(file):
    """Where a take keeps its hold latents: beside its video, named after the same take."""
    stem = os.path.splitext(file)[0]
    head, _, counter = stem.rstrip("_").rpartition("_")
    return (head if counter.isdigit() and head else stem) + ".hold.safetensors"


def saved_hold(path, first, stop, total):
    """The saved latent for frames first..stop of a take, or None when there is none for exactly that part."""
    count = stop - first
    if not os.path.exists(path) or count < 5 or (count - 5) % 17:
        return None
    try:
        import safetensors.torch

        saved = safetensors.torch.load_file(path)
    except Exception:  # an unreadable file: the frames are encoded as before
        return None
    steps = video_steps(count)
    if stop == total and "tail" in saved and saved["tail"].shape[2] >= steps:
        return {"samples": saved["tail"][:, :, -steps:].float()}
    if first == 0 and "head" in saved and saved["head"].shape[2] >= steps:
        return {"samples": saved["head"][:, :, :steps].float()}
    return None


def output_file(file):
    base = os.path.realpath(folder_paths.get_output_directory())
    path = os.path.realpath(os.path.join(base, file))
    if not path.startswith(base + os.sep) or not os.path.exists(path):
        raise ValueError("no file at %s in the output folder" % file)
    return path


class H3LoadStill(io.ComfyNode):
    """A picture from ComfyUI's output folder. The editor keeps references there, with the project."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3LoadStill",
            display_name="H3 Load Still",
            category="minimax/h3",
            description="Read a picture from the output folder. Used by the editor for references and first and last frames.",
            inputs=[io.String.Input("file", tooltip="Path inside the output folder.")],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    def execute(cls, file) -> io.NodeOutput:
        with Image.open(output_file(file)) as image:
            pixels = np.array(ImageOps.exif_transpose(image).convert("RGB")).astype(np.float32) / 255.0
        return io.NodeOutput(torch.from_numpy(pixels)[None])


class H3LoadSound(io.ComfyNode):
    """A sound from ComfyUI's output folder, such as the voice a cast member speaks with."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3LoadSound",
            display_name="H3 Load Sound",
            category="minimax/h3",
            description="Read a sound file from the output folder. Used by the editor for cast voices.",
            inputs=[
                io.String.Input("file", tooltip="Path inside the output folder."),
                io.Float.Input("start", default=0.0, min=0.0, max=36000.0, step=0.001, tooltip="Seconds into the file to start reading."),
                io.Float.Input("seconds", default=30.0, min=0.1, max=600.0, step=0.5, tooltip="Reads at most this much."),
            ],
            outputs=[io.Audio.Output(display_name="audio")],
        )

    @classmethod
    def execute(cls, file, start, seconds) -> io.NodeOutput:
        rate = 48000
        # Decoding from the start and trimming after is sample accurate, unlike seeking in compressed audio.
        pcm = subprocess.run(
            media.command(["ffmpeg", "-v", "error", "-i", output_file(file), "-vn",
                           "-af", "atrim=start=%.6f:duration=%.6f" % (start, seconds),
                           "-ac", "2", "-ar", str(rate), "-f", "f32le", "-"]),
            capture_output=True, check=False)
        if pcm.returncode != 0 or not pcm.stdout:
            raise ValueError("could not read sound from %s: %s" % (file, pcm.stderr.decode(errors="replace")[-300:]))
        samples = torch.from_numpy(np.frombuffer(pcm.stdout, dtype=np.float32).copy()).reshape(-1, 2).t()
        return io.NodeOutput({"waveform": samples[None], "sample_rate": rate})


class H3LoadClip(io.ComfyNode):
    """Part of a rendered clip from ComfyUI's output folder, with its sound."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3LoadClip",
            display_name="H3 Load Clip",
            category="minimax/h3",
            description="Read frames and sound from a clip in the output folder. Used by the editor to extend and bridge clips.",
            inputs=[
                io.String.Input("file", tooltip="Path inside the output folder."),
                io.Int.Input("first", default=0, min=0, max=100000),
                io.Int.Input("last", default=0, min=0, max=100000, tooltip="Stops before this frame. 0 reads to the end."),
            ],
            outputs=[io.Image.Output(display_name="frames"), io.Audio.Output(display_name="audio"),
                     io.Latent.Output(display_name="hold", tooltip="These frames as the latent they were sampled in, when the clip saved one and they are its first or last frames. Connect it to H3 Take's source_hold or end_hold.")],
        )

    @classmethod
    def execute(cls, file, first, last) -> io.NodeOutput:
        path = output_file(file)
        video = InputImpl.VideoFromFile(path).get_components()
        frames = video.images
        stop = last or frames.shape[0]
        audio = video.audio
        if audio is not None:
            rate = audio["sample_rate"]
            rate_first = int(first / float(FPS) * rate)
            rate_stop = int(stop / float(FPS) * rate)
            audio = {"waveform": audio["waveform"][..., rate_first:rate_stop], "sample_rate": rate}
        return io.NodeOutput(frames[first:stop], audio, saved_hold(hold_file(path), first, stop, frames.shape[0]))


HOLD_STEPS = 12   # 39 frames, the hold the editor's extensions and bridges use


class H3SaveHold(io.ComfyNode):
    """Keeps the first and last frames of a sampled take as latents, for the clips that will extend it."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3SaveHold",
            display_name="H3 Save Hold",
            category="minimax/h3",
            description="Save the first and last 39 frames of a sampled H3 take as latents, beside its video. A later take that extends or bridges this one holds them as they are, with no decode and encode in between.",
            inputs=[
                io.Latent.Input("latent", tooltip="The sampled take, before it is decoded."),
                io.String.Input("file", tooltip="Path inside the output folder. Name it after the take's video."),
            ],
            outputs=[],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, latent, file) -> io.NodeOutput:
        import safetensors.torch

        base = os.path.realpath(folder_paths.get_output_directory())
        path = os.path.realpath(os.path.join(base, file))
        if not path.startswith(base + os.sep) or not path.endswith(".safetensors"):
            raise ValueError("%s is not a .safetensors path inside the output folder" % file)
        video = latent["samples"]
        if getattr(video, "is_nested", False):
            video = video.tensors[0]
        steps = min(HOLD_STEPS, video.shape[2])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        safetensors.torch.save_file({"head": video[:, :, :steps].detach().cpu().half().contiguous(),
                                     "tail": video[:, :, -steps:].detach().cpu().half().contiguous()}, path)
        return io.NodeOutput()


class H3Lora(io.ComfyNode):
    """A LoRA on an H3 model: renumbered for HyperFlow, added at run time on quantized bases."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3Lora",
            display_name="H3 LoRA",
            category="minimax/h3",
            description=(
                "Apply a LoRA trained on MiniMax H3. On int8 or fp8 weights and HyperFlow's factorized layers it is "
                "added at run time, so small updates are not rounded away or skipped. On a pruned base its "
                "time-conditioning part is added at run time too. HyperFlow H3 Flash prunes some of H3's blocks, so "
                "there the LoRA's later blocks move down to land on the layers they were trained for. Several H3 LoRA "
                "nodes in a row stack, each with its own optional sigma window."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("lora_name", options=folder_paths.get_filename_list("loras")),
                io.Float.Input("strength", default=1.0, min=-4.0, max=4.0, step=0.05),
                io.Combo.Input("mode", options=list(LORA_MODES), default="auto", optional=True,
                               tooltip="auto adds the LoRA at run time where that keeps it sharp. merge folds it into "
                                       "the weights: the least memory, but softer on int8 and fp8 weights."),
                io.Float.Input("start_sigma", default=1.0, min=0.0, max=1.0, step=0.005, optional=True,
                               tooltip="Highest noise level (sigma) where this LoRA runs. 1.0 runs from the first step; 0.68 "
                                       "skips early structure steps so composition stays locked and detail steps get the LoRA."),
                io.Float.Input("end_sigma", default=0.0, min=0.0, max=1.0, step=0.005, optional=True,
                               tooltip="Lowest noise level (sigma) where this LoRA runs. 0.0 runs to the final step."),
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, model, lora_name, strength, mode="auto", start_sigma=1.0, end_sigma=0.0) -> io.NodeOutput:
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        return io.NodeOutput(h3lora.apply(model, path, lora_name, strength, mode or "auto", start_sigma, end_sigma))


class H3SmallTextEncoder(io.ComfyNode):
    """A Qwen3-VL 4B or 8B in place of H3's 32B text encoder, through a ClipProj projection."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3SmallTextEncoder",
            display_name="H3 Small Text Encoder",
            category="minimax/h3",
            description=(
                "Load a Qwen3-VL 4B or 8B text encoder and project its output into what MiniMax H3 reads from its 32B "
                "encoder, with a ClipProj projection from models/clip_projections. It takes about a third of the memory. "
                "Connect it wherever the 32B's CLIP goes. It knows less than the 32B: named people and speech in "
                "languages other than English suffer most."
            ),
            inputs=[
                io.Combo.Input("clip_name", options=folder_paths.get_filename_list("text_encoders"),
                               tooltip="A Qwen3-VL 4B or 8B, any precision. A text-only Qwen3 loads but ignores the prompt."),
                io.Combo.Input("projection", options=clipproj.list_projections(),
                               tooltip="mmh3-4b-... for a 4B, mmh3-8b-... for an 8B."),
            ],
            outputs=[io.Clip.Output(display_name="clip")],
        )

    @classmethod
    def execute(cls, clip_name, projection) -> io.NodeOutput:
        return io.NodeOutput(clipproj.load(clip_name, projection))


class H3HyperFlowLoader(io.ComfyNode):
    """A HyperFlow H3 Flash checkpoint, read from its GGUF or its INT8 safetensors file."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3HyperFlowLoader",
            display_name="H3 HyperFlow Loader",
            category="minimax/h3",
            description=(
                "Load a HyperFlow H3 Flash checkpoint: MiniMax H3 distilled to 4, 6 or 8 steps, with some blocks "
                "pruned. It reads a GGUF file in any quantization, or an INT8 convrot safetensors file, which runs "
                "on ComfyUI's INT8 kernels. Sample it with H3 HyperFlow Sigmas, the euler sampler and no CFG."
            ),
            inputs=[
                io.Combo.Input("unet_name", options=hyperflow.files()),
                io.Combo.Input("weights_in", options=list(hyperflow.WEIGHTS_IN), default="auto", optional=True,
                               tooltip="For a GGUF file only. disk keeps the weights that do not fit on the GPU in the "
                                       "file and streams them, for machines with little RAM. ram copies them into RAM. "
                                       "auto picks disk with --fast-disk or a fast NVMe drive."),
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def validate_inputs(cls, unet_name) -> bool | str:
        # Checked here instead of against the list, so a workflow made on another machine still
        # runs: the file may sit in another folder, and Windows lists folders with a backslash.
        if not isinstance(unet_name, str):
            return True
        try:
            hyperflow.model_path(unet_name)
        except FileNotFoundError:
            base = unet_name.replace("\\", "/").rsplit("/", 1)[-1]
            return (f"{base} is not installed. Put the file in models/diffusion_models/hyperflow, "
                    "then refresh ComfyUI.")
        return True

    @classmethod
    def execute(cls, unet_name, weights_in="auto") -> io.NodeOutput:
        return io.NodeOutput(hyperflow.load(hyperflow.model_path(unet_name), weights_in or "auto"))


class H3HyperFlowSigmas(io.ComfyNode):
    """The step schedule a HyperFlow checkpoint samples."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3HyperFlowSigmas",
            display_name="H3 HyperFlow Sigmas",
            category="minimax/h3",
            description=(
                "The step schedule of a HyperFlow checkpoint: 4 steps for speed, 6 to balance speed and quality, "
                "8 for quality. Use it with SamplerCustomAdvanced, euler and no CFG."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Int.Input("steps", default=6, min=4, max=8, step=2, optional=True),
            ],
            outputs=[io.Sigmas.Output(display_name="sigmas")],
        )

    @classmethod
    def execute(cls, model, steps=6) -> io.NodeOutput:
        return io.NodeOutput(hyperflow.sigmas(model, 6 if steps is None else steps))


class H3RefineLatent(io.ComfyNode):
    """The latent for a refine pass: the upscaled picture to sample again, with the take's held parts and its sound kept."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3RefineLatent",
            display_name="H3 Refine Latent",
            category="minimax/h3",
            description=(
                "Prepare a finished H3 latent, upscaled to a larger size, for a few more steps at that size. "
                "Held frames come from the take at the larger size and stay as they are, and so does the sound."
            ),
            inputs=[
                io.Latent.Input("upscaled", tooltip="The sampled H3 latent with its picture upscaled, for example by MiniMax H3 Latent Upscale With Model."),
                io.Latent.Input("take", tooltip="The latent of the same H3 Take at the larger size, with its held frames and mask."),
            ],
            outputs=[io.Latent.Output(display_name="latent")],
        )

    @classmethod
    def execute(cls, upscaled, take) -> io.NodeOutput:
        import comfy.nested_tensor

        streams, held = upscaled["samples"], take["samples"]
        if not getattr(streams, "is_nested", False) or not getattr(held, "is_nested", False):
            raise ValueError("H3 Refine Latent needs H3 latents with picture and sound")
        video, audio = streams.tensors
        held_video = held.tensors[0]
        if video.shape != held_video.shape:
            raise ValueError("the upscaled picture is %s but the take is %s. Upscale to the take's size"
                             % (tuple(video.shape), tuple(held_video.shape)))
        mask = take.get("noise_mask")
        if mask is not None:
            video_mask = mask.tensors[0].to(device=video.device, dtype=video.dtype)
            # Where the take holds frames (mask 0), sample from its own frames, not from the upscaled ones.
            video = video * video_mask + held_video.to(video) * (1.0 - video_mask)
        else:
            video_mask = torch.ones_like(video[:, :1])
        out = dict(upscaled)
        out["samples"] = comfy.nested_tensor.NestedTensor((video, audio))
        out["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, torch.zeros_like(audio[:, :1])))
        return io.NodeOutput(out)


NODE_CLASS_MAPPINGS = {"H3Take": H3Take, "H3LoadClip": H3LoadClip, "H3SaveHold": H3SaveHold, "H3LoadStill": H3LoadStill, "H3LoadSound": H3LoadSound,
                       "H3Lora": H3Lora, "H3RefineLatent": H3RefineLatent, "H3HyperFlowLoader": H3HyperFlowLoader,
                       "H3HyperFlowSigmas": H3HyperFlowSigmas, "H3SmallTextEncoder": H3SmallTextEncoder}
NODE_DISPLAY_NAME_MAPPINGS = {"H3Take": "H3 Take", "H3LoadClip": "H3 Load Clip", "H3SaveHold": "H3 Save Hold", "H3LoadStill": "H3 Load Still",
                              "H3LoadSound": "H3 Load Sound", "H3Lora": "H3 LoRA", "H3RefineLatent": "H3 Refine Latent",
                              "H3HyperFlowLoader": "H3 HyperFlow Loader", "H3HyperFlowSigmas": "H3 HyperFlow Sigmas", "H3SmallTextEncoder": "H3 Small Text Encoder"}
# The editor's graphs name each node's twin, an id no other pack registers (see twins.py).
NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = with_twins(NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS)
