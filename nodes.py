"""One H3 take. Calls ComfyUI's own MiniMax nodes. Does not import any other pack."""

import json
import logging
import os
import subprocess
from itertools import islice

import folder_paths
import numpy as np
import torch
from comfy_api.latest import InputImpl, io, ui
from PIL import Image, ImageOps

from .latent_writer import carry_tail, encode_audio, encode_video, fit_frames, pooled_region_steps, write_holds
from . import h3lora
from .twins import with_twins
from . import hyperflow
from . import hybrid
from . import clipproj
from . import sprite
from . import media
from . import holds
from . import style_timeline
from . import vosr2
from . import motion_replace

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
                io.String.Input("style_markers", default="[]", multiline=True, optional=True,
                                tooltip='Experimental: [{"frame": 0, "style": "Watercolor"}, {"frame": 123, "style": "Clay animation"}]. Styles blend over video time. Keep the shot in prompt. Connect positive to H3 Style Guider. Each distinct style adds a model evaluation per sampling step.'),
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
                io.Latent.Input("source_hold", optional=True, tooltip="The held frames of source_video, and their sound, as the latents they were sampled in, from H3 Load Clip. Used in place of encoding them again, so a chain of extensions does not lose detail or brightness at each join."),
                io.Latent.Input("end_hold", optional=True, tooltip="The same for end_video."),
                io.Latent.Input("carry", optional=True, tooltip="A finished H3 latent. Its last hold_frames open this take, with no re-encode. Leave video_mode off."),
                io.Int.Input("hold_frames", default=39, min=5, max=3600, step=1),
                io.Int.Input("end_hold_frames", default=0, min=0, max=3600, step=1, optional=True, tooltip="With video_mode bridge: how many frames of end_video the take holds at its end, when that is fewer than hold_frames. 5 or 22 fit a take of 39 + 51k frames. 0 holds as many as hold_frames."),
                io.Combo.Input("strength", options=list(STRENGTHS) + ["custom"], default="hold", tooltip="How much the held video may change. light tints it, medium and heavy keep the shot but follow the prompt, regenerate keeps nothing."),
                io.Float.Input("keep", default=0.0, min=0.0, max=1.0, step=0.0001, tooltip="Raw mask value, read only when strength is custom. The useful range is about 0.95 to 0.996."),
                io.Combo.Input("align", options=["av", "video"], default="av", tooltip="av snaps holds to 39, 90, 141, so picture and sound share a boundary."),
                io.Mask.Input("region", optional=True, tooltip="White areas of the held video regenerate. One mask or one per frame."),
                io.Int.Input("video_feather", default=0, min=0, max=20, tooltip="Latent steps of soft edge where held video meets generated video."),
                io.Float.Input("context_noise", default=0.0, min=0.0, max=1.0, step=0.001, optional=True, tooltip="Lets the held frames far from the join be partly redrawn, rising from nothing at the join to this mask value at the far end of the hold, so a chain of extensions does not pile up small shifts of colour and texture. The frames at the join stay held. 0 holds every frame as it is."),
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
                ref_audios=None, audio_align="video", source_hold=None, end_hold=None, end_hold_frames=0,
                context_noise=0.0, style_markers="[]") -> io.NodeOutput:
        keep = resolve_keep(strength, keep)
        second = None
        if video_mode == "bridge":
            if source_video is None or end_video is None:
                raise ValueError("a bridge needs source_video and end_video")
            span, end_span = bridge_spans(
                frame_count(source_video, "source video"), frame_count(end_video, "end video"),
                length, hold_frames=hold_frames, align=align, keep=keep, end_hold_frames=end_hold_frames or None,
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

        markers = style_timeline.schedule(style_markers, length)
        styles = list(dict.fromkeys(row["style"] for row in markers))
        pictures = {"ref_image_%s" % i: image for i, (_, image) in enumerate(decided["pictures"], 1)}

        def encode(text):
            if checkpoint == "fl2va":
                return MiniMaxH3ImageToVideo.execute(clip, vae, text, width, height, length,
                                                    first_frame=start_image, last_frame=end_image)
            return MiniMaxH3ReferenceToVideo.execute(clip, text, width, height, length,
                                                    ref_image_size=ref_image_size, vae=vae, audio_vae=audio_vae,
                                                    ref_images=pictures or None, ref_videos=dict(videos) or None,
                                                    ref_video_audios=tracks or None, ref_audios=dict(audios) or None)

        def styled(text):
            if "__h3_style__" in prompt:
                return prompt.replace("__h3_style__", text)
            return (text + "\n" + prompt) if text else prompt

        out = encode(styled(styles[0]) if styles else prompt)
        pinned = [pin for pin in decided["pins"] if pin[0].startswith("mid")] if checkpoint == "fl2va" else decided["pins"]
        positive, latent = out[0], out[1]

        images = {"start": start_image, "end": end_image}
        for i, image in enumerate(mids, 1):
            images["mid %s" % i] = image
        for role, frame in pinned:
            positive = MiniMaxH3AddGuide.execute(positive, latent, frame, vae=vae, image=images[role])[0]

        if styles:
            variants = [positive]
            for style in styles[1:]:
                variant = encode(styled(style))[0]
                for role, frame in pinned:
                    variant = MiniMaxH3AddGuide.execute(variant, latent, frame, vae=vae, image=images[role])[0]
                variants.append(variant)
            timeline = {"markers": markers, "length": length, "conditioning": variants}
            positive = [[text, dict(meta, h3_style_timeline=timeline)] for text, meta in positive]

        if span is not None or window is not None:
            # A carried take brings its own sound for the held span. A track given with it replaces that.
            track = source_audio if carry is None or audio_hold == "full" else None
            pieces = [(span, source_video, track, window, source_hold)]
            if second is not None:
                pieces.append(second)
            latent = cls.hold(latent, vae, audio_vae, pieces, length, width, height,
                              video_feather, audio_feather, region, carry, context_noise or 0.0)

        card = "; ".join(decided["labels"])
        if styles:
            card += "; experimental style blend, %s encodings: use H3 Style Guider" % len(styles)
        if decided["length_note"] == "short":
            card += "; length is below the trained 124-frame range"
        return io.NodeOutput(positive, latent, card)

    @staticmethod
    def hold(latent, vae, audio_vae, pieces, length, width, height, video_feather, audio_feather, region, carry=None,
             context_noise=0.0):
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
                    video_latent = holds.held_picture(held, steps, height, width)
                    if video_latent is None:
                        held = None       # the sound follows the picture: both from the latents, or both from the file
                        start = span["source_start"]
                        video_latent = encode_video(vae, frames[start:start + span["hold"]], width, height, steps)
                video_curve = merge_curves(
                    video_curve, hold_curve(total_steps, video_first, steps, base=span["keep"], feather=video_feather,
                               context_noise=context_noise)
                )
            if window is not None:
                audio_first = window["first"]
                if carry is not None and audio is None:
                    audio_latent = audio_tail
                else:
                    audio_latent = holds.held_sound(held, window, latent["samples"].unbind()[1])
                    if audio_latent is None:
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


class H3StyleGuider(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="H3StyleGuider", display_name="H3 Style Guider (experimental)", category="minimax/h3",
                         description="Blend separately encoded styles across video time in one H3 latent. Use with H3 Take's style_markers. Audio follows the first style. Multiple styles cost multiple model evaluations per step.",
                         inputs=[io.Model.Input("model"), io.Conditioning.Input("conditioning")], outputs=[io.Guider.Output()])

    @classmethod
    def execute(cls, model, conditioning):
        return io.NodeOutput(style_timeline.make_guider(model, conditioning))


def output_file(file):
    base = os.path.realpath(folder_paths.get_output_directory())
    path = os.path.realpath(os.path.join(base, file))
    if not path.startswith(base + os.sep) or not os.path.exists(path):
        raise ValueError("no file at %s in the output folder" % file)
    return path


class H3VOSR2Video(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="H3VOSR2Video", display_name="H3 VOSR2 Video (experimental)", category="minimax/h3",
                         description="Upscale a rendered video 2× frame by frame with the optional ComfyUI-VOSR2 pack.",
                         inputs=[io.Custom("VOSR2_MODEL").Input("model"), io.String.Input("file"),
                                 io.String.Input("filename_prefix", default="video/h3_vosr2")],
                         is_output_node=True, outputs=[])

    @classmethod
    def execute(cls, model, file, filename_prefix):
        import comfy.model_management
        import nodes as comfy_nodes
        from server import PromptServer

        upscale = comfy_nodes.NODE_CLASS_MAPPINGS.get("VOSR2Upscale")
        if upscale is None:
            raise ValueError(vosr2.SETUP)
        source = output_file(file)
        folder, stem, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())
        name = "%s_%05d_.mp4" % (stem, counter)

        def transform(rgb):
            image = torch.from_numpy(rgb).unsqueeze(0).float().div_(255)
            enlarged = upscale.execute(model=model, image=image, **vosr2.SETTINGS)[0]
            return enlarged[0].detach().to("cpu").clamp(0, 1).mul_(255).round_().to(torch.uint8).contiguous().numpy()

        server = PromptServer.instance
        prompt_id = server.last_prompt_id

        def progress(frame, total):
            server.send_sync("h3take_upscale_frame", {"prompt_id": prompt_id, "value": frame, "max": total}, server.client_id)

        vosr2.stream_video(source, os.path.join(folder, name), transform, comfy.model_management.throw_exception_if_processing_interrupted, progress)
        return io.NodeOutput(ui=ui.PreviewVideo([ui.SavedResult(name, subfolder, io.FolderType.output)]))


class H3LoadStill(io.ComfyNode):
    """A picture from ComfyUI's output folder. The editor keeps references there, with the project."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3LoadStill",
            display_name="H3 Load Still",
            category="minimax/h3",
            description="Read a picture from the output folder. Used by the editor for references and first and last frames.",
            inputs=[
                io.String.Input("file", tooltip="Path inside the output folder."),
                io.Combo.Input("backdrop", options=["none", *sprite.BACKDROPS], default="none", optional=True,
                               tooltip="A colour lays a transparent picture on it, and moves a picture drawn on another flat colour "
                                       "onto it. none drops the transparency, which leaves a cutout on black."),
                io.Int.Input("width", default=0, min=0, max=8192, optional=True,
                             tooltip="With height, fit the whole picture inside this size and fill the rest. 0 leaves the picture as it is."),
                io.Int.Input("height", default=0, min=0, max=8192, optional=True),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    def execute(cls, file, backdrop="none", width=0, height=0) -> io.NodeOutput:
        with Image.open(output_file(file)) as image:
            image = sprite.still(ImageOps.exif_transpose(image), backdrop, width, height)
        pixels = np.array(image).astype(np.float32) / 255.0
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
                io.Boolean.Input("reference_only", default=False, optional=True,
                                 tooltip="Read just a short picture reference, without sound or held latents."),
                io.Boolean.Input("frames_only", default=False, optional=True,
                                 tooltip="Decode the original pixels without opening sound or saved latents."),
            ],
            outputs=[io.Image.Output(display_name="frames"), io.Audio.Output(display_name="audio"),
                     io.Latent.Output(display_name="hold", tooltip="These frames as the latents they were sampled in, when the clip saved them and the frames start on a multiple of 17. Their sound comes along when they also start on a multiple of 3 frames. Connect it to H3 Take's source_hold or end_hold.")],
        )

    @classmethod
    def execute(cls, file, first, last, reference_only=False, frames_only=False) -> io.NodeOutput:
        path = output_file(file)
        if reference_only or frames_only:
            import av

            if reference_only and not 2 * FPS <= last - first <= 15 * FPS:
                raise ValueError("a video reference needs 2 to 15 seconds")
            with av.open(path) as container:
                stream = container.streams.video[0]
                width, height = stream.width, stream.height
                if reference_only:
                    scale = min(1, 768 / max(width, height))
                    width, height = (max(2, round(side * scale / 2) * 2) for side in (width, height))
                images = [frame.reformat(width, height, format="rgb24").to_ndarray()
                          for frame in islice(container.decode(stream), first, last or None)]
            if not images or (last and len(images) != last - first):
                raise ValueError("the source video ends before the selected frame range")
            return io.NodeOutput(torch.from_numpy(np.stack(images)).float().div_(255), None, None)
        video = InputImpl.VideoFromFile(path).get_components()
        frames = video.images
        stop = last or frames.shape[0]
        audio = video.audio
        if audio is not None:
            rate = audio["sample_rate"]
            rate_first = int(first / float(FPS) * rate)
            rate_stop = int(stop / float(FPS) * rate)
            audio = {"waveform": audio["waveform"][..., rate_first:rate_stop], "sample_rate": rate}
        return io.NodeOutput(frames[first:stop], audio, holds.part(holds.hold_path(path), first, stop, frames.shape[0]))


class H3SpriteGuide(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="H3SpriteGuide", display_name="H3 Sprite Motion Guide", category="minimax/h3",
            inputs=[io.Combo.Input("direction", options=list(sprite.DIRECTIONS)),
                    io.Combo.Input("camera", options=list(sprite.CAMERAS)),
                    io.Int.Input("poses", default=7, min=4, max=12),
                    io.Int.Input("frames", default=sprite.FRAMES, min=22, max=141, step=17, optional=True),
                    io.Combo.Input("canvas", options=list(sprite.CANVASES), default="portrait", optional=True)],
            outputs=[io.Image.Output(display_name="motion guide")])

    @classmethod
    def execute(cls, direction, camera, poses, frames=sprite.FRAMES, canvas="portrait"):
        frames_arr = sprite.jog(direction, camera, poses, frames=int(frames), canvas=canvas or "portrait")
        return io.NodeOutput(torch.from_numpy(frames_arr).float().div_(255))


class H3DefringeMatte(io.ComfyNode):
    """Remove estimated backdrop colour from soft foreground matte edges to prevent white halos on dark terrain."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3DefringeMatte",
            display_name="H3 Defringe Matte",
            category="minimax/h3",
            description=(
                "Un-premultiply the estimated border backdrop colour from partially covered foreground pixels "
                "(BiRefNet or similar soft mattes) and tighten alpha coverage, eliminating halos on game sprites and props. "
                "On a flat backdrop of any colour it also keys out the backdrop a matte leaves inside or beside the subject."
            ),
            inputs=[
                io.Image.Input("images"),
                io.Mask.Input("mask"),
                io.Float.Input("alpha_low", default=0.12, min=0.0, max=0.5, step=0.01, optional=True),
                io.Float.Input("alpha_high", default=0.88, min=0.5, max=1.0, step=0.01, optional=True),
                io.Boolean.Input("invert_mask", default=False, optional=True,
                                 tooltip="Flip mask polarity when the upstream background remover outputs background=1."),
                io.Boolean.Input("key_backdrop", default=True, optional=True,
                                 tooltip="On a flat backdrop of any colour, also remove what the matte keeps by mistake: "
                                         "pockets of backdrop the subject encloses and patches beside it. Ignored on a scene."),
                io.Combo.Input("alpha", options=list(sprite.ALPHAS), default="soft", optional=True,
                               tooltip="soft keeps partly covered edge pixels. hard makes every pixel fully in or out."),
                io.Image.Input("depth", optional=True,
                               tooltip="A depth map for each image, from any depth node, near bright or near dark. On a flat "
                                       "backdrop it tells leftover backdrop from the subject's own surfaces of that colour."),
            ],
            outputs=[
                io.Image.Output(display_name="rgba"),
                io.Mask.Output(display_name="matte"),
            ],
        )

    @classmethod
    def execute(cls, images, mask, alpha_low=0.12, alpha_high=0.88, invert_mask=False, key_backdrop=True, alpha="soft",
                depth=None):
        masks = mask.detach().float().cpu().numpy()
        if masks.ndim == 2:
            masks = masks[None]
        if invert_mask:
            masks = 1.0 - masks
        rgba_batch = []
        matte_batch = []
        for idx, frame_t in enumerate(images):
            frame = Image.fromarray((frame_t.detach().cpu().clamp(0, 1).numpy() * 255).round().astype(np.uint8))
            m = masks[min(idx, len(masks) - 1)]
            d = None if depth is None else depth[min(idx, len(depth) - 1)][..., 0].detach().float().cpu().numpy()
            out_im = sprite.with_matte(frame, m, alpha_low=alpha_low, alpha_high=alpha_high, key=key_backdrop, depth=d)
            if alpha == "hard":
                out_im = sprite.harden(out_im)
            arr = np.asarray(out_im, dtype=np.float32) / 255.0
            rgba_batch.append(arr)
            matte_batch.append(arr[:, :, 3])
        return io.NodeOutput(
            torch.from_numpy(np.stack(rgba_batch)),
            torch.from_numpy(np.stack(matte_batch)),
        )


class H3SpriteSheet(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        optional = [io.BackgroundRemoval.Input("bg_model", optional=True)] if hasattr(io, "BackgroundRemoval") else []
        return io.Schema(node_id="H3SpriteSheet", display_name="H3 Sprite Sheet", category="minimax/h3",
            inputs=[io.Image.Input("images"), io.Int.Input("poses", default=7, min=4, max=12),
                    io.String.Input("filename_prefix"), io.String.Input("layout", default=""),
                    io.Boolean.Input("idle", default=False, optional=True),
                    io.Boolean.Input("full_action", default=False, optional=True,
                                     tooltip="Sample a prompted gesture from beginning to end."),
                    io.Combo.Input("cell_size", options=[str(n) for n in sprite.EXPORT_SIZES], optional=True),
                    io.Boolean.Input("export_24fps", default=False, optional=True),
                    io.Int.Input("tile_width", default=0, min=0, max=sprite.MAX_TILE, step=2, optional=True,
                                 tooltip="Custom even tile width (32..512). 0 uses cell_size."),
                    io.Int.Input("tile_height", default=0, min=0, max=sprite.MAX_TILE, step=2, optional=True,
                                 tooltip="Custom even tile height (32..512). 0 uses cell_size."),
                    io.Combo.Input("canvas", options=list(sprite.CANVASES), default="portrait", optional=True),
                    io.Float.Input("global_scale", default=0.0, min=0.0, max=16.0, step=0.01, optional=True,
                                   tooltip="Optional shared character scale in tile px per canvas px. 0 auto-scales."),
                    io.Boolean.Input("save_webp", default=False, optional=True),
                    io.Boolean.Input("save_cache", default=True, optional=True,
                                     tooltip="Cache lossless defringed RGBA frames beside the strip for GPU-free re-exports."),
                    io.Combo.Input("alpha", options=list(sprite.ALPHAS), default="soft", optional=True,
                                   tooltip="soft keeps partly covered edge pixels. hard makes every pixel fully in or out, for pixel art."),
                    io.Image.Input("depth", optional=True,
                                   tooltip="A depth map for each frame, from any depth node, near bright or near dark. On a flat "
                                           "backdrop it tells backdrop left inside or beside the subject from the subject's own "
                                           "surfaces of the backdrop's colour."),
                    *optional],
            outputs=[], is_output_node=True)

    @classmethod
    def execute(cls, images, poses, filename_prefix, layout="", idle=False, bg_model=None, cell_size="64", full_action=False,
                export_24fps=False, tile_width=0, tile_height=0, canvas="portrait", global_scale=0.0, save_webp=False,
                save_cache=True, alpha="soft", depth=None):
        base = os.path.realpath(folder_paths.get_output_directory())
        if depth is not None and len(depth) != len(images):
            raise ValueError("The depth maps must be one for each frame")
        dest = os.path.realpath(os.path.join(base, filename_prefix))
        if os.path.commonpath((base, dest)) != base or dest == base:
            raise ValueError("Sprite sheets must be saved inside the output folder")
        frames = [Image.fromarray((f.detach().cpu().clamp(0, 1).numpy() * 255).round().astype(np.uint8)) for f in images]
        def masks_for(indices):
            return bg_model.encode_image(images[indices]).detach().float().cpu().numpy()

        def depth_for(indices):
            return depth[indices][..., 0].detach().float().cpu().numpy()

        cell_size = int(cell_size)
        custom_tile = (int(tile_width), int(tile_height)) if (tile_width and tile_height) else None
        tile_w, tile_h = sprite.validate_tile_size(custom_tile) if custom_tile else (cell_size, cell_size)
        cache_path = (dest + ".frames") if save_cache else None
        scale_override = float(global_scale) if global_scale and float(global_scale) > 0 else None
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        full, indexed, transform = sprite.strips(
            frames, poses, json.loads(layout) if layout else None,
            masks_for if bg_model is not None else None, idle=idle, cell_size=cell_size,
            full_action=full_action, export_24fps=export_24fps,
            tile_size=custom_tile, canvas=canvas or "portrait", global_scale=scale_override,
            cache_dir=cache_path, alpha=alpha or "soft", depth_for=depth_for if depth is not None else None,
        )
        fps = sprite.playback_fps(transform, poses, idle, export_24fps)
        metadata = {"poses": full.width // tile_w, "source_poses": poses, "fps": fps,
                    "tile": tile_w, "tile_size": [tile_w, tile_h],
                    "pivot": [sprite.PIVOT[0] * tile_w // sprite.TILE, sprite.PIVOT[1] * tile_h // sprite.TILE],
                    "layout": transform, "canvas": canvas or "portrait", "frames": len(frames),
                    "cutout": "foreground model" if bg_model is not None else "background key"}
        if alpha == "hard":
            metadata["alpha"] = "hard"
        if depth is not None:
            metadata["depth"] = True
        if cache_path and os.path.isdir(cache_path):
            metadata["frame_cache"] = os.path.basename(cache_path)
        for label, sheet in (("illustrated", full), ("indexed", indexed)):
            filename = dest + "." + label + ".png"
            sheet.save(filename, format="PNG")
            metadata[label] = os.path.basename(filename)
        if save_webp:
            webp_name = dest + ".illustrated.webp"
            full.save(webp_name, format="WEBP", quality=84, alpha_quality=90, method=6)
            metadata["illustrated_webp"] = os.path.basename(webp_name)
        return io.NodeOutput(ui={"sprite": [metadata]})


class H3SaveHold(io.ComfyNode):
    """Keeps a sampled take's latents, picture and sound, for the clips that will extend or bridge it."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3SaveHold",
            display_name="H3 Save Hold",
            category="minimax/h3",
            description="Save a sampled H3 take's latents, picture and sound, beside its video. A later take that extends or bridges this one holds them as they are, with no decode and encode in between, from any cut on a multiple of 17 frames.",
            inputs=[
                io.Latent.Input("latent", tooltip="The sampled take, before it is decoded."),
                io.String.Input("file", tooltip="Path inside the output folder. Name it after the take's video."),
                io.Latent.Input("final", optional=True, tooltip="For a take that is enlarged and refined: the latent after the refine pass. A take that extends this one holds it in its own refine pass."),
                io.Boolean.Input("partial", default=False, optional=True, tooltip="Mark latents saved for an edited range so extend/bridge holds ignore them while draft finalization can still reuse them."),
            ],
            outputs=[],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, latent, file, final=None, partial=False) -> io.NodeOutput:
        base = os.path.realpath(folder_paths.get_output_directory())
        path = os.path.realpath(os.path.join(base, file))
        if not path.startswith(base + os.sep) or not path.endswith(".safetensors"):
            raise ValueError("%s is not a .safetensors path inside the output folder" % file)
        holds.save(path, latent["samples"], final["samples"] if final is not None else None, partial=bool(partial))
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


class H3HybridLoader(io.ComfyNode):
    """Load a base MiniMax H3 safetensors model and selectively overlay transformer block tensor families from another."""

    @classmethod
    def define_schema(cls):
        models = [name for name in folder_paths.get_filename_list("diffusion_models") if name.lower().endswith(".safetensors")]
        return io.Schema(
            node_id="H3HybridLoader",
            display_name="H3 Hybrid Block Loader",
            category="minimax/h3",
            description=(
                "Load a base MiniMax H3 safetensors checkpoint (typically fl2va) and selectively swap matching "
                "transformer block tensor families from an overlay checkpoint (typically ref2va) on load, "
                "preserving quantization siblings (.comfy_quant, weight_scale) without saving a merged file."
            ),
            inputs=[
                io.Combo.Input("base_unet", options=models,
                               tooltip="Primary diffusion model (usually fl2va) that provides the backbone."),
                io.Combo.Input("overlay_unet", options=models,
                               tooltip="Secondary diffusion model (usually ref2va) whose selected block tensors overlay the base."),
                io.String.Input("blocks", default=hybrid.DEFAULT_BLOCKS,
                                tooltip="Block indices or ranges to swap (0-49), e.g. '45-49', '25-49', '0-10, 45-49', or 'all'."),
                io.Combo.Input("scope", options=list(hybrid.SCOPES), default=hybrid.DEFAULT_SCOPE,
                               tooltip="Which tensor families inside each selected block to overlay: adaln, all, attn, mlp, or attn_mlp."),
                io.Combo.Input("weight_dtype", options=list(hybrid.WEIGHT_DTYPES), default="default", optional=True,
                               tooltip="Optional weight dtype override, matching ComfyUI's Load Diffusion Model."),
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def validate_inputs(cls, base_unet, overlay_unet, blocks, scope="adaln", weight_dtype="default") -> bool | str:
        if isinstance(blocks, str):
            try:
                hybrid.parse_blocks(blocks)
            except ValueError as error:
                return str(error)
        if isinstance(scope, str) and scope not in hybrid.SCOPES:
            return f"scope must be one of {', '.join(hybrid.SCOPES)}"
        if isinstance(weight_dtype, str) and weight_dtype not in hybrid.WEIGHT_DTYPES:
            return f"weight_dtype must be one of {', '.join(hybrid.WEIGHT_DTYPES)}"
        for label, name in (("base_unet", base_unet), ("overlay_unet", overlay_unet)):
            if not isinstance(name, str):
                continue
            try:
                hybrid.resolve_model_path(name)
            except FileNotFoundError:
                base = name.replace("\\", "/").rsplit("/", 1)[-1]
                return f"{label}: {base} is not installed in models/diffusion_models."
        return True

    @classmethod
    def execute(cls, base_unet, overlay_unet, blocks=hybrid.DEFAULT_BLOCKS, scope=hybrid.DEFAULT_SCOPE,
                weight_dtype="default") -> io.NodeOutput:
        return io.NodeOutput(hybrid.load(base_unet, overlay_unet, blocks, scope or hybrid.DEFAULT_SCOPE,
                                         weight_dtype or "default"))


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


class H3LoadTake(io.ComfyNode):
    """A rendered take's latents as sampled, picture and sound, to carry on from: finishing a draft at final size."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3LoadTake",
            display_name="H3 Load Take",
            category="minimax/h3",
            description=(
                "Read the latents an H3 take was sampled as, picture and sound, from the file saved beside its video. "
                "Upscale them and refine them at a larger size to finish a draft: the same shot, at final detail."
            ),
            inputs=[
                io.String.Input("file", tooltip="The take's video, as a path inside the output folder."),
                io.Int.Input("first", default=0, min=0, max=100000, optional=True,
                             tooltip="Optional start frame on the 17-frame grid when refining only a part of the draft."),
                io.Int.Input("last", default=0, min=0, max=100000, optional=True,
                             tooltip="Optional exclusive end frame (0 loads the whole take)."),
            ],
            outputs=[io.Latent.Output(display_name="latent")],
        )

    @classmethod
    def execute(cls, file, first=0, last=0) -> io.NodeOutput:
        import comfy.nested_tensor

        video, audio = holds.load_take(holds.hold_path(output_file(file)), first, last)
        return io.NodeOutput({"samples": comfy.nested_tensor.NestedTensor((video, audio))})


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


def _sam3_track_person(sam_model, sam_clip, images, subject="person", detect_threshold=0.35, track_threshold=0.5):
    """Detect and track the primary subject across an IMAGE batch [N, H, W, C] via native SAM3 nodes."""
    from comfy_extras.nodes_sam3 import SAM3_Detect, SAM3_TrackToMask, SAM3_VideoTrack

    tokens = sam_clip.tokenize(subject or "person")
    cond = sam_clip.encode_from_tokens_scheduled(tokens)

    init_mask = None
    first_masks = SAM3_Detect.execute(
        model=sam_model, image=images[:1], conditioning=cond,
        threshold=detect_threshold, refine_iterations=2, individual_masks=True,
    )[0]
    if first_masks is not None and first_masks.ndim == 3 and first_masks.shape[0] > 0:
        areas = (first_masks > 0.5).float().sum(dim=(1, 2))
        if float(areas.max().item()) > 0.0:
            best_idx = int(areas.argmax().item())
            init_mask = first_masks[best_idx:best_idx + 1]

    track = SAM3_VideoTrack.execute(
        images=images, model=sam_model, initial_mask=init_mask, conditioning=cond,
        detection_threshold=track_threshold, max_objects=1, detect_interval=1,
    )[0]
    return SAM3_TrackToMask.execute(track_data=track, object_indices="")[0]


def _slice_with_pad(tensor, start, length):
    total = tensor.shape[0]
    if total == 0:
        raise ValueError("cannot slice an empty batch")
    s = max(0, min(int(start), total - 1))
    e = s + max(1, int(length))
    piece = tensor[s:min(e, total)]
    if piece.shape[0] < int(length):
        idx = motion_replace.reflect_pad_indices(piece.shape[0], int(length))
        piece = piece[torch.tensor(idx, device=piece.device, dtype=torch.long)]
    return piece.contiguous()


class H3GatedSDPose(io.ComfyNode):
    """SDPose extractor that never falls back to full-image hallucination on absent-person frames,
    and reuses cached per-frame keypoints beside the source video when available."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3GatedSDPose",
            display_name="H3 Gated SDPose Extractor",
            category="MinimaxH3",
            description="Run SDPose only inside detected person boxes, reusing cached keypoints beside the source video when available.",
            inputs=[
                io.Image.Input("image"),
                io.Model.Input("model", optional=True),
                io.Vae.Input("vae", optional=True),
                io.BoundingBox.Input("bboxes", optional=True),
                io.Int.Input("batch_size", default=8, min=1, max=64, optional=True),
                io.Float.Input("min_area_frac", default=0.005, min=0.0, max=0.5, step=0.001, optional=True),
                io.Int.Input("min_valid_body_kps", default=2, min=0, max=18, optional=True),
                io.Float.Input("kp_score_thresh", default=0.3, min=0.0, max=1.0, step=0.05, optional=True),
                io.String.Input("source_file", default="", optional=True),
                io.Int.Input("first", default=0, min=0, max=100000, optional=True),
                io.Int.Input("total_frames", default=0, min=0, max=100000, optional=True),
            ],
            outputs=[io.Custom("POSE_KEYPOINT").Output("keypoints")],
        )

    @classmethod
    def execute(cls, image, model=None, vae=None, bboxes=None, batch_size=8, min_area_frac=0.005,
                min_valid_body_kps=2, kp_score_thresh=0.3, source_file="", first=0, total_frames=0) -> io.NodeOutput:
        import logging

        count, height, width, _ = image.shape
        src_path = output_file(source_file) if source_file else None
        kp_frames = None
        if src_path:
            kp_frames = motion_replace.load_pose_cache(src_path, first, first + count, min_area_frac=min_area_frac)
            if kp_frames is not None:
                logging.info("[H3GatedSDPose] Reused cached keypoints for %s frames %s..%s",
                             os.path.basename(src_path), first, first + count)
        if kp_frames is None:
            if model is None or vae is None or bboxes is None:
                raise ValueError("H3GatedSDPose cache miss: connect model, vae, and bboxes to extract keypoints")
            from comfy_extras.nodes_sdpose import SDPoseKeypointExtractor

            gated, _ = motion_replace.gate_bboxes(bboxes, count, height, width, min_area_frac=min_area_frac)
            kp_frames = SDPoseKeypointExtractor.execute(model, vae, image, batch_size=batch_size, bboxes=gated)[0]
            if src_path:
                motion_replace.save_pose_cache(
                    src_path, first, kp_frames, total_frames=total_frames or (first + count),
                    min_area_frac=min_area_frac,
                )
        cleaned = motion_replace.filter_pose_keypoints(
            kp_frames, min_valid_body_kps=min_valid_body_kps, score_thresh=kp_score_thresh
        )
        return io.NodeOutput(cleaned)


class H3IsolateCharacter(io.ComfyNode):
    """Segment the character reference still with SAM3 onto a clean neutral backdrop, caching the cutout beside it."""

    BACKDROPS = {"white": 1.0, "light_gray": 0.85, "mid_gray": 0.5}

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3IsolateCharacter",
            display_name="H3 Isolate Character",
            category="MinimaxH3",
            description="Strip distracting backgrounds and props from a character still using SAM3, reusing cached cutouts.",
            inputs=[
                io.Image.Input("image"),
                io.Model.Input("sam_model", optional=True),
                io.Clip.Input("sam_clip", optional=True),
                io.String.Input("subject", default="person", optional=True),
                io.Combo.Input("backdrop", options=list(cls.BACKDROPS), default="white", optional=True),
                io.Float.Input("threshold", default=0.35, min=0.05, max=0.95, step=0.05, optional=True),
                io.String.Input("character_file", default="", optional=True),
            ],
            outputs=[io.Image.Output(display_name="isolated")],
        )

    @classmethod
    def execute(cls, image, sam_model=None, sam_clip=None, subject="person", backdrop="white",
                threshold=0.35, character_file="") -> io.NodeOutput:
        import logging

        char_path = output_file(character_file) if character_file else None
        if char_path:
            cached = motion_replace.load_cutout_cache(char_path, subject=subject, backdrop=backdrop, threshold=threshold)
            if cached is not None:
                logging.info("[H3IsolateCharacter] Reused cached cutout for %s", os.path.basename(char_path))
                return io.NodeOutput(torch.from_numpy(cached).to(device=image.device, dtype=image.dtype))
        if sam_model is None or sam_clip is None:
            raise ValueError("H3IsolateCharacter cache miss: connect sam_model and sam_clip")
        from comfy_extras.nodes_sam3 import SAM3_Detect

        tokens = sam_clip.tokenize(subject or "person")
        cond = sam_clip.encode_from_tokens_scheduled(tokens)
        masks = SAM3_Detect.execute(
            model=sam_model, image=image[:1], conditioning=cond,
            threshold=threshold, refine_iterations=4, individual_masks=False,
        )[0]
        if masks is None or masks.numel() == 0 or float((masks > 0.5).float().sum().item()) <= 0.0:
            return io.NodeOutput(image)
        alpha = masks[:1].to(device=image.device, dtype=image.dtype)
        if alpha.shape[-2:] != image.shape[1:3]:
            alpha = torch.nn.functional.interpolate(
                alpha[:, None], size=tuple(image.shape[1:3]), mode="bilinear", align_corners=False
            )[:, 0]
        alpha = alpha.clamp(0.0, 1.0)[..., None]
        fill = float(cls.BACKDROPS.get(backdrop, 1.0))
        comp = (image[..., :3] * alpha + fill * (1.0 - alpha)).clamp(0.0, 1.0)
        if char_path:
            motion_replace.save_cutout_cache(
                char_path, comp[0].detach().cpu().float().numpy(),
                subject=subject, backdrop=backdrop, threshold=threshold,
            )
        return io.NodeOutput(comp)


class H3TrackSubjectMask(io.ComfyNode):
    """Track one performer with SAM3 and produce both a tight raw mask and an exit-safe expanded mask,
    caching raw per-frame masks beside the source video for instant reuse across re-renders and splits."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3TrackSubjectMask",
            display_name="H3 Track Subject Mask",
            category="MinimaxH3",
            description="Track the performer with SAM3 and dilate safely, reusing cached masks beside the source video when available.",
            inputs=[
                io.Image.Input("images"),
                io.Model.Input("sam_model", optional=True),
                io.Clip.Input("sam_clip", optional=True),
                io.String.Input("subject", default="person", optional=True),
                io.Int.Input("expansion_px", default=32, min=0, max=256, optional=True),
                io.Float.Input("detect_threshold", default=0.35, min=0.05, max=0.95, step=0.05, optional=True),
                io.Float.Input("track_threshold", default=0.5, min=0.05, max=0.95, step=0.05, optional=True),
                io.String.Input("source_file", default="", optional=True),
                io.Int.Input("first", default=0, min=0, max=100000, optional=True),
                io.Int.Input("total_frames", default=0, min=0, max=100000, optional=True),
            ],
            outputs=[
                io.Mask.Output(display_name="raw_mask"),
                io.Mask.Output(display_name="expanded_mask"),
            ],
        )

    @classmethod
    def execute(cls, images, sam_model=None, sam_clip=None, subject="person", expansion_px=32,
                detect_threshold=0.35, track_threshold=0.5, source_file="", first=0, total_frames=0) -> io.NodeOutput:
        import logging

        count, height, width = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
        src_path = output_file(source_file) if source_file else None
        raw_np = None
        if src_path:
            cached = motion_replace.load_mask_cache(
                src_path, first, first + count, subject=subject,
                detect_threshold=detect_threshold, track_threshold=track_threshold,
            )
            if cached is not None:
                logging.info("[H3TrackSubjectMask] Reused cached SAM3 masks for %s frames %s..%s",
                             os.path.basename(src_path), first, first + count)
                if cached.shape[1] != height or cached.shape[2] != width:
                    cached = motion_replace._resize_mask_nearest_or_bilinear(cached, height, width).astype(np.float32)
                raw_np = cached
        if raw_np is None:
            if sam_model is None or sam_clip is None:
                raise ValueError("H3TrackSubjectMask cache miss: connect sam_model and sam_clip")
            raw = _sam3_track_person(
                sam_model, sam_clip, images, subject=subject,
                detect_threshold=detect_threshold, track_threshold=track_threshold,
            )
            raw_np = raw.detach().cpu().numpy().astype(np.float32)
            if src_path:
                motion_replace.save_mask_cache(
                    src_path, first, raw_np, total_frames=total_frames or (first + count),
                    subject=subject, detect_threshold=detect_threshold, track_threshold=track_threshold,
                )
        expanded_np = motion_replace.expand_mask_safe(raw_np, expansion_px=expansion_px)
        return io.NodeOutput(torch.from_numpy(raw_np).float(), torch.from_numpy(expanded_np))


class H3SliceBatch(io.ComfyNode):
    """Slice an IMAGE and/or MASK batch for one relay window, cloning the last frame if snapped past the end."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3SliceBatch",
            display_name="H3 Slice Batch",
            category="MinimaxH3",
            description="Slice a frame window from an image batch and/or mask batch with safe tail padding.",
            inputs=[
                io.Int.Input("start", default=0, min=0, max=100000),
                io.Int.Input("length", default=124, min=1, max=3600),
                io.Image.Input("images", optional=True),
                io.Mask.Input("mask", optional=True),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                io.Mask.Output(display_name="mask"),
            ],
        )

    @classmethod
    def execute(cls, start=0, length=124, images=None, mask=None) -> io.NodeOutput:
        out_img = _slice_with_pad(images, start, length) if images is not None else torch.zeros((1, 1, 1, 3), dtype=torch.float32)
        if mask is not None:
            m = mask.unsqueeze(0) if mask.ndim == 2 else mask
            out_mask = _slice_with_pad(m, start, length)
        else:
            out_mask = torch.zeros((1, 1, 1), dtype=torch.float32)
        return io.NodeOutput(out_img, out_mask)


class H3ApplyEditMask(io.ComfyNode):
    """Lock the source video latent outside a binary 1-4-4-4-4 / 32px pooled subject mask,
    with optional pure-latent carry from a previous relay window."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3ApplyEditMask",
            display_name="H3 Apply Edit Mask",
            category="MinimaxH3",
            description="Preserve source background latents on Standard H3 and HyperFlow Flash with strict 32px cell pooling and optional windowed relay carry.",
            inputs=[
                io.Latent.Input("latent"),
                io.Vae.Input("vae"),
                io.Image.Input("source_video"),
                io.Mask.Input("mask"),
                io.Latent.Input("prev_latent", optional=True),
                io.Int.Input("overlap_frames", default=22, min=0, max=124, optional=True),
            ],
            outputs=[io.Latent.Output(display_name="latent")],
        )

    @classmethod
    def execute(cls, latent, vae, source_video, mask, prev_latent=None, overlap_frames=22) -> io.NodeOutput:
        import comfy.nested_tensor

        video, audio = latent["samples"].unbind()
        _, _, steps, lat_h, lat_w = video.shape
        target_h, target_w = lat_h * 16, lat_w * 16
        total_frames = latent_frames(steps)

        src = source_video[..., :3]
        if src.shape[0] < total_frames:
            src = torch.cat([src, src[-1:].expand(total_frames - src.shape[0], -1, -1, -1)], dim=0)
        else:
            src = src[:total_frames]
        source_lat = vae.encode(fit_frames(src, target_w, target_h)).to(video).clone()

        video_mask = pooled_region_steps(mask, steps, lat_h, lat_w).to(device=video.device, dtype=torch.float32)
        if prev_latent is not None and int(overlap_frames) > 0:
            ov_steps = video_steps(int(overlap_frames))
            prev_video, _ = prev_latent["samples"].unbind()
            if prev_video.shape[2] < ov_steps or tuple(prev_video.shape[3:]) != (lat_h, lat_w):
                raise ValueError("previous relay window latent does not match overlap/canvas (%s vs %s)"
                                 % (tuple(prev_video.shape), (ov_steps, lat_h, lat_w)))
            # Pure latent-space carry across the 22-frame overlap (zero VAE re-encode drift),
            # softening the final overlap step inside the subject mask to avoid a hard cut cliff.
            source_lat[:, :, :ov_steps] = prev_video[:, :, -ov_steps:].to(source_lat)
            video_mask = motion_replace.apply_overlap_mask(video_mask, ov_steps, tail_ramp=0.45)

        audio_mask = torch.ones_like(audio, dtype=torch.float32)

        out = dict(latent)
        out["samples"] = comfy.nested_tensor.NestedTensor((source_lat, audio))
        out["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))
        return io.NodeOutput(out)


class H3JoinWindow(io.ComfyNode):
    """Join two decoded relay windows cleanly across their latent-locked overlap."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3JoinWindow",
            display_name="H3 Join Window",
            category="MinimaxH3",
            description="Stitch sequential windowed relay outputs across their locked overlap.",
            inputs=[
                io.Image.Input("prev_images"),
                io.Image.Input("next_images"),
                io.Int.Input("overlap_frames", default=22, min=0, max=124, optional=True),
                io.Int.Input("blend_frames", default=4, min=0, max=22, optional=True),
                io.Int.Input("trim_to", default=0, min=0, max=100000, optional=True),
            ],
            outputs=[io.Image.Output(display_name="images")],
        )

    @classmethod
    def execute(cls, prev_images, next_images, overlap_frames=22, blend_frames=4, trim_to=0) -> io.NodeOutput:
        joined = motion_replace.join_window_frames(
            prev_images.detach().cpu().numpy(),
            next_images.detach().cpu().numpy(),
            overlap_frames=overlap_frames,
            blend_frames=blend_frames,
        )
        if int(trim_to) > 0:
            joined = joined[:int(trim_to)]
        return io.NodeOutput(torch.from_numpy(joined).to(device=prev_images.device, dtype=prev_images.dtype))


class H3CompositeSourcePlate(io.ComfyNode):
    """Zero-halo, zero-flyaway source plate compositor using asymmetric M_src|M_gen union and local illumination matching."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3CompositeSourcePlate",
            display_name="H3 Composite Source Plate",
            category="MinimaxH3",
            description="Carry the exact source plate outside the asymmetric source+generated union with local illumination harmonization.",
            inputs=[
                io.Image.Input("generated"),
                io.Image.Input("source"),
                io.Mask.Input("source_mask"),
                io.Model.Input("sam_model", optional=True),
                io.Clip.Input("sam_clip", optional=True),
                io.String.Input("subject", default="person", optional=True),
                io.Int.Input("src_expand_px", default=18, min=0, max=96, optional=True),
                io.Int.Input("gen_expand_px", default=4, min=0, max=64, optional=True),
                io.Float.Input("feather_sigma", default=4.0, min=0.0, max=32.0, step=0.5, optional=True),
            ],
            outputs=[io.Image.Output(display_name="images")],
        )

    @classmethod
    def execute(cls, generated, source, source_mask, sam_model=None, sam_clip=None,
                subject="person", src_expand_px=18, gen_expand_px=4, feather_sigma=4.0) -> io.NodeOutput:
        gen_mask_np = None
        if sam_model is not None and sam_clip is not None:
            gen_mask = _sam3_track_person(sam_model, sam_clip, generated, subject=subject)
            gen_mask_np = gen_mask.detach().cpu().numpy()

        comp = motion_replace.composite_plate_harmonized(
            generated.detach().cpu().numpy(),
            source.detach().cpu().numpy(),
            source_mask.detach().cpu().numpy(),
            gen_mask_np=gen_mask_np,
            src_expand_px=src_expand_px,
            gen_expand_px=gen_expand_px,
            feather_sigma=feather_sigma,
        )
        return io.NodeOutput(torch.from_numpy(comp).to(device=generated.device, dtype=generated.dtype))


def _find_adaln_curve():
    """Locate the standard 8-channel adaln_t_table required to drive H3 Fun ControlNet on HyperFlow Flash."""
    from safetensors import safe_open

    preferred = [
        "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    ]
    candidates = folder_paths.get_filename_list("diffusion_models")
    ordered_files = [f for name in preferred for f in candidates if f.replace("\\", "/").endswith(name)]
    ordered_files.extend(
        f for f in candidates
        if f not in ordered_files and "minimax_h3" in f.lower() and "hyperflow" not in f.lower() and f.endswith(".safetensors")
    )
    for rel in ordered_files:
        path = folder_paths.get_full_path("diffusion_models", rel)
        if not path:
            continue
        try:
            with safe_open(path, framework="pt", device="cpu") as ckpt:
                if "adaln_t_table" in ckpt.keys():
                    return ckpt.get_tensor("adaln_t_table")
        except Exception:
            continue
    raise FileNotFoundError(
        "HyperFlow Flash pose control requires adaln_t_table from a standard MiniMax H3 checkpoint in models/diffusion_models"
    )


class H3FlashControlNetApply(io.ComfyNode):
    """Apply MiniMax H3 Fun ControlNet Union (v1 5-block or v2.0 10-block) to 49-block HyperFlow Flash Ref2VA."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3FlashControlNetApply",
            display_name="H3 Flash ControlNet Apply",
            category="MinimaxH3",
            description="Adapt H3 Fun ControlNet Union v1/v2.0 to 49-block HyperFlow Flash Ref2VA.",
            inputs=[
                io.Model.Input("model"),
                io.Vae.Input("vae"),
                io.ModelPatch.Input("model_patch"),
                io.Float.Input("strength", default=0.75, min=0.0, max=4.0, step=0.01, optional=True),
                io.Float.Input("start_percent", default=0.0, min=0.0, max=1.0, step=0.01, optional=True),
                io.Float.Input("end_percent", default=1.0, min=0.0, max=1.0, step=0.01, optional=True),
                io.Image.Input("control_video", optional=True),
                io.Mask.Input("mask", optional=True),
                io.Image.Input("source_video", optional=True),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, vae, model_patch, strength=0.75, start_percent=0.0, end_percent=1.0,
                control_video=None, mask=None, source_video=None) -> io.NodeOutput:
        from comfy.patcher_extension import WrappersMP
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3FunControlBlockPatch, MiniMaxH3FunControlPatch

        if strength == 0 or (control_video is None and mask is None):
            return io.NodeOutput(model)

        backbone = model.get_model_object("diffusion_model")
        if len(backbone.blocks) != 49 or not hasattr(backbone, "endpoint_time_embedder"):
            raise ValueError("H3FlashControlNetApply expects 49-block HyperFlow Flash Ref2VA")

        orig_layers = tuple(model_patch.model.injection_layers)
        flash_layers = motion_replace.flash_controlnet_layer_map(orig_layers)
        curve = _find_adaln_curve()

        class FlashFunPatch(MiniMaxH3FunControlPatch):
            def diffusion_model_wrapper(self, executor, x, timestep, context, transformer_options=None, **kwargs):
                original = backbone._time_embedding
                previous_override = vars(backbone).get("_time_embedding")
                self.control_time = None

                def embed(unique_t, device, dtype):
                    if self.active:
                        table = curve.to(device=device)
                        times = torch.tensor(unique_t, device=device, dtype=torch.float32)
                        position = times.clamp(0, 1) * (len(table) - 1)
                        lower = position.floor().long().clamp(max=len(table) - 2)
                        self.control_time = torch.lerp(
                            table[lower], table[lower + 1], (position - lower).unsqueeze(1)
                        )
                    return original(unique_t, device, dtype)

                backbone._time_embedding = embed
                try:
                    return super().diffusion_model_wrapper(
                        executor, x, timestep, context, transformer_options or {}, **kwargs
                    )
                finally:
                    if previous_override is None:
                        del backbone._time_embedding
                    else:
                        backbone._time_embedding = previous_override
                    self.control_time = None

            def after_block(self, block_index, args, out):
                if self.active:
                    if self.control_time is None or self.control_time.shape != (args["t_emb"].shape[0], 8):
                        raise RuntimeError("Flash and ControlNet timestep rows did not align")
                    args = dict(args, t_emb=self.control_time)
                return super().after_block(block_index, args, out)

        result = model.clone()
        sampling = result.get_model_object("model_sampling")
        patch = FlashFunPatch(
            model_patch,
            vae,
            control_video[..., :3].movedim(-1, 1) if control_video is not None else None,
            mask,
            source_video[..., :3].movedim(-1, 1) if mask is not None and source_video is not None else None,
            strength,
            float(sampling.percent_to_sigma(start_percent)),
            float(sampling.percent_to_sigma(end_percent)),
        )
        result.add_wrapper(WrappersMP.DIFFUSION_MODEL, patch.diffusion_model_wrapper)
        blocks = result.model_options.get("transformer_options", {}).get("patches_replace", {}).get("dit", {})
        for original_index, flash_index in zip(orig_layers, flash_layers, strict=True):
            replaced = MiniMaxH3FunControlBlockPatch(patch, original_index, blocks.get(("double_block", flash_index)))
            result.set_model_patch_replace(replaced, "dit", "double_block", flash_index)
        return io.NodeOutput(result)


class H3SwapGuard(io.ComfyNode):
    """Detect when Window 0 of a masked character replacement stayed stuck on the source person and auto-retry once."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3SwapGuard",
            display_name="H3 Swap Guard",
            category="MinimaxH3",
            description="Check masked latent change after Window 0 and retry once with boosted character-swap LoRA and a fresh seed if the swap did not take.",
            inputs=[
                io.Latent.Input("sampled"),
                io.Latent.Input("masked_source"),
                io.Model.Input("model"),
                io.Conditioning.Input("conditioning"),
                io.Sampler.Input("sampler"),
                io.Sigmas.Input("sigmas"),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                io.String.Input("swap_lora", default="", optional=True),
                io.Float.Input("boost_strength", default=0.35, min=0.0, max=1.5, step=0.05, optional=True),
                io.Float.Input("min_change", default=0.42, min=0.0, max=2.0, step=0.02, optional=True),
            ],
            outputs=[io.Latent.Output(display_name="latent")],
        )

    @classmethod
    def execute(cls, sampled, masked_source, model, conditioning, sampler, sigmas, seed=0,
                swap_lora="", boost_strength=0.35, min_change=0.42) -> io.NodeOutput:
        import logging
        from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced

        mask_nested = masked_source.get("noise_mask")
        if mask_nested is None:
            return io.NodeOutput(sampled)
        samp_video, _ = sampled["samples"].unbind()
        src_video, _ = masked_source["samples"].unbind()
        video_mask, _ = mask_nested.unbind()
        score = motion_replace.latent_swap_score(
            samp_video.detach().cpu().float().numpy(),
            src_video.detach().cpu().float().numpy(),
            video_mask.detach().cpu().float().numpy(),
        )
        if score is None or score >= float(min_change):
            return io.NodeOutput(sampled)

        logging.info("[H3SwapGuard] Window 0 masked change %.3f < %.3f; retrying once with boosted swap LoRA and fresh seed",
                     score, float(min_change))
        retry_model = model
        if swap_lora and float(boost_strength) > 0.0:
            lora_path = folder_paths.get_full_path("loras", swap_lora)
            if lora_path:
                retry_model = h3lora.apply(retry_model, lora_path, swap_lora, float(boost_strength), "auto", 1.0, 0.0)
        retry_seed = (int(seed) + 7919) & 0x7fffffff
        noise = RandomNoise.execute(retry_seed)[0]
        guider = BasicGuider.execute(retry_model, conditioning)[0]
        retried = SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, masked_source)[0]
        retry_video, _ = retried["samples"].unbind()
        retry_score = motion_replace.latent_swap_score(
            retry_video.detach().cpu().float().numpy(),
            src_video.detach().cpu().float().numpy(),
            video_mask.detach().cpu().float().numpy(),
        )
        if retry_score is not None and retry_score > score:
            logging.info("[H3SwapGuard] Retry improved masked change %.3f -> %.3f; keeping retried Window 0",
                         score, retry_score)
            return io.NodeOutput(retried)
        return io.NodeOutput(sampled)


NODE_CLASS_MAPPINGS = {"H3SpriteGuide": H3SpriteGuide, "H3DefringeMatte": H3DefringeMatte, "H3SpriteSheet": H3SpriteSheet, "H3Take": H3Take, "H3LoadClip": H3LoadClip, "H3SaveHold": H3SaveHold, "H3LoadStill": H3LoadStill, "H3LoadSound": H3LoadSound,
                       "H3VOSR2Video": H3VOSR2Video,
                       "H3StyleGuider": H3StyleGuider,
                       "H3Lora": H3Lora, "H3LoadTake": H3LoadTake, "H3RefineLatent": H3RefineLatent, "H3HyperFlowLoader": H3HyperFlowLoader,
                       "H3HybridLoader": H3HybridLoader,
                       "H3HyperFlowSigmas": H3HyperFlowSigmas, "H3SmallTextEncoder": H3SmallTextEncoder,
                       "H3GatedSDPose": H3GatedSDPose, "H3IsolateCharacter": H3IsolateCharacter,
                       "H3TrackSubjectMask": H3TrackSubjectMask, "H3SliceBatch": H3SliceBatch,
                       "H3ApplyEditMask": H3ApplyEditMask, "H3JoinWindow": H3JoinWindow,
                       "H3SwapGuard": H3SwapGuard,
                       "H3CompositeSourcePlate": H3CompositeSourcePlate, "H3FlashControlNetApply": H3FlashControlNetApply}
NODE_DISPLAY_NAME_MAPPINGS = {"H3SpriteGuide": "H3 Sprite Motion Guide", "H3DefringeMatte": "H3 Defringe Matte", "H3SpriteSheet": "H3 Sprite Sheet", "H3Take": "H3 Take", "H3LoadClip": "H3 Load Clip", "H3SaveHold": "H3 Save Hold", "H3LoadStill": "H3 Load Still",
                              "H3VOSR2Video": "H3 VOSR2 Video (experimental)",
                              "H3StyleGuider": "H3 Style Guider (experimental)",
                              "H3LoadSound": "H3 Load Sound", "H3Lora": "H3 LoRA", "H3LoadTake": "H3 Load Take", "H3RefineLatent": "H3 Refine Latent",
                              "H3HyperFlowLoader": "H3 HyperFlow Loader", "H3HybridLoader": "H3 Hybrid Block Loader", "H3HyperFlowSigmas": "H3 HyperFlow Sigmas",
                              "H3SmallTextEncoder": "H3 Small Text Encoder",
                              "H3GatedSDPose": "H3 Gated SDPose Extractor", "H3IsolateCharacter": "H3 Isolate Character",
                              "H3TrackSubjectMask": "H3 Track Subject Mask", "H3SliceBatch": "H3 Slice Batch",
                              "H3ApplyEditMask": "H3 Apply Edit Mask", "H3JoinWindow": "H3 Join Window",
                              "H3SwapGuard": "H3 Swap Guard",
                              "H3CompositeSourcePlate": "H3 Composite Source Plate", "H3FlashControlNetApply": "H3 Flash ControlNet Apply"}
# The editor's graphs name each node's twin, an id no other pack registers (see twins.py).
NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = with_twins(NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS)
