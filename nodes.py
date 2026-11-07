"""One H3 take. Calls ComfyUI's own MiniMax nodes. Does not import any other pack."""

import os
import subprocess

import folder_paths
import numpy as np
import torch
from comfy_api.latest import InputImpl, io
from PIL import Image, ImageOps

from .latent_writer import carry_tail, encode_audio, encode_video, write_holds
from .policy import (
    AUDIO_ALIGNS,
    ANCHORS,
    AUDIO_HOLDS,
    CHECKPOINTS,
    FPS,
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
                ref_audios=None, audio_align="video") -> io.NodeOutput:
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
            second = (end_span, end_video, end_audio, end_window)
        elif carry is not None:
            if source_video is not None or video_mode != "off":
                raise ValueError("carry replaces the source video. Disconnect it and leave video_mode off.")
            if audio_hold == "full":
                raise ValueError("carry holds its own tail, so audio_hold must be off or span")
            frames = latent_frames(carry["samples"].unbind()[0].shape[2])
            span = video_span("extend_after", frames, length, hold_frames=hold_frames, align=align, keep=keep)
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
        if (audios or any(has_track) or (window is not None and carry is None)) and audio_vae is None:
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
            pieces = [(span, source_video, source_audio, window)]
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
        for span, frames, audio, window in pieces:
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
                    start = span["source_start"]
                    video_latent = encode_video(vae, frames[start:start + span["hold"]], width, height, steps)
                video_curve = merge_curves(
                    video_curve, hold_curve(total_steps, video_first, steps, base=span["keep"], feather=video_feather)
                )
            if window is not None:
                audio_first = window["first"]
                if carry is not None:
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
            ["ffmpeg", "-v", "error", "-i", output_file(file), "-vn", "-af", "atrim=start=%.6f:duration=%.6f" % (start, seconds),
             "-ac", "2", "-ar", str(rate), "-f", "f32le", "-"],
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
            outputs=[io.Image.Output(display_name="frames"), io.Audio.Output(display_name="audio")],
        )

    @classmethod
    def execute(cls, file, first, last) -> io.NodeOutput:
        video = InputImpl.VideoFromFile(output_file(file)).get_components()
        frames = video.images
        stop = last or frames.shape[0]
        audio = video.audio
        if audio is not None:
            rate = audio["sample_rate"]
            rate_first = int(first / float(FPS) * rate)
            rate_stop = int(stop / float(FPS) * rate)
            audio = {"waveform": audio["waveform"][..., rate_first:rate_stop], "sample_rate": rate}
        return io.NodeOutput(frames[first:stop], audio)


NODE_CLASS_MAPPINGS = {"H3Take": H3Take, "H3LoadClip": H3LoadClip, "H3LoadStill": H3LoadStill, "H3LoadSound": H3LoadSound}
NODE_DISPLAY_NAME_MAPPINGS = {"H3Take": "H3 Take", "H3LoadClip": "H3 Load Clip", "H3LoadStill": "H3 Load Still",
                              "H3LoadSound": "H3 Load Sound"}
