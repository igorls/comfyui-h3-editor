"""One H3 take. Calls ComfyUI's own MiniMax nodes. Does not import any other pack."""

from comfy_api.latest import io

from .latent_writer import carry_tail, encode_audio, encode_video, write_holds
from .policy import (
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
    frame_count,
    frame_to_step,
    hold_curve,
    latent_frames,
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
                audio_vae=None, start_image=None, end_image=None, mid_images=None, source_video=None, carry=None,
                region=None, source_audio=None, ref_images=None, ref_videos=None, ref_video_audios=None,
                ref_audios=None) -> io.NodeOutput:
        keep = resolve_keep(strength, keep)
        if carry is not None:
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
            window = audio_window(audio_hold, span, length, audio_seconds(source_audio))
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
            latent = cls.hold(latent, vae, audio_vae, source_video, source_audio, span, window, length,
                              width, height, video_feather, audio_feather, region, carry)

        card = "; ".join(decided["labels"])
        if decided["length_note"] == "short":
            card += "; length is below the trained 124-frame range"
        return io.NodeOutput(positive, latent, card)

    @staticmethod
    def hold(latent, vae, audio_vae, source_video, source_audio, span, window, length, width, height,
             video_feather, audio_feather, region, carry=None):
        total_steps = video_steps(length)
        video_latent = None
        video_first = 0
        video_curve = None
        audio_latent = None
        audio_first = 0
        audio_curve = None
        if span is not None:
            video_first = frame_to_step(span["insert"])
            steps = video_steps(span["hold"])
            if carry is not None:
                video_latent, audio_tail = carry_tail(carry, steps, audio_ticks(span["hold"]))
                if tuple(video_latent.shape[3:]) != (height // 16, width // 16):
                    raise ValueError("carry is a different size from this take")
            else:
                start = span["source_start"]
                frames = source_video[start:start + span["hold"]]
                video_latent = encode_video(vae, frames, width, height, steps)
            video_curve = hold_curve(total_steps, video_first, steps, base=span["keep"], feather=video_feather)
        if window is not None:
            audio_first = window["first"]
            if carry is not None:
                audio_latent = audio_tail
            else:
                audio_latent = encode_audio(audio_vae, source_audio, window["source_second"], window["ticks"])
            audio_curve = hold_curve(audio_ticks(length), audio_first, audio_latent.shape[-1], feather=audio_feather)
        return write_holds(latent, video_latent, video_first, audio_latent, audio_first,
                           video_curve, audio_curve, region)


NODE_CLASS_MAPPINGS = {"H3Take": H3Take}
NODE_DISPLAY_NAME_MAPPINGS = {"H3Take": "H3 Take"}
