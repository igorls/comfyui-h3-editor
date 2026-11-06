"""One H3 take. Calls ComfyUI's own MiniMax nodes. Does not import any other pack."""

from policy import plan, frame_count, video_span

MAX_SOFT = 3


class H3Take:
    """Pin a start still, an end still, or both. Extra pictures stay soft references.

    A source video is a held span: freeze it, and generate frames before it,
    after it, or both. keep 0 freezes those frames. keep 1 regenerates them.
    A start or end pin is one still. Voice is timbre only. Sampling, the
    checkpoint, and the sigma shift stay on the graph so they remain visible.
    """

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "audio_vae": ("VAE",),
            "start_image": ("IMAGE",),
            "end_image": ("IMAGE",),
            "voice": ("AUDIO",),
            "source_video": ("IMAGE",),
            "source_audio": ("AUDIO",),
        }
        for i in range(1, MAX_SOFT + 1):
            optional["ref_image_%s" % i] = ("IMAGE",)
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 768, "min": 32, "max": 4096, "step": 32}),
                "height": ("INT", {"default": 1152, "min": 32, "max": 4096, "step": 32}),
                "length": ("INT", {"default": 124, "min": 5, "max": 362, "step": 17}),
                "anchor": (["none", "start", "end", "both"], {"default": "start"}),
                "video_mode": (["off", "v2v", "extend_after", "extend_before", "extend_both"], {"default": "off"}),
                "hold_frames": ("INT", {"default": 39, "min": 5, "max": 362, "step": 1}),
                "keep": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "align": (["av", "video"], {"default": "av"}),
                "ref_image_size": (["match", "max"], {"default": "match"}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("positive", "latent", "labels")
    FUNCTION = "take"
    CATEGORY = "minimax/h3"
    DESCRIPTION = (
        "One H3 take on ComfyUI's own nodes. Stills pin the start or the end. "
        "A source video is held, and new frames can be generated before it, after it, or both."
    )

    def take(
        self,
        clip,
        vae,
        prompt,
        width,
        height,
        length,
        anchor,
        video_mode,
        hold_frames,
        keep,
        align,
        ref_image_size,
        audio_vae=None,
        start_image=None,
        end_image=None,
        voice=None,
        source_video=None,
        source_audio=None,
        **refs,
    ):
        source_frames = frame_count(source_video, "source video") if source_video is not None else 0
        span = video_span(
            video_mode,
            source_frames,
            length,
            hold_frames=hold_frames,
            align=align,
            keep=keep,
        )
        if span is not None and source_audio is None and audio_vae is not None:
            pass
        soft = [refs.get("ref_image_%s" % i) for i in range(1, MAX_SOFT + 1)]
        decided = plan(
            anchor,
            start=start_image,
            end=end_image,
            soft=soft,
            audio=voice,
            length=length,
            video=span,
        )
        if voice is not None and audio_vae is None:
            raise ValueError("a voice reference needs the H3 audio VAE")
        if span is not None:
            raise ValueError(
                "video hold is planned (%s) but the latent mask writer is not in this node yet. "
                "It will not pin the clip with AddGuide." % decided["labels"][-1]
            )

        from comfy_extras.nodes_minimax_h3 import (
            MiniMaxH3AddGuide,
            MiniMaxH3ReferenceToVideo,
        )

        ref_images = {}
        for i, (_, image) in enumerate(decided["pictures"], 1):
            ref_images["ref_image_%s" % i] = image
        ref_audios = {"ref_audio_1": voice} if voice is not None else None

        out = MiniMaxH3ReferenceToVideo.execute(
            clip,
            prompt,
            width,
            height,
            length,
            ref_image_size=ref_image_size,
            vae=vae,
            audio_vae=audio_vae,
            ref_images=ref_images or None,
            ref_audios=ref_audios,
        )
        positive, latent = out[0], out[1]

        for role, frame_idx in decided["pins"]:
            image = start_image if role == "start" else end_image
            guided = MiniMaxH3AddGuide.execute(
                positive, latent, frame_idx, vae=vae, image=image
            )
            positive = guided[0]

        card = "; ".join(decided["labels"])
        if decided["length_note"] == "short":
            card += "; length is below the trained 124-frame range"
        return (positive, latent, card)


NODE_CLASS_MAPPINGS = {"H3Take": H3Take}
NODE_DISPLAY_NAME_MAPPINGS = {"H3Take": "H3 Take"}
