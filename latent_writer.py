"""Writes held video and audio into an H3 AV latent and builds its nested noise mask.

The mask is 1 where H3 generates and lower where it holds. ComfyUI's sampler
reads latent["noise_mask"] and H3 runs each masked row at its own strength.
Sampling stays on the graph. This module only prepares the latent.
"""

import torch
import torch.nn.functional as F


def fit_frames(frames, width, height):
    """Cover-crop a frame batch onto the take's canvas."""
    frames = frames[..., :3]
    if frames.shape[1] == height and frames.shape[2] == width:
        return frames
    import comfy.utils

    resized = comfy.utils.common_upscale(frames.movedim(-1, 1), width, height, "bicubic", "center")
    return resized.movedim(1, -1).clamp(0.0, 1.0)


def encode_video(vae, frames, width, height, steps):
    latent = vae.encode(fit_frames(frames, width, height))
    if latent.shape[2] != steps:
        raise ValueError("held video encoded to %s latent steps, expected %s" % (latent.shape[2], steps))
    return latent


def audio_pcm(audio, rate, source_second, ticks, hop):
    """Cut exactly ticks * hop samples, so the VAE's generic input crop never moves the start."""
    wave = audio["waveform"][:1]
    if audio["sample_rate"] != rate:
        import comfy.audio

        wave = comfy.audio.resample(wave, audio["sample_rate"], rate)
    start = int(round(source_second * rate))
    need = ticks * hop
    piece = wave[..., start:start + need]
    if piece.shape[-1] < need:
        piece = F.pad(piece, (0, need - piece.shape[-1]))
    return piece


def encode_audio(audio_vae, audio, source_second, ticks):
    rate = getattr(audio_vae, "audio_sample_rate", 32000)
    hop = rate // 40
    piece = audio_pcm(audio, rate, source_second, ticks, hop)
    latent = audio_vae.encode(piece.movedim(1, -1))
    if latent.shape[-1] != ticks:
        raise ValueError("held audio encoded to %s ticks, expected %s" % (latent.shape[-1], ticks))
    return latent


def region_steps(mask, steps, height, width):
    """Resample a MASK batch to one [1, 1, steps, h, w] volume. 1 regenerates that region."""
    mask = mask.reshape((-1, 1) + tuple(mask.shape[-2:])).float()
    mask = F.interpolate(mask, size=(height, width), mode="bilinear", align_corners=False)
    if mask.shape[0] == 1:
        mask = mask.expand(steps, -1, -1, -1)
    else:
        index = torch.linspace(0, mask.shape[0] - 1, steps).round().long()
        mask = mask[index]
    return mask.movedim(0, 1).unsqueeze(0).clamp(0.0, 1.0)


def pooled_region_steps(mask, steps, height, width):
    """Union per-frame pixel masks across H3's 1-4-4-4-4 temporal spans and 32px DiT cells.

    Returns a strictly binary {0.0, 1.0} [1, 1, steps, height, width] latent mask aligned to
    2x2 VAE patch cells (32x32 pixels). Strict binary cell alignment prevents boundary bleed on
    Standard H3 and guarantees every video token hits an exact entry in HyperFlow Flash's
    TwoTimeEmbedder EndpointState lookup (1 -> t_video->r_video, 0 -> stationary t_pin_v).
    All-zero exit frames stay cleanly 0.0.
    """
    spans = [(1, 4, 4, 4, 4)[i % 5] for i in range(steps)]
    total_frames = sum(spans)
    mask = mask.float()
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    elif mask.ndim > 3:
        mask = mask.reshape((-1,) + tuple(mask.shape[-2:]))
    if mask.shape[0] == 1:
        mask = mask.expand(total_frames, -1, -1)
    elif mask.shape[0] < total_frames:
        pad = mask[-1:].expand(total_frames - mask.shape[0], -1, -1)
        mask = torch.cat([mask, pad], dim=0)
    else:
        mask = mask[:total_frames]
    pixel_h, pixel_w = height * 16, width * 16
    if tuple(mask.shape[1:]) != (pixel_h, pixel_w):
        mask = F.interpolate(mask[:, None], size=(pixel_h, pixel_w), mode="bilinear", align_corners=False)[:, 0]
    mask = (mask > 0.5).float()
    rows, offset = [], 0
    for span in spans:
        rows.append(mask[offset:offset + span].amax(dim=0))
        offset += span
    stacked = torch.stack(rows)[:, None]  # [steps, 1, pixel_h, pixel_w]
    pooled = F.max_pool2d(stacked, 32, 32)[:, 0]  # [steps, height//2, width//2]
    lat_mask = pooled.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
    if lat_mask.shape[-2] < height or lat_mask.shape[-1] < width:
        lat_mask = F.pad(lat_mask, (0, width - lat_mask.shape[-1], 0, height - lat_mask.shape[-2]), mode="replicate")
    return lat_mask[:, :height, :width][None, None].contiguous()


def carry_tail(latent, steps, ticks):
    """The last frames of a finished take, still in latent form, so a later pass never re-encodes them."""
    video, audio = latent["samples"].unbind()
    return video[:, :, -steps:], audio[..., -ticks:]


def write_holds(latent, video_latent=None, video_first=0, audio_latent=None, audio_first=0,
                video_curve=None, audio_curve=None, region=None, pooled_region=False):
    """Return a new latent dict with the held streams written and a nested noise mask set."""
    import comfy.nested_tensor

    video, audio = latent["samples"].unbind()
    video = video.clone()
    audio = audio.clone()
    if video_latent is not None:
        video[:, :, video_first:video_first + video_latent.shape[2]] = video_latent.to(video)
    if audio_latent is not None:
        audio[..., audio_first:audio_first + audio_latent.shape[-1]] = audio_latent.to(audio)

    steps, height, width = video.shape[2:]
    ticks = audio.shape[-1]
    video_curve = video_curve if video_curve is not None else [1.0] * steps
    audio_curve = audio_curve if audio_curve is not None else [1.0] * ticks
    if len(video_curve) != steps or len(audio_curve) != ticks:
        raise ValueError("mask curves do not match the latent (%s, %s)" % (steps, ticks))

    video_mask = torch.tensor(video_curve, dtype=torch.float32, device=video.device)
    video_mask = video_mask.view(1, 1, steps, 1, 1).expand(1, 1, steps, height, width).contiguous()
    if region is not None:
        mapper = pooled_region_steps if pooled_region else region_steps
        video_mask = torch.maximum(video_mask, mapper(region, steps, height, width).to(video_mask))
    audio_mask = torch.tensor(audio_curve, dtype=torch.float32, device=audio.device)
    audio_mask = audio_mask.view(1, 1, 1, ticks).expand(1, 1, audio.shape[2], ticks).contiguous()

    out = dict(latent)
    out["samples"] = comfy.nested_tensor.NestedTensor((video, audio))
    out["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))
    return out
