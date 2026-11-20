"""Needs torch (and optionally ComfyUI on the path). Runs pooled_region_steps locally with torch."""

import sys
import types
import unittest
from unittest import mock

try:
    import torch
    from latent_writer import (
        audio_pcm,
        carry_tail,
        encode_audio,
        encode_video,
        pooled_region_steps,
        region_steps,
        write_holds,
    )
except ImportError:
    torch = None

try:
    import comfy.nested_tensor
    HAS_COMFY = True
except ImportError:
    HAS_COMFY = False


class _StubNestedTensor:
    def __init__(self, tensors):
        self.tensors = tuple(tensors)
        self.is_nested = True

    def unbind(self):
        return self.tensors


def _nested(tensors):
    if HAS_COMFY:
        return comfy.nested_tensor.NestedTensor(tensors)
    return _StubNestedTensor(tensors)


def _comfy_patch():
    if HAS_COMFY:
        return mock.nullcontext() if hasattr(mock, "nullcontext") else _NullContext()
    comfy_mod = types.ModuleType("comfy")
    nested_mod = types.ModuleType("comfy.nested_tensor")
    nested_mod.NestedTensor = _StubNestedTensor
    comfy_mod.nested_tensor = nested_mod
    return mock.patch.dict(sys.modules, {"comfy": comfy_mod, "comfy.nested_tensor": nested_mod})


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _Video:
    def encode(self, frames):
        steps = 2 if frames.shape[0] <= 5 else ((frames.shape[0] - 5) // 17) * 5 + 2
        return torch.ones(1, 24, steps, frames.shape[1] // 16, frames.shape[2] // 16)


class _Audio:
    audio_sample_rate = 32000

    def encode(self, pcm):
        return torch.ones(1, 32, 2, pcm.shape[1] // 800)


def _empty(steps=12, ticks=100, h=4, w=6):
    video = torch.zeros(1, 24, steps, h, w)
    audio = torch.zeros(1, 32, 2, ticks)
    return {"samples": _nested((video, audio))}


@unittest.skipIf(torch is None or not HAS_COMFY, "needs torch and ComfyUI")
class WriterTest(unittest.TestCase):
    def test_writes_the_hold_and_masks_it(self):
        curve = [0.0] * 7 + [1.0] * 5
        out = write_holds(_empty(), video_latent=torch.ones(1, 24, 7, 4, 6), video_curve=curve)
        video, audio = out["samples"].unbind()
        self.assertEqual(video[:, :, :7].min().item(), 1.0)
        self.assertEqual(video[:, :, 7:].max().item(), 0.0)
        mask_video, mask_audio = out["noise_mask"].unbind()
        self.assertEqual(tuple(mask_video.shape), (1, 1, 12, 4, 6))
        self.assertEqual(mask_video[0, 0, :7].max().item(), 0.0)
        self.assertEqual(mask_video[0, 0, 7:].min().item(), 1.0)
        self.assertEqual(mask_audio.min().item(), 1.0)
        self.assertTrue(out["noise_mask"].is_nested)

    def test_does_not_touch_the_incoming_latent(self):
        latent = _empty()
        write_holds(latent, video_latent=torch.ones(1, 24, 2, 4, 6), video_curve=[0.0] * 2 + [1.0] * 10)
        self.assertEqual(latent["samples"].unbind()[0].abs().max().item(), 0.0)
        self.assertNotIn("noise_mask", latent)

    def test_audio_lands_at_its_tick(self):
        curve = [1.0] * 20 + [0.0] * 30 + [1.0] * 50
        out = write_holds(_empty(), audio_latent=torch.ones(1, 32, 2, 30), audio_first=20, audio_curve=curve)
        audio = out["samples"].unbind()[1]
        self.assertEqual(audio[..., 19].max().item(), 0.0)
        self.assertEqual(audio[..., 20].min().item(), 1.0)
        self.assertEqual(audio[..., 50].max().item(), 0.0)

    def test_a_region_reopens_part_of_the_hold(self):
        region = torch.zeros(1, 64, 96)
        region[:, :, 48:] = 1.0
        out = write_holds(_empty(), video_latent=torch.ones(1, 24, 12, 4, 6), video_curve=[0.0] * 12, region=region)
        mask = out["noise_mask"].unbind()[0]
        self.assertEqual(mask[0, 0, 0, 0, 0].item(), 0.0)
        self.assertEqual(mask[0, 0, 0, 0, 5].item(), 1.0)

    def test_region_batches_spread_across_steps(self):
        region = torch.zeros(3, 8, 8)
        region[2] = 1.0
        out = region_steps(region, 12, 4, 6)
        self.assertEqual(tuple(out.shape), (1, 1, 12, 4, 6))
        self.assertEqual(out[0, 0, 0].max().item(), 0.0)
        self.assertEqual(out[0, 0, 11].min().item(), 1.0)

    def test_carry_takes_the_last_steps_and_ticks(self):
        latent = _empty()
        video, audio = latent["samples"].unbind()
        video[:, :, -3:] = 2.0
        audio[..., -5:] = 3.0
        tail_video, tail_audio = carry_tail(latent, 3, 5)
        self.assertEqual(tuple(tail_video.shape), (1, 24, 3, 4, 6))
        self.assertEqual(tail_video.min().item(), 2.0)
        self.assertEqual(tail_audio.shape[-1], 5)
        self.assertEqual(tail_audio.min().item(), 3.0)

    def test_curves_must_match_the_latent(self):
        with self.assertRaises(ValueError):
            write_holds(_empty(), video_curve=[1.0] * 3)

    def test_pcm_is_exactly_on_the_audio_grid(self):
        audio = {"waveform": torch.ones(1, 2, 32000), "sample_rate": 32000}
        self.assertEqual(audio_pcm(audio, 32000, 0.5, 65, 800).shape[-1], 52000)
        self.assertEqual(audio_pcm(audio, 32000, 0.9, 65, 800).shape[-1], 52000)

    def test_encoders_check_their_lengths(self):
        frames = torch.zeros(39, 64, 96, 3)
        self.assertEqual(encode_video(_Video(), frames, 96, 64, 12).shape[2], 12)
        with self.assertRaises(ValueError):
            encode_video(_Video(), frames, 96, 64, 9)
        audio = {"waveform": torch.ones(1, 2, 96000), "sample_rate": 32000}
        self.assertEqual(encode_audio(_Audio(), audio, 0.0, 65).shape[-1], 65)


@unittest.skipIf(torch is None, "needs torch")
class PooledRegionTest(unittest.TestCase):
    def test_1_4_4_4_4_temporal_union_and_32px_binary_dit_cell_pooling(self):
        # 7 latent steps cover spans (1, 4, 4, 4, 4, 1, 4) = 22 frames; lat 4x6 -> pixel 64x96.
        mask = torch.zeros(22, 64, 96)
        # Single pixel in top-left 32x32 cell on frame 3 (falls inside step 1 = frames 1..4)
        mask[3, 5, 7] = 1.0
        # Single pixel in bottom-right 32x32 cell on frame 17 (exact step 5 = 1-frame span [17..18))
        mask[17, 60, 90] = 1.0

        pooled = pooled_region_steps(mask, steps=7, height=4, width=6)
        self.assertEqual(tuple(pooled.shape), (1, 1, 7, 4, 6))
        # Strictly binary {0.0, 1.0}
        unique = set(pooled.unique().tolist())
        self.assertLessEqual(unique, {0.0, 1.0})

        # Step 0 (frame 0) and steps 2, 3, 4, 6 stay completely 0.0
        for s in (0, 2, 3, 4, 6):
            self.assertEqual(pooled[0, 0, s].max().item(), 0.0, f"step {s} should be 0")

        # Step 1 activates the full 2x2 latent cell [0:2, 0:2] and nothing else
        self.assertEqual(pooled[0, 0, 1, 0:2, 0:2].min().item(), 1.0)
        self.assertEqual(pooled[0, 0, 1, 2:, :].max().item(), 0.0)
        self.assertEqual(pooled[0, 0, 1, :, 2:].max().item(), 0.0)

        # Step 5 (frame 17) activates ONLY the 2x2 cell [2:4, 4:6]
        self.assertEqual(pooled[0, 0, 5, 2:4, 4:6].min().item(), 1.0)
        self.assertEqual(pooled[0, 0, 5, :2, :].max().item(), 0.0)
        self.assertEqual(pooled[0, 0, 5, :, :4].max().item(), 0.0)

    def test_exit_frames_stay_cleanly_zero(self):
        mask = torch.zeros(22, 64, 96)
        mask[:5, 0:32, 0:32] = 1.0  # steps 0 and 1 active; steps 2..6 all-zero exit frames
        pooled = pooled_region_steps(mask, steps=7, height=4, width=6)
        self.assertEqual(pooled[0, 0, 0, 0:2, 0:2].min().item(), 1.0)
        self.assertEqual(pooled[0, 0, 1, 0:2, 0:2].min().item(), 1.0)
        self.assertEqual(pooled[0, 0, 2:].max().item(), 0.0)

        # Completely empty clip produces strictly all-zero latent mask
        empty = pooled_region_steps(torch.zeros(22, 64, 96), steps=7, height=4, width=6)
        self.assertEqual(empty.max().item(), 0.0)

    def test_handles_2d_single_frame_short_batch_and_odd_latent_dims(self):
        # 2D mask expands across all steps and pads odd latent dimensions cleanly
        m2d = torch.zeros(48, 80)
        m2d[0:16, 0:16] = 1.0
        pooled = pooled_region_steps(m2d, steps=5, height=3, width=5)
        self.assertEqual(tuple(pooled.shape), (1, 1, 5, 3, 5))
        self.assertEqual(pooled[0, 0, :, 0:2, 0:2].min().item(), 1.0)
        self.assertLessEqual(set(pooled.unique().tolist()), {0.0, 1.0})

    def test_write_holds_defaults_to_region_steps_and_supports_pooled_region(self):
        # Mask with one active pixel on frame 3 inside a 32x32 DiT cell (12 steps = 39 frames, lat 4x6 = 64x96 px)
        region = torch.zeros(39, 64, 96)
        region[3, 4, 4] = 1.0
        with _comfy_patch():
            unpooled = write_holds(
                _empty(steps=12), video_latent=torch.ones(1, 24, 12, 4, 6),
                video_curve=[0.0] * 12, region=region, pooled_region=False,
            )
            pooled = write_holds(
                _empty(steps=12), video_latent=torch.ones(1, 24, 12, 4, 6),
                video_curve=[0.0] * 12, region=region, pooled_region=True,
            )
        m_unpooled = unpooled["noise_mask"].unbind()[0]
        m_pooled = pooled["noise_mask"].unbind()[0]
        # Pooled mode produces a full binary 2x2 cell on step 1 and 0 on exit steps
        self.assertEqual(m_pooled[0, 0, 1, 0:2, 0:2].min().item(), 1.0)
        self.assertEqual(m_pooled[0, 0, 2:].max().item(), 0.0)
        self.assertLessEqual(set(m_pooled.unique().tolist()), {0.0, 1.0})
        # Default (pooled_region=False) uses bilinear region_steps, which does NOT expand to a full 2x2 binary cell
        self.assertLess(m_unpooled[0, 0, 1, 1, 1].item(), 1.0)


if __name__ == "__main__":
    unittest.main()
