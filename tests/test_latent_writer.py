"""Needs torch and ComfyUI on the path. Run inside the container. See README."""

import unittest

try:
    import torch
    import comfy.nested_tensor
    from latent_writer import audio_pcm, encode_audio, encode_video, region_steps, write_holds
except ImportError:
    torch = None


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
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}


@unittest.skipIf(torch is None, "needs torch and ComfyUI")
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


if __name__ == "__main__":
    unittest.main()
