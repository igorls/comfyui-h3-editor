import hashlib
import importlib.util
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

import media
import vosr2


class VosrDiscoveryTest(unittest.TestCase):
    def test_wrapper_interface_and_cached_weights_are_separate(self):
        def execute(model, image, upscale, seed, color_alignment, tile_size, tile_overlap, vae_tile_size, vae_tile_overlap):
            pass

        schema = SimpleNamespace(inputs=[SimpleNamespace(id="model", options=["VOSR2"]), SimpleNamespace(id="dtype", options=["default"])])
        registry = {"VOSR2ModelLoader": SimpleNamespace(define_schema=lambda: schema), "VOSR2Upscale": SimpleNamespace(execute=execute)}
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(vosr2.status(vosr2.capabilities({}, directory))["installed"])
            self.assertEqual(vosr2.capabilities({"VOSR2Upscale": object()}, directory), {vosr2.INCOMPATIBLE})
            row = vosr2.status(vosr2.capabilities(registry, directory))
            self.assertTrue(row["installed"])
            self.assertFalse(row["weights_cached"])
            self.assertIn("first use", row["setup"])
            bundle = Path(directory, "vosr2", "VOSR2")
            for name in ["args.json", "ema_model.safetensors", "dinov2_vitl14.safetensors", "Qwen-Image-vae-2d/config.json",
                         "Qwen-Image-vae-2d/diffusion_pytorch_model.safetensors"]:
                path = bundle / name
                path.parent.mkdir(exist_ok=True, parents=True)
                path.touch()
            self.assertTrue(vosr2.status(vosr2.capabilities(registry, directory))["weights_cached"])


@unittest.skipUnless(importlib.util.find_spec("av") and importlib.util.find_spec("numpy"), "PyAV and NumPy required")
class VosrStreamTest(unittest.TestCase):
    def test_stream_preserves_frames_audio_and_cleans_up_if_interrupted(self):
        import av

        def double(rgb):
            return rgb.repeat(2, axis=0).repeat(2, axis=1)

        def soundtrack(path):
            with av.open(path) as video:
                return [hashlib.sha256(bytes(packet)).hexdigest() for packet in video.demux(audio=0) if packet.size]

        with tempfile.TemporaryDirectory() as work:
            source, target = [str(Path(work, name + ".mp4")) for name in ("take", "upscaled")]
            subprocess.run(media.command(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=24",
                                          "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", "0.5",
                                          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", source]), check=True)
            vosr2.stream_video(source, target, double, lambda: None)
            with av.open(target) as result:
                self.assertEqual([(f.width, f.height) for f in result.decode(video=0)], [(128, 96)] * 12)
            self.assertEqual(soundtrack(source), soundtrack(target))
            Path(target).unlink()
            frames = 0

            def interrupt():
                nonlocal frames
                frames += 1
                if frames == 4:
                    raise RuntimeError("interrupted")

            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                vosr2.stream_video(source, target, double, interrupt)
            self.assertEqual([p.name for p in Path(work).iterdir()], ["take.mp4"])
