import ast
import asyncio
import os
from pathlib import Path
import sys
import tempfile
import subprocess
import unittest
from unittest import mock

import media

try:
    import av
    import numpy as np
except ImportError:  # PyAV ships with ComfyUI, not with a bare Python
    av = None


class FfmpegTest(unittest.TestCase):
    def setUp(self):
        media._found.clear()
        self.addCleanup(media._found.clear)

    def test_only_ffmpeg_is_replaced(self):
        with mock.patch.dict(os.environ, {"H3TAKE_FFMPEG": "/opt/own/ffmpeg"}):
            self.assertEqual(media.command(["ffmpeg", "-i", "a.mp4"]), ["/opt/own/ffmpeg", "-i", "a.mp4"])
            self.assertEqual(media.command(["other", "ffmpeg"]), ["other", "ffmpeg"])

    def test_the_packaged_ffmpeg_comes_before_the_path(self):
        packaged = mock.Mock(get_ffmpeg_exe=lambda: "/site/imageio_ffmpeg/ffmpeg")
        with mock.patch.dict(os.environ, {"H3TAKE_FFMPEG": ""}), mock.patch.dict(sys.modules, {"imageio_ffmpeg": packaged}), \
                mock.patch.object(media.shutil, "which", return_value="/usr/bin/ffmpeg"):
            self.assertEqual(media.ffmpeg(), "/site/imageio_ffmpeg/ffmpeg")

    def test_without_any_ffmpeg_the_error_says_what_to_install(self):
        with mock.patch.dict(os.environ, {"H3TAKE_FFMPEG": ""}), mock.patch.dict(sys.modules, {"imageio_ffmpeg": None}), \
                mock.patch.object(media.shutil, "which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "requirements"):
                media.ffmpeg()
        self.assertEqual(media._found, [])


@unittest.skipIf(av is None, "needs PyAV")
class VideoInfoTest(unittest.TestCase):
    def test_shortest_clips_have_decodable_posters(self):
        # Load the API function at its three media boundaries, without starting ComfyUI.
        source = Path(__file__).parents[1] / "editor_api.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        probe = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "probe_file")

        async def run(args):
            subprocess.run(media.command(args), capture_output=True, check=True)

        with tempfile.TemporaryDirectory() as directory:
            for frames in (1, 12, 13, 24):
                with self.subTest(frames=frames):
                    path = os.path.join(directory, "%d.mp4" % frames)
                    self.make(path, frames, False)
                    namespace = {"os": os, "run": run, "video_info": mock.AsyncMock(return_value=media.video_info(path)),
                                 "make_strip": mock.AsyncMock(return_value=None)}
                    exec(compile(ast.Module(body=[probe], type_ignores=[]), str(source), "exec"), namespace)
                    result = asyncio.run(namespace["probe_file"](path))
                    with av.open(os.path.join(directory, result["poster"])) as image:
                        self.assertEqual(next(image.decode(video=0)).width, 480)

    def make(self, path, frames, sound):
        with av.open(path, "w") as container:
            video = container.add_stream("mpeg4", rate=24)
            video.width, video.height, video.pix_fmt = 64, 48, "yuv420p"
            audio = container.add_stream("aac", rate=48000) if sound else None
            for index in range(frames):
                picture = np.full((48, 64, 3), index * 9 % 255, dtype=np.uint8)
                for packet in video.encode(av.VideoFrame.from_ndarray(picture, format="rgb24")):
                    container.mux(packet)
            for packet in video.encode():
                container.mux(packet)
            if audio:
                samples = av.AudioFrame.from_ndarray(np.zeros((1, 2048), dtype=np.float32), format="fltp", layout="mono")
                samples.sample_rate = 48000
                for packet in audio.encode(samples):
                    container.mux(packet)
                for packet in audio.encode():
                    container.mux(packet)

    def test_frames_are_counted_and_sound_is_seen(self):
        with tempfile.TemporaryDirectory() as folder:
            silent, voiced = os.path.join(folder, "a.mp4"), os.path.join(folder, "b.mp4")
            self.make(silent, 30, False)
            self.make(voiced, 7, True)
            self.assertEqual(media.video_info(silent), {"width": 64, "height": 48, "sound": False, "frames": 30})
            self.assertEqual(media.video_info(voiced)["frames"], 7)
            self.assertTrue(media.has_sound(voiced))
            self.assertNotIn("frames", media.video_info(silent, count=False))

    def test_spectrogram_maps_frequency_and_reuses_its_cached_picture(self):
        try:
            media.ffmpeg()
        except RuntimeError as error:
            self.skipTest(str(error))
        with tempfile.TemporaryDirectory() as folder:
            source = os.path.join(folder, "tones.wav")
            subprocess.run(media.command(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                                          "-f", "lavfi", "-i", "sine=frequency=1760:duration=1", "-filter_complex", "[0:a][1:a]concat=n=2:v=0:a=1", source]),
                           capture_output=True, check=True)
            made = media.spectrogram(source)
            self.assertAlmostEqual(made["seconds"], 2, places=2)
            with av.open(os.path.join(folder, made["filename"])) as picture:
                pixels = next(picture.decode(video=0)).to_ndarray(format="rgb24")
            self.assertEqual(pixels.shape, (256, 2048, 3))
            low = pixels[:, 200:800].mean(axis=(1, 2)).argmax()
            high = pixels[:, 1250:1850].mean(axis=(1, 2)).argmax()
            self.assertLess(high, low, "higher frequencies belong higher in the picture")
            with mock.patch.object(media.subprocess, "run", side_effect=AssertionError("cache missed")):
                self.assertEqual(media.spectrogram(source), made)

    def test_a_silent_video_has_no_spectrogram(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "silent.mp4")
            self.make(path, 10, False)
            self.assertIsNone(media.spectrogram(path))


if __name__ == "__main__":
    unittest.main()
