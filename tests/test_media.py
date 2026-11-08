import os
import sys
import tempfile
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


if __name__ == "__main__":
    unittest.main()
