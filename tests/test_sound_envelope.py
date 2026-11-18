import array
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import project


def ffmpeg(*args, capture=False):
    result = subprocess.run(["ffmpeg", "-y", "-nostdin", "-v", "error", *args], check=True,
                            stdout=subprocess.PIPE if capture else subprocess.DEVNULL, timeout=30)
    return result.stdout


class SoundEnvelopeTest(unittest.TestCase):
    def test_points_are_source_anchored_and_round_trip_in_ops_and_summary(self):
        data = project.new_project("sound test")
        ref = {}
        project.apply_op(data, {"op": "add_sound", "as": "music", "asset": {"file": "song.wav"},
                                "seconds": 4}, ref)
        fields = {"offset": 0.5, "start": 24, "frames": 36, "fade_in": 6, "fade_out": 12,
                  "gain_points": [{"frame": 25, "gain": 0}, {"frame": 12, "gain": 1}]}
        project.apply_op(data, {"op": "update_sound", "sound": "$music", "fields": fields}, ref)
        sound = project.describe(data)["sounds"][0]
        self.assertEqual([point["frame"] for point in sound["gain_points"]], [12, 25])
        self.assertEqual((sound["fade_in"], sound["fade_out"]), (6, 12))
        for bad in ({"fade_in": -1}, {"fade_out": 1.5}, {"gain_points": [{}]},
                    {"gain_points": [{"frame": 97, "gain": 1}]},
                    {"gain_points": [{"frame": 10, "gain": 1}] * 2}):
            with self.subTest(fields=bad), self.assertRaises(ValueError):
                project.update_sound(data, sound["sound"], bad)
        project.update_sound(data, sound["sound"], {"gain_points": [], "fade_in": 0, "fade_out": 0})
        self.assertEqual(project.sound_envelope_filter(data["sounds"][0]), "")

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg required for audio export")
    def test_encoded_mix_and_subrange_follow_the_same_envelope(self):
        data = project.new_project("sound test")
        clip, item = project.add_clip(data, {"action": "Black picture."})
        project.accept_take(data, clip["id"], {"id": "take", "frames": 96, "file": "black.mp4", "quality": "draft"})
        item.update({"in": 0, "out": 96})
        project.update_item(data, item["id"], {"fade_in": 24,
                            "gain_points": [{"frame": 24, "gain": 0.5}, {"frame": 72, "gain": 0.5}]})
        sound = project.add_sound(data, {"file": "tone.wav"}, 4, {
            "offset": 0.5, "start": 12, "frames": 48, "gain": 1.5, "fade_in": 12, "fade_out": 18,
            "gain_points": [{"frame": 12, "gain": 0.4}, {"frame": 24, "gain": 1},
                            {"frame": 36, "gain": 0.25}, {"frame": 48, "gain": 0.5}]})

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            video = os.fspath(path / "black.mp4")
            audio = os.fspath(path / "tone.wav")
            # The original clip owns the right speaker. Its audio must survive mixing and trimming too.
            ffmpeg("-f", "lavfi", "-i", "color=black:s=64x64:r=24:d=4", "-f", "lavfi",
                   "-i", "aevalsrc=0|0.05*sin(2*PI*500*t):s=48000:d=4", "-t", "4",
                   "-c:v", "libx264", "-c:a", "aac", video)
            ffmpeg("-f", "lavfi", "-i", "aevalsrc=0.1*sin(2*PI*1000*t)|0:s=48000:d=4", audio)

            def mix(scene, name):
                target = os.fspath(path / name)
                cmd = project.export_command(scene, {clip["id"]: video}, target, width=64, height=64,
                                             sound_paths={sound["id"]: audio})
                subprocess.run(cmd, check=True, timeout=30)
                samples = array.array("f")
                samples.frombytes(ffmpeg("-i", target, "-vn", "-ac", "2", "-f", "f32le", "-", capture=True))
                return samples[::2], samples[1::2]

            full, full_clip = mix(data, "full.mp4")
            part, part_clip = mix(project.slice_project(data, 16, 56), "range.mp4")

        def level(samples, frame):
            center = frame * 2000
            window = samples[center - 240:center + 240]
            self.assertEqual(len(window), 480, "audio ended before the expected frame")
            return math.sqrt(2 * sum(v * v for v in window) / len(window)) / 0.1

        self.assertAlmostEqual(level(full, 6), 0, delta=0.01)
        self.assertGreaterEqual(len(part), 40 * 2000, "audio must cover all 40 frames of the range")
        self.assertAlmostEqual(level(full_clip, 90), 0.4375, delta=0.025)
        self.assertAlmostEqual(level(part_clip, 39), 0.25, delta=0.025)
        self.assertAlmostEqual(level(full_clip, 20), 0.5 * (1 - 0.5 * 20 / 24) * 20 / 24, delta=0.025)
        self.assertAlmostEqual(level(part_clip, 4), level(full_clip, 20), delta=0.025)
        # Positions in the placed sound, with independently calculated linear gains.
        for at, gain in ((6, 0.7 * 0.5), (9, 0.85 * 0.75), (12, 1), (18, 0.625), (24, 0.25),
                         (36, 0.5 * 12 / 18), (42, 0.5625 * 6 / 18)):
            with self.subTest(at=at):
                self.assertAlmostEqual(level(full, 12 + at), gain * 1.5, delta=0.035)
                self.assertAlmostEqual(level(part, 12 + at - 16), level(full, 12 + at), delta=0.025)
