import copy
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import project as store


class CropRangeTest(unittest.TestCase):
    def setUp(self):
        self.project = store.new_project("crop")
        self.clip, self.first = store.add_clip(self.project, {"action": "Picture."})
        store.accept_take(self.project, self.clip["id"], {"id": "take", "quality": "draft", "file": "take.mp4",
                                                          "size": [160, 96], "frames": 24})
        self.second = store.place_clip(self.project, self.clip["id"], 24)
        self.crop = {"x": 0.5, "y": 0.25, "w": 0.5, "h": 0.5}

    def test_crop_belongs_to_placement_and_survives_split(self):
        store.apply_op(self.project, {"op": "update_item", "item": self.second["id"], "fields": {"crop": self.crop}}, {})
        self.assertNotIn("crop", self.first)
        right = store.split_item(self.project, self.second["id"], 36)
        self.assertEqual(right["crop"], self.crop)
        self.assertIsNot(right["crop"], self.second["crop"])
        self.assertEqual(store.describe(self.project)["timeline"][-1]["crop"], self.crop)
        store.update_item(self.project, self.second["id"], {"crop": None})
        self.assertNotIn("crop", self.second)
        for invalid in ({}, dict(self.crop, x=1), dict(self.crop, w=0), dict(self.crop, h=float("nan")), dict(self.crop, x=True)):
            with self.assertRaises(ValueError):
                store.update_item(self.project, self.second["id"], {"crop": invalid})

    def test_invalid_export_ranges_and_effect_dependency(self):
        for span in ({}, {"from": 2, "to": 2}, {"from": -1, "to": 1}, {"from": 0, "to": 49},
                     {"from": 0, "to": True}, {"from": 2.3, "to": 3}):
            with self.assertRaises(ValueError):
                store.export_plan(self.project, span=span)
        effect = {"start": 24, "frames": 4}
        stamp = store.effect_stamp(self.project, effect)
        self.first["crop"] = self.crop
        self.assertEqual(stamp, store.effect_stamp(self.project, effect))
        self.second["crop"] = self.crop
        self.assertNotEqual(stamp, store.effect_stamp(self.project, effect))

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not installed")
    def test_rendered_range_matches_full_cut_inside_transition_and_gap(self):
        ffmpeg = shutil.which("ffmpeg")
        with tempfile.TemporaryDirectory() as folder:
            def path(name):
                return str(Path(folder) / name)

            subprocess.run([ffmpeg, "-loglevel", "error", "-f", "lavfi", "-i",
                            "color=red:s=160x96:r=24:d=1,drawbox=x=80:y=0:w=80:h=96:c=blue:t=fill",
                            "-f", "lavfi", "-i", "sine=f=400:d=1:r=48000", "-c:v", "libx264", "-c:a", "aac",
                            path("take.mp4")], check=True)
            paths = {self.clip["id"]: path("take.mp4")}
            self.first["start"] = 5
            self.second.update(start=21, crop=self.crop, transition={"type": "dissolve", "frames": 8})
            self.project["timeline"].append(dict(copy.deepcopy(self.second), id="last", start=55, transition=None))

            def render(name, span=None):
                subprocess.run(store.export_command(self.project, paths, path(name), quality="best", span=span), check=True, timeout=15)
                if span:
                    samples = len(subprocess.check_output([ffmpeg, "-loglevel", "error", "-i", path(name),
                                                          "-map", "0:a", "-ac", "1", "-f", "s16le", "-"])) // 2
                    expected = (span["to"] - span["from"]) * 48000 // store.FPS
                    self.assertGreaterEqual(samples, expected - 48)  # MP4 edit lists can round by 1 ms.
                    self.assertLess(samples, expected + 1024)       # One AAC packet of encoder padding.
                return subprocess.check_output([ffmpeg, "-loglevel", "error", "-i", path(name), "-vf", "scale=1:1",
                                                "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])

            full = render("full.mp4")
            self.assertEqual(len(full), 79 * 3)
            self.assertGreater(full[10 * 3], 80)     # Red and blue in the wide placement.
            self.assertLess(full[35 * 3], 30)        # Only blue in the cropped placement.
            self.assertGreater(full[35 * 3 + 2], 190)
            for first, last in ((1, 3), (24, 27), (44, 58), (77, 78)):
                with self.subTest(first=first, last=last):
                    span = {"from": first, "to": last}
                    part = render("part_%d.mp4" % first, span)
                    self.assertEqual(len(part), (last - first) * 3)
                    self.assertLessEqual(max(abs(a - b) for a, b in zip(part, full[first * 3:last * 3])), 5)
                    self.assertEqual(store.export_plan(self.project, span=span)["frames"], last - first)
            self.project["timeline"] = [self.first]
            for speed in (0.25, 1.25, 4):
                self.first.update(start=0, speed=speed)
                end = store.scene_end(self.project)
                part = render("speed_%s.mp4" % speed, {"from": end - 1, "to": end})
                self.assertEqual(len(part), 3)
