import hashlib
import json
import unittest

import numpy as np
from PIL import Image, ImageDraw

import project
import sprite


class SpriteProjectTest(unittest.TestCase):
    def setUp(self):
        self.project = project.new_project("Sprites")
        self.project["profile"] = "hyperflow"
        self.clip, self.item = project.add_clip(self.project, {"action": "A dancer leaps.", "seconds": 5})
        self.id = self.clip["id"]

    def plan(self, quality="draft"):
        return project.plan_render(self.project, "Sprites", self.id, quality, seed=77)

    def test_sprite_prefers_fast_installed_family_and_respects_overrides(self):
        self.project["profile"] = "h3"
        project.update_clip(self.project, self.id, {"sprite": {}})
        turbo = project.PROFILES["h3_turbo"]
        available = {turbo["unet"][0], turbo["lora"][0]}
        self.assertEqual(project.project_profile(self.project, available, self.clip)[0], "h3_turbo")
        available.add(project.PROFILES["hyperflow"]["unet"][0])
        self.assertEqual(project.project_profile(self.project, available, self.clip)[0], "hyperflow")
        self.assertEqual(project.project_profile(self.project, set(), self.clip)[0], "h3")
        self.clip["profile"] = "h3"
        self.assertEqual(project.project_profile(self.project, available, self.clip)[0], "h3")

    def test_opt_in_and_restore_normal_timing(self):
        old_length = self.item["out"]
        self.assertNotIn("211", self.plan()[1])
        project.update_clip(self.project, self.id, {"sprite": {}})
        job, graph = self.plan()
        self.assertEqual(self.item["out"], 39)
        self.assertEqual(job["size"], [256, 384])
        self.assertEqual(graph["10"]["inputs"]["length"], 39)
        self.assertEqual(graph["10"]["inputs"]["ref_videos.ref_video_0"], ["210", 0])
        self.assertEqual(graph["210"]["inputs"], {"direction": "S", "camera": "rpg", "poses": 7})
        self.assertEqual(graph["211"]["inputs"]["images"], graph["18"]["inputs"]["images"])
        self.assertNotIn("9", graph)
        project.update_clip(self.project, self.id, {"sprite": None})
        self.assertEqual(self.item["out"], old_length)
        self.assertNotIn("211", self.plan()[1])

    def test_custom_guide_and_prompt_motion(self):
        custom = {"guide": {"file": "dance.mp4", "subfolder": "my/footage"}}
        project.update_clip(self.project, self.id, {"sprite": custom})
        _, graph = self.plan()
        self.assertEqual(
            graph["210"],
            {"class_type": "H3LoadClip", "inputs": {"file": "my/footage/dance.mp4", "first": 0, "last": 39}},
        )
        self.assertNotIn("running in place", graph["10"]["inputs"]["prompt"])
        project.update_clip(self.project, self.id, {"sprite": {"motion": "prompt", "camera": "overhead"}})
        _, graph = self.plan()
        self.assertNotIn("210", graph)
        self.assertNotIn("<Video 1>", graph["10"]["inputs"]["prompt"])
        self.assertIn("90 degrees", graph["10"]["inputs"]["prompt"])

    def test_finish_uses_selected_take_after_shot_changes(self):
        project.update_clip(self.project, self.id, {"sprite": {"direction": "W", "poses": 8}})
        draft, _ = self.plan()
        layout = {"start": 8, "height": 150.0, "center": 128, "ground": 300}
        take = dict(
            id=draft["take"],
            quality="draft",
            seed=17,
            saved=draft["saved"],
            models=draft["models"],
            frames=39,
            subfolder="h3take_projects/Sprites/clips",
            file="example_00001_.mp4",
            sprite={"layout": layout},
        )
        self.clip["takes"].append(take)
        project.update_clip(
            self.project, self.id, {"action": "Revised.", "sprite": {"direction": "N", "poses": 12}}
        )
        job, graph = self.plan("finish")
        self.assertEqual((job["size"], job["finished"], job["seed"]), ([512, 768], take["id"], 17))
        self.assertEqual(graph["15"]["class_type"], "H3LoadTake")
        self.assertEqual(json.loads(graph["211"]["inputs"]["layout"]), layout)
        self.assertEqual(graph["211"]["inputs"]["poses"], 8)
        self.assertEqual(job["saved"]["sprite"]["direction"], "W")
        self.assertEqual(self.clip["sprite"]["direction"], "N")

    def test_rejects_invalid_config_and_incompatible_guides(self):
        for value in (
            False,
            {"direction": "SE"},
            {"poses": True},
            {"poses": 13},
            {"camera": "flat"},
            {"guide": {"file": "a.mp4", "subfolder": "../escape"}},
            {"guide": {"file": "a.mp4"}, "motion": "prompt"},
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                sprite.settings(value)
        project.update_clip(
            self.project, self.id, {"mid_guides": [{"frame": 50, "file": "pose.png", "subfolder": ""}]}
        )
        with self.assertRaises(ValueError):
            project.update_clip(self.project, self.id, {"sprite": {}})
        self.assertNotIn("sprite", self.clip)


class SpriteImageTest(unittest.TestCase):
    def test_orthographic_guides_are_periodic_and_directional(self):
        signatures = set()
        for camera in sprite.CAMERAS:
            for direction in sprite.DIRECTIONS:
                frames = sprite.jog(direction, camera, 7)
                self.assertEqual(frames.shape, (39, 384, 256, 3))
                np.testing.assert_array_equal(frames[0], frames[14])
                self.assertFalse(np.array_equal(frames[0], frames[7]))
                signatures.add(hashlib.sha256(frames[:14].tobytes()).hexdigest())
        self.assertEqual(len(signatures), 12)

    def test_game_size_alpha_palette_and_draft_geometry(self):
        frame = Image.new("RGB", (256, 384), "white")
        draw = ImageDraw.Draw(frame)
        draw.rectangle((110, 110, 145, 253), fill="#2355a9")
        draw.rectangle((115, 115, 140, 135), fill="white")
        full, indexed, layout = sprite.strips([frame] * 39, 7)
        self.assertEqual(full.size, (448, 64))
        self.assertEqual(indexed.mode, "P")
        self.assertEqual(indexed.info["transparency"], 0)
        self.assertLessEqual(np.asarray(indexed).max(), 15)
        cell = full.crop((0, 0, 64, 64))
        self.assertEqual(cell.getchannel("A").getbbox(), (26, 11, 38, 59))
        self.assertEqual(cell.getpixel((32, 15)), (255, 255, 255, 255))
        self.assertEqual(indexed.convert("RGBA").getpixel((32, 50)), (35, 85, 169, 255))
        final, _, inherited = sprite.strips([frame.resize((512, 768))] * 39, 7, layout)
        self.assertEqual(layout, inherited)
        self.assertEqual(final.size, full.size)
        with self.assertRaisesRegex(ValueError, "silhouette"):
            sprite.strips([Image.new("RGB", frame.size, "white")] * 39, 7)
