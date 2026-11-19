import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

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

    def test_idle_uses_planted_motion_and_optional_installed_matte(self):
        project.update_clip(self.project, self.id, {"sprite": {"motion": "idle"}})
        files = {sprite.MATTE_PREFIX + "seg/birefnet.safetensors"}
        _, graph = project.plan_render(self.project, "Sprites", self.id, "draft", seed=77, available=files)
        self.assertNotIn("210", graph)
        self.assertNotIn("<Video 1>", graph["10"]["inputs"]["prompt"])
        self.assertIn("planted", graph["10"]["inputs"]["prompt"])
        self.assertTrue(graph["211"]["inputs"]["idle"])
        self.assertEqual(graph["212"]["inputs"]["bg_removal_name"], "seg/birefnet.safetensors")
        self.assertEqual(graph["211"]["inputs"]["bg_model"], ["212", 0])
        self.assertNotIn("bg_model", self.plan()[1]["211"]["inputs"])
        project.update_clip(self.project, self.id, {"sprite": {"cutout": "edge"}})
        _, graph = project.plan_render(self.project, "Sprites", self.id, "draft", seed=77, available=files)
        self.assertNotIn("212", graph)
        self.assertNotIn("cutout", graph["210"]["inputs"])

    def test_finish_uses_selected_take_after_shot_changes(self):
        project.update_clip(self.project, self.id, {"sprite": {"direction": "W", "poses": 8}})
        draft, draft_graph = self.plan()
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
        sampler = graph[graph["91"]["inputs"]["samples"][0]]
        guider = graph[sampler["inputs"]["guider"][0]]
        conditioning = graph[guider["inputs"]["conditioning"][0]]["inputs"]
        self.assertIn("facing west", conditioning["prompt"])
        self.assertEqual(conditioning["prompt"], draft_graph["10"]["inputs"]["prompt"])
        self.assertEqual(graph[conditioning["ref_videos.ref_video_0"][0]], draft_graph["210"])
        self.assertEqual(job["saved"]["sprite"]["direction"], "W")
        self.assertEqual(self.clip["sprite"]["direction"], "N")

    def test_rejects_invalid_config_and_incompatible_guides(self):
        for value in (
            False,
            {"direction": "NN"},
            {"poses": True},
            {"poses": 13},
            {"camera": "flat"},
            {"guide": {"file": "a.mp4", "subfolder": "../escape"}},
            {"guide": {"file": "a.mp4"}, "motion": "prompt"},
            {"guide": {"file": "a.mp4"}, "motion": "idle"},
            {"motion": "bad"}, {"cutout": "bad"},
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
        self.assertEqual(len(signatures), len(sprite.DIRECTIONS) * len(sprite.CAMERAS))

    def test_mirrored_atlas_preserves_pose_order_and_reflects_the_pivot(self):
        with tempfile.TemporaryDirectory() as tmp:
            strip = Image.new("RGBA", (128, 64))
            strip.putpixel((5, 20), (255, 0, 0, 255))
            strip.putpixel((69, 20), (0, 255, 0, 255))
            path = Path(tmp) / "poses.png"
            strip.save(path)
            rows = [("run_NE", path, dict(poses=2, pivot=[27, 59], fps=12), dict(mirrored=True))]
            for style in ("illustrated", "indexed"):
                with zipfile.ZipFile(io.BytesIO(sprite.pack_atlas(rows, 64, style))) as archive:
                    manifest = json.loads(archive.read("sprites.json"))
                    preview = archive.read("preview.html").decode()
                    embedded = preview.split('<script id="bundled-atlas" type="application/json">', 1)[1]
                    self.assertEqual(json.loads(embedded.split("</script>", 1)[0]), manifest)
                    meta = manifest["animations"]["run_NE"]
                    with Image.open(io.BytesIO(archive.read("sprites.png"))) as image:
                        if style == "indexed":
                            self.assertEqual(image.mode, "P")
                        art = image.convert("RGBA")
                self.assertEqual(meta["anchor"], {"x": 37, "y": 59, "span": 0})
                self.assertEqual(art.getpixel((58, 20)), (255, 0, 0, 255))
                self.assertEqual(art.getpixel((122, 20)), (0, 255, 0, 255))
                self.assertEqual(art.getpixel((57, 20))[3], 0)

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
        self.assertEqual(cell.getchannel("A").point(lambda v: 255 if v >= 128 else 0).getbbox(), (26, 11, 38, 59))
        self.assertEqual(cell.getpixel((32, 15)), (255, 255, 255, 255))
        np.testing.assert_allclose(indexed.convert("RGBA").getpixel((32, 50)), (35, 85, 169, 255), atol=1)
        final, _, inherited = sprite.strips([frame.resize((512, 768))] * 39, 7, layout)
        self.assertEqual(layout, inherited)
        self.assertEqual(final.size, full.size)
        with self.assertRaisesRegex(ValueError, "silhouette"):
            sprite.strips([Image.new("RGB", frame.size, "white")] * 39, 7)

    def test_selected_pose_masks_open_holes_and_keep_white_clothing(self):
        frame = Image.new("RGB", (256, 384), "white")
        draw = ImageDraw.Draw(frame)
        draw.rectangle((70, 100, 180, 260), fill="#5a3621")
        draw.rectangle((90, 130, 160, 170), fill="white")
        draw.rectangle((90, 190, 160, 237), fill="white")
        mask = np.zeros((384, 256), np.float32)
        mask[100:261, 70:181] = 1
        mask[190:238, 90:161] = 0
        layout = {"start": 8, "height": 160, "center": 125, "ground": 260}
        def load(indices):
            self.assertEqual(indices, list(range(8, 22, 2)))
            return [mask] * 7
        full, indexed, saved = sprite.strips([frame]*39, 7, layout, load)
        self.assertEqual(full.getpixel((32, 26)), (255, 255, 255, 255))
        self.assertEqual(full.getpixel((32, 45))[3], 0)
        self.assertEqual(indexed.convert("RGBA").getpixel((32, 45))[3], 0)
        self.assertEqual(saved, layout)

    def test_idle_bakes_a_return_without_duplicate_endpoints(self):
        frames = []
        for n in range(39):
            frame = Image.new("RGB", (256,384), "white")
            ImageDraw.Draw(frame).rectangle((80+n,100,140+n,240), fill="navy")
            frames.append(frame)
        layout = {"start": 6, "height": 140, "center": 123, "ground": 240}
        strip, indexed, _ = sprite.strips(frames, 7, layout, idle=True)
        self.assertEqual(strip.size, (12*64, 64))
        for sheet in (strip,indexed):
            cells = [sheet.crop((i*64,0,(i+1)*64,64)).tobytes() for i in range(12)]
            self.assertEqual(cells[7:], cells[5:0:-1])
            self.assertNotEqual(cells[0], cells[-1])

    def test_export_sizes_sample_source_detail_with_the_same_framing(self):
        frame = Image.new("RGB", (512, 768), "white")
        draw = ImageDraw.Draw(frame)
        for x in range(190, 322):
            draw.line((x, 200, x, 518), fill="black" if x % 8 < 4 else "gray")
        layout = {"start": 8, "height": 160, "center": 128, "ground": 260}
        small, _, _ = sprite.strips([frame] * 39, 7, layout)
        for size in (128, 256):
            full, indexed, reused = sprite.strips([frame] * 39, 7, layout, idle=True, cell_size=size)
            self.assertEqual(reused, layout)
            self.assertEqual(full.size, (12 * size, size))
            self.assertEqual(indexed.size, full.size)
            self.assertEqual(indexed.info["transparency"], 0)
            self.assertLessEqual(np.asarray(indexed).max(), 15)
            first = full.crop((0, 0, size, size))
            bbox = first.getchannel("A").point(lambda a: 255 if a >= 128 else 0).getbbox()
            self.assertLessEqual(abs(bbox[3] - 59 * size / 64), 1)
            baseline = small.crop((0, 0, 64, 64)).resize(first.size, Image.Resampling.NEAREST)
            center = (size // 2 - size // 16, size // 2, size // 2 + size // 16, size // 2 + 1)
            self.assertGreater(np.asarray(first.crop(center))[..., 0].std(),
                               np.asarray(baseline.crop(center))[..., 0].std() + 2)
        with self.assertRaisesRegex(ValueError, "cell size"):
            sprite.strips([frame] * 39, 7, layout, cell_size=96)
