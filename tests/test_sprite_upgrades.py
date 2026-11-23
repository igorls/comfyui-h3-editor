import io
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import h3image
import project
import sprite


def _make_box_frame(size, box, fill=(40, 90, 160), bg="white"):
    im = Image.new("RGB", size, bg)
    ImageDraw.Draw(im).rectangle(box, fill=fill)
    return im


class SpriteCanvasLengthAndTileTest(unittest.TestCase):
    def test_square_and_landscape_canvas_and_legal_lengths(self):
        proj = project.new_project("RPG")
        proj["profile"] = "hyperflow"
        clip, item = project.add_clip(proj, {"action": "Elf mage walks.", "seconds": 5})
        project.update_clip(proj, clip["id"], {
            "sprite": {"direction": "E", "camera": "rpg", "poses": 7, "canvas": "square", "frames": 56, "tile_size": [160, 160]}
        })
        self.assertEqual(item["out"], 56)
        job, graph = project.plan_render(proj, "RPG", clip["id"], "draft", seed=11)
        self.assertEqual(job["size"], [512, 512])
        self.assertEqual(graph["10"]["inputs"]["length"], 56)
        self.assertEqual(graph["210"]["inputs"]["frames"], 56)
        self.assertEqual(graph["210"]["inputs"]["canvas"], "square")
        self.assertEqual(graph["211"]["inputs"]["tile_width"], 160)
        self.assertEqual(graph["211"]["inputs"]["tile_height"], 160)
        self.assertEqual(graph["211"]["inputs"]["canvas"], "square")

        # Landscape draft/final sizes
        self.assertEqual(sprite.canvas_size({"canvas": "landscape"}, "draft"), (384, 256))
        self.assertEqual(sprite.canvas_size({"canvas": "landscape"}, "final"), (768, 512))
        self.assertEqual(sprite.canvas_size({"canvas": "portrait"}, "draft"), (256, 384))
        self.assertEqual(sprite.canvas_size({"canvas": "portrait"}, "final"), (512, 768))

        # Jog guide honors arbitrary legal length & square canvas
        guide = sprite.jog("SE", "rpg", 7, frames=56, canvas="square")
        self.assertEqual(guide.shape, (56, 512, 512, 3))
        np.testing.assert_array_equal(guide[0], guide[14])

    def test_56f_and_73f_strips_and_custom_tile_sizes(self):
        frames_56 = [_make_box_frame((512, 512), (180, 120, 330, 420)) for _ in range(56)]
        full, indexed, layout = sprite.strips(frames_56, 12, tile_size=(160, 160), canvas="square")
        self.assertEqual(full.size, (12 * 160, 160))
        self.assertEqual(indexed.size, full.size)
        self.assertGreaterEqual(layout["start"], 6)
        self.assertLess(layout["start"] + 24, 56)

        frames_73 = [_make_box_frame((512, 512), (180, 120, 330, 420)) for _ in range(73)]
        full73, _, _ = sprite.strips(frames_73, 12, tile_size=(144, 176), canvas="square")
        self.assertEqual(full73.size, (12 * 144, 176))

        for bad_len in (5, 17, 21, 40, 50):
            with self.assertRaises(ValueError):
                sprite.validate_frames(bad_len)
        for bad_tile in (30, 514, 95, (160, 161), (32, 600), "160"):
            with self.assertRaises(ValueError):
                sprite.validate_tile_size(bad_tile)


class H3ImagePropAndTurnaroundTest(unittest.TestCase):
    def test_native_size_enforces_4mp_floor_and_32px_grid(self):
        self.assertEqual(h3image.resolve_size("square"), (2048, 2048))
        self.assertEqual(h3image.resolve_size("portrait"), (1664, 2496))
        self.assertEqual(h3image.resolve_size("landscape"), (2496, 1664))
        for bad in ((512, 768), (1024, 1024), (2000, 2000), (2048, 2050)):
            with self.assertRaises(ValueError):
                h3image.validate_native_size(*bad)

    def test_prop_state_variant_and_turnaround_graphs(self):
        proj = project.new_project("Glade")
        proj["style"] = "Cozy fantasy cel-shaded illustration"
        avail = {sprite.MATTE_PREFIX + "birefnet.safetensors"}

        # Static prop with style ref + top canopy
        job, graph = project.plan_prop_render(proj, "Glade", {
            "kind": "prop",
            "name": "tree_oak",
            "description": "Mossy oak tree",
            "style_ref": {"file": "plate_S.png", "subfolder": "refs"},
            "top_canopy": True,
            "output_size": [256, 288],
            "pivot": [128, 268],
        }, seed=42, available=avail)
        self.assertEqual(job["native_size"], [2048, 2048])
        self.assertEqual(job["output_size"], [256, 288])
        self.assertEqual(job["pivot"], [128, 268])
        self.assertEqual(graph["10"]["class_type"], "PH3ImageConditioning")
        self.assertEqual(graph["4"]["class_type"], "MiniMaxH3SigmaShift")
        self.assertEqual(graph["10"]["inputs"]["images.image_1"], ["41", 0])
        self.assertEqual(graph["41"]["inputs"]["file"], "refs/plate_S.png")
        self.assertIn("top foliage canopy", graph["10"]["inputs"]["prompt"])
        self.assertEqual(graph["22"]["class_type"], "H3DefringeMatte")
        self.assertEqual(graph["15"]["class_type"], "SamplerCustomAdvanced")
        self.assertEqual(graph["14"]["class_type"], "ManualSigmas")

        # Paired state variant (lantern_unlit -> lantern_lit)
        _, vgraph = project.plan_prop_render(proj, "Glade", {
            "kind": "state_variant",
            "name": "lantern_lit",
            "description": "Glowing cyan-emerald runes and lantern flame",
            "style_ref": {"file": "plate_S.png", "subfolder": "refs"},
            "base_prop": {"file": "lantern_unlit.png", "subfolder": "props"},
        }, seed=43, available=avail)
        self.assertEqual(vgraph["10"]["inputs"]["images.image_1"], ["41", 0])
        self.assertEqual(vgraph["10"]["inputs"]["images.image_2"], ["42", 0])
        self.assertIn("pixel-for-pixel", vgraph["10"]["inputs"]["prompt"])

        # Turnaround back plate from front + side
        _, tgraph = project.plan_prop_render(proj, "Glade", {
            "kind": "turnaround",
            "name": "hero_back",
            "view": "back",
            "proportions": "chibi",
            "holding": "a glowing crystal staff",
            "style_ref": {"file": "plate_S.png", "subfolder": "refs"},
            "base_prop": {"file": "plate_E.png", "subfolder": "refs"},
        }, seed=44, available=avail)
        self.assertIn("2.5-head chibi", tgraph["10"]["inputs"]["prompt"])
        self.assertIn("rear back view", tgraph["10"]["inputs"]["prompt"])
        self.assertEqual(tgraph["10"]["inputs"]["images.image_2"], ["42", 0])

    def test_paired_alignment_iou_and_premultiplied_defringe(self):
        # Synthetic prop with soft white-blended anti-aliased border on white backdrop
        raw = Image.new("RGB", (256, 256), (255, 255, 255))
        arr = np.asarray(raw, dtype=np.float32) / 255.0
        mask = np.zeros((256, 256), dtype=np.float32)
        mask[40:220, 70:186] = 1.0
        # Soft 50% halo border blended with pure white
        mask[38:40, 70:186] = 0.5
        mask[220:222, 70:186] = 0.5
        fg = np.array([0.15, 0.25, 0.12], dtype=np.float32)
        blended = mask[..., None] * fg + (1.0 - mask[..., None]) * 1.0
        frame = Image.fromarray((blended * 255).round().astype(np.uint8), "RGB")

        out = h3image.process_prop(frame, mask=mask, output_size=(144, 176), pivot=(72, 162))
        self.assertEqual(out.size, (144, 176))
        out_arr = np.asarray(out, dtype=np.float32) / 255.0
        semi = (out_arr[..., 3] > 0.05) & (out_arr[..., 3] < 0.95)
        if semi.any():
            # Un-premultiplied border stays dark mossy green, not white halo!
            self.assertLess(float(out_arr[..., :3][semi].mean()), 0.35)

        # Paired IoU check: identical -> 1.0, shifted -> < 0.9
        self.assertEqual(h3image.paired_alignment_iou(out, out), 1.0)
        shifted = Image.new("RGBA", (144, 176), (0, 0, 0, 0))
        shifted.paste(out, (30, 0))
        self.assertLess(h3image.paired_alignment_iou(out, shifted), 0.7)

    def test_sprite_settings_accepts_static_prop_mode(self):
        cfg = sprite.settings({"mode": "prop", "camera": "rpg", "tile_size": [144, 176]})
        self.assertEqual(cfg["mode"], "prop")
        self.assertEqual(cfg["tile_size"], [144, 176])
        with self.assertRaises(ValueError):
            sprite.settings({"mode": "prop", "motion": "idle"})


class GlobalScaleWebPAndCacheTest(unittest.TestCase):
    def test_global_layout_prevents_full_action_shrinking_and_edge_clips(self):
        # Idle: compact narrow body; Cast: same body height + wide extended staff/wings
        idle_frames = [_make_box_frame((512, 512), (216, 160, 296, 360), fill=(30, 60, 120)) for _ in range(39)]
        cast_frames = []
        for i in range(39):
            im = _make_box_frame((512, 512), (216, 160, 296, 360), fill=(30, 60, 120))
            if 10 <= i <= 25:
                ImageDraw.Draw(im).rectangle((60, 220, 452, 245), fill=(180, 40, 40))
            cast_frames.append(im)

        idle_boxes = [sprite.cutout(f).getchannel("A").getbbox() for f in idle_frames]
        cast_boxes = [sprite.cutout(f).getchannel("A").getbbox() for f in cast_frames]
        shared = sprite.compute_global_layout(
            {"idle": idle_boxes, "cast": cast_boxes},
            tile_size=(160, 160),
            reference="idle",
        )
        self.assertEqual(shared["tile"], [160, 160])
        self.assertEqual(shared["pivot"], [80, 148])

        idle_strip, _, _ = sprite.strips(
            idle_frames, 7, tile_size=(160, 160), canvas="square", global_scale=shared["scale"]
        )
        cast_strip, _, _ = sprite.strips(
            cast_frames, 7, full_action=True, tile_size=(160, 160), canvas="square", global_scale=shared["scale"]
        )
        # Measure body heights in output pixels — must match identically between idle and cast!
        idle_box = idle_strip.crop((0, 0, 160, 160)).getchannel("A").point(lambda v: 255 if v >= 128 else 0).getbbox()
        cast_boxes_out = [
            cast_strip.crop((i * 160, 0, (i + 1) * 160, 160)).getchannel("A").point(lambda v: 255 if v >= 128 else 0).getbbox()
            for i in range(7)
        ]
        # Every cast frame sits strictly inside tile margins (0 edge clips)
        for b in cast_boxes_out:
            self.assertGreater(b[0], 0)
            self.assertLess(b[2], 160)
            self.assertGreater(b[1], 0)
            self.assertLess(b[3], 160)
        # Body height on idle equals body height on cast (zero character shrinking!)
        self.assertLessEqual(abs((idle_box[3] - idle_box[1]) - (cast_boxes_out[0][3] - cast_boxes_out[0][1])), 1)

    def test_pack_atlas_supports_nonsquare_tiles_and_webp(self):
        with tempfile.TemporaryDirectory() as tmp:
            strip = Image.new("RGBA", (3 * 160, 160), (0, 0, 0, 0))
            for i in range(3):
                ImageDraw.Draw(strip).ellipse((i * 160 + 40, 20, i * 160 + 120, 140), fill=(60 + i * 40, 120, 90, 255))
            path = Path(tmp) / "run_S.png"
            strip.save(path)
            rows = [("run_S", path, dict(poses=3, pivot=[80, 144], fps=24), dict(mirrored=False))]
            blob = sprite.pack_atlas(rows, (160, 160), "illustrated", include_webp=True)
            with zipfile.ZipFile(io.BytesIO(blob)) as zf:
                names = set(zf.namelist())
                self.assertIn("sprites.png", names)
                self.assertIn("sprites.webp", names)
                manifest = json.loads(zf.read("sprites.json"))
                self.assertEqual(manifest["image_webp"], "sprites.webp")
                self.assertEqual(manifest["animations"]["run_S"]["frame_size"], {"w": 160, "h": 160})
                with Image.open(io.BytesIO(zf.read("sprites.webp"))) as webp_im:
                    self.assertEqual(webp_im.size, (480, 160))

            blob_no_webp = sprite.pack_atlas(rows, (160, 160), "illustrated", include_webp=False)
            with zipfile.ZipFile(io.BytesIO(blob_no_webp)) as zf:
                names = set(zf.namelist())
                self.assertIn("sprites.png", names)
                self.assertNotIn("sprites.webp", names)
                manifest = json.loads(zf.read("sprites.json"))
                self.assertNotIn("image_webp", manifest)

    def test_lossless_frame_cache_reexports_without_redecoding(self):
        frames = [_make_box_frame((256, 384), (95, 110, 155, 250), fill=(25 + n * 3, 80, 150)) for n in range(39)]
        with tempfile.TemporaryDirectory() as tmp:
            cache = os.path.join(tmp, "take.frames")
            _, _, layout = sprite.strips(frames, 7, cache_dir=cache)
            # Re-export at 160x160 directly from lossless cache
            from_cache, idx_cache = sprite.export_from_cache(cache, layout, 7, (160, 160))
            direct, _ , _ = sprite.strips(frames, 7, layout, tile_size=(160, 160))
            self.assertEqual(from_cache.size, (7 * 160, 160))
            self.assertEqual(idx_cache.size, from_cache.size)
            self.assertEqual(from_cache.tobytes(), direct.tobytes())
            # Loading all cached frames (indices=None) works for sprite_self_pin
            all_cached = sprite.load_frame_cache(cache)
            self.assertEqual(len(all_cached), 7)
            self.assertEqual(all_cached[0].mode, "RGBA")


class SpriteEfficiencyPinningAndViewRoutingTest(unittest.TestCase):
    def setUp(self):
        self.proj = project.new_project("HeroKit")
        self.proj["profile"] = "hyperflow"
        self.hero = project.add_cast(self.proj, {
            "name": "Lyra",
            "file": "plate_S.png",
            "subfolder": "refs",
            "view": "front",
            "pictures": [
                {"file": "plate_E.png", "subfolder": "refs", "view": "side"},
                {"file": "plate_N.png", "subfolder": "refs", "view": "back"},
                {"file": "plate_W.png", "subfolder": "refs", "view": "left"},
            ],
        })
        self.clip, _ = project.add_clip(self.proj, {
            "action": "@Lyra raises her staff.",
            "cast": [self.hero["id"]],
        })

    def test_view_tagged_cast_filtering_by_direction(self):
        expected = {
            "S": ["refs/plate_S.png"],
            "N": ["refs/plate_N.png"],
            "E": ["refs/plate_E.png"],
            "W": ["refs/plate_W.png"],
            "SE": ["refs/plate_S.png", "refs/plate_E.png"],
            "SW": ["refs/plate_S.png", "refs/plate_W.png"],
            "NE": ["refs/plate_N.png", "refs/plate_E.png"],
            "NW": ["refs/plate_N.png", "refs/plate_W.png"],
        }
        for direction, want in expected.items():
            project.update_clip(self.proj, self.clip["id"], {"sprite": {"direction": direction, "canvas": "square"}})
            shown = project.cast_pictures(self.proj, self.clip)
            got = [f"{p['subfolder']}/{p['file']}" for p in shown[0][1]]
            self.assertEqual(got, want, f"direction {direction}")

    def test_idle_and_full_action_auto_pin_and_strip_loop_wording(self):
        # Idle auto-pins direction-matched plate to both start & end
        project.update_clip(self.proj, self.clip["id"], {
            "sprite": {"direction": "N", "motion": "idle", "canvas": "square"}
        })
        _, graph = project.plan_render(self.proj, "HeroKit", self.clip["id"], "draft", seed=9)
        self.assertEqual(graph["10"]["inputs"]["anchor"], "both")
        self.assertEqual(graph["40"]["inputs"]["file"], "refs/plate_N.png")
        self.assertEqual(graph["41"]["inputs"]["file"], "refs/plate_N.png")

        # Named full_action auto-pins start & end and does NOT say "Loop the movement"
        project.update_clip(self.proj, self.clip["id"], {
            "sprite": {"direction": "E", "motion": "prompt", "animation": "cast", "canvas": "square"}
        })
        _, cgraph = project.plan_render(self.proj, "HeroKit", self.clip["id"], "draft", seed=10)
        self.assertEqual(cgraph["10"]["inputs"]["anchor"], "both")
        self.assertEqual(cgraph["40"]["inputs"]["file"], "refs/plate_E.png")
        self.assertEqual(cgraph["41"]["inputs"]["file"], "refs/plate_E.png")
        self.assertNotIn("Loop the movement", cgraph["10"]["inputs"]["prompt"])
        self.assertIn("Perform the full gesture", cgraph["10"]["inputs"]["prompt"])

    def test_pinned_heading_picture_replaces_views_of_other_headings(self):
        # A north-east clip pinned to its own picture: the back and side views would turn the body mid-take.
        pin = {"file": "plate_NE.png", "subfolder": "refs"}
        project.update_clip(self.proj, self.clip["id"], {
            "sprite": {"direction": "NE", "motion": "idle", "canvas": "square"}, "first": pin, "last": pin,
        })
        shown = project.cast_pictures(self.proj, self.clip)
        self.assertEqual([p["file"] for p in shown[0][1]], ["plate_NE.png"])
        _, graph = project.plan_render(self.proj, "HeroKit", self.clip["id"], "draft", seed=3)
        take = graph["10"]["inputs"]
        self.assertEqual(graph[take["ref_images.ref_image_0"][0]]["inputs"]["file"], "refs/plate_NE.png")
        self.assertNotIn("ref_images.ref_image_1", take)
        self.assertIn("<Subject 1> is", take["prompt"])
        self.assertIn("in <Picture 3>.", take["prompt"])

        # A cardinal heading with its own view keeps that view beside the pin.
        project.update_clip(self.proj, self.clip["id"], {"sprite": {"direction": "N", "motion": "idle"}})
        shown = project.cast_pictures(self.proj, self.clip)
        self.assertEqual([p["file"] for p in shown[0][1]], ["plate_N.png"])

    def test_diagonal_idle_is_not_auto_pinned_to_another_heading(self):
        project.update_clip(self.proj, self.clip["id"], {"sprite": {"direction": "NE", "motion": "idle"}})
        self.assertIsNone(project.sprite_auto_pin(self.proj, self.clip))
        _, graph = project.plan_render(self.proj, "HeroKit", self.clip["id"], "draft", seed=4)
        self.assertEqual(graph["10"]["inputs"]["anchor"], "none")
        self.assertEqual(len(project.cast_pictures(self.proj, self.clip)[0][1]), 2)

    def test_sprite_stills_load_on_white_and_pins_fit_the_canvas(self):
        pin = {"file": "plate_NE.png", "subfolder": "refs"}
        project.update_clip(self.proj, self.clip["id"], {
            "sprite": {"direction": "NE", "motion": "idle", "canvas": "square"}, "first": pin, "last": pin,
        })
        _, graph = project.plan_render(self.proj, "HeroKit", self.clip["id"], "draft", seed=3)
        want = {"file": "refs/plate_NE.png", "backdrop": "white", "width": 512, "height": 512}
        self.assertEqual(graph["40"]["inputs"], want)
        self.assertEqual(graph["41"]["inputs"], want)
        self.assertEqual(graph["50"]["inputs"], {"file": "refs/plate_NE.png", "backdrop": "white"})
        # A clip that is not a sprite loads its pictures as before.
        self.assertEqual(project.still_node(pin), {"class_type": "H3LoadStill", "inputs": {"file": "refs/plate_NE.png"}})

        cutout = Image.new("RGBA", (256, 384), (0, 0, 0, 0))
        ImageDraw.Draw(cutout).rectangle((100, 20, 150, 370), fill=(20, 20, 80, 255))
        out = sprite.still(cutout, "white", 512, 512)
        self.assertEqual((out.mode, out.size), ("RGB", (512, 512)))
        self.assertEqual(out.getpixel((2, 2)), (255, 255, 255))      # the filled side
        self.assertEqual(out.getpixel((90, 256)), (255, 255, 255))   # transparency, laid on white
        box = Image.eval(out.convert("L"), lambda v: 255 if v < 128 else 0).getbbox()
        self.assertTrue(box[1] > 0 and box[3] < 512, box)            # head and feet inside the canvas
        self.assertEqual(sprite.still(cutout).getpixel((2, 2)), (0, 0, 0))

    def test_sprite_skips_audio_decode_and_save_hold_when_not_refining(self):
        project.update_clip(self.proj, self.clip["id"], {"sprite": {"direction": "S", "canvas": "square"}})
        _, graph = project.plan_render(self.proj, "HeroKit", self.clip["id"], "draft", seed=5)
        self.assertNotIn("17", graph)
        self.assertNotIn("audio", graph["18"]["inputs"])
        self.assertNotIn("20", graph)

        # When refine=True is explicitly requested, H3SaveHold stays while audio decode is still skipped
        project.update_clip(self.proj, self.clip["id"], {"sprite": {"direction": "S", "refine": True}})
        _, rgraph = project.plan_render(self.proj, "HeroKit", self.clip["id"], "draft", seed=5)
        self.assertNotIn("17", rgraph)
        self.assertIn("20", rgraph)

    def test_self_pin_helper_and_graph_wiring(self):
        # Frame 14 touches edge; neighbors 13 & 15 are clean -> pick_self_pin_frame skips 14 cleanly
        frames = [_make_box_frame((256, 256), (80, 60, 176, 200), fill=(40 + i, 80, 120)) for i in range(39)]
        frames[14] = _make_box_frame((256, 256), (0, 60, 176, 200), fill=(54, 80, 120))
        idx, rgb, pins = sprite.pick_self_pin_frame(frames, stride=14)
        self.assertIn(idx, (13, 15))
        self.assertEqual(rgb.mode, "RGB")
        self.assertEqual(pins, [0, 14, 28])

        # Seam score on a periodic synthetic clip is ~0.0
        periodic = [frames[i % 14] for i in range(39)]
        periodic[14] = periodic[0]
        self.assertLess(sprite.seam_score(periodic, 6, 14), 0.01)

        # Graph wiring for self_pin on 56-frame walk (stride=26 -> pins 0, 26, 52)
        project.update_clip(self.proj, self.clip["id"], {
            "sprite": {
                "direction": "S", "canvas": "square", "frames": 56,
                "self_pin": {"file": "pin_f26.png", "subfolder": "pins", "stride": 26},
            }
        })
        _, sgraph = project.plan_render(self.proj, "HeroKit", self.clip["id"], "draft", seed=12)
        self.assertEqual(sgraph["10"]["inputs"]["anchor"], "start")
        self.assertEqual(sgraph["40"]["inputs"]["file"], "pins/pin_f26.png")
        self.assertEqual(sgraph["10"]["inputs"]["mid_frames"], "26, 52")
        self.assertEqual(sgraph["209"]["inputs"]["file"], "pins/pin_f26.png")

    def test_sprite_draft_with_cast_does_not_inherit_live_action_video_project_style(self):
        self.proj["style"] = "live-action footage of a creator talking to her camera, 35mm lens, natural skin"
        project.update_clip(self.proj, self.clip["id"], {"sprite": {"direction": "E", "canvas": "square"}})
        _, graph = project.plan_render(self.proj, "HeroKit", self.clip["id"], "draft", seed=7)
        prompt = graph["10"]["inputs"]["prompt"]
        self.assertNotIn("creator talking to her camera", prompt)
        self.assertNotIn("live-action footage", prompt)
        self.assertIn("exact art style", prompt)

        # Style-ref 4 MP prop also locks to style_ref instead of leaking the video project's live-action look
        _, pgraph = project.plan_prop_render(self.proj, "HeroKit", {
            "kind": "prop", "name": "lantern", "description": "Carved stone lantern",
            "style_ref": {"file": "plate_S.png", "subfolder": "refs"},
        }, seed=8)
        self.assertNotIn("creator talking to her camera", pgraph["10"]["inputs"]["prompt"])

    def test_birefnet_matte_remeasures_layout_on_shaded_backdrop_preventing_half_size_sprites(self):
        # Simulate a non-uniform warm studio gradient backdrop where cheap background-key cutout()
        # sees the whole 512x512 canvas, while BiRefNet cleanly isolates the 260px character.
        grad = np.zeros((512, 512, 3), dtype=np.uint8)
        grad[:, :, 0] = np.linspace(180, 245, 512, dtype=np.uint8)[None, :]
        grad[:, :, 1] = np.linspace(160, 230, 512, dtype=np.uint8)[:, None]
        grad[:, :, 2] = 175
        base = Image.fromarray(grad, "RGB")
        ImageDraw.Draw(base).rectangle((190, 140, 322, 400), fill=(40, 80, 180))
        frames = [base.copy() for _ in range(39)]

        mask = np.zeros((512, 512), dtype=np.float32)
        mask[140:400, 190:322] = 1.0

        full, _, layout = sprite.strips(
            frames, 7, masks_for=lambda idx: [mask for _ in idx], canvas="square"
        )
        # Layout must reflect the real 260px BiRefNet silhouette, NOT the 512px canvas!
        self.assertAlmostEqual(layout["height"], 260.0, delta=2.0)
        self.assertAlmostEqual(layout["ground"], 400.0, delta=2.0)
        # Rendered tile body height should match BODY (~48px), never half-size (~24px)!
        cell0 = full.crop((0, 0, 64, 64))
        bb = cell0.getchannel("A").point(lambda v: 255 if v >= 128 else 0).getbbox()
        self.assertIsNotNone(bb)
        self.assertGreaterEqual(bb[3] - bb[1], 45)
        self.assertLessEqual(bb[3] - bb[1], 50)


class NodesAndApiRegistrationTest(unittest.TestCase):
    def test_nodes_and_editor_api_syntax_and_registrations(self):
        import ast

        root = Path(__file__).resolve().parent.parent
        nodes_src = (root / "nodes.py").read_text(encoding="utf-8")
        nodes_tree = ast.parse(nodes_src, filename="nodes.py")
        classes = {n.name for n in nodes_tree.body if isinstance(n, ast.ClassDef)}
        self.assertIn("H3DefringeMatte", classes)
        self.assertIn("H3SpriteGuide", classes)
        self.assertIn("H3SpriteSheet", classes)
        self.assertIn('"H3DefringeMatte": H3DefringeMatte', nodes_src)
        self.assertIn('"H3DefringeMatte": "H3 Defringe Matte"', nodes_src)

        api_src = (root / "editor_api.py").read_text(encoding="utf-8")
        ast.parse(api_src, filename="editor_api.py")
        self.assertIn('("GET", "/h3take/api/queue", "render_queue")', api_src)
        self.assertIn('("POST", "/h3take/api/queue/cancel", "cancel_queue")', api_src)
        self.assertIn('("POST", "/h3take/api/queue/clear-history", "clear_queue_history")', api_src)
        self.assertIn('("POST", "/h3take/api/project/{name}/sprite-self-pin", "sprite_self_pin")', api_src)
        self.assertIn('("POST", "/h3take/api/project/{name}/props", "sprite_prop")', api_src)
        self.assertIn("export_from_cache", api_src)


class CutoutTest(unittest.TestCase):
    """A matte and the backdrop's own colour together, on a flat backdrop of any colour."""

    def subject(self, ground, size=(256, 384)):
        frame = Image.new("RGB", size, ground)
        ImageDraw.Draw(frame).rectangle((70, 100, 180, 300), fill="#5a3621")
        mask = np.zeros((size[1], size[0]), np.float32)
        mask[100:301, 70:181] = 1
        return frame, mask

    def test_a_patch_of_backdrop_the_matte_kept_is_removed_on_any_flat_colour(self):
        for ground in ("white", "black", "#ff00ff", "#2f8f4f"):
            with self.subTest(ground=ground):
                frame, mask = self.subject(ground)
                mask[150:162, 181:189] = 1  # a lump of backdrop beside the subject, called foreground
                before = np.asarray(sprite.with_matte(frame, mask, key=False))[..., 3]
                after = np.asarray(sprite.with_matte(frame, mask))[..., 3]
                self.assertEqual(before[155, 185], 255)
                self.assertEqual(after[155, 185], 0)
                self.assertEqual(after[200, 120], 255)
                self.assertEqual(after[10, 10], 0)

    def test_a_backdrop_coloured_surface_of_the_subject_stays(self):
        frame, mask = self.subject("white")
        draw = ImageDraw.Draw(frame)
        draw.rectangle((90, 130, 160, 170), fill="white")   # a white shirt inside the subject
        draw.rectangle((150, 200, 180, 280), fill="white")  # a white sleeve on the subject's outline
        out = np.asarray(sprite.with_matte(frame, mask))
        self.assertEqual(tuple(out[150, 120]), (255, 255, 255, 255))
        self.assertEqual(tuple(out[240, 170]), (255, 255, 255, 255))

    def test_a_scene_behind_the_subject_is_left_to_the_matte(self):
        rng = np.random.default_rng(7)
        scene = rng.integers(0, 255, (384, 256, 3), dtype=np.uint8)
        frame = Image.fromarray(scene, "RGB")
        ImageDraw.Draw(frame).rectangle((70, 100, 180, 300), fill="#5a3621")
        mask = np.zeros((384, 256), np.float32)
        mask[100:301, 70:181] = 1
        self.assertFalse(sprite.backdrop(np.asarray(frame, dtype=np.float32) / 255)[2])
        out = np.asarray(sprite.with_matte(frame, mask))[..., 3]
        np.testing.assert_array_equal(out, (mask * 255).astype(np.uint8))

    def test_without_a_matte_the_backdrop_colour_cuts_with_a_soft_clean_edge(self):
        frame, _ = self.subject("#ff00ff")
        big = frame.resize((1024, 1536), Image.Resampling.NEAREST).resize((256, 384), Image.Resampling.LANCZOS)
        out = np.asarray(sprite.with_matte(big))
        self.assertEqual(out[10, 10, 3], 0)
        self.assertEqual(tuple(out[200, 120]), (0x5a, 0x36, 0x21, 255))
        visible = out[out[..., 3] > 40]
        # No magenta left on the edge: blue never rises far above the subject's own.
        self.assertLess(int(visible[:, 2].max()), 0x21 + 40)

    def test_enclosed_pockets_go_only_when_asked(self):
        frame, _ = self.subject("white")
        ImageDraw.Draw(frame).rectangle((110, 220, 140, 280), fill="white")
        self.assertEqual(np.asarray(sprite.with_matte(frame))[250, 125, 3], 255)
        self.assertEqual(np.asarray(sprite.with_matte(frame, pockets=True))[250, 125, 3], 0)

    def test_hard_alpha_leaves_no_half_transparent_pixel(self):
        frame, mask = self.subject("white", (512, 512))
        soft_mask = np.asarray(Image.fromarray((mask[:384] * 255).astype(np.uint8)).resize((512, 512)), np.float32) / 255
        frames = [frame.resize((512, 512))] * 39
        load = lambda idx: [soft_mask for _ in idx]
        soft, _, _ = sprite.strips(frames, 7, masks_for=load, canvas="square")
        hard, _, _ = sprite.strips(frames, 7, masks_for=load, canvas="square", alpha="hard")
        self.assertTrue(((np.asarray(soft)[..., 3] > 0) & (np.asarray(soft)[..., 3] < 255)).any())
        self.assertEqual(set(np.unique(np.asarray(hard)[..., 3])), {0, 255})

    def depth_map(self, frame_size=(256, 384), near=((70, 100, 180, 300),), far_value=.3, near_value=.5):
        depth = np.full((frame_size[1], frame_size[0]), far_value, np.float32)
        for x0, y0, x1, y1 in near:
            depth[y0:y1 + 1, x0:x1 + 1] = near_value
        return depth

    def test_depth_keeps_a_small_backdrop_coloured_part_of_the_subject(self):
        frame, mask = self.subject("white")
        ImageDraw.Draw(frame).rectangle((175, 150, 180, 160), fill="white")  # a white cuff on the outline
        self.assertEqual(np.asarray(sprite.with_matte(frame, mask))[155, 178, 3], 0)
        for depth in (self.depth_map(), 1 / self.depth_map()):  # near bright, then near dark
            self.assertEqual(np.asarray(sprite.with_matte(frame, mask, depth=depth))[155, 178, 3], 255)

    def test_depth_removes_a_large_patch_of_backdrop_the_matte_kept(self):
        frame, mask = self.subject("white")
        mask[100:301, 181:215] = 1  # a slab of backdrop, far too large for the size rule
        self.assertEqual(np.asarray(sprite.with_matte(frame, mask))[200, 200, 3], 255)
        out = np.asarray(sprite.with_matte(frame, mask, depth=self.depth_map()))[..., 3]
        self.assertEqual(out[200, 200], 0)
        self.assertEqual(out[200, 120], 255)

    def test_depth_opens_enclosed_pockets_and_keeps_an_enclosed_white_surface(self):
        frame, mask = self.subject("white")
        draw = ImageDraw.Draw(frame)
        draw.rectangle((110, 220, 140, 280), fill="white")  # a gap between the legs: backdrop shows through
        draw.rectangle((90, 130, 160, 170), fill="white")   # a white shirt
        depth = self.depth_map()
        depth[220:281, 110:141] = .3
        for m in (mask, None):
            out = np.asarray(sprite.with_matte(frame, m, depth=depth))[..., 3]
            self.assertEqual(out[250, 125], 0)
            self.assertEqual(out[150, 125], 255)

    def test_a_depth_map_that_cannot_tell_the_two_apart_is_ignored(self):
        frame, mask = self.subject("white")
        mask[150:162, 181:189] = 1
        flat = np.full((384, 256), .4, np.float32)
        np.testing.assert_array_equal(np.asarray(sprite.with_matte(frame, mask, depth=flat)),
                                      np.asarray(sprite.with_matte(frame, mask)))

    def test_a_depth_map_of_another_size_is_fitted_to_the_frame(self):
        frame, mask = self.subject("white")
        mask[100:301, 181:215] = 1
        small = np.asarray(Image.fromarray(self.depth_map(), "F").resize((128, 192), Image.Resampling.BILINEAR))
        self.assertEqual(np.asarray(sprite.with_matte(frame, mask, depth=small))[200, 200, 3], 0)

    def test_depth_is_used_only_when_its_nodes_and_weights_are_installed(self):
        self.assertIsNone(sprite.depth_model([]))
        self.assertIsNone(sprite.depth_model(["depth_pro_2:depthpro2-l.safetensors"]))  # the focal file would be downloaded
        both = ["depth_pro_2:depthpro2-s.safetensors", "depth_pro_2:depthpro2-focal.safetensors"]
        self.assertEqual(sprite.depth_model(both), "S")
        self.assertEqual(sprite.depth_model(both + ["depth_pro_2:depthpro2-h.safetensors", "depth_pro_2:depthpro2-l.safetensors"]), "L")
        nodes, link = sprite.depth_nodes("L", ["18", 0], 213)
        self.assertEqual([n["class_type"] for n in nodes.values()], list(sprite.DEPTH_NODES))
        self.assertEqual(link, ["215", 0])
        self.assertEqual(nodes["214"]["inputs"]["images"], ["18", 0])
        self.assertEqual(sprite.settings({"direction": "S", "depth": "on"})["depth"], "on")
        with self.assertRaisesRegex(ValueError, "off or on"):
            sprite.settings({"direction": "S", "depth": "auto"})

    def test_the_sheet_passes_each_pose_its_depth_map(self):
        frame, mask = self.subject("white", (512, 512))
        frame = frame.resize((512, 512))
        ImageDraw.Draw(frame).rectangle((0, 0, 511, 511), fill="white")
        ImageDraw.Draw(frame).rectangle((190, 140, 322, 400), fill="#5a3621")
        mask = np.zeros((512, 512), np.float32)
        mask[140:401, 190:361] = 1  # the matte also takes a slab of backdrop at the subject's right
        depth = self.depth_map((512, 512), near=((190, 140, 322, 400),))
        asked = []

        def depth_for(indices):
            asked.append(list(indices))
            return [depth for _ in indices]

        load = lambda idx: [mask for _ in idx]
        plain, _, layout = sprite.strips([frame] * 39, 7, masks_for=load, canvas="square")
        deep, _, _ = sprite.strips([frame] * 39, 7, layout, load, canvas="square", depth_for=depth_for)
        self.assertEqual(len(asked), 1)
        self.assertLess((np.asarray(deep)[..., 3] > 128).sum(), (np.asarray(plain)[..., 3] > 128).sum() * .9)

    def test_alpha_reaches_the_sheet_node_and_is_validated(self):
        self.assertEqual(sprite.settings({"direction": "S", "alpha": "hard"})["alpha"], "hard")
        with self.assertRaisesRegex(ValueError, "soft or hard"):
            sprite.settings({"direction": "S", "alpha": "crisp"})


if __name__ == "__main__":
    unittest.main()
