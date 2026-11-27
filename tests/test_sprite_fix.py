import ast
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw

try:
    import torch
    import latent_writer
except ImportError:
    torch = None
    latent_writer = None

import policy
import project
import sprite

ROOT = Path(__file__).resolve().parent.parent


class _StubNestedTensor:
    def __init__(self, tensors):
        self.tensors = tuple(tensors)
        self.is_nested = True

    def unbind(self):
        return self.tensors


def _comfy_patch():
    comfy_mod = types.ModuleType("comfy")
    nested_mod = types.ModuleType("comfy.nested_tensor")
    nested_mod.NestedTensor = _StubNestedTensor
    comfy_mod.nested_tensor = nested_mod
    return mock.patch.dict(sys.modules, {"comfy": comfy_mod, "comfy.nested_tensor": nested_mod})


def _empty_latent(steps=7, ticks=37, h=8, w=6):
    video = torch.zeros(1, 24, steps, h, w)
    audio = torch.zeros(1, 32, 2, ticks)
    return {"samples": _StubNestedTensor((video, audio))}


def _make_sprite_frames(count=39, size=(256, 384), bg="white"):
    frames = []
    for i in range(count):
        im = Image.new("RGB", size, bg)
        draw = ImageDraw.Draw(im)
        shift = (i % 7) * 2
        draw.rectangle((88 + shift, 96, 168 + shift, 280), fill=(40 + i * 3, 85, 160))
        frames.append(im)
    return frames


def _build_cached_sheet(tmp_dir, count=39, poses=7, size=(256, 384), canvas="portrait",
                        tile_size=(64, 64), idle=False):
    frames = _make_sprite_frames(count=count, size=size)
    cache_dir = os.path.join(tmp_dir, "c1_t1.frames")
    full, indexed, layout = sprite.strips(
        frames, poses, idle=idle, tile_size=tile_size, canvas=canvas, cache_dir=cache_dir,
    )
    stem = "c1_t1"
    full.save(os.path.join(tmp_dir, stem + ".illustrated.png"), format="PNG")
    indexed.save(os.path.join(tmp_dir, stem + ".indexed.png"), format="PNG")
    sheet = {
        "poses": full.width // tile_size[0],
        "source_poses": poses,
        "fps": sprite.playback_fps(layout, poses, idle),
        "tile": tile_size[0],
        "tile_size": list(tile_size),
        "pivot": [sprite.PIVOT[0] * tile_size[0] // sprite.TILE, sprite.PIVOT[1] * tile_size[1] // sprite.TILE],
        "layout": layout,
        "canvas": canvas,
        "frames": count,
        "cutout": "background key",
        "frame_cache": os.path.basename(cache_dir),
        "illustrated": stem + ".illustrated.png",
        "indexed": stem + ".indexed.png",
    }
    return frames, cache_dir, sheet, stem


@unittest.skipIf(torch is None, "needs torch")
class WriteHoldsSingleFrameTest(unittest.TestCase):
    def test_region_frame_pooled_opens_only_single_target_step(self):
        # 7 steps = 22 frames (spans 1, 4, 4, 4, 4, 1, 4); latent 8x6 = 128x96 px.
        region = torch.zeros(1, 128, 96)
        region[0, 10, 12] = 1.0  # top-left 32x32 cell -> latent [0:2, 0:2]
        with _comfy_patch():
            out = latent_writer.write_holds(
                _empty_latent(steps=7, h=8, w=6),
                video_latent=torch.ones(1, 24, 7, 8, 6),
                video_curve=[0.0] * 7,
                region=region,
                pooled_region=True,
                region_frame=7,  # step 2 (frames 5..8)
            )
        mask = out["noise_mask"].unbind()[0]
        self.assertEqual(tuple(mask.shape), (1, 1, 7, 8, 6))
        self.assertLessEqual(set(mask.unique().tolist()), {0.0, 1.0})
        target = policy.frame_to_step(7)
        self.assertEqual(target, 2)
        for s in range(7):
            if s == target:
                self.assertEqual(mask[0, 0, s, 0:2, 0:2].min().item(), 1.0)
                self.assertEqual(mask[0, 0, s, 2:, :].max().item(), 0.0)
                self.assertEqual(mask[0, 0, s, :, 2:].max().item(), 0.0)
            else:
                self.assertEqual(mask[0, 0, s].max().item(), 0.0, f"step {s} must stay frozen")

    def test_region_frame_clamps_and_handles_step_0_and_step_5(self):
        region = torch.ones(1, 128, 96)
        for frame, expected_step in ((0, 0), (17, 5), (9999, 6)):
            with _comfy_patch():
                out = latent_writer.write_holds(
                    _empty_latent(steps=7, h=8, w=6),
                    video_curve=[0.0] * 7,
                    region=region,
                    pooled_region=True,
                    region_frame=frame,
                )
            mask = out["noise_mask"].unbind()[0]
            for s in range(7):
                if s == expected_step:
                    self.assertEqual(mask[0, 0, s].min().item(), 1.0)
                else:
                    self.assertEqual(mask[0, 0, s].max().item(), 0.0)

    def test_region_frame_unpooled_preserves_continuous_values_on_target_step_only(self):
        region = torch.zeros(1, 128, 96)
        region[0, 64:, 48:] = 0.65
        with _comfy_patch():
            out = latent_writer.write_holds(
                _empty_latent(steps=7, h=8, w=6),
                video_curve=[0.0] * 7,
                region=region,
                pooled_region=False,
                region_frame=10,  # step 3
            )
        mask = out["noise_mask"].unbind()[0]
        self.assertAlmostEqual(mask[0, 0, 3, 6, 4].item(), 0.65, places=4)
        self.assertEqual(mask[0, 0, 3, 0, 0].item(), 0.0)
        for s in (0, 1, 2, 4, 5, 6):
            self.assertEqual(mask[0, 0, s].max().item(), 0.0)

    def test_region_frame_minus_one_preserves_all_step_behavior(self):
        region = torch.zeros(1, 128, 96)
        region[0, :32, :32] = 1.0
        with _comfy_patch():
            out = latent_writer.write_holds(
                _empty_latent(steps=7, h=8, w=6),
                video_curve=[0.0] * 7,
                region=region,
                pooled_region=True,
                region_frame=-1,
            )
        mask = out["noise_mask"].unbind()[0]
        for s in range(7):
            self.assertEqual(mask[0, 0, s, 0:2, 0:2].min().item(), 1.0)
            self.assertEqual(mask[0, 0, s, 2:, :].max().item(), 0.0)

    def test_nodes_h3take_defines_and_forwards_region_frame_and_pooled_region(self):
        src = (ROOT / "nodes.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        h3take = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "H3Take")
        methods = {n.name: n for n in h3take.body if isinstance(n, ast.FunctionDef)}
        exec_args = [a.arg for a in methods["execute"].args.args]
        hold_args = [a.arg for a in methods["hold"].args.args]
        self.assertIn("region_frame", exec_args)
        self.assertIn("pooled_region", exec_args)
        self.assertIn("region_frame", hold_args)
        self.assertIn("pooled_region", hold_args)
        schema_src = ast.get_source_segment(src, methods["define_schema"])
        self.assertIn('"region_frame"', schema_src)
        self.assertIn('"pooled_region"', schema_src)


class TileMaskProjectionTest(unittest.TestCase):
    def test_tile_mask_round_trips_through_render_cells(self):
        layout = {"start": 6, "height": 184.0, "center": 128.0, "ground": 280.0}
        for canvas, frame_size, tile_size in (
            ("portrait", (256, 384), (64, 64)),
            ("portrait", (512, 768), (128, 128)),
            ("square", (512, 512), (144, 176)),
        ):
            mask_tile = Image.new("L", (128, 128), 0)
            ImageDraw.Draw(mask_tile).rectangle((32, 24, 96, 104), fill=255)
            frame_mask = sprite.tile_mask_to_frame_mask(
                mask_tile, frame_size, layout, tile_size, canvas=canvas,
            )
            self.assertEqual(frame_mask.shape, (frame_size[1], frame_size[0]))
            self.assertEqual(frame_mask.dtype, np.float32)
            self.assertGreaterEqual(float(frame_mask.min()), 0.0)
            self.assertLessEqual(float(frame_mask.max()), 1.0)
            self.assertGreater(float(frame_mask.max()), 0.99)
            # Far corner outside tile crop must be strictly 0.0
            self.assertEqual(float(frame_mask[0, 0]), 0.0)

            # Render the projected mask back through _render_cells and check alignment
            rgb = (frame_mask * 255).round().astype(np.uint8)
            rgba = Image.fromarray(np.stack([rgb, rgb, rgb, np.full_like(rgb, 255)], axis=2), "RGBA")
            ref_w, ref_h = sprite.canvas_size({"canvas": canvas}, "draft")
            atlas = sprite._render_cells({0: rgba}, [0], layout, tile_size[0], tile_size[1], ref_w, ref_h)
            rendered_r = np.asarray(atlas)[:, :, 0]
            # Center of the painted tile box (50%, 50%) must be bright white (>240)
            self.assertGreater(int(rendered_r[tile_size[1] // 2, tile_size[0] // 2]), 240)
            # Corners of the tile (outside [32..96, 24..104] in 128x128) must be 0
            self.assertEqual(int(rendered_r[2, 2]), 0)
            self.assertEqual(int(rendered_r[tile_size[1] - 3, tile_size[0] - 3]), 0)

    def test_accepts_rgba_stroke_mask(self):
        layout = {"start": 6, "height": 184.0, "center": 128.0, "ground": 280.0}
        rgba_mask = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        ImageDraw.Draw(rgba_mask).rectangle((20, 20, 44, 44), fill=(255, 80, 80, 255))
        fm = sprite.tile_mask_to_frame_mask(rgba_mask, (256, 384), layout, (64, 64), canvas="portrait")
        self.assertGreater(float(fm.max()), 0.4)
        self.assertEqual(float(fm[0, 0]), 0.0)


class PoseCachePatchHelpersTest(unittest.TestCase):
    def test_pose_source_index_and_backup_idempotency(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, cache_dir, sheet, _ = _build_cached_sheet(tmp, idle=True)
            order, src, orig = sprite.pose_source_index(sheet, 0)
            self.assertEqual(len(order), 12)  # 7 poses ping-pong = 12
            self.assertEqual(src, orig)
            # Return leg pose 7 mirrors pose 5
            _, src5, _ = sprite.pose_source_index(sheet, 5)
            _, src7, _ = sprite.pose_source_index(sheet, 7)
            self.assertEqual(src5, src7)

            with self.assertRaises(ValueError):
                sprite.pose_source_index(sheet, 99)

            b1 = sprite.backup_pose_frame(cache_dir, src)
            orig_bytes = Path(b1).read_bytes()
            # Mutate cached file and call backup again: orig backup must NOT be overwritten
            Image.new("RGBA", (256, 384), (1, 2, 3, 255)).save(os.path.join(cache_dir, "f%04d.png" % src))
            b2 = sprite.backup_pose_frame(cache_dir, src)
            self.assertEqual(b1, b2)
            self.assertEqual(Path(b2).read_bytes(), orig_bytes)

    def test_apply_pose_alpha_erase_and_restore_with_defringe(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames, cache_dir, sheet, _ = _build_cached_sheet(tmp)
            _, src, _ = sprite.pose_source_index(sheet, 2)
            frame_path = os.path.join(cache_dir, "f%04d.png" % src)
            before = np.asarray(Image.open(frame_path).convert("RGBA"))
            self.assertGreater(int((before[:, :, 3] > 200).sum()), 1000)

            # Erase center of tile
            erase = Image.new("L", (64, 64), 0)
            ImageDraw.Draw(erase).rectangle((16, 16, 48, 48), fill=255)
            res = sprite.apply_pose_alpha(cache_dir, sheet, 2, erase_mask=erase)
            self.assertEqual(res["method"], "alpha")
            self.assertEqual(sheet["patched_poses"]["2"]["method"], "alpha")
            after_erase = np.asarray(Image.open(frame_path).convert("RGBA"))
            self.assertLess(int((after_erase[:, :, 3] > 200).sum()), int((before[:, :, 3] > 200).sum()))

            # Restore back using raw frame
            res2 = sprite.apply_pose_alpha(cache_dir, sheet, 2, restore_mask=erase, raw_frame=frames[src])
            self.assertEqual(res2["source_index"], src)
            after_restore = np.asarray(Image.open(frame_path).convert("RGBA"))
            self.assertGreater(int((after_restore[:, :, 3] > 200).sum()), int((after_erase[:, :, 3] > 200).sum()))

    def test_composite_pose_repair_bit_exact_outside_mask_and_full_replace(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, cache_dir, sheet, _ = _build_cached_sheet(tmp)
            _, src, _ = sprite.pose_source_index(sheet, 3)
            frame_path = os.path.join(cache_dir, "f%04d.png" % src)
            orig_arr = np.asarray(Image.open(frame_path).convert("RGBA"))

            repaired = Image.new("RGBA", (256, 384), (220, 40, 60, 255))
            mask_tile = Image.new("L", (64, 64), 0)
            ImageDraw.Draw(mask_tile).rectangle((22, 22, 42, 42), fill=255)
            proj = sprite.tile_mask_to_frame_mask(mask_tile, (256, 384), sheet["layout"], (64, 64))

            res = sprite.composite_pose_repair(cache_dir, sheet, 3, repaired, mask_tile=mask_tile, feather_px=3)
            self.assertEqual(res["method"], "ai")
            self.assertEqual(sheet["patched_poses"]["3"]["method"], "ai")
            comp = np.asarray(Image.open(frame_path).convert("RGBA"))

            # Every pixel outside the projected mask MUST be bit-exact equal to original!
            outside = proj == 0.0
            self.assertTrue(outside.any())
            np.testing.assert_array_equal(comp[outside], orig_arr[outside])

            # Deep inside the mask must match repaired color (220, 40, 60, 255)
            ys, xs = np.where(proj > 0.99)
            cy, cx = int(np.median(ys)), int(np.median(xs))
            self.assertGreater(int(comp[cy, cx, 0]), 200)
            self.assertLess(int(comp[cy, cx, 1]), 60)

            # Unmasked replacement replaces the entire frame
            full_rep = Image.new("RGBA", (256, 384), (10, 200, 30, 255))
            sprite.composite_pose_repair(cache_dir, sheet, 3, full_rep, mask_tile=None)
            comp_full = np.asarray(Image.open(frame_path).convert("RGBA"))
            np.testing.assert_array_equal(comp_full[0, 0], np.array([10, 200, 30, 255], dtype=np.uint8))

    def test_nudge_repack_and_revert_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, cache_dir, sheet, stem = _build_cached_sheet(tmp)
            orig_order, orig_src, _ = sprite.pose_source_index(sheet, 1)
            orig_frame_bytes = Path(os.path.join(cache_dir, "f%04d.png" % orig_src)).read_bytes()
            self.assertNotIn("indices", sheet["layout"])

            # Create stale resized variants that repack_take_from_cache must delete
            stale_files = [
                stem + ".128px.illustrated.png",
                stem + ".128px.indexed.png",
                stem + ".256px.illustrated.png",
                stem + ".128x160px.24fps.indexed.png",
            ]
            for sf in stale_files:
                Path(os.path.join(tmp, sf)).write_bytes(b"stale")
            keep_other = os.path.join(tmp, "other_take.128px.illustrated.png")
            Path(keep_other).write_bytes(b"keep")

            # Nudge pose 1 to orig_src + 1 with a distinct bright green silhouette
            new_idx = orig_src + 1
            nudged_img = Image.new("RGBA", (256, 384), (0, 0, 0, 0))
            ImageDraw.Draw(nudged_img).rectangle((90, 100, 170, 280), fill=(20, 230, 50, 255))
            nres = sprite.nudge_pose_frame(cache_dir, sheet, 1, new_idx, nudged_img)
            self.assertEqual(nres["source_index"], new_idx)
            self.assertEqual(nres["orig_index"], orig_src)
            self.assertEqual(sheet["layout"]["indices"][1], new_idx)
            self.assertEqual(sheet["patched_poses"]["1"]["method"], "nudge")

            # Repack should rewrite base strips and remove all stale *px* variants for stem
            rres = sprite.repack_take_from_cache(cache_dir, sheet, tmp, stem)
            self.assertEqual(sorted(rres["removed_variants"]), sorted(stale_files))
            for sf in stale_files:
                self.assertFalse(os.path.exists(os.path.join(tmp, sf)))
            self.assertTrue(os.path.exists(keep_other))
            self.assertTrue(os.path.isfile(os.path.join(tmp, stem + ".illustrated.png")))
            self.assertTrue(os.path.isfile(os.path.join(tmp, stem + ".indexed.png")))

            # Verify pose 1 cell in repacked strip shows the green nudged silhouette
            repacked = Image.open(os.path.join(tmp, stem + ".illustrated.png")).convert("RGBA")
            cell1 = np.asarray(repacked.crop((64, 0, 128, 64)))
            self.assertGreater(int(cell1[:, :, 1].max()), 200)

            # Revert pose 1 -> restores original index, cleans materialized indices, restores original frame
            rev = sprite.revert_pose_frame(cache_dir, sheet, 1)
            self.assertTrue(rev["reverted"])
            self.assertEqual(rev["source_index"], orig_src)
            self.assertNotIn("patched_poses", sheet)
            self.assertNotIn("indices", sheet["layout"])
            self.assertEqual(Path(os.path.join(cache_dir, "f%04d.png" % orig_src)).read_bytes(), orig_frame_bytes)
            _, restored_src, _ = sprite.pose_source_index(sheet, 1)
            self.assertEqual(restored_src, orig_src)


class PlanSpritePoseFixTest(unittest.TestCase):
    def _sample_project(self, cutout="foreground model"):
        proj = project.new_project("HeroSprites")
        proj["profile"] = "hyperflow"
        proj["cast"] = [{
            "id": "cast_1",
            "name": "Aria",
            "file": "aria_front.png",
            "subfolder": "h3take_projects/HeroSprites/refs",
            "refmod": {"file": "aria.safetensors", "strength": 0.9},
        }]
        proj["loras"] = [{"file": "pixel_style.safetensors", "strength": 0.7, "start_sigma": 0.9, "on": True}]
        clip, _ = project.add_clip(proj, {
            "action": "Aria jogging energetically.",
            "cast": ["cast_1"],
            "seconds": 2,
        })
        project.update_clip(proj, clip["id"], {
            "sprite": {"direction": "S", "camera": "rpg", "poses": 7, "canvas": "portrait", "frames": 39},
        })
        take = {
            "id": "t1",
            "file": "c1_t1.mp4",
            "subfolder": "h3take_projects/HeroSprites/clips",
            "quality": "final",
            "frames": 39,
            "size": [512, 768],
            "saved": {"action": clip["action"], "cast": list(clip["cast"]), "sprite": dict(clip["sprite"])},
            "sprite": {
                "poses": 7,
                "source_poses": 7,
                "fps": 12,
                "tile": 64,
                "tile_size": [64, 64],
                "pivot": [32, 59],
                "layout": {"start": 6, "height": 180.0, "center": 128.0, "ground": 280.0},
                "canvas": "portrait",
                "frames": 39,
                "cutout": cutout,
            },
        }
        return proj, clip, take

    def test_masked_and_unmasked_pose_fix_graphs_and_windows(self):
        proj, clip, take = self._sample_project(cutout="foreground model")
        avail = {
            project.PROFILES["hyperflow"]["unet"][0],
            project.MODEL_CHOICES["clip"][0],
            project.MODEL_CHOICES["vae"][0],
            project.MODEL_CHOICES["audio_vae"][0],
            "pixel_style.safetensors",
            project.REFMOD_PREFIX + "aria.safetensors",
            sprite.MATTE_PREFIX + "birefnet.safetensors",
        }

        # Pose 0 -> source_index = 6 -> window [0, 22), local_frame = 6
        plan_masked = project.plan_sprite_pose_fix(
            proj, "HeroSprites", clip, take, 0, "fix left hand fingers",
            mask_asset={"file": "mask_p0.png", "subfolder": "h3take_projects/HeroSprites/tmp"},
            strength="full", seed=1234, available=avail,
        )
        self.assertEqual(plan_masked["seed"], 1234)
        self.assertEqual(plan_masked["window"], {"start": 0, "length": 22, "local_frame": 6, "source_index": 6})
        self.assertEqual(plan_masked["save_node"], "25")
        g = plan_masked["graph"]

        # Exact model, LoRA, RefMod, cast ref reuse
        self.assertEqual(g["1"]["class_type"], "H3HyperFlowLoader")
        self.assertEqual(g["100"]["class_type"], "H3Lora")
        self.assertEqual(g["100"]["inputs"]["lora_name"], "pixel_style.safetensors")
        self.assertEqual(g["72"]["class_type"], "MiniMaxH3RefModsLoader")
        self.assertEqual(g["73"]["class_type"], "MiniMaxH3RefModApply")
        self.assertEqual(g["50"]["class_type"], "H3LoadStill")
        self.assertEqual(g["10"]["inputs"]["ref_images.ref_image_0"], ["50", 0])

        # Windowed H3LoadClip + H3Take single-frame spatial+temporal hold
        self.assertEqual(g["6"]["class_type"], "H3LoadClip")
        self.assertEqual(g["6"]["inputs"]["first"], 0)
        self.assertEqual(g["6"]["inputs"]["last"], 22)
        self.assertTrue(g["6"]["inputs"]["frames_only"])
        self.assertEqual(g["7"]["class_type"], "H3LoadStill")
        self.assertEqual(g["8"]["class_type"], "ImageToMask")
        t_in = g["10"]["inputs"]
        self.assertEqual(t_in["width"], 512)
        self.assertEqual(t_in["height"], 768)
        self.assertEqual(t_in["length"], 22)
        self.assertEqual(t_in["video_mode"], "v2v")
        self.assertEqual(t_in["hold_frames"], 22)
        self.assertEqual(t_in["strength"], "hold")
        self.assertEqual(t_in["keep"], 0.0)
        self.assertEqual(t_in["align"], "video")
        self.assertEqual(t_in["audio_hold"], "off")
        self.assertEqual(t_in["source_video"], ["6", 0])
        self.assertEqual(t_in["source_hold"], ["6", 2])
        self.assertEqual(t_in["region"], ["8", 0])
        self.assertEqual(t_in["region_frame"], 6)
        self.assertTrue(t_in["pooled_region"])
        self.assertIn("Fix frame: fix left hand fingers", t_in["prompt"])
        self.assertIn("Single full-body game character", t_in["prompt"])

        # Single-frame decode + BiRefNet cutout + SaveImage; NO audio decode or SaveVideo
        self.assertEqual(g["16"]["class_type"], "VAEDecode")
        self.assertEqual(g["17"]["class_type"], "ImageFromBatch")
        self.assertEqual(g["17"]["inputs"]["batch_index"], 6)
        self.assertEqual(g["17"]["inputs"]["length"], 1)
        self.assertEqual(g["22"]["class_type"], "LoadBackgroundRemovalModel")
        self.assertEqual(g["23"]["class_type"], "RemoveBackground")
        self.assertEqual(g["24"]["class_type"], "H3DefringeMatte")
        self.assertEqual(g["25"]["class_type"], "SaveImage")
        self.assertEqual(g["25"]["inputs"]["images"], ["24", 0])
        self.assertNotIn("21", g)  # full strength skips SplitSigmas
        for forbidden in ("VAEDecodeAudio", "CreateVideo", "SaveVideo", "H3SaveHold"):
            self.assertNotIn(forbidden, {node["class_type"] for node in g.values()})

        # Pose 6 -> source_index = 18 -> window [17, 39), local_frame = 1, unmasked SolidMask, strength="medium"
        plan_unmasked = project.plan_sprite_pose_fix(
            proj, "HeroSprites", clip, take, 6, "redraw boot cleanly",
            mask_asset=None, strength="medium", seed=5678, available=avail,
        )
        self.assertEqual(plan_unmasked["window"], {"start": 17, "length": 22, "local_frame": 1, "source_index": 18})
        g2 = plan_unmasked["graph"]
        self.assertNotIn("7", g2)
        self.assertEqual(g2["8"]["class_type"], "SolidMask")
        self.assertEqual(g2["8"]["inputs"], {"value": 1.0, "width": 512, "height": 768})
        self.assertEqual(g2["10"]["inputs"]["region_frame"], 1)
        self.assertEqual(g2["17"]["inputs"]["batch_index"], 1)
        self.assertEqual(g2["21"]["class_type"], "SplitSigmas")
        self.assertEqual(g2["15"]["inputs"]["sigmas"], ["21", 1])

    def test_background_key_skips_birefnet_and_strength_validation(self):
        proj, clip, take = self._sample_project(cutout="background key")
        plan = project.plan_sprite_pose_fix(proj, "HeroSprites", clip, take, 2, "fix cape", strength="heavy", seed=9)
        g = plan["graph"]
        self.assertNotIn("22", g)
        self.assertNotIn("23", g)
        self.assertNotIn("24", g)
        self.assertEqual(g["25"]["inputs"]["images"], ["17", 0])
        self.assertEqual(g["21"]["class_type"], "SplitSigmas")
        self.assertEqual(g["21"]["inputs"]["step"], 1)

        with self.assertRaises(ValueError):
            project.plan_sprite_pose_fix(proj, "HeroSprites", clip, take, 2, "fix cape", strength="bogus")


if __name__ == "__main__":
    unittest.main()
