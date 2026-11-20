"""Unit tests for pure motion_replace helpers (runs locally on CPU without ComfyUI or GPU)."""

import unittest
import numpy as np

import motion_replace


class WindowRelayTest(unittest.TestCase):
    def test_snap_grid_frames_snaps_up_onto_5_plus_17k(self):
        self.assertEqual(motion_replace.snap_grid_frames(1), 5)
        self.assertEqual(motion_replace.snap_grid_frames(5), 5)
        self.assertEqual(motion_replace.snap_grid_frames(48), 56)
        self.assertEqual(motion_replace.snap_grid_frames(124), 124)
        self.assertEqual(motion_replace.snap_grid_frames(125), 141)
        self.assertEqual(motion_replace.snap_grid_frames(141), 141)

    def test_plan_relay_windows_single_and_multi_window(self):
        # Single window up to 141 frames (~5.88s)
        self.assertEqual(
            motion_replace.plan_relay_windows(120),
            [{"start": 0, "length": 124, "overlap": 0}],
        )
        self.assertEqual(
            motion_replace.plan_relay_windows(141),
            [{"start": 0, "length": 141, "overlap": 0}],
        )

        # 9.58s (230 frames) -> 2 windows: 124f + 141f (overlap 22f, stride 102f)
        w2 = motion_replace.plan_relay_windows(230)
        self.assertEqual(w2, [
            {"start": 0, "length": 124, "overlap": 0},
            {"start": 102, "length": 141, "overlap": 22},
        ])
        self.assertGreaterEqual(w2[-1]["start"] + w2[-1]["length"], 230)

        # 30s (720 frames) -> linear tiling covering all 720 frames with exact 22f overlap
        w30 = motion_replace.plan_relay_windows(720)
        self.assertEqual(len(w30), 7)
        self.assertEqual(w30[0]["overlap"], 0)
        for prev, nxt in zip(w30[:-1], w30[1:]):
            self.assertEqual(nxt["overlap"], 22)
            self.assertEqual(nxt["start"] - prev["start"], 102)
            self.assertEqual(prev["length"], 124)
        self.assertGreaterEqual(w30[-1]["start"] + w30[-1]["length"], 720)
        self.assertLessEqual(w30[-1]["length"], 141)
        self.assertEqual((w30[-1]["length"] - 5) % 17, 0)

    def test_plan_relay_windows_rejects_bad_stride(self):
        with self.assertRaisesRegex(ValueError, "multiple of 17"):
            motion_replace.plan_relay_windows(230, window_frames=124, overlap_frames=20)
        with self.assertRaisesRegex(ValueError, "multiple of 17"):
            motion_replace.plan_relay_windows(230, window_frames=22, overlap_frames=22)

    def test_join_window_frames_stitches_length_and_midpoint_crossfade(self):
        a = np.zeros((124, 4, 4, 3), dtype=np.float32)
        b = np.ones((141, 4, 4, 3), dtype=np.float32)
        joined = motion_replace.join_window_frames(a, b, overlap_frames=22, blend_frames=4)
        self.assertEqual(joined.shape, (124 + 141 - 22, 4, 4, 3))
        # Midpoint of 22f overlap is frame 111 (cut at 124 - 22 + 9 = 111)
        self.assertEqual(float(joined[110].max()), 0.0)
        # 4 blended frames have weights 0.2, 0.4, 0.6, 0.8
        for k, expected in enumerate((0.2, 0.4, 0.6, 0.8)):
            self.assertAlmostEqual(float(joined[111 + k, 0, 0, 0]), expected, places=5)
        self.assertEqual(float(joined[115].min()), 1.0)

        # Identical values across overlap stay exact; zero overlap concatenates cleanly
        same = motion_replace.join_window_frames(b[:40], b[:30], overlap_frames=22, blend_frames=4)
        self.assertEqual(same.shape[0], 48)
        self.assertAlmostEqual(float(np.abs(same - 1.0).max()), 0.0, places=6)
        concat = motion_replace.join_window_frames(a[:10], b[:12], overlap_frames=0)
        self.assertEqual(concat.shape[0], 22)


class LayerMapTest(unittest.TestCase):
    def test_maps_v1_and_v2_onto_49_block_flash(self):
        self.assertEqual(
            motion_replace.flash_controlnet_layer_map((0, 10, 20, 30, 40)),
            (0, 9, 19, 29, 39),
        )
        self.assertEqual(
            motion_replace.flash_controlnet_layer_map((0, 5, 10, 15, 20, 25, 30, 35, 40, 45)),
            (0, 5, 9, 14, 19, 24, 29, 34, 39, 44),
        )

    def test_rejects_pruned_block_9(self):
        with self.assertRaisesRegex(ValueError, "block 9"):
            motion_replace.flash_controlnet_layer_map((0, 9, 20))


class PoseGatingTest(unittest.TestCase):
    def test_empty_and_tiny_frames_get_zero_area_sentinel(self):
        bboxes = [
            [{"x": 20.0, "y": 40.0, "width": 120.0, "height": 300.0}],
            [],
            [{"x": 0.0, "y": 0.0, "width": 4.0, "height": 4.0}],
        ]
        gated, present = motion_replace.gate_bboxes(bboxes, 3, 672, 384, min_area_frac=0.005)
        self.assertEqual(present, [True, False, False])
        self.assertEqual(gated[0][0]["width"], 120.0)
        self.assertEqual(gated[1], [{"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0}])
        self.assertEqual(gated[2], [{"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0}])

    def test_handles_empty_list_and_single_dict_inputs(self):
        gated, present = motion_replace.gate_bboxes([], 2, 100, 100)
        self.assertEqual(present, [False, False])
        self.assertEqual(len(gated), 2)
        gated2, present2 = motion_replace.gate_bboxes({"x": 0, "y": 0, "width": 50, "height": 50}, 2, 100, 100)
        self.assertEqual(present2, [True, True])
        self.assertEqual(gated2[1][0]["width"], 50)

    def test_filters_low_confidence_ghost_people(self):
        good = [0.5, 0.5, 0.9] * 4 + [0.0, 0.0, 0.0] * 14
        ghost = [0.5, 0.5, 0.8] + [0.0, 0.0, 0.1] * 17
        frames = [
            {"people": [{"pose_keypoints_2d": good}]},
            {"people": [{"pose_keypoints_2d": ghost}]},
            {"people": []},
        ]
        out = motion_replace.filter_pose_keypoints(frames, min_valid_body_kps=2, score_thresh=0.3)
        self.assertEqual([len(f["people"]) for f in out], [1, 0, 0])
        self.assertIs(motion_replace.filter_pose_keypoints(frames, min_valid_body_kps=0), frames)


class MaskAndPlateTest(unittest.TestCase):
    def test_expand_mask_safe_dilates_present_and_keeps_exit_empty(self):
        masks = np.zeros((2, 64, 64), dtype=np.float32)
        masks[0, 30:34, 30:34] = 1.0
        expanded = motion_replace.expand_mask_safe(masks, expansion_px=6)
        self.assertGreater(expanded[0].sum(), masks[0].sum())
        self.assertEqual(expanded[1].max(), 0.0)
        # 2D input is promoted cleanly
        expanded_2d = motion_replace.expand_mask_safe(masks[0], expansion_px=4)
        self.assertEqual(expanded_2d.shape, (1, 64, 64))

    def test_harmonized_plate_eliminates_outer_halo_and_preserves_exit_frames(self):
        N, H, W = 2, 128, 128
        src = np.full((N, H, W, 3), 0.4, dtype=np.float32)
        gen = np.full((N, H, W, 3), 0.55, dtype=np.float32)
        m_src = np.zeros((N, H, W), dtype=np.float32)
        m_gen = np.zeros((N, H, W), dtype=np.float32)
        m_src[0, 40:96, 40:88] = 1.0
        m_gen[0, 40:96, 40:80] = 1.0
        gen[0, 40:96, 40:80] = [0.1, 0.8, 0.9]

        out = motion_replace.composite_plate_harmonized(
            gen, src, m_src, gen_mask_np=m_gen, src_expand_px=12, gen_expand_px=4, feather_sigma=2.0
        )
        # Far outside union: exact source plate
        self.assertAlmostEqual(float(np.abs(out[0, 0:20, 0:128] - src[0, 0:20, 0:128]).max()), 0.0, places=5)
        # Inside generated character: untouched cyan
        self.assertAlmostEqual(float(out[0, 68, 60, 1]), 0.8, places=2)
        # Uncovered background sliver: harmonized back near 0.40 source wall
        self.assertLess(float(np.abs(out[0, 68, 84, 0] - 0.4)), 0.04)
        # Exit frame 1: bit-exact source plate
        self.assertEqual(float(np.abs(out[1] - src[1]).max()), 0.0)

    def test_spatially_varying_harmonization_matches_dark_ceiling_and_bright_wall_locally(self):
        N, H, W = 1, 160, 160
        src = np.zeros((N, H, W, 3), dtype=np.float32)
        src[:, :80, :] = 0.2
        src[:, 80:, :] = 0.7
        gen = src.copy()
        gen[:, :80, :] += 0.10
        gen[:, 80:, :] -= 0.10
        m_src = np.zeros((N, H, W), dtype=np.float32)
        m_gen = np.zeros((N, H, W), dtype=np.float32)
        m_src[0, 60:110, 60:100] = 1.0
        m_gen[0, 72:100, 68:92] = 1.0
        gen[0, 72:100, 68:92] = [0.9, 0.2, 0.2]

        out = motion_replace.composite_plate_harmonized(
            gen, src, m_src, gen_mask_np=m_gen, src_expand_px=12, gen_expand_px=4, feather_sigma=2.0, harm_sigma=14.0
        )
        self.assertLess(float(out[0, 62, 80, 0]), 0.24)
        self.assertGreater(float(out[0, 106, 80, 0]), 0.66)
        self.assertAlmostEqual(float(out[0, 86, 80, 0]), 0.9, places=2)

    def test_harmonized_plate_handles_mismatched_resolution_and_none_gen_mask(self):
        src = np.full((2, 48, 32, 3), 0.3, dtype=np.float32)
        gen = np.full((3, 96, 64, 3), 0.7, dtype=np.float32)
        m_src = np.zeros((2, 48, 32), dtype=np.float32)
        m_src[:, 10:30, 8:24] = 1.0
        out = motion_replace.composite_plate_harmonized(gen, src, m_src, gen_mask_np=None)
        self.assertEqual(out.shape, (3, 96, 64, 3))
        self.assertAlmostEqual(float(out[0, 0, 0, 0]), 0.3, places=2)


class AutoStyleAndSwapGuardTest(unittest.TestCase):
    def test_classify_character_style_separates_cel_shaded_anime_from_textured_photo(self):
        # Cel-shaded 2D illustration: large flat blocks with sharp dark outlines
        cel = np.full((128, 128, 3), 210, dtype=np.uint8)
        cel[20:100, 30:98] = [240, 200, 180]
        cel[20:100, 30:32] = [20, 20, 20]
        cel[20:100, 96:98] = [20, 20, 20]
        cel[60:100, 32:96] = [40, 80, 200]
        self.assertEqual(motion_replace.classify_character_style(cel), "preserve")

        # Realistic photo: continuous shading + natural sensor/skin micro-texture
        rng = np.random.default_rng(7)
        grad = np.linspace(50, 190, 128, dtype=np.float32)[None, :, None]
        photo = np.clip(grad + rng.normal(0, 6.0, size=(128, 128, 3)), 0, 255).astype(np.uint8)
        self.assertEqual(motion_replace.classify_character_style(photo), "photoreal")

    def test_latent_swap_score_detects_non_swap_vs_swap_and_empty_mask(self):
        rng = np.random.default_rng(1)
        src = rng.normal(0, 1.0, size=(1, 4, 8, 6, 6)).astype(np.float32)
        mask = np.zeros((1, 1, 8, 6, 6), dtype=np.float32)
        self.assertIsNone(motion_replace.latent_swap_score(src, src, mask))
        mask[:, :, :, 2:4, 2:4] = 1.0
        stuck = src + rng.normal(0, 0.15, size=src.shape).astype(np.float32)
        swapped = src.copy()
        swapped[:, :, :, 2:4, 2:4] = rng.normal(0.5, 1.0, size=(1, 4, 8, 2, 2)).astype(np.float32)
        self.assertLess(motion_replace.latent_swap_score(stuck, src, mask), 0.3)
        self.assertGreater(motion_replace.latent_swap_score(swapped, src, mask), 0.8)

    def test_apply_overlap_mask_locks_head_and_ramps_final_overlap_step(self):
        mask = np.zeros((1, 1, 12, 4, 4), dtype=np.float32)
        mask[:, :, :, 1:3, 1:3] = 1.0
        out = motion_replace.apply_overlap_mask(mask, ov_steps=7, tail_ramp=0.45)
        self.assertEqual(float(out[:, :, :6].max()), 0.0)
        self.assertAlmostEqual(float(out[0, 0, 6, 1, 1]), 0.45, places=5)
        self.assertEqual(float(out[0, 0, 6, 0, 0]), 0.0)
        self.assertEqual(float(out[0, 0, 7, 1, 1]), 1.0)

    def test_reflect_pad_indices_mirrors_tail_instead_of_freezing(self):
        self.assertEqual(motion_replace.reflect_pad_indices(5, 8), [0, 1, 2, 3, 4, 3, 2, 1])
        self.assertEqual(motion_replace.reflect_pad_indices(1, 4), [0, 0, 0, 0])
        self.assertEqual(motion_replace.reflect_pad_indices(6, 4), [0, 1, 2, 3])


class PreprocessingCacheTest(unittest.TestCase):
    def test_mask_cache_merges_partial_splits_and_invalidates_on_file_change(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "dance.mp4")
            with open(src, "wb") as f:
                f.write(b"v1-video-bytes")
            m1 = np.zeros((10, 16, 12), dtype=np.float32)
            m1[:, 4:10, 3:8] = 1.0
            m2 = np.zeros((12, 16, 12), dtype=np.float32)
            m2[:, 2:6, 5:10] = 1.0

            self.assertFalse(motion_replace.has_mask_cache(src, 0, 10))
            motion_replace.save_mask_cache(src, 0, m1, total_frames=30)
            self.assertTrue(motion_replace.has_mask_cache(src, 0, 10))
            self.assertFalse(motion_replace.has_mask_cache(src, 8, 20))
            # Merge second overlapping split 8..20 -> contiguous coverage 0..20!
            motion_replace.save_mask_cache(src, 8, m2, total_frames=30)
            self.assertTrue(motion_replace.has_mask_cache(src, 0, 20))
            loaded = motion_replace.load_mask_cache(src, 2, 14)
            self.assertEqual(loaded.shape, (12, 16, 12))
            self.assertTrue(np.array_equal(loaded[:6], m1[2:8]))
            self.assertTrue(np.array_equal(loaded[6:], m2[:6]))
            # Modifying source file invalidates cache automatically.
            with open(src, "wb") as f:
                f.write(b"v2-modified-video-bytes-longer")
            self.assertFalse(motion_replace.has_mask_cache(src, 0, 10))
            self.assertIsNone(motion_replace.load_mask_cache(src, 0, 10))

    def test_pose_cache_merges_splits_and_handles_numpy_values(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "dance.mp4")
            with open(src, "wb") as f:
                f.write(b"video-bytes")
            kp1 = [{"people": [{"pose_keypoints_2d": np.array([1.5, 2.5, 0.9], dtype=np.float32)}]}] * 5
            kp2 = [{"people": [{"pose_keypoints_2d": [3.0, 4.0, 0.8]}]}] * 5
            motion_replace.save_pose_cache(src, 0, kp1, total_frames=20)
            motion_replace.save_pose_cache(src, 5, kp2, total_frames=20)
            self.assertTrue(motion_replace.has_pose_cache(src, 0, 10))
            self.assertFalse(motion_replace.has_pose_cache(src, 0, 12))
            sub = motion_replace.load_pose_cache(src, 3, 7)
            self.assertEqual(len(sub), 4)
            self.assertEqual(sub[0]["people"][0]["pose_keypoints_2d"][:2], [1.5, 2.5])
            self.assertEqual(sub[3]["people"][0]["pose_keypoints_2d"][:2], [3.0, 4.0])

    def test_cutout_cache_roundtrips_rgb_and_invalidates_on_threshold_or_source_change(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            char = os.path.join(tmp, "hero.png")
            with open(char, "wb") as f:
                f.write(b"png-bytes-1")
            rgb = np.full((32, 24, 3), 0.5, dtype=np.float32)
            rgb[8:24, 6:18] = [0.2, 0.6, 0.9]
            self.assertFalse(motion_replace.has_cutout_cache(char))
            motion_replace.save_cutout_cache(char, rgb, subject="person", backdrop="white", threshold=0.35)
            self.assertTrue(motion_replace.has_cutout_cache(char, subject="person", backdrop="white", threshold=0.35))
            self.assertFalse(motion_replace.has_cutout_cache(char, subject="person", backdrop="white", threshold=0.5))
            loaded = motion_replace.load_cutout_cache(char, subject="person", backdrop="white", threshold=0.35)
            self.assertEqual(loaded.shape, (1, 32, 24, 3))
            self.assertLess(float(np.abs(loaded[0] - rgb).max()), 0.01)

    def test_cache_summaries_and_preview_sheets(self):
        import io
        import os
        import tempfile
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "clip.mp4")
            with open(src, "wb") as f:
                f.write(b"video-data")
            m = np.zeros((6, 48, 36), dtype=np.float32)
            m[:, 12:36, 10:26] = 1.0
            motion_replace.save_mask_cache(src, 4, m, total_frames=20)
            m_sum = motion_replace.mask_cache_summary(src)
            self.assertEqual(m_sum["frames"], 20)
            self.assertEqual(m_sum["cached_frames"], 6)
            self.assertEqual(m_sum["spans"], [[4, 10]])
            self.assertGreater(m_sum["size_bytes"], 0)

            kp = [{"canvas_width": 36, "canvas_height": 48,
                   "people": [{"pose_keypoints_2d": [18, 10, 0.9, 18, 20, 0.9, 12, 20, 0.9, 10, 28, 0.9, 10, 34, 0.9,
                                                    24, 20, 0.9, 26, 28, 0.9, 26, 34, 0.9, 15, 32, 0.9, 15, 40, 0.9,
                                                    15, 46, 0.9, 21, 32, 0.9, 21, 40, 0.9, 21, 46, 0.9,
                                                    16, 8, 0.9, 20, 8, 0.9, 14, 9, 0.9, 22, 9, 0.9]}]}] * 4
            motion_replace.save_pose_cache(src, 0, kp, total_frames=20)
            p_sum = motion_replace.pose_cache_summary(src)
            self.assertEqual(p_sum["cached_frames"], 4)
            self.assertEqual(p_sum["detected_frames"], 4)
            self.assertEqual(p_sum["spans"], [[0, 4]])

            char = os.path.join(tmp, "hero.png")
            with open(char, "wb") as f:
                f.write(b"char-bytes")
            motion_replace.save_cutout_cache(char, np.full((24, 24, 3), 0.7, dtype=np.float32))
            c_sum = motion_replace.cutout_cache_summary(char)
            self.assertIsNotNone(c_sum)
            self.assertGreater(c_sum["size_bytes"], 0)

            # Verify both contact sheets render valid JPEGs
            rgbs = [np.full((48, 36, 3), 80, dtype=np.uint8) for _ in range(3)]
            mask_jpeg = motion_replace.render_mask_sheet(rgbs, [4, 6, 9], [m[0], m[2], m[5]], expand_px=6)
            img_m = Image.open(io.BytesIO(mask_jpeg))
            self.assertEqual(img_m.format, "JPEG")
            self.assertGreater(img_m.size[0], 100)

            pose_jpeg = motion_replace.render_pose_sheet(rgbs, [0, 1, 2], kp[:3])
            img_p = Image.open(io.BytesIO(pose_jpeg))
            self.assertEqual(img_p.format, "JPEG")
            self.assertGreater(img_p.size[0], 100)


if __name__ == "__main__":
    unittest.main()
