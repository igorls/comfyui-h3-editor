import unittest
from policy import plan, legal_length, video_span


class _Still:
    shape = (1, 8, 8, 3)


class _Clip:
    shape = (22, 8, 8, 3)


class PolicyTest(unittest.TestCase):
    def test_legal_grid(self):
        self.assertTrue(legal_length(124))
        self.assertTrue(legal_length(56))
        self.assertFalse(legal_length(30))

    def test_soft_ref_is_not_a_pin(self):
        out = plan("start", start=_Still(), soft=(_Still(),), length=124)
        self.assertEqual([role for role, _ in out["pictures"]], ["start", "soft"])
        self.assertEqual(out["pins"], [("start", 0)])
        self.assertIn("<Picture 2> soft", out["labels"])

    def test_both_ends_are_the_first_two_pictures(self):
        out = plan("both", start=_Still(), end=_Still(), soft=(_Still(),), audio=object(), length=73)
        self.assertEqual(out["pins"], [("start", 0), ("end", 72)])
        self.assertEqual(out["labels"][0], "<Picture 1> start")
        self.assertEqual(out["labels"][1], "<Picture 2> end")
        self.assertEqual(out["labels"][-1], "<Audio 1> timbre")
        self.assertEqual(out["length_note"], "short")

    def test_a_start_pin_cannot_be_a_clip(self):
        with self.assertRaises(ValueError) as caught:
            plan("start", start=_Clip(), length=124)
        self.assertIn("held span", str(caught.exception))

    def test_extend_after_holds_the_tail_and_generates_the_rest(self):
        span = video_span("extend_after", source_frames=124, length=192, hold_frames=39)
        self.assertEqual(span["insert"], 0)
        self.assertEqual(span["hold"], 39)
        self.assertEqual(span["generate_before"], 0)
        self.assertEqual(span["generate_after"], 153)
        self.assertEqual(span["slice_from"], "tail")
        self.assertEqual(span["keep"], 0.0)

    def test_extend_before_holds_the_head_at_the_end(self):
        span = video_span("extend_before", source_frames=124, length=192, hold_frames=39)
        self.assertEqual(span["hold"], 39)
        self.assertEqual(span["insert"], 153)
        self.assertEqual(span["generate_before"], 153)
        self.assertEqual(span["generate_after"], 0)
        self.assertEqual(span["slice_from"], "head")
        self.assertTrue(span["audio_exact"])

    def test_extend_both_leaves_room_on_each_side(self):
        span = video_span("extend_both", source_frames=124, length=243, hold_frames=90)
        self.assertEqual(span["hold"], 90)
        self.assertGreaterEqual(span["generate_before"], 5)
        self.assertGreaterEqual(span["generate_after"], 5)
        self.assertEqual(span["generate_before"] + span["hold"] + span["generate_after"], 243)
        self.assertEqual(span["slice_from"], "middle")

    def test_v2v_is_the_same_length_with_a_keep(self):
        span = video_span("v2v", source_frames=124, length=124, keep=0.4)
        self.assertEqual(span["hold"], 124)
        self.assertEqual(span["generate_before"], 0)
        self.assertEqual(span["generate_after"], 0)
        self.assertEqual(span["keep"], 0.4)

    def test_v2v_rejects_a_different_length(self):
        with self.assertRaises(ValueError) as caught:
            video_span("v2v", source_frames=124, length=192)
        self.assertIn("extend_both", str(caught.exception))

    def test_vram_picks_a_trained_span(self):
        from policy import max_block_for_vram, split_span
        self.assertEqual(max_block_for_vram(16), 124)
        self.assertEqual(max_block_for_vram(32), 243)
        self.assertEqual(max_block_for_vram(96), 362)

    def test_the_default_after_region_splits_on_a_16gb_mac(self):
        from policy import split_span
        blocks = split_span(243, max_block=124, overlap=39)
        self.assertEqual([b["length"] for b in blocks], [124, 124, 73])
        self.assertGreaterEqual(blocks[-1]["start"] + blocks[-1]["length"], 243)

    def test_an_off_grid_region_fails_instead_of_hanging(self):
        from policy import split_span
        with self.assertRaises(ValueError) as caught:
            split_span(248, max_block=124, overlap=39)
        self.assertIn("remaining", str(caught.exception))

    def test_a_long_take_becomes_sequential_passes(self):
        from policy import split_span
        blocks = split_span(600, max_block=243, overlap=39)
        self.assertGreater(len(blocks), 1)
        self.assertEqual(blocks[0]["hold"], 0)
        self.assertTrue(all(b["length"] <= 243 for b in blocks))
        self.assertTrue(all(b["hold"] == 39 for b in blocks[1:]))
        self.assertGreaterEqual(blocks[-1]["start"] + blocks[-1]["length"], 600)


if __name__ == "__main__":
    unittest.main()
