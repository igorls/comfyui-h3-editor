import unittest
from policy import (
    audio_ticks,
    audio_window,
    frame_to_step,
    hold_curve,
    latent_frames,
    legal_length,
    parse_frames,
    plan,
    resolve_keep,
    video_span,
    video_steps,
)


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
        out = plan("both", start=_Still(), end=_Still(), soft=(_Still(),), ref_audios=1, length=73)
        self.assertEqual(out["pins"], [("start", 0), ("end", 72)])
        self.assertEqual(out["labels"][1], "<Picture 1> start")
        self.assertEqual(out["labels"][2], "<Picture 2> end")
        self.assertEqual(out["labels"][-1], "<Audio 1> reference")
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

    def test_named_strengths_and_custom(self):
        self.assertEqual(resolve_keep("hold"), 0.0)
        self.assertEqual(resolve_keep("medium"), 0.99)
        self.assertEqual(resolve_keep("regenerate"), 1.0)
        self.assertEqual(resolve_keep("custom", 0.42), 0.42)
        with self.assertRaises(ValueError):
            resolve_keep("loud")

    def test_text_alone_is_a_legal_take(self):
        out = plan("none", length=124)
        self.assertEqual(out["pins"], [])
        self.assertEqual(out["pictures"], [])

    def test_soundtracks_number_before_standalone_audio(self):
        out = plan("none", ref_videos=[True, False], ref_audios=1, length=124)
        self.assertIn("<Audio 1> soundtrack of <Video 1>", out["labels"])
        self.assertIn("<Video 2> reference", out["labels"])
        self.assertIn("<Audio 2> reference", out["labels"])

    def test_fl2va_refuses_references(self):
        with self.assertRaises(ValueError) as caught:
            plan("start", start=_Still(), soft=(_Still(),), checkpoint="fl2va", length=124)
        self.assertIn("ref2va", str(caught.exception))

    def test_mid_pins_take_no_picture_number(self):
        out = plan("start", start=_Still(), mids=(_Still(), _Still()), mid_frames=[34, 68], length=124)
        self.assertEqual(len(out["pictures"]), 1)
        self.assertEqual(out["pins"], [("start", 0), ("mid 1", 34), ("mid 2", 68)])

    def test_mid_pins_need_one_frame_each(self):
        with self.assertRaises(ValueError):
            plan("none", mids=(_Still(),), mid_frames=[], length=124)
        with self.assertRaises(ValueError):
            plan("none", mids=(_Still(),), mid_frames=[124], length=124)

    def test_parse_frames(self):
        self.assertEqual(parse_frames("30, 60 90"), [30, 60, 90])
        self.assertEqual(parse_frames(""), [])
        with self.assertRaises(ValueError):
            parse_frames("thirty")

    def test_latent_grid_matches_the_native_formula(self):
        self.assertEqual(video_steps(5), 2)
        self.assertEqual(video_steps(22), 7)
        self.assertEqual(video_steps(124), 37)
        self.assertEqual(frame_to_step(0), 0)
        self.assertEqual(frame_to_step(1), 1)
        self.assertEqual(frame_to_step(4), 1)
        self.assertEqual(frame_to_step(5), 2)
        self.assertEqual(frame_to_step(16), 4)
        self.assertEqual(frame_to_step(17), 5)
        self.assertEqual(frame_to_step(153), 45)

    def test_every_legal_insert_starts_a_step_group(self):
        for insert in range(0, 340, 17):
            self.assertEqual(frame_to_step(insert) % 5, 0)

    def test_audio_clock(self):
        self.assertEqual(audio_ticks(39), 65)
        self.assertEqual(audio_ticks(124), 207)

    def test_span_audio_follows_the_held_video(self):
        span = video_span("extend_after", source_frames=124, length=192, hold_frames=39)
        window = audio_window("span", span, 192, 124 / 24)
        self.assertEqual(window["first"], 0)
        self.assertEqual(window["ticks"], 65)
        self.assertAlmostEqual(window["source_second"], 85 / 24)

    def test_full_audio_lines_up_with_the_held_video(self):
        span = video_span("extend_before", source_frames=124, length=192, hold_frames=39)
        window = audio_window("full", span, 192, 124 / 24)
        self.assertEqual(window["first"], 255)
        self.assertEqual(window["ticks"], 65)
        self.assertLess(abs(window["source_second"]), 1 / 40)

    def test_full_audio_without_video_starts_at_zero_and_stops_with_the_track(self):
        window = audio_window("full", None, 124, 2.0)
        self.assertEqual((window["first"], window["ticks"]), (0, 80))

    def test_audio_hold_needs_audio(self):
        with self.assertRaises(ValueError):
            audio_window("full", None, 124, 0)
        with self.assertRaises(ValueError):
            audio_window("span", None, 124, 2.0)
        self.assertIsNone(audio_window("off", None, 124, 0))

    def test_hold_curve_holds_and_generates(self):
        values = hold_curve(10, 2, 4, base=0.0)
        self.assertEqual(values, [1, 1, 0, 0, 0, 0, 1, 1, 1, 1])

    def test_hold_curve_feathers_only_edges_that_meet_generated_material(self):
        values = hold_curve(10, 0, 6, base=0.0, feather=2)
        self.assertEqual(values[0], 0.0)
        self.assertGreater(values[5], values[4])
        self.assertGreater(values[4], 0.0)
        self.assertLess(values[5], 1.0)
        self.assertEqual(values[6], 1.0)

    def test_hold_curve_respects_a_partial_keep(self):
        self.assertEqual(hold_curve(4, 1, 2, base=0.5), [1.0, 0.5, 0.5, 1.0])

    def test_source_window_starts_where_the_mode_says(self):
        self.assertEqual(video_span("extend_after", 124, 192, hold_frames=39)["source_start"], 85)
        self.assertEqual(video_span("extend_before", 124, 192, hold_frames=39)["source_start"], 0)
        self.assertEqual(video_span("extend_both", 124, 243, hold_frames=90)["source_start"], 17)
        self.assertEqual(video_span("v2v", 124, 124)["source_start"], 0)

    def test_latent_frames_inverts_video_steps(self):
        for frames in (5, 22, 39, 124, 243, 498):
            self.assertEqual(latent_frames(video_steps(frames)), frames)


if __name__ == "__main__":
    unittest.main()
