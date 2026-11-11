import unittest
from policy import (
    audio_ticks,
    audio_window,
    bridge_spans,
    frame_to_step,
    hold_curve,
    HYPERFLOW_DROPPED,
    dotted_lora_key,
    editor_id,
    hyperflow_lora_key,
    premapped_blocks,
    latent_frames,
    legal_length,
    merge_curves,
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

    def test_a_bridge_holds_both_ends(self):
        first, last = bridge_spans(73, 73, 141)
        self.assertEqual((first["insert"], first["hold"], first["source_start"]), (0, 39, 34))
        self.assertEqual((last["insert"], last["hold"], last["source_start"]), (102, 39, 0))
        self.assertEqual(first["generate_after"], 63)
        self.assertEqual(last["generate_before"], 63)
        self.assertTrue(last["audio_exact"])
        self.assertEqual(frame_to_step(last["insert"]) % 5, 0)

    def test_a_bridge_needs_room_between_its_holds(self):
        with self.assertRaises(ValueError):
            bridge_spans(73, 73, 73)

    def test_merging_curves_keeps_the_stronger_hold(self):
        first = hold_curve(6, 0, 2)
        last = hold_curve(6, 4, 2)
        self.assertEqual(merge_curves(first, last), [0, 0, 1, 1, 0, 0])
        self.assertEqual(merge_curves(None, last), last)

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

    def test_full_audio_aligned_to_the_take_ignores_the_held_video(self):
        span = video_span("extend_before", source_frames=124, length=192, hold_frames=39)
        window = audio_window("full", span, 192, 30.0, align="take")
        self.assertEqual((window["first"], window["ticks"], window["source_second"]), (0, 320, 0))

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



def block_of(key):
    return int(key.split("blocks.")[1].split(".")[0])


class PremappedBlocksTest(unittest.TestCase):
    def test_a_file_says_which_blocks_it_was_renumbered_around(self):
        self.assertEqual(premapped_blocks({"hyperflow.dropped_blocks": "6,21,22"}), HYPERFLOW_DROPPED[47])
        self.assertEqual(premapped_blocks({"hyperflow.dropped_blocks": " 9 "}), HYPERFLOW_DROPPED[49])
        for nothing in (None, {}, {"hyperflow.dropped_blocks": ""}, {"hyperflow.dropped_blocks": "six"}, {"hyperflow.variant": "fl2va"}):
            self.assertIsNone(premapped_blocks(nothing))


class EditorIdTest(unittest.TestCase):
    def test_a_pack_node_has_a_twin_id_of_its_own(self):
        self.assertEqual([editor_id(n) for n in ("H3Take", "H3Lora", "H3LivePreview")],
                         ["H3EditorTake", "H3EditorLora", "H3EditorLivePreview"])
        # A twin has no twin, and other packs' nodes keep their ids.
        self.assertEqual([editor_id(n) for n in ("H3EditorLora", "UNETLoader", "MiniMaxH3SigmaShift")],
                         ["H3EditorLora", "UNETLoader", "MiniMaxH3SigmaShift"])


class DottedLoraKeyTest(unittest.TestCase):
    LAYERS = {name.replace(".", "_"): name for name in (
        "blocks.3.attn.qkv_proj", "blocks.3.attn.out_proj", "blocks.3.mlp.fc1", "blocks.3.adaln_proj.linear",
        "token_refiner.blocks.0.mlp.fc2")}

    def test_a_trainers_key_takes_the_models_layer_name(self):
        for key, named in (
                ("lora_unet_blocks_3_attn_qkv_proj.lora_down.weight", "blocks.3.attn.qkv_proj.lora_down.weight"),
                ("lora_unet_blocks_3_attn_out_proj.alpha", "blocks.3.attn.out_proj.alpha"),
                ("lora_unet_blocks_3_adaln_proj_linear.lora_up.weight", "blocks.3.adaln_proj.linear.lora_up.weight"),
                ("lora_unet_token_refiner_blocks_0_mlp_fc2.lora_up.weight", "token_refiner.blocks.0.mlp.fc2.lora_up.weight")):
            self.assertEqual(dotted_lora_key(key, self.LAYERS), named)

    def test_other_keys_stay(self):
        for key in ("blocks.3.mlp.fc1.lora_A.weight", "lora_unet_blocks_99_mlp_fc1.alpha", "lora_te_text_model.alpha"):
            self.assertEqual(dotted_lora_key(key, self.LAYERS), key)


class HyperFlowLoraKeyTest(unittest.TestCase):
    REF2VA = HYPERFLOW_DROPPED[49]
    FL2VA = HYPERFLOW_DROPPED[47]

    def moved(self, block, dropped):
        key = hyperflow_lora_key("diffusion_model.blocks.%s.mlp.fc1.lora_A.weight" % block, dropped)
        return None if key is None else block_of(key)

    def test_on_ref2va_blocks_after_9_move_down_by_one(self):
        self.assertEqual([self.moved(b, self.REF2VA) for b in (8, 9, 10, 25, 49)], [8, None, 9, 24, 48])
        self.assertEqual(hyperflow_lora_key("lora_unet_blocks_10_attn_qkv_proj.lora_down.weight", self.REF2VA),
                         "lora_unet_blocks_9_attn_qkv_proj.lora_down.weight")

    def test_on_fl2va_blocks_move_past_6_21_and_22(self):
        self.assertEqual([self.moved(b, self.FL2VA) for b in (5, 6, 7, 20, 21, 22, 23, 25, 49)],
                         [5, None, 6, 19, None, None, 20, 22, 46])

    def test_every_kept_block_lands_on_its_own_layer(self):
        for count, dropped in HYPERFLOW_DROPPED.items():
            kept = [self.moved(b, dropped) for b in range(50) if b not in dropped]
            self.assertEqual(kept, list(range(count)))

    def test_other_layers_stay(self):
        for key in ("diffusion_model.token_refiner.blocks.0.mlp.fc1.lora_A.weight",
                    "lora_unet_token_refiner_blocks_1_mlp_fc1.alpha",
                    "diffusion_model.final_layer.video_out.lora_A.weight"):
            self.assertEqual(hyperflow_lora_key(key, self.FL2VA), key)


if __name__ == "__main__":
    unittest.main()
