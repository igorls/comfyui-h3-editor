import unittest

from policy import bridge_spans, legal_length, video_span
from project import (MODELS, active_take, clip_prompt, export_command, frame_of, item_length, linked_length, new_project,
                     render_graph, render_models, slug, snap_frames, tempo_chain)
from prompting import compose_prompt, compose_reference_prompt, tag_subjects


def clip(**extra):
    base = {"id": "c1", "action": "A keeper walks to the window.", "seconds": 5, "takes": []}
    base.update(extra)
    return base


class PromptTest(unittest.TestCase):
    def test_three_core_fields_in_order(self):
        text = compose_prompt("Live-action, cinematic", "A keeper walks.")
        self.assertTrue(text.startswith("integrated_multimodal_description: [Shot 1] Live-action, cinematic, A keeper walks."))
        self.assertLess(text.index("overall_soundscape:"), text.index("non_diegetic_music:"))
        self.assertTrue(text.endswith("non_diegetic_music: N/A"))

    def test_dialogue_uses_the_speaker_format(self):
        text = compose_prompt("", "He turns.", speaker="The old keeper with a low voice", dialogue="Who goes there?")
        self.assertIn("The old keeper with a low voice (S1) says: <d>[English] Who goes there?</d>", text)

    def test_a_first_frame_adds_the_instruction_line(self):
        text = compose_prompt("", "He turns.", first_frame=True)
        self.assertTrue(text.startswith("For the target video, at 0.00 seconds"))
        self.assertIn("\n\nintegrated_multimodal_description:", text)


class ReferencePromptTest(unittest.TestCase):
    tom = {"name": "Tom", "description": "the elderly keeper with a white beard"}
    sea = {"name": "Sea", "description": "the black storm sea"}

    def test_six_sections_in_order(self):
        text = compose_reference_prompt("Cinematic", "@Tom walks.", subjects=[self.tom])
        names = ["subject_definitions:", "summary:", "retention_analysis:", "detailed_description:", "overall_soundscape:", "non_diegetic_music:"]
        self.assertEqual(sorted(text.index(n) for n in names), [text.index(n) for n in names])
        self.assertIn("[reference generation] <Subject 1> walks.", text)
        self.assertIn("<Subject 1> is the elderly keeper with a white beard in <Picture 1>.", text)

    def test_frames_take_the_first_picture_numbers(self):
        text = compose_reference_prompt("", "@Tom walks.", subjects=[self.tom], first_frame=True, last_frame=True)
        self.assertIn("<Picture 1> is the first frame of [Shot 1].", text)
        self.assertIn("<Picture 2> is the last frame of [Shot 1].", text)
        self.assertIn("in <Picture 3>.", text)
        self.assertIn("The shot begins from <Picture 1>.", text)
        self.assertIn("The shot ends on <Picture 2>.", text)
        self.assertIn("[keyframe completion + reference generation]", text)

    def test_an_unmentioned_subject_is_still_placed_in_the_shot(self):
        text = compose_reference_prompt("", "Rain falls.", subjects=[self.tom])
        self.assertIn("[Shot 1] <Subject 1> appears in the shot. Rain falls.", text)

    def test_names_become_tags_whatever_the_case(self):
        self.assertEqual(tag_subjects("@tom and @SEA", [self.tom, self.sea]), "<Subject 1> and <Subject 2>")

    def test_a_speaker_can_be_a_subject(self):
        text = compose_reference_prompt("", "@Tom turns.", subjects=[self.tom], speaker="@Tom", dialogue="Who goes there?")
        self.assertIn("<Subject 1> (S1) says: <d>[English] Who goes there?</d>", text)


class ProjectTest(unittest.TestCase):
    def test_slug_keeps_paths_inside_the_folder(self):
        self.assertEqual(slug("My Scene"), "My_Scene")
        self.assertEqual(slug("../../etc"), "etc")
        with self.assertRaises(ValueError):
            slug("../..")

    def test_lengths_snap_to_the_h3_grid(self):
        self.assertEqual(snap_frames(5), 124)
        self.assertEqual((snap_frames(3.3) - 5) % 17, 0)
        self.assertEqual(snap_frames(0.1), 22)

    def test_a_clip_renders_to_its_own_file(self):
        graph = render_graph(new_project("x"), "x", clip(), "t1", "draft", 3)
        self.assertEqual(graph["19"]["inputs"]["filename_prefix"], "h3take_projects/x/clips/c1_t1")
        self.assertEqual(graph["10"]["inputs"]["width"], 640)
        self.assertEqual(graph["12"]["inputs"]["steps"], 4)
        self.assertIn("integrated_multimodal_description", graph["10"]["inputs"]["prompt"])

    def test_a_clip_renders_with_the_moving_preview(self):
        graph = render_graph(new_project("x"), "x", clip(), "t1", "draft", 3)
        self.assertEqual(graph["13"]["inputs"]["model"], ["9", 0])
        self.assertEqual(graph["9"]["inputs"]["folder"], "h3take_projects/x/preview")

    def test_a_prompt_only_clip_renders_on_hyperflow_fl2va(self):
        graph = render_graph(new_project("x"), "x", clip(), "t1", "draft", 3)
        self.assertEqual(graph["1"]["class_type"], "HyperFlowGGUFLoader")
        self.assertEqual(graph["1"]["inputs"]["unet_name"], MODELS["unet_text"])
        self.assertEqual(graph["10"]["inputs"]["checkpoint"], "fl2va")
        self.assertEqual(graph["12"], {"class_type": "HyperFlowSigmas", "inputs": {"model": ["5", 0], "steps": 4}})

    def test_a_clip_with_references_renders_on_hyperflow_ref2va(self):
        graph = render_graph(new_project("x"), "x", clip(first={"file": "f.png"}), "t1", "draft", 3)
        self.assertEqual(graph["1"]["inputs"]["unet_name"], MODELS["unet"])
        self.assertEqual(graph["10"]["inputs"]["checkpoint"], "ref2va")

    def test_a_final_is_doubled_by_the_upscaler(self):
        graph = render_graph(new_project("x"), "x", clip(), "t1", "final", 3)
        self.assertEqual(graph["12"]["inputs"]["steps"], 6)
        self.assertEqual(graph["22"]["inputs"]["images"], ["16", 0])
        self.assertEqual(graph["27"]["class_type"], "VSRDecode")
        self.assertEqual(graph["18"]["inputs"]["images"], ["27", 0])
        self.assertEqual(graph["18"]["inputs"]["audio"], ["17", 0])
        self.assertEqual(render_models(graph)["upscaler"], MODELS["upscaler"])

    def test_a_draft_is_not_upscaled(self):
        graph = render_graph(new_project("x"), "x", clip(), "t1", "draft", 3)
        self.assertNotIn("27", graph)
        self.assertEqual(graph["18"]["inputs"]["images"], ["16", 0])

    def test_a_project_that_names_a_retired_model_renders_with_the_current_one(self):
        project = new_project("x")
        project["models"] = {"unet": "fasth3_omniref_qad_nvfp4_comfy.safetensors", "clip": MODELS["clip"]}
        graph = render_graph(project, "x", clip(first={"file": "f.png"}), "t1", "draft", 3)
        self.assertEqual(graph["1"]["class_type"], "HyperFlowGGUFLoader")
        self.assertEqual(graph["1"]["inputs"]["unet_name"], MODELS["unet"])

    def test_a_safetensors_unet_keeps_the_plain_loader_and_scheduler(self):
        project = new_project("x")
        project["models"] = {"unet": "other_h3.safetensors"}
        graph = render_graph(project, "x", clip(), "t1", "draft", 3)
        self.assertEqual(graph["1"]["class_type"], "UNETLoader")
        self.assertEqual(graph["1"]["inputs"]["unet_name"], "other_h3.safetensors")
        self.assertEqual(graph["12"]["class_type"], "BasicScheduler")
        self.assertEqual(graph["10"]["inputs"]["checkpoint"], "ref2va")

    def test_an_empty_clip_is_not_rendered(self):
        with self.assertRaises(ValueError):
            render_graph(new_project("x"), "x", clip(action=" "), "t1", "draft", 3)

    def test_prompt_uses_the_project_style(self):
        project = new_project("x")
        project["style"] = "Claymation"
        self.assertIn("[Shot 1] Claymation, A keeper", clip_prompt(project, clip()))

    def test_active_take_falls_back_to_the_newest(self):
        takes = [{"id": "t1"}, {"id": "t2"}]
        self.assertEqual(active_take({"takes": takes, "active": "t1"})["id"], "t1")
        self.assertEqual(active_take({"takes": takes})["id"], "t2")
        self.assertIsNone(active_take({"takes": []}))


class CastTest(unittest.TestCase):
    asset = {"file": "a.png", "subfolder": "p/assets"}

    def project(self):
        project = new_project("x")
        project["cast"] = [
            {"id": "a", "name": "Tom", "description": "the keeper", **self.asset},
            {"id": "b", "name": "Sea", "description": "the sea", "file": "b.png", "subfolder": "p/assets"},
        ]
        return project

    def test_a_clip_only_gets_the_cast_it_chose(self):
        graph = render_graph(self.project(), "x", clip(cast=["b"]), "t1", "draft", 3)
        self.assertEqual(graph["50"]["inputs"]["file"], "p/assets/b.png")
        self.assertEqual(graph["10"]["inputs"]["ref_images.ref_image_0"], ["50", 0])
        self.assertNotIn("51", graph)
        self.assertIn("subject_definitions:", graph["10"]["inputs"]["prompt"])

    def test_no_references_keeps_the_three_field_prompt(self):
        graph = render_graph(self.project(), "x", clip(), "t1", "draft", 3)
        self.assertTrue(graph["10"]["inputs"]["prompt"].startswith("integrated_multimodal_description:"))

    def test_frames_set_the_anchor(self):
        for extra, anchor in (({"first": self.asset}, "start"), ({"last": self.asset}, "end"),
                              ({"first": self.asset, "last": self.asset}, "both"), ({}, "none")):
            graph = render_graph(self.project(), "x", clip(**extra), "t1", "draft", 3)
            self.assertEqual(graph["10"]["inputs"]["anchor"], anchor)
        graph = render_graph(self.project(), "x", clip(first=self.asset), "t1", "draft", 3)
        self.assertEqual(graph["10"]["inputs"]["start_image"], ["40", 0])

    def test_a_held_end_ignores_a_frame_there(self):
        self.assertIsNone(frame_of({"first": self.asset, "link": {"mode": "extend_after"}}, "first"))
        self.assertEqual(frame_of({"last": self.asset, "link": {"mode": "extend_after"}}, "last"), self.asset)
        self.assertIsNone(frame_of({"last": self.asset, "link": {"mode": "extend_before"}}, "last"))
        self.assertIsNone(frame_of({"first": self.asset, "link": {"mode": "bridge"}}, "first"))


class LinkTest(unittest.TestCase):
    side = {"file": "a.mp4", "subfolder": "p/clips", "first": 0, "last": 73}

    def test_extend_lengths_start_on_the_shared_boundary(self):
        for seconds in (1, 2.5, 4, 9):
            length = linked_length({"mode": "extend_after", "hold": 39}, seconds)
            self.assertTrue(legal_length(length))
            self.assertEqual((length - 39) % 51, 0)
            self.assertTrue(video_span("extend_before", 39, length, hold_frames=39)["audio_exact"])

    def test_bridge_lengths_put_the_far_hold_on_the_shared_boundary(self):
        for seconds in (1, 2, 4, 8):
            length = linked_length({"mode": "bridge", "hold": 39}, seconds)
            self.assertTrue(legal_length(length))
            self.assertTrue(bridge_spans(39, 39, length)[1]["audio_exact"])

    def test_extending_after_reads_the_tail_the_timeline_shows(self):
        link = {"mode": "extend_after", "hold": 39, "a": dict(self.side, last=60)}
        graph = render_graph(new_project("x"), "x", clip(link=link, seconds=2), "t1", "draft", 3)
        self.assertEqual(graph["6"]["inputs"], {"file": "p/clips/a.mp4", "first": 21, "last": 60})
        self.assertEqual(graph["10"]["inputs"]["video_mode"], "extend_after")
        self.assertEqual(graph["10"]["inputs"]["source_audio"], ["6", 1])
        self.assertEqual(graph["10"]["inputs"]["audio_hold"], "span")

    def test_extending_before_reads_the_head_the_timeline_shows(self):
        link = {"mode": "extend_before", "hold": 39, "a": dict(self.side, first=10)}
        graph = render_graph(new_project("x"), "x", clip(link=link, seconds=2), "t1", "draft", 3)
        self.assertEqual((graph["6"]["inputs"]["first"], graph["6"]["inputs"]["last"]), (10, 49))

    def test_a_bridge_reads_both_clips(self):
        link = {"mode": "bridge", "hold": 39, "a": self.side, "b": dict(self.side, file="b.mp4")}
        graph = render_graph(new_project("x"), "x", clip(link=link, seconds=3), "t1", "draft", 3)
        self.assertEqual(graph["6"]["inputs"]["first"], 34)
        self.assertEqual(graph["7"]["inputs"], {"file": "p/clips/b.mp4", "first": 0, "last": 39})
        self.assertEqual(graph["10"]["inputs"]["end_video"], ["7", 0])

    def test_a_bridge_is_guided_onto_its_arrival_frame(self):
        link = {"mode": "bridge", "hold": 39, "a": self.side, "b": dict(self.side, file="b.mp4")}
        graph = render_graph(new_project("x"), "x", clip(link=link, seconds=3), "t1", "draft", 3)
        length = graph["10"]["inputs"]["length"]
        self.assertEqual(graph["10"]["inputs"]["mid_frames"], str(length - 39))
        self.assertEqual(graph["10"]["inputs"]["mid_images.mid_image_0"], ["8", 0])
        self.assertEqual(graph["8"]["inputs"]["image"], ["7", 0])


class ExportTest(unittest.TestCase):
    def setUp(self):
        self.project = new_project("x")
        self.project["clips"] = [clip(id="a", takes=[{"id": "t1", "quality": "draft"}]), clip(id="b", takes=[{"id": "t1", "quality": "draft"}])]

    def filter_of(self, args):
        return args[args.index("-filter_complex") + 1]

    def test_clips_join_in_timeline_order_not_list_order(self):
        self.project["timeline"] = [
            {"clip": "b", "start": 100, "in": 0, "out": 50},
            {"clip": "a", "start": 0, "in": 10, "out": 110},
        ]
        args = export_command(self.project, {"a": "a.mp4", "b": "b.mp4"}, "o.mp4")
        self.assertLess(args.index("a.mp4"), args.index("b.mp4"))
        self.assertIn("trim=start_frame=10:end_frame=110", self.filter_of(args))
        self.assertIn("concat=n=2:v=1:a=1", self.filter_of(args))
        self.assertEqual(args[args.index("-map") + 1], "[k0v]")

    def test_a_gap_becomes_black_and_silence(self):
        self.project["timeline"] = [{"clip": "a", "start": 24, "in": 0, "out": 24}]
        text = self.filter_of(export_command(self.project, {"a": "a.mp4"}, "o.mp4"))
        self.assertIn("color=c=black", text)
        self.assertIn("anullsrc", text)
        self.assertIn("concat=n=2", text)

    def test_unrendered_clips_are_skipped(self):
        self.project["timeline"] = [{"clip": "a", "start": 0, "in": 0, "out": 24}, {"clip": "b", "start": 24, "in": 0, "out": 24}]
        args = export_command(self.project, {"a": "a.mp4"}, "o.mp4")
        self.assertNotIn("b.mp4", args)

    def test_speed_changes_timing_on_the_timeline_only(self):
        self.assertEqual(item_length({"in": 0, "out": 100, "speed": 2}), 50)
        self.assertEqual(item_length({"in": 0, "out": 100}), 100)
        self.project["timeline"] = [{"clip": "a", "start": 0, "in": 10, "out": 110, "speed": 2}]
        text = self.filter_of(export_command(self.project, {"a": "a.mp4"}, "o.mp4"))
        self.assertIn("trim=start_frame=10:end_frame=110,setpts=(PTS-STARTPTS)/2.00000", text)
        self.assertIn("atempo=2", text.replace("atempo=2.00000", "atempo=2"))

    def test_bad_speeds_are_refused_instead_of_looping(self):
        for bad in (0, -1, float("inf"), float("-inf"), float("nan"), 5, 0.1, "fast"):
            with self.assertRaises(ValueError):
                tempo_chain(bad)
            with self.assertRaises(ValueError):
                item_length({"in": 0, "out": 10, "speed": bad})
        self.project["timeline"] = [{"clip": "a", "start": 0, "in": 0, "out": 48, "speed": -2}]
        with self.assertRaises(ValueError):
            export_command(self.project, {"a": "a.mp4"}, "o.mp4")

    def test_large_speed_changes_chain_atempo(self):
        self.assertEqual(tempo_chain(1), "")
        self.assertEqual(tempo_chain(4), "atempo=2.0,atempo=2.00000")
        self.assertEqual(tempo_chain(0.25), "atempo=0.5,atempo=0.50000")

    def test_a_transition_crossfades_the_overlap(self):
        self.project["timeline"] = [
            {"clip": "a", "start": 0, "in": 0, "out": 48},
            {"clip": "b", "start": 36, "in": 0, "out": 48, "transition": {"type": "black", "frames": 12}},
        ]
        args = export_command(self.project, {"a": "a.mp4", "b": "b.mp4"}, "o.mp4")
        text = self.filter_of(args)
        self.assertIn("xfade=transition=fadeblack:duration=0.50000:offset=1.50000", text)
        self.assertIn("acrossfade=d=0.50000", text)
        self.assertNotIn("concat", text)

    def test_the_cut_takes_the_size_of_its_largest_picture(self):
        self.project["timeline"] = [{"clip": "a", "start": 0, "in": 0, "out": 24}, {"clip": "b", "start": 24, "in": 0, "out": 24}]
        text = self.filter_of(export_command(self.project, {"a": "a.mp4", "b": "b.mp4"}, "o.mp4"))
        self.assertIn("scale=640:352:", text)
        self.project["clips"][1]["takes"] = [{"id": "t2", "quality": "final", "upscale": 2}]
        text = self.filter_of(export_command(self.project, {"a": "a.mp4", "b": "b.mp4"}, "o.mp4"))
        self.assertIn("scale=2688:1536:", text)

    def test_nothing_to_export_says_so(self):
        with self.assertRaises(ValueError):
            export_command(self.project, {}, "o.mp4")


if __name__ == "__main__":
    unittest.main()
