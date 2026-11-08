import unittest

from policy import bridge_spans, legal_length, video_span
from project import (MODELS, PROFILES, REFINE_TAIL, active_take, add_clip, clip_prompt, export_command, finals_status,
                     frame_of, installed_profile, item_length, linked_length, new_project, pick_file, plan_render,
                     profile_status, project_models, render_graph, render_models, set_project, slug, snap_frames,
                     tempo_chain)

HYPERFLOW = set(PROFILES["hyperflow"]["unet"] + PROFILES["hyperflow"]["unet_text"])
BASE_TURBO = set(PROFILES["h3_turbo"]["unet"] + PROFILES["h3_turbo"]["unet_text"] + [PROFILES["h3_turbo"]["lora"][0]])
FINALS = {MODELS["latent_upscaler"], MODELS["lora"]}
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

    def test_a_final_is_refined_at_the_larger_size(self):
        project = new_project("x")
        project["format"] = "9:16"
        graph = render_graph(project, "x", clip(), "t1", "final", 3)
        self.assertEqual(graph["12"]["inputs"]["steps"], 6)
        self.assertEqual((graph["10"]["inputs"]["width"], graph["10"]["inputs"]["height"]), (768, 1344))
        self.assertEqual((graph["80"]["inputs"]["width"], graph["80"]["inputs"]["height"]), (1088, 1920))
        self.assertEqual(graph["82"]["inputs"]["samples"], ["15", 0])
        self.assertEqual((graph["82"]["inputs"]["width"], graph["82"]["inputs"]["height"]), (1088, 1920))
        self.assertEqual(graph["83"]["inputs"], {"upscaled": ["82", 0], "take": ["80", 1]})
        self.assertEqual(graph["87"]["inputs"], {"sigmas": ["12", 0], "step": 3})
        self.assertEqual(graph["88"]["inputs"]["conditioning"], ["80", 0])
        self.assertEqual(graph["90"]["inputs"]["sigmas"], ["87", 1])
        self.assertEqual(graph["18"]["inputs"]["images"], ["91", 0])
        # The sound is the first pass's.
        self.assertEqual(graph["18"]["inputs"]["audio"], ["17", 0])
        self.assertEqual(render_models(graph)["upscaler"], MODELS["latent_upscaler"])

    def test_a_final_samples_both_passes_with_the_realism_slider(self):
        graph = render_graph(new_project("x"), "x", clip(), "t1", "final", 3)
        self.assertEqual(graph["95"]["inputs"], {"model": ["1", 0], "lora_name": MODELS["lora"], "strength": 1.5})
        self.assertEqual(graph["5"]["inputs"]["model"], ["95", 0])
        used = render_models(graph)
        self.assertEqual((used["lora"], used["lora_strength"]), (MODELS["lora"], 1.5))

    def test_the_refine_take_holds_what_the_first_take_holds(self):
        track = {"file": "t.wav", "subfolder": "s", "from": 2.0}
        graph = render_graph(new_project("x"), "x", clip(track=track, first={"file": "f.png"}), "t1", "final", 3)
        first, refine = graph["10"]["inputs"], graph["80"]["inputs"]
        for key in ("audio_hold", "source_audio", "start_image", "anchor", "prompt"):
            self.assertEqual(refine[key], first[key])

    def test_each_profile_builds_its_own_graph(self):
        for name, profile in PROFILES.items():
            project = new_project("x")
            project["profile"] = name
            for quality in ("draft", "final"):
                with self.subTest(profile=name, quality=quality):
                    graph = render_graph(project, "x", clip(first={"file": "f.png"}), "t1", quality, 3)
                    unet = profile["unet"][0]
                    self.assertEqual(graph["1"]["inputs"]["unet_name"], unet)
                    self.assertEqual(graph["10"]["inputs"]["checkpoint"], "ref2va")
                    self.assertEqual(graph["11"]["inputs"]["sampler_name"], profile["sampler"])
                    sigmas = graph["12"]
                    if unet.endswith(".gguf"):
                        self.assertEqual(graph["1"]["class_type"], "HyperFlowGGUFLoader")
                        self.assertEqual(sigmas["class_type"], "HyperFlowSigmas")
                    else:
                        self.assertEqual(graph["1"]["class_type"], "UNETLoader")
                        self.assertEqual((sigmas["class_type"], sigmas["inputs"]["scheduler"]), ("BasicScheduler", "simple"))
                    self.assertEqual(sigmas["inputs"]["steps"], profile["steps"][quality])
                    if profile.get("lora"):
                        self.assertEqual(graph["94"]["inputs"]["model"], ["1", 0])
                        self.assertEqual(graph["94"]["inputs"]["lora_name"], profile["lora"][0])
                    else:
                        self.assertNotIn("94", graph)
                    if quality == "final":
                        self.assertEqual(graph["95"]["inputs"]["model"], ["94", 0] if profile.get("lora") else ["1", 0])
                        self.assertEqual(graph["5"]["inputs"]["model"], ["95", 0])
                        refine = graph["87"]
                        if profile.get("refine_sigmas"):
                            self.assertEqual(refine, {"class_type": "ManualSigmas", "inputs": {"sigmas": REFINE_TAIL}})
                            self.assertEqual(graph["90"]["inputs"]["sigmas"], ["87", 0])
                        else:
                            self.assertEqual(refine["inputs"], {"sigmas": ["12", 0], "step": profile["refine_from"]})
                            self.assertEqual(graph["90"]["inputs"]["sigmas"], ["87", 1])
                    else:
                        self.assertEqual(graph["5"]["inputs"]["model"], ["94", 0] if profile.get("lora") else ["1", 0])

    def test_a_prompt_only_clip_uses_the_familys_fl2va_or_else_its_ref2va(self):
        for name, checkpoint in (("hyperflow", "fl2va"), ("h3", "fl2va"), ("singularity", "ref2va")):
            project = new_project("x")
            project["profile"] = name
            graph = render_graph(project, "x", clip(), "t1", "draft", 3)
            expected = (PROFILES[name].get("unet_text") or PROFILES[name]["unet"])[0]
            self.assertEqual((graph["1"]["inputs"]["unet_name"], graph["10"]["inputs"]["checkpoint"]), (expected, checkpoint))

    def test_a_profiles_files_are_found_among_the_installed_ones(self):
        installed = {"DasiwaMinimaxH3_dasiwaHybridV4Turbo_123.safetensors", "minimax_h3_ref2va_pruned_int8_convrot.safetensors"}
        self.assertEqual(pick_file(PROFILES["dasiwa_turbo"]["unet"], installed), "DasiwaMinimaxH3_dasiwaHybridV4Turbo_123.safetensors")
        self.assertIsNone(pick_file(PROFILES["dasiwa"]["unet"], installed))
        rows = {row["id"]: row for row in profile_status(installed)}
        self.assertTrue(rows["h3"]["available"])
        self.assertEqual(rows["h3_turbo"]["missing"], [PROFILES["h3_turbo"]["lora"][0]])
        self.assertFalse(rows["hyperflow"]["available"])
        project = new_project("x")
        project["profile"] = "h3"
        # Without its fl2va installed, a family renders prompt-only clips on its ref2va.
        self.assertIsNone(project_models(project, {"minimax_h3_ref2va_pruned_int8_convrot.safetensors"})["unet_text"])

    def test_an_unknown_profile_is_refused_and_an_old_one_renders_on_hyperflow(self):
        with self.assertRaises(ValueError):
            set_project(new_project("x"), {"profile": "nope"})
        project = new_project("x")
        project["profile"] = "gone"
        self.assertEqual(render_graph(project, "x", clip(), "t1", "draft", 3)["1"]["inputs"]["unet_name"], MODELS["unet_text"])

    def test_a_draft_is_not_refined(self):
        graph = render_graph(new_project("x"), "x", clip(), "t1", "draft", 3)
        for node in ("80", "91", "95"):
            self.assertNotIn(node, graph)
        self.assertEqual(graph["18"]["inputs"]["images"], ["16", 0])
        self.assertEqual(graph["5"]["inputs"]["model"], ["1", 0])

    def test_a_project_without_a_profile_renders_with_the_first_one_installed(self):
        self.assertEqual(installed_profile(BASE_TURBO), "h3_turbo")
        self.assertEqual(installed_profile(BASE_TURBO | HYPERFLOW), "hyperflow")
        self.assertEqual(installed_profile(set()), "hyperflow")
        self.assertEqual(installed_profile(None), "hyperflow")
        graph = render_graph(new_project("x"), "x", clip(), "t1", "draft", 3, BASE_TURBO | FINALS)
        self.assertEqual(graph["1"]["inputs"]["unet_name"], PROFILES["h3_turbo"]["unet_text"][0])
        self.assertEqual(graph["94"]["inputs"]["lora_name"], PROFILES["h3_turbo"]["lora"][0])
        # A profile the project names is kept, installed or not.
        project = new_project("x")
        project["profile"] = "hyperflow"
        graph = render_graph(project, "x", clip(), "t1", "draft", 3, BASE_TURBO)
        self.assertEqual(graph["1"]["class_type"], "HyperFlowGGUFLoader")

    def test_a_final_renders_without_the_slider_or_upscaler_when_they_are_missing(self):
        project = new_project("x")
        project["format"] = "9:16"
        made, _ = add_clip(project, {"action": "A keeper walks to the window."})
        job, graph = plan_render(project, "x", made["id"], "final", 3, HYPERFLOW)
        for node in ("80", "91", "95"):
            self.assertNotIn(node, graph)
        self.assertEqual(graph["18"]["inputs"]["images"], ["16", 0])
        self.assertNotIn("size", job)
        self.assertNotIn("lora", job["models"])
        self.assertEqual(len(finals_status(HYPERFLOW)), 2)
        self.assertEqual(finals_status(HYPERFLOW | FINALS), [])
        _, graph = plan_render(project, "x", made["id"], "final", 3, HYPERFLOW | FINALS)
        self.assertIn("91", graph)
        self.assertEqual(graph["95"]["inputs"]["lora_name"], MODELS["lora"])

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

    def test_a_recorded_size_must_be_two_numbers_in_range(self):
        self.project["timeline"] = [{"clip": "b", "start": 0, "in": 0, "out": 24}]
        for size in (["1920:1080,drawtext=x", 1080], [1920], [100000, 1080], "1920x1080"):
            self.project["clips"][1]["takes"] = [{"id": "t2", "quality": "final", "size": size}]
            with self.assertRaises(ValueError):
                export_command(self.project, {"b": "b.mp4"}, "o.mp4")

    def test_a_refined_final_sets_the_cut_to_its_size(self):
        self.project["timeline"] = [{"clip": "a", "start": 0, "in": 0, "out": 24}, {"clip": "b", "start": 24, "in": 0, "out": 24}]
        self.project["clips"][1]["takes"] = [{"id": "t2", "quality": "final", "size": [1920, 1088]}]
        text = self.filter_of(export_command(self.project, {"a": "a.mp4", "b": "b.mp4"}, "o.mp4"))
        self.assertIn("scale=1920:1088:", text)

    def test_nothing_to_export_says_so(self):
        with self.assertRaises(ValueError):
            export_command(self.project, {}, "o.mp4")


if __name__ == "__main__":
    unittest.main()
