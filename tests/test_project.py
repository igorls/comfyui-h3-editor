import re
import unittest

from policy import bridge_spans, legal_length, video_span
from project import (MODELS, PRESET_LIMIT, PROFILES, REALISM_LORAS, REFINE_TAIL, active_take, add_cast, add_clip, apply_op, cast_pictures, clean_checkpoints,
                     clip_prompt, describe, encoder_status, update_cast, export_command, export_plan, finals_status,
                     frame_of, installed_profile, is_hyperflow, item_length, linked_length, lora_status, new_project, pick_file,
                     plan_render, profile_status, project_models, render_graph, render_models, set_project, slug,
                     snap_frames, tempo_chain)

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
        self.assertEqual(graph["1"]["class_type"], "H3HyperFlowLoader")
        self.assertEqual(graph["1"]["inputs"]["unet_name"], MODELS["unet_text"])
        self.assertEqual(graph["10"]["inputs"]["checkpoint"], "fl2va")
        self.assertEqual(graph["12"], {"class_type": "H3HyperFlowSigmas", "inputs": {"model": ["5", 0], "steps": 4}})

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
        self.assertEqual(graph["95"]["inputs"], {"model": ["1", 0], "lora_name": MODELS["lora"], "strength": 1.2, "start_sigma": 0.75})
        self.assertEqual(graph["5"]["inputs"]["model"], ["95", 0])
        used = render_models(graph)
        self.assertEqual((used["lora"], used["lora_strength"]), (MODELS["lora"], 1.2))

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
                    if is_hyperflow(unet):
                        self.assertEqual(graph["1"]["class_type"], "H3HyperFlowLoader")
                        self.assertEqual(sigmas["class_type"], "H3HyperFlowSigmas")
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

    def test_hyperflow_takes_any_quantization_when_its_int8_and_q8_are_not_installed(self):
        q4 = {"minimax_h3_ref2va_hyperflow_flash_14b-Q4_K_M.gguf", "minimax_h3_fl2va_hyperflow_flash_14b-Q4_K_M.gguf"}
        self.assertTrue({row["id"]: row for row in profile_status(q4)}["hyperflow"]["available"])
        graph = render_graph(new_project("x"), "x", clip(), "t1", "draft", 3, q4)
        self.assertEqual(graph["1"]["inputs"]["unet_name"], "minimax_h3_fl2va_hyperflow_flash_14b-Q4_K_M.gguf")
        self.assertEqual(graph["1"]["class_type"], "H3HyperFlowLoader")

    def test_hyperflow_renders_on_its_int8_files_when_no_gguf_is_installed(self):
        int8 = {"minimax_h3_ref2va_hyperflow_flash_14b-int8_convrot.safetensors",
                "minimax_h3_fl2va_hyperflow_flash_14b-int8_convrot.safetensors"}
        self.assertTrue({row["id"]: row for row in profile_status(int8)}["hyperflow"]["available"])
        graph = render_graph(new_project("x"), "x", clip(), "t1", "final", 3, int8)
        self.assertEqual(graph["1"]["inputs"]["unet_name"], "minimax_h3_fl2va_hyperflow_flash_14b-int8_convrot.safetensors")
        self.assertEqual(graph["1"]["class_type"], "H3HyperFlowLoader")
        self.assertEqual(graph["12"], {"class_type": "H3HyperFlowSigmas", "inputs": {"model": ["5", 0], "steps": 6}})
        graph = render_graph(new_project("x"), "x", clip(first={"file": "f.png"}), "t1", "draft", 3, int8)
        self.assertEqual(graph["1"]["inputs"]["unet_name"], "minimax_h3_ref2va_hyperflow_flash_14b-int8_convrot.safetensors")

    def test_hyperflow_takes_its_int8_file_before_any_gguf_whatever_folder_each_is_in(self):
        names = PROFILES["hyperflow"]["unet"]
        int8 = "minimax_h3_ref2va_hyperflow_flash_14b-int8_convrot.safetensors"
        q8 = "hyperflow/minimax_h3_ref2va_hyperflow_flash_14b-Q8_0.gguf"
        q4 = "minimax_h3_ref2va_hyperflow_flash_14b-Q4_K_M.gguf"
        self.assertEqual(names[0], "hyperflow/" + int8)
        self.assertEqual(pick_file(names, {int8, q4, q8}), int8)
        self.assertEqual(pick_file(names, {"z/" + int8, "a/" + q4, q8}), "z/" + int8)
        self.assertEqual(pick_file(names, {int8, "a/" + int8, names[0], q8}), names[0])
        # The fl2va file is never taken for ref2va.
        self.assertIsNone(pick_file(names, {"hyperflow/minimax_h3_fl2va_hyperflow_flash_14b-int8_convrot.safetensors"}))

    def test_hyperflow_without_its_int8_file_takes_q8_in_its_folder_then_the_first_by_name(self):
        names = PROFILES["hyperflow"]["unet"]
        q8 = "minimax_h3_ref2va_hyperflow_flash_14b-Q8_0.gguf"
        q4 = "minimax_h3_ref2va_hyperflow_flash_14b-Q4_K_M.gguf"
        self.assertEqual(pick_file(names, {q4, "hyperflow/" + q8}), "hyperflow/" + q8)
        self.assertEqual(pick_file(names, {q4, q8}), q4)
        self.assertEqual(pick_file(names, {"z/" + q4, "a/" + q8}), "z/" + q4)

    def test_a_file_in_a_folder_is_found_as_windows_lists_it(self):
        names = PROFILES["hyperflow"]["unet"]
        int8 = names[0].replace("/", "\\")
        q8 = "hyperflow\\minimax_h3_ref2va_hyperflow_flash_14b-Q8_0.gguf"
        self.assertEqual(pick_file(names, {int8, q8}), int8)
        self.assertEqual(pick_file(names, {q8, "minimax_h3_ref2va_hyperflow_flash_14b-Q4_K_M.gguf"}), q8)
        for name in (int8, q8):
            self.assertTrue({row["id"]: row for row in profile_status({name})}["hyperflow"]["available"], name)
            graph = render_graph(new_project("x"), "x", clip(first={"file": "f.png"}), "t1", "draft", 3, {name})
            self.assertEqual((graph["1"]["class_type"], graph["1"]["inputs"]["unet_name"]), ("H3HyperFlowLoader", name))

    def test_a_hyperflow_file_is_told_by_its_name(self):
        for name in ("minimax_h3_ref2va_hyperflow_flash_14b-Q8_0.gguf", "hyperflow/any.gguf", "hyperflow\\Any.GGUF",
                     "minimax_h3_ref2va_hyperflow_flash_14b-int8_convrot.safetensors",
                     "hyperflow\\minimax_h3_fl2va_hyperflow_flash_14b-int8_convrot.safetensors"):
            self.assertTrue(is_hyperflow(name), name)
        # A folder named hyperflow does not make a plain H3 checkpoint one.
        for name in ("minimax_h3_ref2va_pruned_int8_convrot.safetensors", "hyperflow/other_h3.safetensors", "", None):
            self.assertFalse(is_hyperflow(name), name)

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
        self.assertEqual(graph["1"]["class_type"], "H3HyperFlowLoader")

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
        self.assertEqual(graph["1"]["class_type"], "H3HyperFlowLoader")
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


class LoraTest(unittest.TestCase):
    def test_a_projects_loras_stack_on_every_render_in_their_order(self):
        project = new_project("x")
        set_project(project, {"loras": [{"file": "film.safetensors", "strength": 0.8},
                                        {"file": "grain.safetensors", "strength": -0.5}]})
        draft = render_graph(project, "x", clip(), "t1", "draft", 3)
        self.assertEqual(draft["100"]["inputs"], {"model": ["1", 0], "lora_name": "film.safetensors", "strength": 0.8})
        self.assertEqual(draft["101"]["inputs"], {"model": ["100", 0], "lora_name": "grain.safetensors", "strength": -0.5})
        self.assertEqual(draft["5"]["inputs"]["model"], ["101", 0])
        final = render_graph(project, "x", clip(), "t1", "final", 3)
        # After the Realism Slider, and on the model both passes sample with.
        self.assertEqual(final["100"]["inputs"]["model"], ["95", 0])
        self.assertEqual(final["5"]["inputs"]["model"], ["101", 0])
        self.assertEqual(render_models(final)["loras"], [{"file": "film.safetensors", "strength": 0.8},
                                                         {"file": "grain.safetensors", "strength": -0.5}])
        self.assertNotIn("loras", render_models(render_graph(new_project("x"), "x", clip(), "t1", "draft", 3)))

    def test_a_lora_that_is_off_at_zero_or_missing_is_left_out(self):
        project = new_project("x")
        set_project(project, {"loras": [{"file": "off.safetensors", "on": False}, {"file": "zero.safetensors", "strength": 0},
                                        {"file": "gone.safetensors"}, {"file": "styles/film.safetensors", "strength": 1.2}]})
        self.assertEqual([r["on"] for r in project["loras"]], [False, True, True, True])
        # ComfyUI on Windows lists a file in a folder with a backslash, and its loader takes it that way.
        graph = render_graph(project, "x", clip(), "t1", "draft", 3, HYPERFLOW | {"zero.safetensors", "styles\\film.safetensors"})
        self.assertEqual(graph["100"]["inputs"]["lora_name"], "styles\\film.safetensors")
        self.assertNotIn("101", graph)
        # A project file written by hand cannot break a render.
        project["loras"] = ["film", {"file": 3}, {"file": "a.safetensors", "strength": "strong"}, {"file": "b.safetensors", "strength": 9}]
        graph = render_graph(project, "x", clip(), "t1", "draft", 3)
        self.assertEqual(graph["100"]["inputs"], {"model": ["1", 0], "lora_name": "b.safetensors", "strength": 2.0})

    def test_loras_are_checked_when_they_are_set(self):
        project = new_project("x")
        for bad in ("film.safetensors", [{"strength": 1}], [{"file": "a", "strength": 5}], [{"file": "a", "strength": True}],
                    [{"file": "a"}, {"file": "a"}], [{"file": "a", "weight": 1}], [{"file": str(n)} for n in range(9)]):
            with self.assertRaises(ValueError):
                set_project(project, {"loras": bad})
        set_project(project, {"loras": [{"file": "a.safetensors"}]})
        row = project["loras"][0]
        self.assertEqual((row["file"], row["strength"], row["on"]), ("a.safetensors", 1.0, True))
        self.assertTrue(row["id"])
        set_project(project, {"loras": [dict(row, strength=0.5)]})
        self.assertEqual(project["loras"][0]["id"], row["id"])
        set_project(project, {"loras": []})
        self.assertEqual(project["loras"], [])

    def test_a_project_sets_how_strong_the_realism_slider_is_on_finals(self):
        project = new_project("x")
        set_project(project, {"realism": 0.75})
        graph = render_graph(project, "x", clip(), "t1", "final", 3)
        self.assertEqual(graph["95"]["inputs"]["strength"], 0.75)
        # A draft never runs it.
        self.assertNotIn("95", render_graph(project, "x", clip(), "t1", "draft", 3))
        set_project(project, {"realism": 0})
        graph = render_graph(project, "x", clip(), "t1", "final", 3)
        self.assertNotIn("95", graph)
        self.assertEqual(graph["5"]["inputs"]["model"], ["1", 0])
        set_project(project, {"realism": None})
        self.assertEqual(render_graph(project, "x", clip(), "t1", "final", 3)["95"]["inputs"]["strength"], 1.2)
        for bad in (4, -1, "strong", True):
            with self.assertRaises(ValueError):
                set_project(project, {"realism": bad})

    def test_realism_slider_prefers_v2_and_falls_back_to_v1_when_only_v1_is_installed(self):
        v2, v1 = REALISM_LORAS
        only_v1 = HYPERFLOW | {MODELS["latent_upscaler"], v1}
        both = only_v1 | {v2}
        self.assertEqual(finals_status(only_v1), [])
        self.assertEqual(lora_status(only_v1)["file"], v1)
        self.assertTrue(lora_status(only_v1)["installed"])
        self.assertEqual(render_graph(new_project("x"), "x", clip(), "t1", "final", 3, only_v1)["95"]["inputs"]["lora_name"], v1)
        self.assertEqual(lora_status(both)["file"], v2)
        self.assertEqual(render_graph(new_project("x"), "x", clip(), "t1", "final", 3, both)["95"]["inputs"]["lora_name"], v2)


class EncoderTest(unittest.TestCase):
    INT8 = {"qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "minimax_h3_video_vae_int8_convrot.safetensors",
            "minimax_h3_audio_vae_fp32.safetensors"}

    def test_a_render_takes_the_text_encoder_and_vae_that_are_installed(self):
        graph = render_graph(new_project("x"), "x", clip(), "t1", "draft", 3, HYPERFLOW | self.INT8)
        self.assertEqual(graph["2"]["inputs"]["clip_name"], "qwen3vl_32b_minimax_h3_int8_convrot.safetensors")
        self.assertEqual(graph["3"]["inputs"]["vae_name"], "minimax_h3_video_vae_int8_convrot.safetensors")
        self.assertEqual(graph["4"]["inputs"]["vae_name"], "minimax_h3_audio_vae_fp32.safetensors")
        # With the usual files beside them, the usual files win, and another precision is found by its name.
        both = HYPERFLOW | self.INT8 | {MODELS["clip"], MODELS["vae"], "split/qwen3vl_32b_minimax_h3_fp8.safetensors"}
        graph = render_graph(new_project("x"), "x", clip(), "t1", "draft", 3, both)
        self.assertEqual((graph["2"]["inputs"]["clip_name"], graph["3"]["inputs"]["vae_name"]), (MODELS["clip"], MODELS["vae"]))
        only = render_graph(new_project("x"), "x", clip(), "t1", "draft", 3, HYPERFLOW | {"split/qwen3vl_32b_minimax_h3_fp8.safetensors"})
        self.assertEqual(only["2"]["inputs"]["clip_name"], "split/qwen3vl_32b_minimax_h3_fp8.safetensors")
        # Nothing known, or nothing installed: the usual names, so the graph says what is missing.
        self.assertEqual(render_graph(new_project("x"), "x", clip(), "t1", "draft", 3)["2"]["inputs"]["clip_name"], MODELS["clip"])
        self.assertEqual(len(encoder_status(HYPERFLOW)["missing"]), 3)
        self.assertEqual(encoder_status(HYPERFLOW | self.INT8)["missing"], [])

    def test_a_project_can_name_its_text_encoder_and_vae(self):
        project = new_project("x")
        both = HYPERFLOW | self.INT8 | {MODELS["clip"], MODELS["vae"]}
        set_project(project, {"text_encoder": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "vae": "minimax_h3_video_vae_int8_convrot.safetensors"})
        job, graph = plan_render(project, "x", add_clip(project, {"action": "A keeper climbs."})[0]["id"], "draft", 3, both)
        self.assertEqual(graph["2"]["inputs"]["clip_name"], "qwen3vl_32b_minimax_h3_int8_convrot.safetensors")
        self.assertEqual(graph["3"]["inputs"]["vae_name"], "minimax_h3_video_vae_int8_convrot.safetensors")
        self.assertEqual((job["models"]["text_encoder"], job["models"]["vae"]),
                         ("qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "minimax_h3_video_vae_int8_convrot.safetensors"))
        # A named file this machine lacks gives way to an installed one.
        graph = render_graph(project, "x", clip(), "t1", "draft", 3, HYPERFLOW | {MODELS["clip"], MODELS["vae"]})
        self.assertEqual((graph["2"]["inputs"]["clip_name"], graph["3"]["inputs"]["vae_name"]), (MODELS["clip"], MODELS["vae"]))
        set_project(project, {"text_encoder": None, "vae": None})
        self.assertNotIn("models", project)
        for bad in (3, "", ["a"]):
            with self.assertRaises(ValueError):
                set_project(project, {"vae": bad})


class ClipModelTest(unittest.TestCase):
    def test_a_clip_renders_on_the_model_it_picks(self):
        project = new_project("x")
        project["profile"] = "hyperflow"
        own, _ = add_clip(project, {"action": "A keeper climbs the stairs.", "profile": "h3"})
        other, _ = add_clip(project, {"action": "The lamp turns."})
        job, graph = plan_render(project, "x", own["id"], "draft", 3)
        self.assertEqual(graph["1"], {"class_type": "UNETLoader", "inputs": {
            "unet_name": PROFILES["h3"]["unet_text"][0], "weight_dtype": "default"}})
        self.assertEqual(graph["12"]["inputs"]["steps"], 10)
        self.assertEqual(graph["11"]["inputs"]["sampler_name"], "res_multistep")
        self.assertEqual(job["models"]["profile"], "h3")
        job, graph = plan_render(project, "x", other["id"], "draft", 3)
        self.assertEqual(graph["1"]["class_type"], "H3HyperFlowLoader")
        self.assertEqual(job["models"]["profile"], "hyperflow")

    def test_a_clips_model_brings_its_own_lora_and_refine_steps(self):
        project = new_project("x")
        graph = render_graph(project, "x", clip(profile="h3_turbo"), "t1", "final", 3)
        self.assertEqual(graph["94"]["inputs"]["lora_name"], PROFILES["h3_turbo"]["lora"][0])
        self.assertEqual(graph["87"]["class_type"], "ManualSigmas")
        # A ref2va-only family renders a prompt-only clip on its ref2va.
        graph = render_graph(project, "x", clip(profile="singularity"), "t1", "draft", 3)
        self.assertEqual(graph["1"]["inputs"]["unet_name"], PROFILES["singularity"]["unet"][0])
        self.assertEqual(graph["10"]["inputs"]["checkpoint"], "ref2va")

    def test_a_clip_goes_back_to_the_projects_model(self):
        project = new_project("x")
        project["profile"] = "h3"
        made, _ = add_clip(project, {"action": "A keeper climbs the stairs.", "profile": "hyperflow"})
        apply_op(project, {"op": "update_clip", "clip": made["id"], "fields": {"profile": None}}, {})
        _, graph = plan_render(project, "x", made["id"], "draft", 3)
        self.assertEqual(graph["1"]["class_type"], "UNETLoader")
        with self.assertRaises(ValueError):
            add_clip(project, {"action": "x", "profile": "nope"})
        # A name written by hand that is no profile is passed over.
        self.assertEqual(render_graph(project, "x", clip(profile="nope"), "t1", "draft", 3)["1"]["class_type"], "UNETLoader")


class CheckpointTest(unittest.TestCase):
    def own(self, **extra):
        project = new_project("x")
        set_project(project, {"checkpoints": [dict({"id": "tuned", "title": "Tuned", "unet": "tuned_ref2va.safetensors"}, **extra)],
                              "profile": "tuned"})
        return project

    def test_a_projects_own_checkpoint_renders_on_its_familys_settings(self):
        project = self.own()
        job, graph = plan_render(project, "x", add_clip(project, {"action": "A keeper climbs."})[0]["id"], "final", 3)
        # No fl2va of its own, so a prompt-only clip renders on its ref2va.
        self.assertEqual(graph["1"], {"class_type": "UNETLoader", "inputs": {"unet_name": "tuned_ref2va.safetensors", "weight_dtype": "default"}})
        self.assertEqual(graph["10"]["inputs"]["checkpoint"], "ref2va")
        self.assertEqual(graph["12"]["inputs"], {"model": ["5", 0], "scheduler": "simple", "steps": 20, "denoise": 1.0})
        self.assertEqual(graph["11"]["inputs"]["sampler_name"], "res_multistep")
        self.assertEqual(graph["87"]["inputs"]["step"], 17)
        self.assertEqual((job["models"]["profile"], job["models"]["profile_title"]), ("tuned", "Tuned"))
        self.assertEqual((job["models"]["steps"], job["models"]["sampler"]), (20, "res_multistep"))

    def test_it_takes_its_own_steps_sampler_scheduler_and_text_model(self):
        project = self.own(steps={"draft": 6, "final": 12}, sampler="euler", scheduler="beta", unet_text="tuned_fl2va.safetensors")
        draft = render_graph(project, "x", clip(), "t1", "draft", 3)
        self.assertEqual(draft["1"]["inputs"]["unet_name"], "tuned_fl2va.safetensors")
        self.assertEqual(draft["10"]["inputs"]["checkpoint"], "fl2va")
        self.assertEqual((draft["12"]["inputs"]["steps"], draft["12"]["inputs"]["scheduler"]), (6, "beta"))
        self.assertEqual(draft["11"]["inputs"]["sampler_name"], "euler")
        final = render_graph(project, "x", clip(first={"file": "f.png"}), "t1", "final", 3)
        self.assertEqual(final["1"]["inputs"]["unet_name"], "tuned_ref2va.safetensors")
        # The last three steps are redrawn, as on the family it is based on.
        self.assertEqual((final["12"]["inputs"]["steps"], final["87"]["inputs"]["step"]), (12, 9))

    def test_it_keeps_its_familys_lora_and_loader(self):
        turbo = render_graph(self.own(base="h3_turbo"), "x", clip(), "t1", "final", 3)
        self.assertEqual(turbo["94"]["inputs"]["lora_name"], PROFILES["h3_turbo"]["lora"][0])
        self.assertEqual(turbo["12"]["inputs"]["steps"], 6)
        self.assertEqual(turbo["87"]["class_type"], "ManualSigmas")
        # A HyperFlow file loads as one whatever it is based on, on HyperFlow's own sigmas.
        flash = render_graph(self.own(unet="my_hyperflow_mix.safetensors", base="hyperflow"), "x", clip(), "t1", "draft", 3)
        self.assertEqual(flash["1"]["class_type"], "H3HyperFlowLoader")
        self.assertEqual(flash["12"]["class_type"], "H3HyperFlowSigmas")

    def test_it_can_leave_its_familys_lora_out(self):
        # A checkpoint with its turbo built in: the Turbo family's steps and sampler, without the Turbo LoRA.
        graph = render_graph(self.own(base="h3_turbo", lora=False), "x", clip(), "t1", "draft", 3)
        self.assertNotIn("94", graph)
        self.assertEqual((graph["12"]["inputs"]["steps"], graph["11"]["inputs"]["sampler_name"]), (6, "euler"))
        self.assertIn("94", render_graph(self.own(base="h3_turbo", lora=True), "x", clip(), "t1", "draft", 3))
        with self.assertRaises(ValueError):
            self.own(lora="no")

    def test_the_merged_community_checkpoints_are_built_in(self):
        for name, steps in (("eros", (6, 8)), ("dasiwa_v5", (6, 8))):
            project = new_project("x")
            set_project(project, {"profile": name})
            draft = render_graph(project, "x", clip(), "t1", "draft", 3)
            final = render_graph(project, "x", clip(), "t1", "final", 3)
            self.assertEqual(draft["1"], {"class_type": "UNETLoader", "inputs": {"unet_name": PROFILES[name]["unet"][0], "weight_dtype": "default"}})
            self.assertEqual((draft["12"]["inputs"]["steps"], final["12"]["inputs"]["steps"]), steps)
            # Their turbo is in the weights, so no family LoRA, and a prompt-only clip renders on the one file.
            self.assertNotIn("94", draft)
            self.assertEqual(draft["10"]["inputs"]["checkpoint"], "ref2va")
            self.assertEqual(final["87"]["class_type"], "ManualSigmas")
        found = {"dasiwa_minimax_h3_ref2va_v5xs_pruned_graviton_turbo_w4a8.safetensors"}
        self.assertEqual(render_graph(dict(new_project("x"), profile="dasiwa_v5"), "x", clip(), "t1", "draft", 3, found)["1"]["inputs"]["unet_name"],
                         "dasiwa_minimax_h3_ref2va_v5xs_pruned_graviton_turbo_w4a8.safetensors")
        rows = {row["id"]: row for row in profile_status(set())}
        self.assertEqual((rows["h3_turbo"]["lora"], rows["eros"]["lora"]), (PROFILES["h3_turbo"]["lora"][0], None))

    def test_one_clip_can_render_on_it(self):
        project = self.own()
        set_project(project, {"profile": "hyperflow"})
        made, _ = add_clip(project, {"action": "A keeper climbs.", "profile": "tuned"})
        job, graph = plan_render(project, "x", made["id"], "draft", 3)
        self.assertEqual(graph["1"]["inputs"]["unet_name"], "tuned_ref2va.safetensors")
        # Once the checkpoint is gone, the clip renders on the project's model again.
        set_project(project, {"checkpoints": []})
        _, graph = plan_render(project, "x", made["id"], "draft", 3)
        self.assertEqual(graph["1"]["class_type"], "H3HyperFlowLoader")
        with self.assertRaises(ValueError):
            add_clip(project, {"action": "x", "profile": "tuned"})

    def test_checkpoints_are_checked_when_they_are_set(self):
        project = new_project("x")
        for bad in ("tuned.safetensors", [{"title": "Tuned"}], [{"unet": "a", "base": "nope"}], [{"unet": "a", "id": "h3"}],
                    [{"unet": "a", "id": "m"}, {"unet": "b", "id": "m"}], [{"unet": "a", "steps": {"draft": 0}}],
                    [{"unet": "a", "steps": {"quick": 4}}], [{"unet": "a", "cfg": 3}], [{"unet": str(n)} for n in range(13)]):
            with self.assertRaises(ValueError):
                set_project(project, {"checkpoints": bad})
        with self.assertRaises(ValueError):
            set_project(project, {"profile": "tuned"})
        # The presets every project shares are checked the same way, with more room.
        self.assertEqual(len(clean_checkpoints([{"unet": str(n)} for n in range(13)], PRESET_LIMIT)), 13)
        set_project(project, {"checkpoints": [{"unet": "mixes/tuned_v2.safetensors"}]})
        row = project["checkpoints"][0]
        self.assertEqual((row["title"], row["base"]), ("tuned_v2", "h3"))
        self.assertTrue(row["id"])
        # A project file written by hand cannot break a render.
        project["checkpoints"] = ["tuned", {"id": "h3", "unet": "x"}, {"id": "mine", "unet": "mine.safetensors", "steps": {"draft": "few"}, "sampler": 3}]
        project["profile"] = "mine"
        graph = render_graph(project, "x", clip(), "t1", "draft", 3)
        self.assertEqual((graph["1"]["inputs"]["unet_name"], graph["12"]["inputs"]["steps"]), ("mine.safetensors", 10))


class CastTest(unittest.TestCase):
    asset = {"file": "a.png", "subfolder": "p/assets"}

    def project(self):
        project = new_project("x")
        project["cast"] = [
            {"id": "a", "name": "Tom", "description": "the keeper", **self.asset},
            {"id": "b", "name": "Sea", "description": "the sea", "file": "b.png", "subfolder": "p/assets"},
        ]
        return project

    def test_a_member_brings_all_its_pictures_to_the_shot(self):
        project = self.project()
        update_cast(project, "a", {"pictures": [{"file": "side.png", "subfolder": "p/assets"}, {"file": "back.png", "subfolder": "p/assets"}]})
        graph = render_graph(project, "x", clip(cast=["a", "b"], action="@Tom looks at @Sea."), "t1", "draft", 3)
        self.assertEqual([graph[str(n)]["inputs"]["file"] for n in (50, 51, 52, 53)],
                         ["p/assets/a.png", "p/assets/side.png", "p/assets/back.png", "p/assets/b.png"])
        self.assertEqual(graph["10"]["inputs"]["ref_images.ref_image_3"], ["53", 0])
        prompt = graph["10"]["inputs"]["prompt"]
        self.assertIn("<Subject 1> is the keeper in <Picture 1>, <Picture 2> and <Picture 3>.", prompt)
        self.assertIn("<Subject 2> is the sea in <Picture 4>.", prompt)

    def test_pictures_are_shared_out_within_what_h3_takes(self):
        project = self.project()
        more = [{"file": "t%s.png" % n, "subfolder": "p/assets"} for n in range(7)]
        update_cast(project, "a", {"pictures": more})
        update_cast(project, "b", {"pictures": more[:3]})
        shot = clip(cast=["a", "b"], first={"file": "f.png"})
        # Eight in all beside the first frame: one each, then one more each in turn.
        self.assertEqual([len(pictures) for _, pictures in cast_pictures(project, shot)], [4, 4])
        graph = render_graph(project, "x", shot, "t1", "draft", 3)
        self.assertIn("57", graph)
        self.assertNotIn("58", graph)
        self.assertIn("<Subject 1> is the keeper in <Picture 2>, <Picture 3>, <Picture 4> and <Picture 5>.", graph["10"]["inputs"]["prompt"])
        with self.assertRaises(ValueError):
            update_cast(project, "a", {"pictures": more + more[:1]})
        with self.assertRaises(ValueError):
            update_cast(project, "a", {"pictures": [{"subfolder": "p"}]})
        update_cast(project, "a", {"pictures": []})
        self.assertEqual(describe(project)["cast"][0]["pictures"], 1)

    def test_a_members_refmod_joins_the_conditioning_of_both_passes(self):
        project = self.project()
        update_cast(project, "a", {"refmod": {"file": "tom_v1", "strength": 0.8}})
        final = render_graph(project, "x", clip(cast=["a"], action="@Tom waits."), "t1", "final", 3)
        self.assertEqual(final["72"]["inputs"]["mod_1"], "tom_v1")
        self.assertEqual((final["72"]["inputs"]["strength_1"], final["72"]["inputs"]["mod_2"]), (0.8, "(none)"))
        self.assertEqual(final["73"]["inputs"]["conditioning"], ["10", 0])
        self.assertEqual(final["13"]["inputs"]["conditioning"], ["73", 0])
        self.assertEqual(final["74"]["inputs"]["conditioning"], ["80", 0])
        self.assertEqual(final["88"]["inputs"]["conditioning"], ["74", 0])
        self.assertEqual(render_models(final)["refmods"], [{"file": "tom_v1", "strength": 0.8}])
        # Only for a clip that chose the member, and only when the pack offers the file.
        self.assertNotIn("72", render_graph(project, "x", clip(cast=["b"]), "t1", "draft", 3))
        self.assertNotIn("72", render_graph(project, "x", clip(cast=["a"]), "t1", "draft", 3, HYPERFLOW))
        self.assertIn("72", render_graph(project, "x", clip(cast=["a"]), "t1", "draft", 3, HYPERFLOW | {"refmod:tom_v1"}))
        self.assertEqual(describe(project)["cast"][0]["refmod"], "tom_v1")

    def test_a_member_that_is_only_a_refmod_renders_from_the_prompt(self):
        project = new_project("x")
        made = add_cast(project, {"name": "Mira", "description": "a tall woman in a red scarf.", "refmod": {"file": "mira"}})
        self.assertEqual(made["refmod"], {"file": "mira", "strength": 1.0})
        graph = render_graph(project, "x", clip(cast=[made["id"]], action="@Mira turns. Then @mira smiles."), "t1", "draft", 3)
        # No picture, so the prompt-only model, with her written out the first time she is named.
        self.assertEqual(graph["10"]["inputs"]["checkpoint"], "fl2va")
        self.assertIn("Mira, a tall woman in a red scarf, turns. Then Mira smiles.", graph["10"]["inputs"]["prompt"])
        self.assertEqual(graph["13"]["inputs"]["conditioning"], ["73", 0])
        for bad in ({"strength": 1}, {"file": "v", "strength": 2}, {"file": "v", "copies": 3}, "mira"):
            with self.assertRaises(ValueError):
                update_cast(project, made["id"], {"refmod": bad})
        with self.assertRaises(ValueError):
            update_cast(project, made["id"], {"refmod": None})
        # A project file written by hand cannot break a render.
        made["refmod"] = {"file": "mira", "strength": "strong"}
        self.assertEqual(render_graph(project, "x", clip(cast=[made["id"]]), "t1", "draft", 3)["72"]["inputs"]["strength_1"], 1.0)

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
        # A smaller, lighter export: half the picture, compressed harder.
        full = re.search(r"crop=(\d+):(\d+)", self.filter_of(args))
        small = export_command(self.project, {"a": "a.mp4", "b": "b.mp4"}, "o.mp4", scale=0.5, quality="small")
        self.assertEqual((args[args.index("-crf") + 1], small[small.index("-crf") + 1]), ("16", "23"))
        self.assertIn("crop=%s:%s" % (int(full.group(1)) // 2, int(full.group(2)) // 2), self.filter_of(small))
        for bad in ({"scale": 0.3}, {"quality": "tiny"}):
            with self.assertRaises(ValueError):
                export_command(self.project, {"a": "a.mp4"}, "o.mp4", **bad)
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

    def test_a_plan_says_what_an_export_will_hold(self):
        project = new_project("x")
        project["clips"] = [
            {"id": "a", "name": "A", "takes": [{"id": "t", "quality": "final", "size": [1920, 1088], "frames": 100}], "active": "t"},
            {"id": "b", "name": "B", "takes": [{"id": "u", "quality": "draft", "size": [640, 352], "frames": 50}], "active": "u"},
            {"id": "c", "name": "C", "takes": [], "active": None},
        ]
        project["timeline"] = [{"id": "1", "clip": "a", "start": 0, "in": 0, "out": 100},
                               {"id": "2", "clip": "b", "start": 120, "in": 0, "out": 50},
                               {"id": "3", "clip": "c", "start": 170, "in": 0, "out": 24}]
        plan = export_plan(project)
        self.assertEqual((plan["width"], plan["height"], plan["frames"]), (1920, 1088, 194))
        self.assertEqual((plan["clips"], plan["drafts"], plan["missing"], plan["gaps"]), (2, 1, 1, 1))
        self.assertEqual((export_plan(project, 0.5)["width"], export_plan(project, 0.5)["height"]), (960, 544))

    def test_nothing_to_export_says_so(self):
        with self.assertRaises(ValueError):
            export_command(self.project, {}, "o.mp4")


if __name__ == "__main__":
    unittest.main()
