import unittest

from project import active_take, clip_prompt, export_command, new_project, render_graph, slug, snap_frames
from prompting import compose_prompt


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

    def test_nothing_to_export_says_so(self):
        with self.assertRaises(ValueError):
            export_command(self.project, {}, "o.mp4")


if __name__ == "__main__":
    unittest.main()
