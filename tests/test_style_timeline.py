import json
import types
import unittest
from unittest import mock

import project
import style_timeline as styles

try:
    import torch
    import comfy.samplers
except ImportError:
    torch = None


class StyleTimelineTest(unittest.TestCase):
    def test_markers_are_sorted_validated_and_include_the_starting_look(self):
        rows = styles.schedule('[{"frame": 60, "style": "Clay"}]', 124, "Watercolor")
        self.assertEqual([(row["frame"], row["style"]) for row in rows], [(0, "Watercolor"), (60, "Clay")])
        for bad in ([{"frame": 124, "style": "Clay"}], [{"frame": True, "style": "Clay"}],
                    [{"frame": 4, "style": "A"}, {"frame": 4, "style": "B"}]):
            with self.assertRaises(ValueError):
                styles.clean_markers(bad, 124)

    def test_weights_blend_over_video_frames_and_reach_both_styles(self):
        rows = styles.schedule([{"frame": 0, "style": "A"}, {"frame": 100, "style": "B"}], 124)
        blend = styles.weights(rows, [0, 25, 50, 75, 100, 123])
        self.assertEqual(blend[0], [1, 0.84375, 0.5, 0.15625, 0, 0])
        for parts in zip(*blend):
            self.assertAlmostEqual(sum(parts), 1)
        self.assertEqual(blend[1][-1], 1)

    def test_repeated_styles_share_an_encoding_without_losing_marker_timing(self):
        rows = styles.schedule([{"frame": 0, "style": "A"}, {"frame": 60, "style": "B"}, {"frame": 123, "style": "A"}], 124)
        self.assertEqual(styles.weights(rows, [0, 60, 123]), [[1, 0, 1], [0, 1, 0]])

    def test_implicit_start_is_counted_in_the_marker_limit(self):
        rows = [{"frame": frame, "style": "Ink"} for frame in range(1, 33)]
        with self.assertRaisesRegex(ValueError, "including its starting look"):
            styles.schedule(rows, 40, "Clay")
        resolved = styles.schedule(rows[:-1], 40, "Clay")
        self.assertEqual(len(resolved), 32)
        self.assertEqual(styles.schedule(json.dumps(resolved), 40), resolved)

    def test_real_h3_temporal_grid_is_used_instead_of_uniform_frame_spacing(self):
        self.assertEqual(styles.latent_positions(7, 22), [0, 2.5, 6.5, 10.5, 14.5, 17, 21])

    def test_render_graph_cross_encodes_the_same_shot_at_draft_and_final_size(self):
        data = project.new_project("Styles")
        clip, _ = project.add_clip(data, {"action": "A bird takes flight.", "style_markers": [
            {"frame": 0, "style": "Watercolor"}, {"frame": 100, "style": "Clay"}]})
        available = set(project.PROFILES["hyperflow"]["unet"] + project.PROFILES["hyperflow"]["unet_text"] + [project.MODELS["latent_upscaler"]])
        job, graph = project.plan_render(data, "Styles", clip["id"], "final", 42, available)
        self.assertEqual(graph["13"]["class_type"], "H3StyleGuider")
        self.assertEqual(graph["88"]["class_type"], "H3StyleGuider")
        self.assertEqual(graph["10"]["inputs"]["style_markers"], graph["80"]["inputs"]["style_markers"])
        self.assertNotIn(data["style"], graph["10"]["inputs"]["prompt"])
        self.assertIn("A bird takes flight", graph["10"]["inputs"]["prompt"])
        self.assertEqual(job["saved"]["style_markers"], clip["style_markers"])
        self.assertEqual(len(json.loads(graph["10"]["inputs"]["style_markers"])), 2)


@unittest.skipIf(torch is None, "needs torch and ComfyUI")
class StyleGuiderTest(unittest.TestCase):
    def test_guidance_mixes_video_time_and_keeps_the_first_audio_prediction(self):
        marker_rows = styles.schedule([{"frame": 0, "style": "A"}, {"frame": 21, "style": "B"}], 22)
        variants = [[[torch.zeros(1, 2, 4), {"minimax_token_tags": "a"}]],
                    [[torch.ones(1, 3, 4), {"minimax_token_tags": "b"}]]]
        conditioning = [[variants[0][0][0], {"minimax_keyframes": ["guide"], "h3_style_timeline": {
            "markers": marker_rows, "length": 22, "conditioning": variants}}]]
        patcher = types.SimpleNamespace(model_options={}, is_dynamic=lambda: False)
        guider = styles.make_guider(patcher, conditioning)
        guider.inner_model = types.SimpleNamespace(latent_shapes=[(1, 2, 7, 2, 2), (1, 3, 4)])
        guider.conds = guider.original_conds
        a = torch.cat((torch.zeros(1, 1, 56), torch.full((1, 1, 12), 7.0)), dim=-1)
        b = torch.full_like(a, 10.0)
        with mock.patch.object(comfy.samplers, "sampling_function", side_effect=[a, b]) as predict:
            out = guider.predict_noise(torch.zeros_like(a), torch.tensor([1.0]))
        video = out[..., :56].reshape(1, 2, 7, 2, 2)
        self.assertTrue(torch.equal(video[:, :, 0], torch.zeros(1, 2, 2, 2)))
        self.assertTrue(torch.equal(video[:, :, -1], torch.full((1, 2, 2, 2), 10.0)))
        self.assertTrue(torch.equal(out[..., 56:], torch.full((1, 1, 12), 7.0)))
        self.assertEqual(predict.call_count, 2)
        self.assertEqual(guider.original_conds["style_1"][0]["minimax_keyframes"], ["guide"])


if __name__ == "__main__":
    unittest.main()
