import copy
import unittest

import project


def guide(frame, file="pose.png"):
    return {"frame": frame, "file": file, "subfolder": "h3take_projects/film/assets"}


class MidGuideTest(unittest.TestCase):
    def setUp(self):
        self.data = project.new_project("film")
        self.clip, _ = project.add_clip(self.data, {"action": "Turn", "mid_guides": [guide(68, "b.png"), guide(34)]})

    def test_draft_final_and_snapshot_match_sorted_pictures(self):
        expected = [guide(34), guide(68, "b.png")]
        self.assertEqual(project.describe(self.data)["clips"][0]["mid_guides"], expected)
        for quality, passes in (("draft", ["10"]), ("final", ["10", "80"])):
            job, graph = project.plan_render(self.data, "film", self.clip["id"], quality, seed=11)
            self.assertEqual(job["saved"]["mid_guides"], expected)
            for node in passes:
                take = graph[node]["inputs"]
                self.assertEqual(take["mid_frames"], "34, 68")
                for i, row in enumerate(expected):
                    image, slot = take["mid_images.mid_image_%s" % i]
                    self.assertEqual((graph[image], slot), (project.still_node(row), 0))
        self.clip["mid_guides"][0]["frame"] = 8
        self.assertEqual(job["saved"]["mid_guides"], expected)
        revised, graph = project.plan_render(self.data, "film", self.clip["id"], "draft", seed=11)
        self.assertEqual(graph["10"]["inputs"]["mid_frames"], "8, 68")
        self.assertEqual(revised["saved"]["mid_guides"][0]["frame"], 8)

    def test_clear_legacy_and_invalid_edits(self):
        clip_id = self.clip["id"]
        for invalid in ([guide(0)], [guide(123)], [guide(True)], [guide(1.5)], [guide(17), guide(17)],
                        [guide(n) for n in range(1, 10)], [{"frame": 12}], [guide(12, "audio.wav")], "12"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                project.update_clip(self.data, clip_id, {"mid_guides": invalid})
        with self.assertRaises(ValueError):
            project.update_clip(self.data, clip_id, {"seconds": 1})
        self.assertEqual(self.clip["seconds"], 5)
        edited = copy.deepcopy(self.data)
        edited["clips"][0]["mid_guides"] = [guide(0)]
        with self.assertRaises(ValueError):
            project.merge_editor_save(self.data, edited, self.data)
        project.update_clip(self.data, clip_id, {"seconds": 1, "mid_guides": []})
        self.assertEqual(project.plan_render(self.data, "film", clip_id, "draft")[1]["10"]["inputs"]["mid_frames"], "")
        del self.clip["mid_guides"]
        self.assertNotIn("mid_guides", project.describe(self.data)["clips"][0])
        self.clip["kind"] = "footage"
        with self.assertRaisesRegex(ValueError, "generated clips"):
            project.update_clip(self.data, clip_id, {"mid_guides": [guide(9)]})

    def test_linked_frames_exclude_holds_and_keep_internal_landing(self):
        side = {"file": "source.mp4", "subfolder": "clips", "first": 0, "last": 124}
        for mode, arrival, end, first, last, limit in (
            ("extend_after", None, None, 39, 88, 8), ("extend_before", None, None, 1, 50, 8),
            ("bridge", "pin", None, 39, 88, 8), ("bridge", None, 22, 39, 118, 8),
            ("bridge", None, 39, 39, 101, 7),
        ):
            link = {"mode": mode, "hold": 39, "a": side, "b": side, "arrive": arrival, "end_hold": end}
            self.clip.update(seconds=3, link=link)
            with self.subTest(mode=mode, arrival=arrival, end=end):
                length = project.linked_length(link, 3)
                self.clip["mid_guides"] = [guide(n) for n in range(first, first + limit - 1)] + [guide(last)]
                graph = project.render_graph(self.data, "film", self.clip, "take", "final", 1)
                for node in ("10", "80"):
                    pins = graph[node]["inputs"]
                    numbers = [int(n) for n in pins["mid_frames"].split(",")]
                    self.assertEqual(numbers[:limit], [row["frame"] for row in self.clip["mid_guides"]])
                    if mode == "bridge" and end == 39:
                        self.assertEqual(numbers[-1], length - 39)
                        self.assertEqual(pins["mid_images.mid_image_7"], ["8", 0])
                    else:
                        self.assertEqual(len(numbers), limit)
                for invalid in (first - 1, last + 1):
                    with self.assertRaises(ValueError):
                        project.clean_mid_guides(dict(self.clip, mid_guides=[guide(invalid)]))
                with self.assertRaises(ValueError):
                    project.clean_mid_guides(dict(self.clip, mid_guides=[guide(n) for n in range(first, first + limit + 1)]))
