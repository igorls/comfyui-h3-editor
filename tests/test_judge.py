import unittest

import judge
from project import add_effect, accept_effect_take, export_command, new_project, slice_project


def cut():
    project = new_project("x")
    for n, name in enumerate(("c1", "c2")):
        project["clips"].append({"id": name, "name": name, "action": "", "seconds": 4, "takes": [
            {"id": "t", "file": name + ".mp4", "frames": 96, "quality": "draft"}], "active": "t"})
        project["timeline"].append({"id": "i%d" % n, "clip": name, "start": 96 * n, "in": 0, "out": 96})
    return project


class SliceTest(unittest.TestCase):
    def test_a_range_keeps_the_parts_of_the_clips_inside_it(self):
        part = slice_project(cut(), 90, 120)
        self.assertEqual([(i["clip"], i["start"], i["in"], i["out"]) for i in part["timeline"]], [("c1", 0, 90, 96), ("c2", 6, 0, 24)])

    def test_a_clip_at_another_speed_is_trimmed_in_its_own_frames(self):
        project = cut()
        project["timeline"][0]["speed"] = 2.0          # 96 frames play in 48
        project["timeline"][1]["start"] = 48
        part = slice_project(project, 24, 60)
        self.assertEqual([(i["clip"], i["start"], i["in"], i["out"]) for i in part["timeline"]], [("c1", 0, 48, 96), ("c2", 24, 0, 12)])

    def test_sounds_and_vfx_clips_are_trimmed_with_the_range(self):
        project = cut()
        project["sounds"] = [{"id": "s1", "file": "m.mp3", "seconds": 10.0, "start": 0, "offset": 1.0, "frames": 192, "gain": 1.0, "lane": 0}]
        effect = add_effect(project, {"start": 80, "frames": 40})
        accept_effect_take(project, effect["id"], {"id": "tx", "file": "fx.mp4", "frames": 40})
        part = slice_project(project, 90, 120)
        self.assertEqual([(s["start"], s["frames"], s["offset"]) for s in part["sounds"]], [(0, 30, 4.75)])
        self.assertEqual([(e["start"], e["frames"], e["skip"]) for e in part["effects"]], [(0, 30, 10)])
        args = export_command(part, {"c1": "a.mp4", "c2": "b.mp4"}, "o.mp4", effect_paths={effect["id"]: "fx.mp4"})
        self.assertIn("trim=start_frame=10,setpts=PTS-STARTPTS,", args[args.index("-filter_complex") + 1])

    def test_a_range_must_run_forwards(self):
        for first, last in ((10, 10), (20, 5), (-1, 5)):
            with self.assertRaises(ValueError):
                slice_project(cut(), first, last)


class RubricTest(unittest.TestCase):
    def test_the_default_rubric_becomes_one_score_question_each(self):
        asked = judge.questions(judge.RUBRIC)
        self.assertEqual(set(asked), {row["key"] for row in judge.RUBRIC})
        for row in judge.RUBRIC:
            self.assertEqual(asked[row["key"]]["type"], "score")
            self.assertEqual(asked[row["key"]]["criteria"], row["levels"])
            self.assertTrue(2 <= len(row["levels"]) <= 10)

    def test_a_reply_reads_as_a_level_a_part_and_a_confidence(self):
        rubric = judge.clean_rubric([
            {"key": "subject", "title": "Subject", "instructions": "Rate it.", "levels": ["a", "b", "c", "d", "e"]},
            {"key": "sharp", "instructions": "Is it sharp?", "levels": ["soft", "sharp"]}])
        reply = {"answers": {"subject": {"type": "score", "score": 3.4, "probabilities": {"3": 0.6, "4": 0.4}, "confidence": 0.8}},
                 "usage": {"total_input_tokens": 1234}}
        read = judge.read_answers(rubric, reply)
        row = read["rows"][0]
        self.assertEqual((row["score"], row["of"], row["part"], row["confidence"]), (3.4, 4, 0.85, 0.8))
        self.assertEqual(row["level"], "d")
        self.assertEqual(row["chances"], [0.0, 0.0, 0.0, 0.6, 0.4])
        self.assertIn("error", read["rows"][1])                         # a criterion with no answer says so
        self.assertEqual(read["tokens"], 1234)

    def test_a_check_reads_as_how_likely_its_good_end_is(self):
        rubric = judge.clean_rubric([{"key": "sharp", "instructions": "Is it sharp?", "levels": ["soft", "sharp"]}])
        row = judge.read_answers(rubric, {"answers": {"sharp": {"score": 0.77, "confidence": 0.77}}})["rows"][0]
        self.assertEqual((row["score"], row["of"], row["part"], row["level"]), (0.77, 1, 0.77, "sharp"))

    def test_a_score_outside_the_levels_is_held_to_them(self):
        rubric = judge.clean_rubric([{"key": "subject", "instructions": "Rate it.", "levels": ["a", "b", "c", "d", "e"]}])
        read = judge.read_answers(rubric, {"answers": {"subject": {"score": 9, "confidence": 1}}})
        self.assertEqual((read["rows"][0]["score"], read["rows"][0]["part"]), (4.0, 1.0))

    def test_the_cast_picture_criterion_is_kept_through_a_check_of_the_rubric(self):
        kept = judge.clean_rubric(judge.RUBRIC)
        self.assertEqual([row["key"] for row in kept if row.get("needs") == "reference"], ["match"])

    def test_a_rubric_of_the_callers_is_checked(self):
        good = [{"key": "tone", "instructions": "Rate the tone.", "levels": ["flat", "warm"]}]
        self.assertEqual(judge.clean_rubric(good)[0]["title"], "tone")
        self.assertIs(judge.clean_rubric(None), judge.RUBRIC)
        for bad in ([], "tone", [{"key": "a b", "instructions": "x", "levels": ["a", "b"]}], [{"key": "a", "instructions": "", "levels": ["a", "b"]}],
                    [{"key": "a", "instructions": "x", "levels": ["only one"]}], good + good):
            with self.assertRaises(ValueError):
                judge.clean_rubric(bad)

    def test_refusals_are_explained_in_plain_words(self):
        self.assertIn("API key", judge.explain(403, {"error": {"message": "forbidden"}}))
        self.assertIn("rate limit", judge.explain(429, {}))
        self.assertIn("declined", judge.explain(400, {"error": {"code": "safety", "message": "blocked"}}))


if __name__ == "__main__":
    unittest.main()
