import copy
import unittest

import project as store
from project import (UNKNOWN, accept_take, add_clip, add_linked, apply_op, close_gap, describe, fail_render, finish_render,
                     item_end, merge, new_project, plan_render, add_pending, set_speed, set_transition, split_item, update_clip,
                     update_item)


def take(take_id, frames):
    return {"id": take_id, "file": take_id + ".mp4", "subfolder": "h3take_projects/p/clips", "poster": take_id + ".jpg",
            "frames": frames, "quality": "draft", "seed": 1}


def scene(*lengths):
    """A project with rendered clips side by side."""
    project = new_project("Test")
    for index, frames in enumerate(lengths):
        clip, _ = add_clip(project, {"action": "Shot %s." % index})
        accept_take(project, clip["id"], take("t%s" % index, frames))
    return project


def starts(project):
    return [(i["start"], item_end(i)) for i in store.ordered(project)]


class TimelineTest(unittest.TestCase):
    def test_unrendered_clips_take_their_legal_length(self):
        project = new_project("Test")
        add_clip(project, {"action": "One.", "seconds": 4.5})
        add_clip(project, {"action": "Two."})
        self.assertEqual(starts(project), [(0, 107), (107, 231)])

    def test_a_render_of_another_length_moves_the_next_clip(self):
        project = new_project("Test")
        first, _ = add_clip(project, {"action": "One."})
        add_clip(project, {"action": "Two."})
        accept_take(project, first["id"], take("a", 90))
        self.assertEqual(starts(project), [(0, 90), (90, 214)])

    def test_new_seconds_resize_an_unrendered_clip(self):
        project = new_project("Test")
        first, _ = add_clip(project, {"action": "One."})
        add_clip(project, {"action": "Two."})
        update_clip(project, first["id"], {"seconds": 2})
        self.assertEqual(starts(project), [(0, 56), (56, 180)])

    def test_unknown_fields_are_named(self):
        project = new_project("Test")
        with self.assertRaisesRegex(ValueError, "unknown fields: colour"):
            add_clip(project, {"colour": "red"})

    def test_split_keeps_both_halves_on_the_same_take(self):
        project = scene(124)
        item = project["timeline"][0]
        right = split_item(project, item["id"], 50)
        self.assertEqual((item["in"], item["out"], right["start"], right["in"], right["out"]), (0, 50, 50, 50, 124))

    def test_speed_moves_the_clips_after(self):
        project = scene(124, 124)
        set_speed(project, store.ordered(project)[0]["id"], 2)
        self.assertEqual(starts(project), [(0, 62), (62, 186)])

    def test_a_transition_slides_the_clip_back(self):
        project = scene(124, 124)
        set_transition(project, store.ordered(project)[1]["id"], "dissolve", 12)
        self.assertEqual(starts(project), [(0, 124), (112, 236)])

    def test_close_every_gap(self):
        project = scene(60, 60)
        update_item(project, store.ordered(project)[1]["id"], {"start": 100})
        self.assertEqual(close_gap(project), 40)
        self.assertEqual(starts(project), [(0, 60), (60, 120)])

    def test_a_ripple_trim_pulls_the_next_clip(self):
        project = scene(124, 124)
        update_item(project, store.ordered(project)[0]["id"], {"out": 100}, ripple=True)
        self.assertEqual(starts(project), [(0, 100), (100, 224)])

    def test_out_past_the_take_is_refused(self):
        project = scene(124)
        with self.assertRaisesRegex(ValueError, "124 frames"):
            update_item(project, project["timeline"][0]["id"], {"out": 200})


class LinkedTest(unittest.TestCase):
    def test_extend_after_makes_room(self):
        project = scene(124, 124)
        clip, item = add_linked(project, store.ordered(project)[0]["id"], "extend_after")
        self.assertEqual(clip["link"]["a"]["file"], "t0.mp4")
        self.assertEqual(starts(project), [(0, 124), (124, 226), (226, 350)])

    def test_merge_keeps_the_scene_length(self):
        project = scene(124, 124)
        before = store.scene_end(project)
        clip, _ = add_linked(project, store.ordered(project)[0]["id"], "merge")
        self.assertEqual(store.scene_end(project), before)
        self.assertEqual(clip["link"]["mode"], "bridge")
        self.assertEqual(clip["link"]["a"]["last"], 124 - 32)

    def test_joining_an_unrendered_clip_is_refused(self):
        project = new_project("Test")
        _, item = add_clip(project, {"action": "One."})
        with self.assertRaisesRegex(ValueError, "render Clip 1"):
            add_linked(project, item["id"], "extend_after")


class RenderRecordTest(unittest.TestCase):
    def test_a_render_becomes_the_active_take(self):
        project = new_project("Test")
        clip, _ = add_clip(project, {"action": "One.", "seed": 7})
        job, graph = plan_render(project, "Test", clip["id"], "draft")
        self.assertEqual(graph["14"]["inputs"]["noise_seed"], 7)
        job["id"] = "prompt-1"
        add_pending(project, job)
        with self.assertRaisesRegex(ValueError, "already rendering"):
            plan_render(project, "Test", clip["id"], "draft")
        finish_render(project, "prompt-1", {"filename": "x.mp4", "subfolder": "s"}, {"frames": 124, "poster": "x.jpg"})
        self.assertEqual(project["pending"], [])
        self.assertEqual(store.active_take(clip)["file"], "x.mp4")
        self.assertEqual(clip["takes"][0]["saved"]["action"], "One.")

    def test_a_failed_render_says_why(self):
        project = new_project("Test")
        clip, _ = add_clip(project, {"action": "One."})
        job, _ = plan_render(project, "Test", clip["id"], "draft")
        job["id"] = "prompt-1"
        add_pending(project, job)
        fail_render(project, "prompt-1", "out of memory")
        self.assertEqual((project["pending"], clip["error"]), ([], "out of memory"))
        self.assertEqual(describe(project)["clips"][0]["state"], "failed")


class MergeTest(unittest.TestCase):
    def test_edits_to_different_clips_both_land(self):
        base = scene(124, 124)
        mine, theirs = copy.deepcopy(base), copy.deepcopy(base)
        mine["clips"][0]["action"] = "Mine."
        theirs["clips"][1]["action"] = "Theirs."
        merged = merge(base, mine, theirs)
        self.assertEqual([c["action"] for c in merged["clips"]], ["Mine.", "Theirs."])

    def test_a_stale_copy_cannot_drop_a_take_or_a_job(self):
        base = scene(124)
        mine, theirs = copy.deepcopy(base), copy.deepcopy(base)
        theirs["clips"][0]["takes"].append(take("new", 124))
        theirs["clips"][0]["active"] = "new"
        theirs["pending"] = [{"id": "p2", "clip": "x"}]
        mine["pending"] = []
        mine["clips"][0]["name"] = "Renamed"
        merged = merge(base, mine, theirs)
        clip = merged["clips"][0]
        self.assertEqual(([t["id"] for t in clip["takes"]], clip["active"], clip["name"]), (["t0", "new"], "new", "Renamed"))
        self.assertEqual(merged["pending"], [{"id": "p2", "clip": "x"}])

    def test_new_clips_from_both_sides_are_kept(self):
        base = scene(124)
        mine, theirs = copy.deepcopy(base), copy.deepcopy(base)
        add_clip(mine, {"action": "Mine."})
        add_clip(theirs, {"action": "Theirs."})
        merged = merge(base, mine, theirs)
        self.assertEqual(len(merged["clips"]), 3)
        self.assertEqual(len(merged["timeline"]), 3)

    def test_a_removal_wins_over_an_edit(self):
        base = scene(124, 124)
        mine, theirs = copy.deepcopy(base), copy.deepcopy(base)
        mine["timeline"] = mine["timeline"][1:]
        theirs["timeline"][0]["start"] = 5
        merged = merge(base, mine, theirs)
        self.assertEqual([i["id"] for i in merged["timeline"]], [base["timeline"][1]["id"]])

    def test_a_key_only_one_side_added_survives(self):
        base = scene(124)
        mine, theirs = copy.deepcopy(base), copy.deepcopy(base)
        theirs["cast"] = [{"id": "p1", "name": "Ana"}]
        mine["style"] = "Noir"
        merged = merge(base, mine, theirs)
        self.assertEqual((merged["cast"][0]["name"], merged["style"]), ("Ana", "Noir"))

    def test_an_up_to_date_copy_still_cannot_drop_takes_or_jobs(self):
        current = scene(124)
        current["pending"] = [{"id": "p1", "clip": "x"}]
        mine = copy.deepcopy(current)
        mine["clips"][0]["takes"] = []
        mine["pending"] = []
        mine["style"] = "Noir"
        merged = merge(current, mine, current)
        self.assertEqual((len(merged["clips"][0]["takes"]), merged["pending"], merged["style"]), (1, current["pending"], "Noir"))

    def test_restoring_an_old_revision_keeps_newer_takes(self):
        old = scene(124)
        current = copy.deepcopy(old)
        current["clips"][0]["takes"].append(take("new", 124))
        add_clip(current, {"action": "Added later."})
        restored = merge(current, old, current)
        self.assertEqual(len(restored["clips"]), 1)
        self.assertEqual([t["id"] for t in restored["clips"][0]["takes"]], ["t0", "new"])

    def test_without_a_common_version_nothing_is_dropped(self):
        base = scene(124, 124)
        mine, theirs = copy.deepcopy(base), copy.deepcopy(base)
        mine["timeline"] = mine["timeline"][1:]
        merged = merge(UNKNOWN, mine, theirs)
        self.assertEqual(len(merged["timeline"]), 2)


class BatchTest(unittest.TestCase):
    def test_later_operations_name_earlier_ones(self):
        project = new_project("Test")
        names = {}
        apply_op(project, {"op": "add_clip", "as": "open", "fields": {"action": "One."}}, names)
        apply_op(project, {"op": "update_item", "item": "$open", "fields": {"start": 10}}, names)
        apply_op(project, {"op": "update_clip", "clip": "$open", "fields": {"name": "Opening"}}, names)
        self.assertEqual(project["clips"][0]["name"], "Opening")
        self.assertEqual(project["timeline"][0]["start"], 10)

    def test_an_unknown_name_is_refused(self):
        with self.assertRaisesRegex(ValueError, r"\$nope"):
            apply_op(new_project("Test"), {"op": "remove_clip", "clip": "$nope"}, {})

    def test_the_summary_reads_like_a_timeline(self):
        project = scene(124, 124)
        summary = describe(project)
        self.assertEqual([t["at"] for t in summary["timeline"]], ["00:00:00", "00:05:04"])
        self.assertEqual(summary["length"], "00:10:08")


if __name__ == "__main__":
    unittest.main()
