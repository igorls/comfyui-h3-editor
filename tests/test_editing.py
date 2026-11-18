import copy
import unittest

import project as store
from project import clean_title, copy_name, retarget
from project import (UNKNOWN, accept_take, add_clip, add_linked, apply_op, close_gap, describe, fail_render, finish_render,
                     item_end, merge, new_project, plan_render, add_pending, set_speed, set_transition, split_item, update_clip,
                     update_item)


def take(take_id, frames):
    return {"id": take_id, "file": take_id + ".mp4", "subfolder": "h3take_projects/p/clips", "poster": take_id + ".jpg",
            "frames": frames, "quality": "draft", "seed": 1}


class ProjectFolderTest(unittest.TestCase):
    def test_a_duplicate_points_at_its_own_folder(self):
        data = {"clips": [{"takes": [take("t1", 90)], "first": {"file": "a.png", "subfolder": "h3take_projects/p/assets"}}],
                "cast": [{"file": "b.png", "subfolder": "h3take_projects\\p\\assets"}],
                "other": {"subfolder": "h3take_projects/pp/clips"}, "name": "h3take_projects/p is only a name here"}
        moved = retarget(data, "p", "p_copy")
        self.assertEqual(moved["clips"][0]["takes"][0]["subfolder"], "h3take_projects/p_copy/clips")
        self.assertEqual(moved["clips"][0]["first"]["subfolder"], "h3take_projects/p_copy/assets")
        self.assertEqual(moved["cast"][0]["subfolder"], "h3take_projects/p_copy/assets")
        # Another project whose name only starts the same is left alone, and so is plain text.
        self.assertEqual(moved["other"]["subfolder"], "h3take_projects/pp/clips")
        self.assertEqual(moved["name"], data["name"])
        # The original is not touched.
        self.assertEqual(data["clips"][0]["takes"][0]["subfolder"], "h3take_projects/p/clips")

    def test_a_duplicate_gets_a_name_no_project_has(self):
        self.assertEqual(copy_name("Bag demo", ["Bag_demo"]), "Bag demo copy")
        self.assertEqual(copy_name("Bag demo", ["Bag_demo", "bag_demo_copy", "Bag_demo_copy_2"]), "Bag demo copy 3")

    def test_a_projects_name_needs_letters_or_numbers(self):
        self.assertEqual(clean_title("  Morning teaser "), "Morning teaser")
        for bad in ("", "   ", "!!!", None):
            with self.assertRaises(ValueError):
                clean_title(bad)


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

    def test_a_clip_gets_a_volume_where_it_sits_and_a_split_keeps_it(self):
        project = scene(124)
        item = project["timeline"][0]
        apply_op(project, {"op": "update_item", "item": item["id"], "fields": {"gain": 0.4}}, {})
        self.assertEqual(item["gain"], 0.4)
        self.assertEqual(describe(project)["timeline"][0]["gain"], 0.4)
        right = split_item(project, item["id"], 50)
        self.assertEqual(right["gain"], 0.4)
        # 1 is as rendered: nothing is kept for it.
        apply_op(project, {"op": "update_item", "item": item["id"], "fields": {"gain": 1}}, {})
        self.assertNotIn("gain", item)
        self.assertNotIn("gain", describe(project)["timeline"][0])
        with self.assertRaisesRegex(ValueError, "volume must be a number from 0 to 2"):
            apply_op(project, {"op": "update_item", "item": item["id"], "fields": {"gain": 3}}, {})

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
        # The source stops 9 frames early and the new clip plays those 9 itself: the scene is as long as before.
        self.assertEqual(starts(project), [(0, 115), (115, 226), (226, 350)])

    def test_merge_keeps_the_scene_length(self):
        project = scene(124, 124)
        before = store.scene_end(project)
        clip, item = add_linked(project, store.ordered(project)[0]["id"], "merge")
        self.assertEqual(store.scene_end(project), before)
        self.assertEqual(clip["link"]["mode"], "bridge")
        self.assertEqual(clip["link"]["a"]["last"], 124 - 32)
        self.assertEqual(clip["link"]["b"]["first"], 31)
        first, middle, last = store.ordered(project)
        self.assertEqual((item_end(first), item_end(middle)), (middle["start"], last["start"]))
        self.assertEqual((first["out"], middle["in"], middle["out"], last["in"]), (83, 30, 111, 40))

    def test_joining_an_unrendered_clip_is_refused(self):
        project = new_project("Test")
        _, item = add_clip(project, {"action": "One."})
        with self.assertRaisesRegex(ValueError, "render Clip 1"):
            add_linked(project, item["id"], "extend_after")

    def test_an_extension_plays_the_last_held_frames_itself(self):
        project = scene(124, 124)
        first = store.ordered(project)[0]
        clip, item = add_linked(project, first["id"], "extend_after")
        # The held frames are the source's last 39. The source is cut 9 inside them, and the new clip starts there.
        self.assertEqual(clip["link"]["handover"], 9)
        self.assertEqual((clip["link"]["a"]["first"], clip["link"]["a"]["last"]), (0, 124))
        self.assertEqual((first["in"], first["out"]), (0, 115))
        self.assertEqual((item["in"], item["out"]), (30, 141))
        self.assertEqual(item["start"], item_end(first))

    def test_an_extension_before_plays_the_first_held_frames_itself(self):
        project = scene(124, 124)
        second = store.ordered(project)[1]
        clip, item = add_linked(project, second["id"], "extend_before")
        self.assertEqual(clip["link"]["a"]["first"], 0)
        self.assertEqual((second["in"], second["out"]), (9, 124))
        self.assertEqual((item["in"], item["out"]), (0, 141 - 39 + 9))
        self.assertEqual(second["start"], item_end(item))
        self.assertEqual(store.scene_end(project), 124 + 102 + 124)

    def test_a_bridge_arrives_on_a_short_run_of_the_next_clip(self):
        project = scene(124, 124)
        first, second = store.ordered(project)
        clip, item = add_linked(project, first["id"], "bridge")
        self.assertEqual((clip["link"]["arrive"], clip["link"]["end_hold"], clip["seconds"]), ("short", 22, 6.375))
        # Both clips hand over 9 frames inside what the bridge holds of them: 39 frames of the first, 22 of the next.
        self.assertEqual((first["out"], second["in"]), (115, 9))
        self.assertEqual((clip["link"]["b"]["first"], item["in"], item["out"]), (0, 30, 192 - 22 + 9))
        self.assertEqual((item["start"], second["start"]), (item_end(first), item_end(item)))

    def test_a_bridge_to_a_trimmed_clip_moves_its_start_onto_the_grid(self):
        project = scene(124, 124)
        first, second = store.ordered(project)
        second["in"] = 3
        clip, _ = add_linked(project, first["id"], "bridge")
        self.assertEqual((clip["link"]["b"]["first"], second["in"]), (17, 26))

    def test_a_bridge_needs_22_frames_of_the_next_clip(self):
        project = scene(124, 124)
        first, second = store.ordered(project)
        second["out"] = 20
        with self.assertRaisesRegex(ValueError, "at least 22 frames"):
            add_linked(project, first["id"], "bridge")

    def test_a_bridge_can_arrive_on_one_pinned_frame(self):
        project = scene(124, 124)
        first, second = store.ordered(project)
        clip, item = add_linked(project, first["id"], "bridge", arrive="pin")
        self.assertEqual(clip["link"]["arrive"], "pin")
        # The first clip hands over as for an extension. The next clip is not touched: its first frame is the pin.
        self.assertEqual((first["out"], second["in"]), (115, 0))
        self.assertEqual(clip["link"]["b"]["first"], 0)
        # 39 held frames and 153 new ones, the last of them the pinned frame, which the next clip shows itself.
        self.assertEqual((clip["seconds"], item["in"], item["out"]), (6.375, 30, 192 - 1))
        self.assertEqual((item["start"], second["start"]), (item_end(first), item_end(item)))

    def test_a_bridge_to_a_trimmed_clip_leaves_its_cut_alone(self):
        project = scene(124, 124)
        first, second = store.ordered(project)
        first["out"], second["in"] = 110, 3
        clip, _ = add_linked(project, first["id"], "bridge", arrive="pin")
        self.assertEqual((clip["link"]["a"]["last"], clip["link"]["b"]["first"]), (107, 3))
        self.assertEqual((first["out"], second["in"]), (98, 3))

    def test_a_bridge_can_arrive_on_a_short_clip(self):
        project = scene(124, 124)
        first, second = store.ordered(project)
        second["out"] = 20
        add_linked(project, first["id"], "bridge", arrive="pin")
        held = scene(124, 124)
        first, second = store.ordered(held)
        second["out"] = 20
        with self.assertRaisesRegex(ValueError, "at least 39 frames"):
            add_linked(held, first["id"], "bridge", arrive="hold")

    def test_a_trimmed_arrival_makes_the_bridge_stale_and_is_pinned_again(self):
        project = scene(124, 124)
        first, second = store.ordered(project)
        clip, item = add_linked(project, first["id"], "bridge", arrive="pin")
        accept_take(project, clip["id"], take("tb", 192))
        self.assertFalse(store.is_stale(project, clip))
        second["in"] = 12
        self.assertTrue(store.is_stale(project, clip))
        store.refresh_link(project, clip)
        self.assertEqual((clip["link"]["b"]["first"], second["in"]), (12, 12))
        self.assertFalse(store.is_stale(project, clip))

    def test_arrive_is_for_bridges(self):
        project = scene(124, 124)
        first = store.ordered(project)[0]
        with self.assertRaisesRegex(ValueError, "for a bridge"):
            add_linked(project, first["id"], "extend_after", arrive="pin")
        with self.assertRaisesRegex(ValueError, "one of short, pin, hold"):
            add_linked(project, first["id"], "bridge", arrive="glide")

    def test_a_bridge_that_holds_hands_over_at_both_ends(self):
        project = scene(124, 124)
        first, second = store.ordered(project)
        clip, item = add_linked(project, first["id"], "bridge", arrive="hold")
        self.assertNotIn("arrive", clip["link"])
        self.assertEqual((first["out"], second["in"]), (115, 9))
        self.assertEqual((item["in"], item["out"]), (30, 192 - 39 + 9))
        self.assertEqual((item["start"], second["start"]), (item_end(first), item_end(item)))

    def test_a_first_render_keeps_the_handover(self):
        project = scene(124, 124)
        clip, item = add_linked(project, store.ordered(project)[0]["id"], "extend_after", seconds=2)
        accept_take(project, clip["id"], take("tx", 192))
        self.assertEqual((item["in"], item["out"]), (30, 192))

    def test_extending_a_trimmed_clip_moves_its_cut_onto_the_latent_grid(self):
        # 100 is not a place the picture latent can be cut: the nearest one before it is 90 (39 + 17 * 3).
        project = scene(124, 124)
        first = store.ordered(project)[0]
        update_item(project, first["id"], {"out": 100}, ripple=True)
        clip, item = add_linked(project, first["id"], "extend_after")
        self.assertEqual((clip["link"]["a"]["first"], clip["link"]["a"]["last"]), (0, 90))
        self.assertEqual(first["out"], 81)
        self.assertEqual(item["start"], 81)
        # What followed still touches the new clip.
        self.assertEqual(store.ordered(project)[2]["start"], item_end(item))

    def test_extending_before_a_trimmed_start_moves_it_onto_the_latent_grid(self):
        project = scene(124, 124)
        second = store.ordered(project)[1]
        second["in"] = 20
        clip, item = add_linked(project, second["id"], "extend_before")
        self.assertEqual(clip["link"]["a"]["first"], 34)
        self.assertEqual(second["in"], 43)
        self.assertEqual(second["start"], item_end(item))

    def test_a_bridge_that_holds_moves_both_cuts_onto_the_latent_grid(self):
        project = scene(124, 124)
        first, second = store.ordered(project)
        first["out"], second["in"] = 110, 3
        clip, _ = add_linked(project, first["id"], "bridge", arrive="hold")
        self.assertEqual((clip["link"]["a"]["last"], clip["link"]["b"]["first"]), (107, 17))
        self.assertEqual((first["out"], second["in"]), (98, 26))

    def test_footage_keeps_its_cut(self):
        project = scene(124, 124)
        first = store.ordered(project)[0]
        store.find_clip(project, first["clip"])["kind"] = "footage"
        first["out"] = 100
        clip, _ = add_linked(project, first["id"], "extend_after")
        self.assertEqual(clip["link"]["a"]["last"], 100)
        self.assertEqual(first["out"], 91)

    def test_a_clip_too_short_to_give_frames_up_keeps_its_cut(self):
        project = scene(124, 124)
        first = store.ordered(project)[0]
        first["in"], first["out"] = 60, 100
        clip, _ = add_linked(project, first["id"], "extend_after")
        self.assertEqual(clip["link"]["a"]["last"], 100)

    def test_a_linked_clip_continues_its_sources_track_from_the_held_frames(self):
        project = scene(141, 124)
        first = store.ordered(project)[0]
        store.find_clip(project, first["clip"])["track"] = {"file": "voice.mp3", "subfolder": "s", "from": 2.0}
        clip, _ = add_linked(project, first["id"], "extend_after")
        # The new take's first frame is the source's frame 141 - 39, wherever the source is cut.
        self.assertEqual(clip["track"]["from"], round(2.0 + (141 - 39) / 24.0, 6))


class JoinTest(unittest.TestCase):
    """A linked clip against its source, after both are rendered."""

    def setUp(self):
        self.project = scene(124, 124)
        self.first = store.ordered(self.project)[0]
        self.clip, self.item = add_linked(self.project, self.first["id"], "extend_after")
        accept_take(self.project, self.clip["id"], take("tx", 141))

    def test_a_fresh_join_is_whole(self):
        self.assertFalse(store.is_stale(self.project, self.clip))

    def test_trimming_the_source_breaks_the_join(self):
        update_item(self.project, self.first["id"], {"out": 110}, ripple=True)
        self.assertTrue(store.is_stale(self.project, self.clip))

    def test_a_new_take_of_the_source_breaks_the_join(self):
        accept_take(self.project, self.first["clip"], take("t9", 124))
        self.assertTrue(store.is_stale(self.project, self.clip))

    def test_trimming_the_sources_other_end_does_not(self):
        self.first["in"] = 10
        self.assertFalse(store.is_stale(self.project, self.clip))

    def test_rendering_again_holds_the_same_frames_of_the_sources_new_take(self):
        accept_take(self.project, self.first["clip"], take("t9", 124))
        store.refresh_link(self.project, self.clip)
        self.assertEqual((self.clip["link"]["a"]["file"], self.clip["link"]["a"]["last"]), ("t9.mp4", 124))
        self.assertEqual(self.first["out"], 115)
        self.assertFalse(store.is_stale(self.project, self.clip))

    def test_rendering_again_after_a_trim_joins_at_the_new_cut(self):
        update_item(self.project, self.first["id"], {"out": 110}, ripple=True)
        store.refresh_link(self.project, self.clip)
        # 110 moves back onto the grid at 107, and the source stops 9 frames inside the frames now held.
        self.assertEqual(self.clip["link"]["a"]["last"], 107)
        self.assertEqual(self.first["out"], 98)
        self.assertEqual(self.item["start"], item_end(self.first))
        self.assertFalse(store.is_stale(self.project, self.clip))

    def test_a_join_made_before_handovers_is_read_as_it_was(self):
        project = scene(124, 124)
        first = store.ordered(project)[0]
        side = store.side_of(project, first)
        link = {"mode": "extend_after", "hold": 39, "a": side}
        self.assertEqual(store.linked_range(link, 141), (39, 141))
        project["clips"].append({"id": "cx", "name": "Old", "takes": [take("to", 141)], "active": "to", "link": link})
        project["timeline"].append({"id": "ix", "clip": "cx", "start": 124, "in": 39, "out": 141})
        self.assertFalse(store.is_stale(project, project["clips"][-1]))
        store.refresh_link(project, project["clips"][-1])
        self.assertEqual((first["out"], link["a"]["last"]), (124, 124))


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

    def test_a_final_records_the_size_it_comes_out_at(self):
        project = new_project("Test")
        clip, _ = add_clip(project, {"action": "One."})
        job, _ = plan_render(project, "Test", clip["id"], "final")
        self.assertEqual(job["size"], [1920, 1088])
        job["id"] = "prompt-1"
        add_pending(project, job)
        finish_render(project, "prompt-1", {"filename": "x.mp4", "subfolder": "s"}, {"frames": 124, "poster": "x.jpg"})
        take = store.active_take(clip)
        self.assertEqual(take["size"], [1920, 1088])
        self.assertEqual(take["models"]["lora_strength"], 1.2)
        self.assertEqual(take["models"]["profile"], "hyperflow")

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
    def test_removing_versions_preserves_other_placements_and_late_renders(self):
        data = scene(124)
        clip, item = data["clips"][0], data["timeline"][0]
        for name, frames in (("short", 32), ("last", 124)):
            accept_take(data, clip["id"], take(name, frames))
        item["in"] = 81
        other = dict(copy.deepcopy(clip), id="independent")
        data["clips"].append(other)
        old = copy.deepcopy(data)
        args = {"op": "remove_takes", "clip": clip["id"], "takes": ["t0", "last"]}
        store.apply_op(data, args, {})
        self.assertEqual(([t["id"] for t in clip["takes"]], clip["active"]), (["short"], "short"))
        self.assertEqual((item["in"], item["out"]), (31, 32))
        self.assertEqual(len(other["takes"]), 3)
        accept_take(data, clip["id"], take("arrived", 32))
        for base in (old, data, UNKNOWN):
            stale = copy.deepcopy(old)
            stale["style"] = "Ink"
            saved = store.merge_editor_save(base, stale, data)
            self.assertEqual([t["id"] for t in saved["clips"][0]["takes"]], ["short", "arrived"])
            self.assertEqual(saved["clips"][0]["active"], "arrived")
            self.assertEqual(saved["style"], "Ink")

    def test_removing_unknown_or_last_versions_is_atomic(self):
        data = scene(90)
        clip = data["clips"][0]
        accept_take(data, clip["id"], take("new", 90))
        before = copy.deepcopy(data)
        for ids in ([], ["unknown", "new"], ["t0", "new"], "new"):
            with self.assertRaises((ValueError, KeyError)):
                store.remove_takes(data, clip["id"], ids)
            self.assertEqual(data, before)
        store.remove_takes(data, clip["id"], ["t0"])
        self.assertEqual(clip["active"], "new")
        restored = store.restore_revision(data, before)
        self.assertEqual([t["id"] for t in restored["clips"][0]["takes"]], ["t0", "new"])

    def test_clearing_a_shot_keeps_edits_and_accepts_a_replacement(self):
        data = scene(124)
        clip, item = data["clips"][0], data["timeline"][0]
        item.update({"in": 16, "out": 89})
        split_item(data, item["id"], 32)
        before = copy.deepcopy(data)
        keep = copy.deepcopy(data["timeline"])
        op = {"op": "remove_takes", "clip": clip["id"], "takes": ["t0"], "allow_empty": True}
        self.assertIsNone(store.apply_op(data, op, {})["active"])
        self.assertEqual((clip["action"], data["timeline"]), (before["clips"][0]["action"], keep))
        for base in (before, data, UNKNOWN):
            stale = store.merge_editor_save(base, before, data)
            self.assertEqual((stale["clips"][0]["takes"], stale["clips"][0]["active"]), ([], None))
            self.assertTrue(stale["clips"][0]["awaiting_replacement"])
        accept_take(data, clip["id"], take("again", 124))
        self.assertEqual((clip["active"], data["timeline"]), ("again", keep))
        self.assertNotIn("awaiting_replacement", clip)
        clip["kind"] = "footage"
        with self.assertRaisesRegex(ValueError, "imported footage"):
            store.remove_takes(data, clip["id"], ["again"], allow_empty=True)
        updated = copy.deepcopy(data)
        updated["clips"][0]["takes"][0]["strip"] = {"sprite": "made.jpg"}
        self.assertEqual(store.merge_editor_save(data, updated, data)["clips"][0]["takes"][0]["strip"],
                         {"sprite": "made.jpg"})

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

    def test_an_undone_delete_brings_the_clip_back_in_its_place(self):
        before = scene(124, 124, 124)
        deleted = copy.deepcopy(before)
        store.remove_clip(deleted, deleted["clips"][1]["id"])
        merged = merge(deleted, copy.deepcopy(before), deleted)
        self.assertEqual([c["id"] for c in merged["clips"]], [c["id"] for c in before["clips"]])

    def test_a_clip_the_other_side_added_stays_where_it_was_added(self):
        base = scene(124, 124)
        mine, theirs = copy.deepcopy(base), copy.deepcopy(base)
        mine["clips"][0]["action"] = "Mine."
        add_clip(theirs, {"action": "Theirs."})
        merged = merge(base, mine, theirs)
        self.assertEqual([c["action"] for c in merged["clips"]], ["Mine.", "Shot 1.", "Theirs."])


class LanguageAndVoiceTest(unittest.TestCase):
    def voiced(self, pictured):
        project = new_project("Test")
        project["language"] = "Portuguese"
        voice = {"file": "voice.wav", "subfolder": "h3take_projects/Test/assets"}
        fields = {"name": "Ana", "description": "a calm woman in her thirties", "voice": voice}
        if pictured:
            fields.update({"file": "ana.png", "subfolder": "h3take_projects/Test/assets"})
        person = store.add_cast(project, fields)
        clip, _ = add_clip(project, {"action": "A sunny office.", "speaker": "@Ana", "dialogue": "Bom dia!",
                                     "cast": [person["id"]] if pictured else []})
        return project, clip

    def test_dialogue_keeps_its_language(self):
        project = new_project("Test")
        project["language"] = "Portuguese"
        clip, _ = add_clip(project, {"action": "A sunny office.", "speaker": "A woman", "dialogue": "Bom dia!"})
        self.assertIn("<d>[Portuguese] Bom dia!</d>", store.clip_prompt(project, clip))
        update_clip(project, clip["id"], {"language": "Spanish"})
        self.assertIn("<d>[Spanish] Bom dia!</d>", store.clip_prompt(project, clip))

    def test_a_pictured_speaker_uses_their_voice(self):
        project, clip = self.voiced(pictured=True)
        text = store.clip_prompt(project, clip)
        self.assertIn("<Audio 1> is the voice-timbre reference for <Subject 1> (S1).", text)
        self.assertIn("[reference generation + audio reference]", text)
        self.assertIn("<Subject 1> (S1), using the voice timbre referenced from <Audio 1>, says: <d>[Portuguese] Bom dia!</d>", text)
        graph = store.render_graph(project, "Test", clip, "t1", "draft", 1)
        self.assertEqual(graph["10"]["inputs"]["ref_audios.ref_audio_0"], ["60", 0])
        self.assertEqual(graph["60"]["inputs"]["file"], "h3take_projects/Test/assets/voice.wav")

    def test_an_unseen_narrator_is_written_out(self):
        project, clip = self.voiced(pictured=False)
        text = store.clip_prompt(project, clip)
        self.assertIn("<Audio 1> is the voice-timbre reference for a calm woman in her thirties (S1).", text)
        self.assertIn("A calm woman in her thirties (S1), using the voice timbre", text)
        self.assertNotIn("<Subject", text)
        self.assertNotIn("ref_images.", str(store.render_graph(project, "Test", clip, "t1", "draft", 1)["10"]["inputs"]))

    def test_the_speaker_id_follows_the_identity(self):
        project, clip = self.voiced(pictured=False)
        project["cast"][0]["description"] = "A warm, clear narrator"
        update_clip(project, clip["id"], {"speaker": "@Ana, off-screen"})
        text = store.clip_prompt(project, clip)
        self.assertIn("is the voice-timbre reference for a warm, clear narrator (S1).", text)
        self.assertIn("A warm, clear narrator (S1), off-screen, using the voice timbre referenced from <Audio 1>, says:", text)
        pictured, shot = self.voiced(pictured=True)
        update_clip(pictured, shot["id"], {"speaker": "@Ana, off-screen"})
        self.assertIn("<Subject 1> (S1), off-screen, using the voice", store.clip_prompt(pictured, shot))

    def test_no_line_means_no_voice(self):
        project, clip = self.voiced(pictured=True)
        update_clip(project, clip["id"], {"dialogue": ""})
        self.assertNotIn("<Audio 1>", store.clip_prompt(project, clip))


class TrackTest(unittest.TestCase):
    track = {"file": "narration.mp3", "subfolder": "h3take_projects/Test/assets", "from": 0}

    def presenter(self):
        project = new_project("Test")
        clip, item = add_clip(project, {"action": "A woman talks to the camera.", "seconds": 5, "track": self.track})
        return project, clip, item

    def test_a_clip_is_generated_to_its_slice_of_the_track(self):
        project, clip, _ = self.presenter()
        graph = store.render_graph(project, "Test", clip, "t1", "draft", 1)
        inputs = graph["10"]["inputs"]
        self.assertEqual((inputs["audio_hold"], inputs["audio_align"], inputs["source_audio"]), ("full", "take", ["61", 0]))
        self.assertEqual(graph["61"]["inputs"]["file"], "h3take_projects/Test/assets/narration.mp3")
        self.assertEqual(graph["61"]["inputs"]["start"], 0.0)
        job, _ = plan_render(project, "Test", clip["id"], "draft")
        self.assertAlmostEqual(job["track"]["seconds"], 124 / 24.0)

    def test_an_extension_holds_the_next_slice(self):
        project, clip, item = self.presenter()
        accept_take(project, clip["id"], take("t0", 124))
        after, placed = add_linked(project, item["id"], "extend_after", seconds=4.25, action="She keeps talking.")
        # Its first frame is the first held frame, 39 before the end of the source.
        self.assertAlmostEqual(after["track"]["from"], (124 - 39) / 24.0, places=5)
        # The timeline shows it from frame 39, so the track runs on without a jump at the cut.
        self.assertAlmostEqual(after["track"]["from"] + placed["in"] / 24.0, item_end(item) / 24.0, places=5)
        graph = store.render_graph(project, "Test", after, "t2", "draft", 1)
        self.assertEqual(graph["10"]["inputs"]["source_audio"], ["61", 0])
        self.assertEqual(graph["10"]["inputs"]["source_video"], ["6", 0])

    def test_a_chain_keeps_the_track_in_step(self):
        project, clip, item = self.presenter()
        accept_take(project, clip["id"], take("t0", 124))
        for index in range(3):
            clip, item = add_linked(project, item["id"], "extend_after", seconds=4.25)
            accept_take(project, clip["id"], take("t%s" % (index + 1), clip_frames(clip)))
            self.assertAlmostEqual(clip["track"]["from"] + item["in"] / 24.0, item["start"] / 24.0, places=5)

    def test_extending_before_the_start_of_the_track_is_refused(self):
        project, clip, item = self.presenter()
        accept_take(project, clip["id"], take("t0", 124))
        with self.assertRaisesRegex(ValueError, "too late"):
            add_linked(project, item["id"], "extend_before")

    def test_a_track_needs_full_speed_to_continue(self):
        project, clip, item = self.presenter()
        accept_take(project, clip["id"], take("t0", 124))
        set_speed(project, item["id"], 2)
        with self.assertRaisesRegex(ValueError, "100% speed"):
            add_linked(project, item["id"], "extend_after")

    def test_linked_clips_keep_the_cast(self):
        project, clip, item = self.presenter()
        update_clip(project, clip["id"], {"cast": ["p1"]})
        accept_take(project, clip["id"], take("t0", 124))
        after, _ = add_linked(project, item["id"], "extend_after")
        self.assertEqual(after["cast"], ["p1"])

    def test_linked_clips_keep_the_model_and_loras_of_their_source(self):
        # An extension carries on the same shot: drawn by another model, its look would change at the join.
        project, clip, item = self.presenter()
        loras = [{"file": "style.safetensors", "strength": 0.8}]
        update_clip(project, clip["id"], {"profile": "h3_turbo", "loras": loras})
        accept_take(project, clip["id"], take("t0", 124))
        after, _ = add_linked(project, item["id"], "extend_after")
        self.assertEqual(after["profile"], "h3_turbo")
        self.assertEqual([row["file"] for row in after["loras"]], ["style.safetensors"])
        after["loras"][0]["strength"] = 0.1
        self.assertEqual(clip["loras"][0]["strength"], 0.8)

    def test_a_linked_clip_of_a_clip_on_the_projects_model_names_none(self):
        project, clip, item = self.presenter()
        accept_take(project, clip["id"], take("t0", 124))
        after, _ = add_linked(project, item["id"], "extend_after")
        self.assertNotIn("profile", after)
        self.assertNotIn("loras", after)

    def test_a_track_takes_the_place_of_a_voice(self):
        project, clip, _ = self.presenter()
        store.add_cast(project, {"name": "Ana", "voice": {"file": "v.wav", "subfolder": "s"}})
        update_clip(project, clip["id"], {"speaker": "@Ana", "dialogue": "Oi!"})
        self.assertNotIn("60", store.render_graph(project, "Test", clip, "t1", "draft", 1))

    def test_a_bad_from_is_refused(self):
        with self.assertRaisesRegex(ValueError, "0 or more"):
            add_clip(new_project("Test"), {"action": "x", "track": dict(self.track, **{"from": -1})})


def clip_frames(clip):
    return store.linked_length(clip["link"], clip["seconds"])


class FormatTest(unittest.TestCase):
    def test_a_vertical_project_renders_vertical(self):
        project = new_project("Test")
        store.set_project(project, {"format": "9:16"})
        clip, _ = add_clip(project, {"action": "One."})
        inputs = store.render_graph(project, "Test", clip, "t1", "final", 1)["10"]["inputs"]
        self.assertEqual((inputs["width"], inputs["height"]), (768, 1344))

    def test_an_unknown_format_is_refused(self):
        with self.assertRaisesRegex(ValueError, "16:9, 9:16, 1:1"):
            store.set_project(new_project("Test"), {"format": "4:0"})

    def test_footage_edits_but_never_renders(self):
        project = new_project("Test")
        clip, item = store.add_footage(project, "Screen recording", take("f1", 240))
        self.assertEqual((item["in"], item["out"], clip["seconds"]), (0, 240, 10.0))
        with self.assertRaisesRegex(ValueError, "imported footage"):
            plan_render(project, "Test", clip["id"], "draft")
        extension, _ = add_linked(project, item["id"], "extend_after")
        self.assertEqual(extension["link"]["a"]["file"], "f1.mp4")


class BatchTest(unittest.TestCase):
    def test_later_operations_name_earlier_ones(self):
        project = new_project("Test")
        names = {}
        apply_op(project, {"op": "add_clip", "as": "open", "fields": {"action": "One."}}, names)
        apply_op(project, {"op": "update_item", "item": "$open", "fields": {"start": 10}}, names)
        apply_op(project, {"op": "update_clip", "clip": "$open", "fields": {"name": "Opening"}}, names)
        self.assertEqual(project["clips"][0]["name"], "Opening")
        self.assertEqual(project["timeline"][0]["start"], 10)

    def test_a_new_cast_member_can_be_named_in_the_same_batch(self):
        project = new_project("Test")
        names = {}
        apply_op(project, {"op": "add_cast", "as": "her", "fields": {"name": "Ana", "file": "a.png"}}, names)
        apply_op(project, {"op": "add_clip", "fields": {"action": "Ana talks.", "cast": ["$her"]}}, names)
        self.assertEqual(project["clips"][0]["cast"], [project["cast"][0]["id"]])

    def test_an_unknown_name_is_refused(self):
        with self.assertRaisesRegex(ValueError, r"\$nope"):
            apply_op(new_project("Test"), {"op": "remove_clip", "clip": "$nope"}, {})

    def test_the_summary_reads_like_a_timeline(self):
        project = scene(124, 124)
        summary = describe(project)
        self.assertEqual([t["at"] for t in summary["timeline"]], ["00:00:00", "00:05:04"])
        self.assertEqual(summary["length"], "00:10:08")


class BinTest(unittest.TestCase):
    def test_a_new_bin_can_be_named_in_the_same_batch(self):
        project = new_project("Test")
        names = {}
        apply_op(project, {"op": "add_bin", "as": "broll", "fields": {"name": "B-roll"}}, names)
        apply_op(project, {"op": "add_clip", "as": "shot", "fields": {"action": "Waves.", "bin": "$broll"}}, names)
        apply_op(project, {"op": "add_clip", "fields": {"action": "Loose."}}, names)
        summary = describe(project)
        self.assertEqual(summary["bins"], [{"id": names["broll"]["bin"], "name": "B-roll", "clips": 1}])
        self.assertEqual([c.get("bin") for c in summary["clips"]], [names["broll"]["bin"], None])

    def test_a_clip_can_only_go_in_a_bin_the_project_has(self):
        project = scene(124)
        with self.assertRaisesRegex(KeyError, "no bin"):
            update_clip(project, project["clips"][0]["id"], {"bin": "b-missing"})

    def test_null_takes_a_clip_out_of_its_bin(self):
        project = scene(124)
        found = store.add_bin(project)
        clip = project["clips"][0]
        update_clip(project, clip["id"], {"bin": found["id"]})
        self.assertEqual((found["name"], clip["bin"]), ("Bin 1", found["id"]))
        update_clip(project, clip["id"], {"bin": None})
        self.assertNotIn("bin", clip)

    def test_removing_a_bin_keeps_its_clips(self):
        project = scene(124, 124)
        found = store.add_bin(project, {"name": "Selects"})
        for clip in project["clips"]:
            update_clip(project, clip["id"], {"bin": found["id"]})
        apply_op(project, {"op": "remove_bin", "bin": found["id"]}, {})
        self.assertEqual((project["bins"], len(project["clips"])), ([], 2))
        self.assertFalse(any("bin" in c for c in project["clips"]))

    def test_a_rename_to_nothing_keeps_the_name(self):
        project = new_project("Test")
        found = store.add_bin(project, {"name": "Selects"})
        apply_op(project, {"op": "update_bin", "bin": found["id"], "fields": {"name": "  "}}, {})
        self.assertEqual(found["name"], "Selects")
        apply_op(project, {"op": "update_bin", "bin": found["id"], "fields": {"name": " Keepers "}}, {})
        self.assertEqual(found["name"], "Keepers")

    def test_an_extension_files_itself_with_its_source(self):
        project = scene(124)
        found = store.add_bin(project)
        update_clip(project, project["clips"][0]["id"], {"bin": found["id"]})
        clip, _ = add_linked(project, project["timeline"][0]["id"], "extend_after")
        self.assertEqual(clip["bin"], found["id"])

    def test_an_import_into_a_bin_removed_meanwhile_is_kept_loose(self):
        project = new_project("Test")
        clip, _ = store.add_footage(project, "Beach", take("f1", 96), bin_id="b-gone")
        self.assertNotIn("bin", clip)
        self.assertEqual(len(project["clips"]), 1)

    def test_removing_a_clip_takes_it_off_the_timeline(self):
        project = scene(124, 124)
        gone = project["clips"][0]["id"]
        apply_op(project, {"op": "remove_clip", "clip": gone}, {})
        self.assertEqual([c["id"] for c in project["clips"]], [project["timeline"][0]["clip"]])
        self.assertEqual(len(project["timeline"]), 1)


if __name__ == "__main__":
    unittest.main()
