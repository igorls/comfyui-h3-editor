import unittest

import project as store
from project import (accept_effect_take, add_effect, apply_op, describe, effect_stale, export_command, export_plan, new_project,
                     remove_effect, update_effect)


def cut():
    """A project with one rendered clip of 90 frames on the timeline."""
    project = new_project("x")
    project["clips"].append({"id": "c1", "name": "Shot", "action": "A", "seconds": 3.75, "takes": [
        {"id": "t1", "file": "a.mp4", "subfolder": "p/clips", "frames": 90, "quality": "draft"}], "active": "t1"})
    project["timeline"].append({"id": "i1", "clip": "c1", "start": 0, "in": 0, "out": 90})
    return project


class EffectTest(unittest.TestCase):
    def test_a_new_vfx_clip_starts_as_a_title_and_is_not_rendered(self):
        project = cut()
        effect = add_effect(project, {"start": 12})
        self.assertIn("export function frame", effect["source"])
        self.assertEqual((effect["start"], effect["frames"]), (12, 72))
        self.assertEqual(describe(project)["effects"][0]["state"], "not rendered")

    def test_two_vfx_clips_never_share_a_frame(self):
        project = cut()
        first = add_effect(project, {"start": 0, "frames": 48})
        with self.assertRaises(ValueError):
            add_effect(project, {"start": 47, "frames": 10})
        second = add_effect(project, {"start": 48, "frames": 10})
        with self.assertRaises(ValueError):
            update_effect(project, second["id"], {"start": 40})
        update_effect(project, first["id"], {"frames": 40})
        update_effect(project, second["id"], {"start": 40})

    def test_a_take_goes_out_of_date_when_what_it_draws_changes(self):
        project = cut()
        effect = add_effect(project, {"start": 0, "frames": 48})
        accept_effect_take(project, effect["id"], {"id": "t9", "file": "fx.mp4", "frames": 48})
        self.assertFalse(effect_stale(project, effect))
        self.assertEqual(describe(project)["effects"][0]["state"], "rendered")
        update_effect(project, effect["id"], {"name": "Title card"})
        self.assertFalse(effect_stale(project, effect))                 # a name draws nothing
        update_effect(project, effect["id"], {"params": {"text": "Hello"}})
        self.assertTrue(effect_stale(project, effect))

    def test_a_take_goes_out_of_date_when_the_picture_below_changes(self):
        project = cut()
        effect = add_effect(project, {"start": 0, "frames": 48})
        accept_effect_take(project, effect["id"], {"id": "t9", "file": "fx.mp4", "frames": 48})
        project["timeline"][0]["in"] = 5
        self.assertTrue(effect_stale(project, effect))
        project["timeline"][0]["in"] = 0
        self.assertFalse(effect_stale(project, effect))
        project["clips"][0]["takes"].append({"id": "t2", "file": "b.mp4", "frames": 90, "quality": "draft"})
        project["clips"][0]["active"] = "t2"
        self.assertTrue(effect_stale(project, effect))

    def test_a_clip_outside_its_span_does_not_touch_it(self):
        project = cut()
        effect = add_effect(project, {"start": 0, "frames": 48})
        accept_effect_take(project, effect["id"], {"id": "t9", "file": "fx.mp4", "frames": 48})
        project["clips"].append({"id": "c2", "name": "Next", "action": "B", "seconds": 2, "takes": [], "active": None})
        project["timeline"].append({"id": "i2", "clip": "c2", "start": 90, "in": 0, "out": 48})
        self.assertFalse(effect_stale(project, effect))

    def test_the_export_lays_a_vfx_take_over_its_span(self):
        project = cut()
        effect = add_effect(project, {"start": 24, "frames": 48})
        accept_effect_take(project, effect["id"], {"id": "t9", "file": "fx.mp4", "frames": 48})
        args = export_command(project, {"c1": "/out/a.mp4"}, "/out/cut.mp4", effect_paths={effect["id"]: "/out/fx.mp4"})
        graph = args[args.index("-filter_complex") + 1]
        self.assertIn("/out/fx.mp4", args)
        self.assertIn("overlay=eof_action=pass:enable='between(t,1.00000,2.97917)'", graph)
        self.assertIn("setpts=PTS-STARTPTS+1.00000/TB", graph)
        self.assertEqual(args[args.index("-map") + 1], "[o1]")

    def test_a_vfx_clip_with_no_take_is_left_out_and_counted(self):
        project = cut()
        add_effect(project, {"start": 24, "frames": 48})
        args = export_command(project, {"c1": "/out/a.mp4"}, "/out/cut.mp4", effect_paths={})
        self.assertNotIn("overlay", args[args.index("-filter_complex") + 1])
        plan = export_plan(project)
        self.assertEqual((plan["effects"], plan["effects_missing"], plan["effects_stale"]), (0, 1, 0))

    def test_agents_add_change_and_remove_vfx_clips(self):
        project = cut()
        names = {}
        made = apply_op(project, {"op": "add_effect", "as": "title", "fields": {"start": 0, "frames": 24, "source": "export function frame() {}"}}, names)
        apply_op(project, {"op": "update_effect", "effect": "$title", "fields": {"params": {"size": 0.2}}}, names)
        self.assertEqual(store.find_effect(project, made["effect"])["params"], {"size": 0.2})
        apply_op(project, {"op": "remove_effect", "effect": "$title"}, names)
        self.assertEqual(project["effects"], [])

    def test_a_render_request_waits_for_one_editor_to_claim_it(self):
        project = cut()
        effect = add_effect(project, {"start": 0, "frames": 24})
        apply_op(project, {"op": "render_effect", "effect": effect["id"]}, {})
        self.assertEqual(describe(project)["effects"][0]["state"], "waiting for an open editor to render it")
        self.assertEqual(store.effects_waiting(project), [effect])
        self.assertTrue(store.claim_effect_render(project, effect["id"], "j1"))
        self.assertFalse(store.claim_effect_render(project, effect["id"], "j2"))       # a second editor is turned away
        self.assertEqual(describe(project)["effects"][0]["state"], "rendering")
        accept_effect_take(project, effect["id"], {"id": "t9", "file": "fx.mp4", "frames": 24})
        self.assertEqual(store.effects_waiting(project), [])
        self.assertEqual(describe(project)["effects"][0]["state"], "rendered")

    def test_a_clip_names_the_pictures_its_code_may_draw(self):
        project = cut()
        effect = add_effect(project, {"inputs": {"logo": {"file": "logo.png", "subfolder": "p/assets"}}})
        self.assertEqual(effect["inputs"], {"logo": {"file": "logo.png", "subfolder": "p/assets"}})
        accept_effect_take(project, effect["id"], {"id": "t9", "file": "fx.mp4", "frames": 72})
        update_effect(project, effect["id"], {"inputs": {"logo": {"file": "other.png"}}})
        self.assertTrue(effect_stale(project, effect))                  # another picture draws another take
        for inputs in ({"my logo": {"file": "a.png"}}, {"logo": {}}, {"logo": "a.png"}, ["a.png"]):
            with self.assertRaises(ValueError):
                update_effect(project, effect["id"], {"inputs": inputs})

    def test_bad_fields_are_refused(self):
        project = cut()
        for fields in ({"frames": 0}, {"start": -1}, {"source": 5}, {"params": []}, {"colour": "red"}):
            with self.assertRaises(ValueError):
                add_effect(project, fields)
        with self.assertRaises(KeyError):
            remove_effect(project, "nope")


if __name__ == "__main__":
    unittest.main()
