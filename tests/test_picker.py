import unittest

from project import project_card


def take(name, poster=True):
    return {"id": name, "file": name + ".mp4", "subfolder": "h3take_projects/a/clips", "poster": name + ".jpg" if poster else None, "frames": 90}


class ProjectCardTest(unittest.TestCase):
    def test_an_empty_project_has_no_picture_and_no_length(self):
        card = project_card({"name": "Morning teaser"})
        self.assertEqual(card, {"name": "Morning teaser", "format": "16:9", "clips": 0, "frames": 0, "picture": None, "rendering": 0})

    def test_the_picture_is_the_first_rendered_clip_on_the_timeline(self):
        project = {"format": "9:16", "pending": [{"clip": "c2"}],
                   "clips": [{"id": "c1", "takes": []}, {"id": "c2", "takes": [take("old"), take("new")], "active": "old"},
                             {"id": "c3", "takes": [take("later")]}],
                   "timeline": [{"id": "i3", "clip": "c3", "start": 180, "in": 0, "out": 90},
                                {"id": "i1", "clip": "c1", "start": 0, "in": 0, "out": 90},
                                {"id": "i2", "clip": "c2", "start": 90, "in": 0, "out": 90, "speed": 2}]}
        card = project_card(project)
        self.assertEqual(card["picture"], {"file": "old.jpg", "subfolder": "h3take_projects/a/clips"})
        self.assertEqual((card["format"], card["clips"], card["frames"], card["rendering"]), ("9:16", 3, 270, 1))

    def test_a_project_with_nothing_on_the_timeline_shows_a_rendered_clip_from_its_bin(self):
        card = project_card({"clips": [{"id": "c1", "takes": [take("sound only", poster=False)]}, {"id": "c2", "takes": [take("shot")]}], "timeline": []})
        self.assertEqual((card["clips"], card["picture"]["file"]), (0, "shot.jpg"))

    def test_a_broken_timeline_item_is_left_out_instead_of_failing_the_list(self):
        card = project_card({"format": "4:0", "timeline": [{"id": "i1", "clip": "c1", "start": 0}, "junk",
                                                          {"id": "i2", "clip": "c1", "start": 10, "in": 0, "out": 40}]})
        self.assertEqual((card["format"], card["clips"], card["frames"]), ("16:9", 1, 50))


if __name__ == "__main__":
    unittest.main()
