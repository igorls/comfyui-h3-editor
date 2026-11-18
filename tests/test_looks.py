import json
import os
import tempfile
import unittest

import looks


class LookLibraryTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = os.path.join(self.folder.name, "user", "h3take", "looks.json")

    def test_builtins_are_available_without_creating_a_file(self):
        rows = looks.library(self.path)
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(row["builtin"] and row["style"] for row in rows))
        rows[0]["style"] = "changed"
        self.assertNotEqual(looks.library(self.path)[0]["style"], "changed")
        self.assertFalse(os.path.exists(self.path))

    def test_saving_and_updating_survives_a_new_library_read(self):
        saved = looks.save(self.path, "My style", "Soft light, ink outlines")
        self.assertEqual(looks.library(self.path)[-1], saved)
        updated = looks.save(self.path, "Renamed", "Strong shadows", saved["id"])
        self.assertEqual(updated["id"], saved["id"])
        self.assertEqual(looks.library(self.path)[-1]["style"], "Strong shadows")
        with open(self.path, encoding="utf-8") as file:
            self.assertEqual(json.load(file), [updated])

    def test_deleting_a_library_entry_does_not_change_a_project_copy(self):
        saved = looks.save(self.path, "Night", "Blue moonlight")
        project = {"style": saved["style"]}
        looks.remove(self.path, saved["id"])
        self.assertEqual(len(looks.library(self.path)), 8)
        self.assertEqual(project["style"], "Blue moonlight")

    def test_names_cannot_accidentally_overwrite_another_saved_look(self):
        saved = looks.save(self.path, "Night", "Blue moonlight")
        with self.assertRaisesRegex(ValueError, "already has that name"):
            looks.save(self.path, "night", "Red")
        with self.assertRaisesRegex(ValueError, "gone"):
            looks.save(self.path, "Night", "Red", "builtin_noir")
        self.assertEqual(looks.library(self.path)[-1], saved)

    def test_invalid_input_and_corrupt_libraries_are_never_overwritten(self):
        for name, style in (("", "text"), ("a" * 81, "text"), ("Name", " "), ("Name", None)):
            with self.assertRaises(ValueError):
                looks.save(self.path, name, style)
        os.makedirs(os.path.dirname(self.path))
        with open(self.path, "w", encoding="utf-8") as file:
            file.write("not json")
        with self.assertRaises(ValueError):
            looks.save(self.path, "Name", "Text")
        with open(self.path, encoding="utf-8") as file:
            self.assertEqual(file.read(), "not json")


if __name__ == "__main__":
    unittest.main()
