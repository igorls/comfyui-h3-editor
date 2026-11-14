import unittest

from project import add_cast, cast_assets, cast_edited, cast_from_catalog, catalog_state, find_cast, save_to_catalog, update_cast


def moved(found, subfolder):
    return {key: {"file": "copy_" + key[1], "subfolder": subfolder} for key in found}


class CatalogTest(unittest.TestCase):
    def setUp(self):
        self.project = {"cast": [], "clips": []}
        self.person = add_cast(self.project, {"name": "Marta", "description": "a woman in a cream blouse", "file": "marta.png",
                                              "subfolder": "h3take_projects/a/assets",
                                              "voice": {"file": "marta.wav", "subfolder": "h3take_projects/a/assets"}})
        self.catalog = []

    def save(self, person):
        return save_to_catalog(self.catalog, person, moved(cast_assets(person), "h3take_projects/_cast/files"))

    def test_saving_copies_the_member_into_the_catalog_and_links_it(self):
        entry = self.save(self.person)
        self.assertEqual((entry["name"], entry["version"]), ("Marta", 1))
        self.assertEqual((entry["file"], entry["subfolder"]), ("copy_marta.png", "h3take_projects/_cast/files"))
        self.assertEqual(entry["voice"]["file"], "copy_marta.wav")
        self.assertNotIn("catalog", entry)
        self.assertEqual(self.person["catalog"]["id"], entry["id"])
        self.assertEqual(catalog_state(self.person, self.catalog), "current")
        # The project keeps its own files.
        self.assertEqual(self.person["file"], "marta.png")

    def test_a_change_in_the_project_shows_and_saving_again_updates_the_entry(self):
        entry = self.save(self.person)
        update_cast(self.project, self.person["id"], {"description": "a woman in a green apron"})
        self.assertTrue(cast_edited(self.person))
        self.assertEqual(catalog_state(self.person, self.catalog), "changed")
        again = self.save(self.person)
        self.assertIs(again, entry)
        self.assertEqual((entry["version"], entry["description"], len(self.catalog)), (2, "a woman in a green apron", 1))
        self.assertEqual(catalog_state(self.person, self.catalog), "current")

    def test_another_project_brings_a_copy_in_and_sees_a_newer_version(self):
        entry = self.save(self.person)
        other = {"cast": [], "clips": []}
        copy = cast_from_catalog(other, entry, moved(cast_assets(entry), "h3take_projects/b/assets"))
        self.assertEqual((copy["name"], copy["file"], copy["subfolder"]), ("Marta", "copy_copy_marta.png", "h3take_projects/b/assets"))
        self.assertEqual(copy["voice"]["subfolder"], "h3take_projects/b/assets")
        self.assertEqual(catalog_state(copy, self.catalog), "current")
        # The first project changes her and saves her to the catalog: the second sees a newer version.
        update_cast(self.project, self.person["id"], {"description": "a woman in a green apron"})
        self.save(self.person)
        self.assertEqual(catalog_state(copy, self.catalog), "newer")
        self.assertFalse(cast_edited(copy))
        cast_from_catalog(other, entry, moved(cast_assets(entry), "h3take_projects/b/assets"), copy["id"])
        updated = find_cast(other, copy["id"])
        self.assertEqual((updated["description"], updated["catalog"]["version"]), ("a woman in a green apron", 2))
        self.assertEqual(catalog_state(updated, self.catalog), "current")

    def test_an_update_replaces_what_the_entry_no_longer_has(self):
        update_cast(self.project, self.person["id"], {"pictures": [{"file": "side.png", "subfolder": "h3take_projects/a/assets"}]})
        entry = self.save(self.person)
        other = {"cast": [], "clips": []}
        copy = cast_from_catalog(other, entry, {})
        self.assertEqual(len(copy["pictures"]), 1)
        update_cast(self.project, self.person["id"], {"pictures": [], "voice": None})
        self.save(self.person)
        cast_from_catalog(other, entry, {}, copy["id"])
        self.assertNotIn("pictures", copy)
        self.assertNotIn("voice", copy)

    def test_a_member_not_from_the_catalog_or_removed_from_it(self):
        self.assertIsNone(catalog_state(self.person, self.catalog))
        self.save(self.person)
        self.assertEqual(catalog_state(self.person, []), "missing")


if __name__ == "__main__":
    unittest.main()
