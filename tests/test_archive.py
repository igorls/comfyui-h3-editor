import json
import os
import shutil
import tempfile
import unittest
import zipfile

import archive
import project as project_module

ROOT = "h3take_projects"


def touch(base, path, size=10):
    full = os.path.join(base, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "wb") as f:
        f.write(b"x" * size)


def take(clip, take_id, folder="Demo"):
    return {"id": take_id, "file": "%s_%s_00001_.mp4" % (clip, take_id), "subfolder": "%s/%s/clips" % (ROOT, folder),
            "poster": "%s_%s_00001_.jpg" % (clip, take_id), "quality": "draft", "frames": 49, "seed": 1}


def demo(base, folder="Demo"):
    """A project folder: clip ca on the timeline with two takes, clip cb off it with one, footage cf off it, a cast
    picture, a sound with its peaks, a VFX clip with two takes, and this machine's history and previews."""
    data = {
        "name": "Demo", "profile": "hyperflow", "loras": [{"file": "a.safetensors", "strength": 1}],
        "clips": [
            {"id": "ca", "name": "A", "action": "a", "takes": [take("ca", "t1"), take("ca", "t2")], "active": "t2"},
            {"id": "cb", "name": "B", "action": "b", "takes": [take("cb", "t3")], "active": "t3"},
            {"id": "cf", "name": "F", "kind": "footage", "takes": [take("cf", "t4")], "active": "t4"},
        ],
        "timeline": [{"id": "i1", "clip": "ca", "start": 0, "in": 0, "out": 49}],
        "cast": [{"id": "p1", "name": "Lumi", "file": "pic.png", "subfolder": "%s/%s/assets" % (ROOT, folder)}],
        "sounds": [{"id": "s1", "file": "song.wav", "subfolder": "%s/%s/assets" % (ROOT, folder)}],
        "effects": [{"id": "x1", "name": "FX", "takes": [take("x1", "t5"), take("x1", "t6")], "active": "t6"}],
        "pending": ["job"], "revision": 9,
    }
    for clip, take_id in (("ca", "t1"), ("ca", "t2"), ("cb", "t3"), ("cf", "t4"), ("x1", "t5"), ("x1", "t6")):
        for end in ("_00001_.mp4", "_00001_.jpg", "_00001_.strip.jpg", "_00001_.peaks.json", ".hold.safetensors"):
            touch(base, "clips/%s_%s%s" % (clip, take_id, end))
    touch(base, "assets/pic.png")
    touch(base, "assets/song.wav")
    touch(base, "assets/song.peaks.json")
    touch(base, "assets/unused.png")
    touch(base, "exports/cut.mp4", 100)
    touch(base, "history/000001.json")
    touch(base, "preview/live.mp4")
    touch(base, "clips/scratch.tmp")
    with open(os.path.join(base, "project.json"), "w") as f:
        json.dump(data, f)
    return data


class ArchiveTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.base = os.path.join(self.tmp, "Demo")
        os.makedirs(self.base)
        self.data = demo(self.base)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def files(self, keep):
        return archive.files_for(self.base, ROOT, "Demo", archive.trimmed(self.data, keep), keep)[0]

    def test_everything_leaves_out_only_this_machines_history_and_previews(self):
        paths = self.files("all")
        self.assertIn("exports/cut.mp4", paths)
        self.assertIn("clips/cb_t3_00001_.mp4", paths)
        self.assertIn("assets/unused.png", paths)
        self.assertFalse(any(p.startswith(("history/", "preview/")) or p.endswith(".tmp") for p in paths))

    def test_the_cut_keeps_the_shown_takes_with_their_companions_and_drops_the_rest(self):
        paths = set(self.files("cut"))
        for end in ("_00001_.mp4", "_00001_.jpg", "_00001_.strip.jpg", "_00001_.peaks.json", ".hold.safetensors"):
            self.assertIn("clips/ca_t2" + end, paths)
        self.assertIn("clips/cf_t4_00001_.mp4", paths)            # footage stays, on the timeline or not
        self.assertIn("clips/x1_t6_00001_.mp4", paths)            # the VFX clip's active take
        self.assertTrue({"assets/pic.png", "assets/song.wav", "assets/song.peaks.json"} <= paths)
        for gone in ("clips/ca_t1_00001_.mp4", "clips/cb_t3_00001_.mp4", "clips/x1_t5_00001_.mp4", "exports/cut.mp4", "assets/unused.png"):
            self.assertNotIn(gone, paths)
        data = archive.trimmed(self.data, "cut")
        clips = {c["id"]: c for c in data["clips"]}
        self.assertEqual([t["id"] for t in clips["ca"]["takes"]], ["t2"])
        self.assertEqual((clips["cb"]["takes"], clips["cb"]["active"]), ([], None))
        self.assertEqual(data["pending"], [])
        self.assertNotIn("revision", data)

    def test_no_renders_keeps_text_pictures_sounds_and_footage(self):
        paths = set(self.files("text"))
        self.assertIn("clips/cf_t4_00001_.mp4", paths)
        self.assertTrue({"assets/pic.png", "assets/song.wav"} <= paths)
        self.assertFalse(any(p.startswith(("clips/ca_", "clips/cb_", "clips/x1_")) for p in paths))
        data = archive.trimmed(self.data, "text")
        self.assertEqual([c["active"] for c in data["clips"]], [None, None, "t4"])
        self.assertEqual(data["clips"][0]["action"], "a")
        self.assertEqual(data["effects"][0]["takes"], [])

    def test_the_levels_get_smaller(self):
        sizes = archive.measure(self.base, ROOT, "Demo", self.data)
        self.assertGreater(sizes["all"]["bytes"], sizes["cut"]["bytes"])
        self.assertGreater(sizes["cut"]["bytes"], sizes["text"]["bytes"])

    def test_a_zip_opens_as_the_same_project(self):
        target = os.path.join(self.tmp, "demo.zip")
        manifest = archive.pack(self.base, ROOT, "Demo", self.data, "cut", target, release="0.6.0")
        self.assertEqual((manifest["release"], manifest["keep"], manifest["models"]["loras"]), ("0.6.0", "cut", ["a.safetensors"]))
        opened = os.path.join(self.tmp, "Opened")
        read_manifest, data = archive.unpack(target, opened)
        self.assertEqual(read_manifest["folder"], "Demo")
        self.assertTrue(os.path.isfile(os.path.join(opened, "clips", "ca_t2_00001_.mp4")))
        self.assertFalse(os.path.exists(os.path.join(opened, "project.json")))     # the server commits it, retargeted
        moved = project_module.retarget(data, "Demo", "Opened")
        self.assertEqual(moved["clips"][0]["takes"][0]["subfolder"], "%s/Opened/clips" % ROOT)

    def zip_with(self, members, manifest=None):
        target = os.path.join(self.tmp, "bad.zip")
        with zipfile.ZipFile(target, "w") as zipped:
            zipped.writestr("manifest.json", json.dumps(manifest or {"format": archive.FORMAT, "format_version": 1}))
            zipped.writestr("project.json", json.dumps({"clips": []}))
            for name, body in members:
                if isinstance(body, zipfile.ZipInfo):
                    zipped.writestr(body, "x")
                else:
                    zipped.writestr(name, body)
        return target

    def test_a_path_that_climbs_out_is_refused(self):
        with self.assertRaises(archive.ArchiveError):
            archive.read(self.zip_with([("files/../../evil.py", "x")]))

    def test_a_file_outside_files_is_refused(self):
        with self.assertRaises(archive.ArchiveError):
            archive.read(self.zip_with([("evil.py", "x")]))

    def test_a_link_is_refused(self):
        link = zipfile.ZipInfo("files/clips/link")
        link.external_attr = (0o120777 << 16)
        with self.assertRaises(archive.ArchiveError):
            archive.read(self.zip_with([("", link)]))

    def test_a_zip_that_is_not_a_project_is_refused(self):
        target = os.path.join(self.tmp, "other.zip")
        with zipfile.ZipFile(target, "w") as zipped:
            zipped.writestr("readme.txt", "hello")
        with self.assertRaisesRegex(archive.ArchiveError, "not an H3 Editor project"):
            archive.read(target)

    def test_a_zip_from_a_newer_editor_says_so(self):
        with self.assertRaisesRegex(archive.ArchiveError, "newer"):
            archive.read(self.zip_with([], {"format": archive.FORMAT, "format_version": 99, "release": "9.0.0"}))

    def test_a_failed_open_leaves_nothing_behind(self):
        target = self.zip_with([("files/ok.txt", "x"), ("files/../escape.txt", "x")])
        opened = os.path.join(self.tmp, "Half")
        with self.assertRaises(archive.ArchiveError):
            archive.unpack(target, opened)
        self.assertFalse(os.path.exists(opened))

    def test_a_taken_name_gets_a_number(self):
        slug = project_module.slug
        self.assertEqual(archive.free_name("Demo", ["Other"], slug), "Demo")
        self.assertEqual(archive.free_name("Demo", ["Demo", "Demo_2"], slug), "Demo 3")


if __name__ == "__main__":
    unittest.main()
