import tempfile
import unittest
from pathlib import Path
from unittest import mock

import hold_cache


class ProjectCacheTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.projects = Path(tmp.name) / "h3take_projects"
        self.clips = self.projects / "one" / "clips"
        self.clips.mkdir(parents=True)
        self.data = {"clips": [{"takes": [{"file": "take_00001__edit.mp4", "subfolder": "h3take_projects/one/clips"}]}],
                     "link": {"file": "join_00005_.mp4", "subfolder": "h3take_projects\\one\\clips"}}

    def report(self, scope=None, version=None):
        return hold_cache.project_cache(self.projects, "one", self.data, scope, version)

    def test_scopes_preserve_media_and_count_actual_files(self):
        kept = ["take.hold.safetensors", "join.hold.safetensors"]
        others = ["model.safetensors", "take_00001__edit.mp4", "take.hold.safetensors.tmp", "take.jpg"]
        for name in kept + others + ["gone.hold.safetensors"]:
            (self.clips / name).write_bytes(b"123")
        for directory in (self.projects / "two" / "clips", self.clips / "nested.hold.safetensors",
                          self.clips.parent / "assets"):
            directory.mkdir(parents=True)
            (directory / "gone.hold.safetensors").write_bytes(b"keep")
        report = self.report()
        self.assertEqual((report["all"]["files"], report["all"]["bytes"], report["unused"]["bytes"]), (3, 9, 3))
        report = self.report("unused", report["unused"]["version"])
        self.assertEqual(report["removed"], {"files": 1, "bytes": 3})
        self.assertEqual(report["all"]["bytes"], 6)
        report = self.report("all", report["all"]["version"])
        self.assertEqual(report["removed"], {"files": 2, "bytes": 6})
        self.assertEqual(report["all"]["bytes"], 0)
        self.assertEqual(self.report("all", report["all"]["version"])["removed"]["files"], 0)
        for name in others:
            self.assertEqual((self.clips / name).read_bytes(), b"123")
        self.assertEqual(len(list(self.projects.rglob("gone.hold.safetensors"))), 3)

    def test_new_reference_invalidates_unused_cleanup(self):
        (self.clips / "gone.hold.safetensors").touch()
        old = self.report()
        self.data["clips"][0]["takes"].append({"file": "gone_00001_.mp4", "subfolder": "h3take_projects/one/clips"})
        with self.assertRaises(hold_cache.CacheConflict):
            self.report("unused", old["unused"]["version"])
        self.assertEqual(self.report()["all"]["files"], 1)

    def test_render_queued_after_review_blocks_cleanup(self):
        path = self.clips / "gone.hold.safetensors"
        path.touch()
        version = self.report()["all"]["version"]
        self.data["pending"] = [{"id": "rendering"}]
        self.assertTrue(self.report()["rendering"])
        with self.assertRaisesRegex(hold_cache.CacheConflict, "renders to finish"):
            self.report("all", version)
        self.assertTrue(path.exists())

    def test_partial_cleanup_reports_locked_files_and_can_be_retried(self):
        paths = [self.clips / (name + ".hold.safetensors") for name in ("a", "b")]
        for path in paths:
            path.write_bytes(b"123")
        unlink = Path.unlink

        def remove(path):
            if path == paths[1]:
                raise PermissionError("busy")
            unlink(path)

        version = self.report()["all"]["version"]
        with mock.patch.object(Path, "unlink", remove):
            report = self.report("all", version)
        self.assertEqual((report["failed"], report["removed"]["bytes"], report["all"]["files"]), (1, 3, 1))
        self.assertEqual(self.report("all", report["all"]["version"])["all"]["files"], 0)

    def test_linked_files_and_clip_folders_are_never_cleaned(self):
        other = self.clips.parent / "other"
        other.mkdir()
        target = other / "important.hold.safetensors"
        target.touch()
        link = self.clips / "linked.hold.safetensors"
        try:
            link.symlink_to(target)
        except OSError as error:
            self.skipTest(str(error))
        self.assertEqual(self.report()["all"]["files"], 0)
        link.unlink()
        self.clips.rmdir()
        self.clips.symlink_to(other, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "linked"):
            self.report()
        self.assertTrue(target.exists())

    def test_project_without_clips_has_an_empty_cache(self):
        self.clips.rmdir()
        self.assertEqual(self.report()["all"]["files"], 0)
        self.assertFalse(self.clips.exists())
