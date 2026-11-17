"""Get the models: the setup list, what counts as missing, and the downloader's safety rules."""

import hashlib
import io
import os
import re
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

import fetch
import project as store

DOWNLOADS = [entry for entry in store.SETUP_FILES if not entry.get("manual")]
EVERY_FIRST_NAME = {entry["needs"][0] for entry in store.SETUP_FILES}


class SetupListTest(unittest.TestCase):
    def test_each_file_lands_under_the_name_a_render_asks_for(self):
        for entry in store.SETUP_FILES:
            self.assertEqual(entry["needs"][0].rsplit("/", 1)[-1], (entry.get("path") or entry["needs"][0]).rsplit("/", 1)[-1], entry["id"])

    def test_the_default_profiles_files_come_from_its_own_lists(self):
        ids = {entry["id"]: entry for entry in store.SETUP_FILES}
        self.assertEqual(ids["unet"]["needs"], store.PROFILES[store.DEFAULT_PROFILE]["unet"])
        self.assertEqual(ids["unet_text"]["needs"], store.PROFILES[store.DEFAULT_PROFILE]["unet_text"])
        for key in ("clip", "vae", "audio_vae"):
            self.assertIn(store.MODEL_CHOICES[key], [entry["needs"] for entry in store.SETUP_FILES])
        self.assertEqual(ids["latent_upscaler"]["needs"], [store.MODELS["latent_upscaler"]])
        self.assertEqual(ids["realism"]["needs"], list(store.REALISM_LORAS))
        self.assertEqual(store.DEFAULT_PROFILE, "hyperflow")

    def test_every_download_is_pinned_and_checked(self):
        for entry in DOWNLOADS:
            url = store.setup_url(entry)
            self.assertTrue(fetch.allowed(url), url)
            self.assertRegex(url, r"^https://huggingface\.co/[\w.-]+/[\w.-]+/resolve/[0-9a-f]{40}/")
            self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")
            self.assertGreater(entry["bytes"], 0)
            self.assertTrue(entry["license"][1].startswith("https://"))

    def test_a_fresh_install_downloads_about_55_gb(self):
        status = store.setup_status(set())
        self.assertEqual(status["missing"], [entry["id"] for entry in DOWNLOADS])
        self.assertAlmostEqual(status["bytes"] / 1e9, 54.6, delta=0.1)
        self.assertEqual(status["manual"], ["realism"])
        self.assertFalse(status["ready"])


class MissingTest(unittest.TestCase):
    def test_it_misses_what_the_profile_encoder_and_finals_checks_miss(self):
        status = store.setup_status(set())
        missing = {row["file"] for row in status["files"] if not row["installed"]}
        hyperflow = next(row for row in store.profile_status(set()) if row["id"] == store.DEFAULT_PROFILE)
        self.assertLessEqual(set(hyperflow["missing"]), missing)
        for note in store.encoder_status(set())["missing"]:
            self.assertTrue(any(name in note for name in missing), note)
        for note in store.finals_status(set()):
            self.assertTrue(any(name in note for name in missing), note)

    def test_with_every_file_in_place_nothing_is_missing_anywhere(self):
        status = store.setup_status(EVERY_FIRST_NAME)
        self.assertEqual((status["missing"], status["manual"], status["bytes"], status["ready"]), ([], [], 0, True))
        hyperflow = next(row for row in store.profile_status(EVERY_FIRST_NAME) if row["id"] == store.DEFAULT_PROFILE)
        self.assertTrue(hyperflow["available"])
        self.assertEqual(store.encoder_status(EVERY_FIRST_NAME)["missing"], [])
        self.assertEqual(store.finals_status(EVERY_FIRST_NAME), [])

    def test_another_precision_or_a_gguf_counts_as_installed(self):
        available = {"qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "minimax_h3_video_vae_int8_convrot.safetensors",
                     "minimax_h3_audio_vae_fp32.safetensors", "minimax_h3_ref2va_hyperflow_flash_14b-Q4_K_M.gguf",
                     "elsewhere\\minimax_h3_fl2va_hyperflow_flash_14b-int8_convrot.safetensors"}
        status = store.setup_status(available)
        self.assertEqual(status["missing"], ["latent_upscaler"])
        self.assertTrue(status["ready"])
        found = {row["id"]: row["found"] for row in status["files"]}
        self.assertEqual(found["unet"], "minimax_h3_ref2va_hyperflow_flash_14b-Q4_K_M.gguf")

    def test_drafts_are_ready_without_the_finals_files(self):
        drafts = {entry["needs"][0] for entry in store.SETUP_FILES if entry["for"] == "draft"}
        status = store.setup_status(drafts)
        self.assertTrue(status["ready"])
        self.assertEqual(status["missing"], ["latent_upscaler"])
        self.assertEqual(status["bytes"], 690592992)


class FakeResponse(io.BytesIO):
    def __init__(self, body, status=200, headers=None, check=None):
        super().__init__(body)
        self.status = status
        self.headers = headers or {}
        self.check = check

    def read(self, size=-1):
        if self.check:
            self.check()
        return super().read(min(size, 7) if size and size > 0 else size)


class FakeOpener:
    """Serves one body, honouring Range unless told not to, and records each request."""

    def __init__(self, body, ranges=True, check=None):
        self.body, self.ranges, self.check, self.requests = body, ranges, check, []

    def open(self, request, timeout=None):
        self.requests.append(request)
        asked = request.get_header("Range")
        if asked and self.ranges:
            start = int(re.match(r"bytes=(\d+)-", asked).group(1))
            return FakeResponse(self.body[start:], 206, {"Content-Range": "bytes %d-%d/%d" % (start, len(self.body) - 1, len(self.body))}, self.check)
        return FakeResponse(self.body, 200, {}, self.check)


class DownloadTest(unittest.TestCase):
    BODY = bytes(range(256)) * 40

    def setUp(self):
        self.folder = tempfile.mkdtemp()
        self.dest = os.path.join(self.folder, "hyperflow", "model.safetensors")
        self.part = self.dest + fetch.PART
        self.item = {"id": "unet", "url": "https://huggingface.co/a/b/resolve/%s/model.safetensors" % ("0" * 40), "dest": self.dest,
                     "bytes": len(self.BODY), "sha256": hashlib.sha256(self.BODY).hexdigest()}
        self.real = fetch.opener

    def tearDown(self):
        fetch.opener = self.real

    def serve(self, **options):
        served = FakeOpener(self.BODY, **options)
        fetch.opener = lambda: served
        return served

    def test_it_writes_to_a_part_file_and_renames_it_when_it_checks_out(self):
        seen = []
        self.serve(check=lambda: seen.append((os.path.exists(self.dest), os.path.exists(self.part))))
        self.assertTrue(fetch.fetch(self.item))
        self.assertTrue(seen)
        # While it came, the file had its .part name only.
        self.assertTrue(all(not done for done, _ in seen))
        with open(self.dest, "rb") as f:
            self.assertEqual(f.read(), self.BODY)
        self.assertFalse(os.path.exists(self.part))

    def test_a_file_already_there_is_never_touched(self):
        os.makedirs(os.path.dirname(self.dest))
        with open(self.dest, "wb") as f:
            f.write(b"mine")
        served = self.serve()
        self.assertFalse(fetch.fetch(self.item))
        self.assertEqual(served.requests, [])
        with open(self.dest, "rb") as f:
            self.assertEqual(f.read(), b"mine")

    def test_placing_never_replaces_a_file_that_appeared_meanwhile(self):
        os.makedirs(os.path.dirname(self.dest))
        with open(self.part, "wb") as f:
            f.write(self.BODY)
        with open(self.dest, "wb") as f:
            f.write(b"mine")
        self.assertFalse(fetch.place(self.part, self.dest))
        with open(self.dest, "rb") as f:
            self.assertEqual(f.read(), b"mine")
        self.assertFalse(os.path.exists(self.part))

    def test_a_file_that_does_not_match_its_hash_is_deleted(self):
        self.serve()
        self.item["sha256"] = "f" * 64
        with self.assertRaises(RuntimeError):
            fetch.fetch(self.item)
        self.assertFalse(os.path.exists(self.dest))
        self.assertFalse(os.path.exists(self.part))

    def test_it_carries_on_from_a_part_file(self):
        os.makedirs(os.path.dirname(self.dest))
        with open(self.part, "wb") as f:
            f.write(self.BODY[:3000])
        served = self.serve()
        self.assertTrue(fetch.fetch(self.item))
        self.assertEqual(served.requests[0].get_header("Range"), "bytes=3000-")
        with open(self.dest, "rb") as f:
            self.assertEqual(f.read(), self.BODY)

    def test_a_server_that_sends_the_whole_file_again_starts_it_over(self):
        os.makedirs(os.path.dirname(self.dest))
        with open(self.part, "wb") as f:
            f.write(b"x" * 3000)
        self.serve(ranges=False)
        self.assertTrue(fetch.fetch(self.item))
        with open(self.dest, "rb") as f:
            self.assertEqual(f.read(), self.BODY)

    def test_stopping_keeps_what_came(self):
        stop = threading.Event()
        calls = []

        def check():
            calls.append(1)
            if len(calls) == 3:
                stop.set()

        self.serve(check=check)
        with self.assertRaises(fetch.Stopped):
            fetch.fetch(self.item, stop)
        self.assertFalse(os.path.exists(self.dest))
        self.assertGreater(os.path.getsize(self.part), 0)


class UrlRulesTest(unittest.TestCase):
    def test_only_https_on_hugging_face(self):
        self.assertTrue(fetch.allowed("https://huggingface.co/a/b/resolve/x/y.safetensors"))
        for url in ("http://huggingface.co/a/b/resolve/x/y", "https://huggingface.co.example.com/y", "https://example.com/huggingface.co/y",
                    "https://user:pass@huggingface.co/y", "file:///etc/passwd", "ftp://huggingface.co/y", ""):
            self.assertFalse(fetch.allowed(url), url)

    def test_a_plan_with_another_url_is_refused_before_anything_starts(self):
        plan = [{"id": "x", "url": "https://example.com/x", "dest": os.path.abspath("x"), "bytes": 1, "sha256": "0" * 64}]
        with self.assertRaises(fetch.Refused):
            fetch.start(plan)
        self.assertFalse(fetch.running())

    def test_a_plan_without_a_hash_is_refused(self):
        plan = [{"id": "x", "url": "https://huggingface.co/a/b/resolve/x/y", "dest": os.path.abspath("x"), "bytes": 1, "sha256": ""}]
        with self.assertRaises(fetch.Refused):
            fetch.check_plan(plan)

    def test_a_redirect_off_https_is_refused(self):
        handler = fetch.HttpsOnly()
        request = urllib.request.Request("https://huggingface.co/a")
        with self.assertRaises(urllib.error.HTTPError):
            handler.redirect_request(request, None, 302, "Found", {}, "http://cdn.example.com/a")
        self.assertIsNotNone(handler.redirect_request(request, None, 302, "Found", {}, "https://cdn-lfs.hf.co/a"))


class JobTest(unittest.TestCase):
    def test_a_job_downloads_each_file_and_reports_it(self):
        body = b"model bytes" * 100
        folder = tempfile.mkdtemp()
        plan = [{"id": name, "url": "https://huggingface.co/a/b/resolve/%s/%s" % ("0" * 40, name), "dest": os.path.join(folder, name),
                 "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()} for name in ("one", "two")]
        with open(plan[1]["dest"], "wb") as f:
            f.write(b"mine")
        real = fetch.opener
        fetch.opener = lambda: FakeOpener(body)
        placed = []
        try:
            fetch.start(plan, placed.append)
            for _ in range(200):
                if not fetch.running():
                    break
                threading.Event().wait(0.02)
        finally:
            fetch.opener = real
        status = fetch.status()
        self.assertEqual(status["state"], "done")
        self.assertEqual([row["state"] for row in status["files"]], ["done", "skipped"])
        self.assertEqual([item["id"] for item in placed], ["one"])
        with open(plan[1]["dest"], "rb") as f:
            self.assertEqual(f.read(), b"mine")


if __name__ == "__main__":
    unittest.main()
