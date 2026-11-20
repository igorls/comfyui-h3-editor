"""CPU unit tests for editor_api motion model downloader routes, mux_motion_audio, assemble_prompt_edit, and edit.motion."""

import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import fetch
import project as store

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_editor_api(models_dir, output_dir, node_mappings=None, file_lists=None):
    """Import the real editor_api module in an isolated package namespace with stubbed ComfyUI globals."""
    pkg_name = "_h3_test_pkg"
    file_lists = file_lists or {}

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.models_dir = str(models_dir)
    folder_paths.folder_names_and_paths = {
        k: ([os.path.join(str(models_dir), k)], {".safetensors"})
        for k in ("diffusion_models", "unet_gguf", "loras", "latent_upscale_models",
                  "text_encoders", "clip", "vae", "checkpoints", "model_patches", "background_removal")
    }
    folder_paths.filename_list_cache = {}
    folder_paths.get_output_directory = lambda: str(output_dir)
    folder_paths.get_input_directory = lambda: str(output_dir)
    folder_paths.get_user_directory = lambda: str(output_dir)
    folder_paths.get_folder_paths = lambda kind: [os.path.join(str(models_dir), kind)]
    folder_paths.get_filename_list = lambda kind: list(file_lists.get(kind, []))

    comfy_nodes = types.ModuleType("nodes")
    comfy_nodes.NODE_CLASS_MAPPINGS = dict(node_mappings or {})

    # Wire a lightweight package pointing at REPO_ROOT so editor_api's relative imports resolve cleanly.
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(REPO_ROOT)]
    patched = {
        "folder_paths": folder_paths,
        "nodes": comfy_nodes,
        pkg_name: pkg,
        f"{pkg_name}.fetch": fetch,
        f"{pkg_name}.project": store,
    }
    for sub in ("archive", "cuts", "listen", "lips", "scene", "detail", "media", "vosr2", "looks", "patches", "policy"):
        patched[f"{pkg_name}.{sub}"] = importlib.import_module(sub)

    with mock.patch.dict(sys.modules, patched):
        spec = importlib.util.spec_from_file_location(f"{pkg_name}.editor_api", REPO_ROOT / "editor_api.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"{pkg_name}.editor_api"] = mod
        spec.loader.exec_module(mod)
    return mod, folder_paths, comfy_nodes


class _FakeRequest:
    def __init__(self, body=None, query=None):
        self._body = body
        self.can_read_body = body is not None
        self.query = query or {}

    async def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class MotionApiRoutesTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.models_dir = os.path.join(self.tmp.name, "models")
        self.output_dir = os.path.join(self.tmp.name, "output")
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        self.node_mappings = {name: object() for name in (*store.SAM3_NODES, *store.MOTION_NODES)}
        self.file_lists = {
            "diffusion_models": [store.PROFILES["hyperflow"]["unet"][0]],
            "checkpoints": [],
            "loras": [],
            "model_patches": [],
            "vae": [],
        }
        self.api, self.folder_paths, self.comfy_nodes = _load_editor_api(
            self.models_dir, self.output_dir, self.node_mappings, self.file_lists,
        )

    def tearDown(self):
        self.tmp.cleanup()

    def _run_with_stubs(self):
        return mock.patch.dict(sys.modules, {"folder_paths": self.folder_paths, "nodes": self.comfy_nodes})

    async def test_available_models_discovers_sam3_and_motion_checkpoints(self):
        self.file_lists["checkpoints"] = [store.MOTION_SAM3, store.MOTION_POSE]
        self.file_lists["model_patches"] = [store.MOTION_CONTROL_V2]
        with self._run_with_stubs():
            avail = self.api.available_models()
        self.assertIn("sam3-nodes", avail)
        self.assertIn("motion-nodes", avail)
        self.assertIn("motion-checkpoints:" + store.MOTION_SAM3, avail)
        self.assertIn("motion-control:" + store.MOTION_CONTROL_V2, avail)

    async def test_motion_setup_get_download_and_cancel_routes(self):
        # Routes are registered
        route_paths = {(m, p) for m, p, _ in self.api.ROUTES}
        self.assertIn(("GET", "/h3take/api/models/motion"), route_paths)
        self.assertIn(("POST", "/h3take/api/models/motion/download"), route_paths)
        self.assertIn(("POST", "/h3take/api/models/motion/cancel"), route_paths)

        with self._run_with_stubs():
            resp = await self.api.motion_model_setup(_FakeRequest())
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.text)
            self.assertFalse(data["ready"])
            self.assertIn("replace", data["groups"])
            self.assertIn("pose", data["groups"])
            self.assertIn("sharp", data["groups"])
            self.assertEqual(data["groups"]["replace"]["missing"], ["sam3", "swap_lora"])

            # Bad group query returns 400
            bad = await self.api.motion_model_setup(_FakeRequest(query={"groups": "nope"}))
            self.assertEqual(bad.status, 400)

            # POST /download builds exact pinned plan for requested groups and invokes fetch.start
            started = []
            with mock.patch.object(fetch, "start", side_effect=lambda plan, cb: started.append((plan, cb))), \
                 mock.patch.object(self.api, "free_space", return_value=[]):
                dl = await self.api.motion_model_download(_FakeRequest(body={"groups": ["replace"]}))
            self.assertEqual(dl.status, 200)
            self.assertEqual(len(started), 1)
            plan = started[0][0]
            self.assertEqual([item["id"] for item in plan], ["sam3", "swap_lora"])
            for item in plan:
                self.assertTrue(fetch.allowed(item["url"]), item["url"])
                self.assertRegex(item["sha256"], r"^[0-9a-f]{64}$")

            # Disk short returns 507
            short_rows = [{"folder": self.models_dir, "needs": 2_000_000_000, "free": 100_000_000}]
            with mock.patch.object(self.api, "free_space", return_value=short_rows):
                short_resp = await self.api.motion_model_download(_FakeRequest(body={"groups": ["replace"]}))
            self.assertEqual(short_resp.status, 507)

            # Active download conflict returns 409
            with mock.patch.object(fetch, "start", side_effect=fetch.Refused("already running")), \
                 mock.patch.object(self.api, "free_space", return_value=[]):
                busy_resp = await self.api.motion_model_download(_FakeRequest(body={"groups": ["replace"]}))
            self.assertEqual(busy_resp.status, 409)

            # Invalid groups body returns 400
            bad_body = await self.api.motion_model_download(_FakeRequest(body={"groups": "replace"}))
            self.assertEqual(bad_body.status, 400)

            # Cancel calls fetch.cancel and returns report
            with mock.patch.object(fetch, "cancel") as cancel_mock:
                cancel_resp = await self.api.motion_model_cancel(_FakeRequest())
            self.assertEqual(cancel_resp.status, 200)
            cancel_mock.assert_called_once()


class MotionAudioAndPromptEditTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.models_dir = os.path.join(self.tmp.name, "models")
        self.output_dir = os.path.join(self.tmp.name, "output")
        self.clips_dir = os.path.join(self.output_dir, "h3take_projects", "p", "clips")
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.clips_dir, exist_ok=True)
        self.api, self.folder_paths, self.comfy_nodes = _load_editor_api(self.models_dir, self.output_dir)

    def tearDown(self):
        self.tmp.cleanup()

    def _run_with_stubs(self):
        return mock.patch.dict(sys.modules, {"folder_paths": self.folder_paths, "nodes": self.comfy_nodes})

    async def test_mux_motion_audio_preserves_sharp_resolution_and_handles_exact_and_trimmed_frames(self):
        source_path = os.path.join(self.clips_dir, "src.mp4")
        rendered_path = os.path.join(self.clips_dir, "out.mp4")
        with open(source_path, "wb") as f:
            f.write(b"src")
        with open(rendered_path, "wb") as f:
            f.write(b"ren")

        recorded = []

        async def fake_run(args):
            recorded.append(list(args))
            with open(args[-1], "wb") as f:
                f.write(b"muxed")
            return ""

        # Case 1: multi-window exact frame count -> -c:v copy, never scales down 1088x1920!
        with self._run_with_stubs(), \
             mock.patch.object(self.api, "video_info", side_effect=[
                 {"width": 1088, "height": 1920, "frames": 230},
                 {"width": 1088, "height": 1920, "frames": 230},
             ]), \
             mock.patch.object(self.api.media, "has_sound", return_value=True), \
             mock.patch.object(self.api, "run", side_effect=fake_run):
            out = await self.api.mux_motion_audio(rendered_path, {
                "file": "src.mp4", "subfolder": "h3take_projects/p/clips", "in": 0, "out": 230,
            })
        self.assertEqual(out, rendered_path)
        cmd = recorded[-1]
        self.assertIn("-c:v", cmd)
        self.assertEqual(cmd[cmd.index("-c:v") + 1], "copy")
        self.assertNotIn("-shortest", cmd)
        self.assertFalse(any("scale=" in arg for arg in cmd))

        # Case 2: single-window padded (73f -> 60f) with silent source -> trims without scaling + anullsrc
        with self._run_with_stubs(), \
             mock.patch.object(self.api, "video_info", side_effect=[
                 {"width": 1088, "height": 1920, "frames": 73},
                 {"width": 1088, "height": 1920, "frames": 60},
             ]), \
             mock.patch.object(self.api.media, "has_sound", return_value=False), \
             mock.patch.object(self.api, "run", side_effect=fake_run):
            await self.api.mux_motion_audio(rendered_path, {
                "file": "src.mp4", "subfolder": "h3take_projects/p/clips", "in": 12, "out": 72,
            })
        cmd2 = recorded[-1]
        self.assertIn("anullsrc=r=48000:cl=stereo", cmd2)
        self.assertIn("trim=end_frame=60,setpts=PTS-STARTPTS", cmd2)
        self.assertFalse(any("scale=" in arg for arg in cmd2))

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "needs ffmpeg and ffprobe")
    async def test_mux_motion_audio_real_ffmpeg_keeps_sharp_resolution_and_exact_frames(self):
        source_path = os.path.join(self.clips_dir, "real_src.mp4")
        rendered_path = os.path.join(self.clips_dir, "real_ren.mp4")
        # 544x960 source (3s = 72f) with sine audio; 1088x1920 rendered Sharp output (73f padded -> 60f)
        subprocess.run([
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "color=c=blue:s=544x960:r=24:d=3",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=3",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", source_path,
        ], check=True)
        subprocess.run([
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "color=c=green:s=1088x1920:r=24:d=3.042",
            "-frames:v", "73", "-c:v", "libx264", "-pix_fmt", "yuv420p", rendered_path,
        ], check=True)

        def probe(path, count=True):
            out = subprocess.check_output([
                "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                "-show_entries", "stream=width,height,nb_read_frames", "-of", "json", path,
            ], text=True)
            s = json.loads(out)["streams"][0]
            return {"width": int(s["width"]), "height": int(s["height"]), "frames": int(s["nb_read_frames"])}

        async def probe_async(path, count=True):
            return probe(path, count)

        with self._run_with_stubs(), \
             mock.patch.object(self.api, "video_info", side_effect=probe_async), \
             mock.patch.object(self.api.media, "has_sound", return_value=True):
            await self.api.mux_motion_audio(rendered_path, {
                "file": "real_src.mp4", "subfolder": "h3take_projects/p/clips", "in": 6, "out": 66,
            })

        result = probe(rendered_path)
        self.assertEqual((result["width"], result["height"], result["frames"]), (1088, 1920, 60))

    async def test_assemble_prompt_edit_preserves_full_clip_sharp_resolution_and_keeps_partial_at_source(self):
        source_path = os.path.join(self.clips_dir, "src.mp4")
        rendered_path = os.path.join(self.clips_dir, "ren.mp4")
        with open(source_path, "wb") as f:
            f.write(b"src")
        with open(rendered_path, "wb") as f:
            f.write(b"ren")

        recorded = []

        async def fake_run(args):
            recorded.append(list(args))
            with open(args[-1], "wb") as f:
                f.write(b"edited")
            return ""

        # Full-clip Sharp edit (0..120 of 120): preserves 1088x1920 instead of scaling down to 544x960
        with self._run_with_stubs(), \
             mock.patch.object(self.api, "video_info", side_effect=[
                 {"width": 544, "height": 960},
                 {"width": 1088, "height": 1920, "frames": 124},
                 {"width": 1088, "height": 1920, "frames": 120},
             ]), \
             mock.patch.object(self.api, "run", side_effect=fake_run):
            await self.api.assemble_prompt_edit(rendered_path, {
                "file": "src.mp4", "subfolder": "h3take_projects/p/clips", "in": 0, "out": 120, "frames": 120,
            })
        fc_full = recorded[-1][recorded[-1].index("-filter_complex") + 1]
        self.assertIn("scale=1088:1920:", fc_full)

        # Partial-clip edit (24..96 of 120): stays at original 544x960 to match prefix/suffix
        with self._run_with_stubs(), \
             mock.patch.object(self.api, "video_info", side_effect=[
                 {"width": 544, "height": 960},
                 {"width": 1088, "height": 1920, "frames": 73},
                 {"width": 544, "height": 960, "frames": 120},
             ]), \
             mock.patch.object(self.api, "run", side_effect=fake_run):
            await self.api.assemble_prompt_edit(rendered_path, {
                "file": "src.mp4", "subfolder": "h3take_projects/p/clips", "in": 24, "out": 96, "frames": 120,
            })
        fc_partial = recorded[-1][recorded[-1].index("-filter_complex") + 1]
        self.assertIn("scale=544:960:", fc_partial)

    async def test_backwards_compatible_edit_motion_supports_replace_up_to_30s_and_sharp_size(self):
        avail = set(store.PROFILES["hyperflow"]["unet"] + store.PROFILES["hyperflow"]["unet_text"]) | {
            "sam3-nodes",
            "motion-checkpoints:" + store.MOTION_SAM3,
            store.MOTION_SWAP_LORA,
            store.RELAYVSR_MODEL,
            store.RELAYVSR_VAE,
        }
        project = store.new_project("Legacy")
        store.set_project(project, {"format": "9:16", "sizes": {"final": "sharp"}})
        take = {"id": "orig", "file": "src.mp4", "subfolder": "h3take_projects/legacy/clips",
                "frames": 240, "size": [544, 960]}
        src, _ = store.add_footage(project, "src", take)
        job, graph = store.plan_render(project, "legacy", src["id"], "final", seed=11, available=avail, edit={
            "take": "orig", "in": 0, "out": 240, "prompt": "",
            "motion": {"character": {"file": "char.png"}, "mode": "replace", "style": "preserve"},
        })
        # Full-clip Sharp edit keeps 1088x1920 on job["size"] and runs 2-window relay
        self.assertEqual(job["size"], [1088, 1920])
        self.assertIn("328", graph)
        self.assertEqual(graph["328"]["inputs"]["trim_to"], 240)

    async def test_check_motion_cache_and_pipeline_report_and_clear(self):
        import numpy as np
        from PIL import Image

        folder = "demo_motion"
        clips_dir = os.path.join(self.output_dir, store.ROOT, folder, "clips")
        assets_dir = os.path.join(self.output_dir, store.ROOT, folder, "assets")
        os.makedirs(clips_dir, exist_ok=True)
        os.makedirs(assets_dir, exist_ok=True)

        src_path = os.path.join(clips_dir, "dance.mp4")
        with open(src_path, "wb") as f:
            f.write(b"source-mp4-bytes")
        char_path = os.path.join(assets_dir, "hero.png")
        Image.new("RGB", (32, 32), (180, 120, 90)).save(char_path)

        # Populate sidecar mask, pose, and cutout caches
        store.motion_replace.save_mask_cache(src_path, 0, np.ones((73, 16, 12), dtype=np.float32), total_frames=120)
        store.motion_replace.save_pose_cache(src_path, 0, [{"people": []}] * 73, total_frames=120)
        store.motion_replace.save_cutout_cache(char_path, np.full((32, 32, 3), 0.5, dtype=np.float32))

        src_rel = f"{store.ROOT}/{folder}/clips/dance.mp4"
        char_obj = {"file": "hero.png", "subfolder": f"{store.ROOT}/{folder}/assets"}

        with self._run_with_stubs():
            hits = self.api.check_motion_cache({"file": src_rel, "first": 0, "last": 73}, char_obj, {})
            self.assertEqual(hits, {"mask": True, "pose": True, "cutout": True})

            project = store.new_project("Demo Motion")
            store.set_project(project, {"format": "9:16"})
            src_clip, _ = store.add_footage(project, "dance", {
                "id": "t0", "file": "dance.mp4", "subfolder": f"{store.ROOT}/{folder}/clips",
                "frames": 120, "size": [544, 960],
            })
            m_clip, _ = store.add_motion_clip(
                project, src_clip["id"], first=0, last=72,
                motion={"character": char_obj, "mode": "replace", "style": "auto", "strength": 0.35},
            )

            avail = set(store.PROFILES["hyperflow"]["unet"] + store.PROFILES["hyperflow"]["unet_text"]) | {
                "sam3-nodes", "motion-checkpoints:" + store.MOTION_SAM3,
                store.MOTION_SWAP_LORA, "sdpose-nodes",
                "motion-det:" + store.MOTION_DETECTOR, "motion-pose:" + store.MOTION_POSE,
            }
            report = self.api._motion_pipeline_report(project, folder, m_clip["id"], available=avail)
            self.assertEqual(report["clip"]["id"], m_clip["id"])
            by_id = {s["id"]: s for s in report["clip"]["stages"]}
            self.assertEqual(by_id["source"]["status"], "ready")
            self.assertEqual(by_id["cutout"]["status"], "cached")
            self.assertEqual(by_id["segmentation"]["status"], "cached")
            self.assertEqual(by_id["pose"]["status"], "cached")
            self.assertIn("/motion/preview?", by_id["segmentation"]["preview_url"])
            self.assertIn("/motion/preview?", by_id["pose"]["preview_url"])
            self.assertEqual(len(report["caches"]), 3)

            # Clear mask cache via handler
            req = _FakeRequest(body={"kind": "mask", "file": src_rel})
            req.match_info = {"name": folder}
            resp = await self.api.motion_cache_clear(req)
            self.assertEqual(resp.status, 200)
            self.assertEqual(json.loads(resp.text)["removed"], 1)
            self.assertFalse(store.motion_replace.has_mask_cache(src_path, 0, 73))
            self.assertTrue(store.motion_replace.has_pose_cache(src_path, 0, 73))

            # Clear remaining caches with kind="all"
            req_all = _FakeRequest(body={"kind": "all"})
            req_all.match_info = {"name": folder}
            resp_all = await self.api.motion_cache_clear(req_all)
            self.assertEqual(json.loads(resp_all.text)["removed"], 2)
            self.assertFalse(store.motion_replace.has_pose_cache(src_path, 0, 73))
            self.assertFalse(store.motion_replace.has_cutout_cache(char_path))


if __name__ == "__main__":
    unittest.main()
