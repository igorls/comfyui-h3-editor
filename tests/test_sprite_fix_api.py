"""Unit tests for the 5 Sprite Fix Pose HTTP endpoints and rev-aware sized_sprite() in editor_api.py."""

import base64
import importlib
import importlib.util
import io
import json
import os
import pathlib
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
from PIL import Image

import fetch
import project as store
import sprite

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _make_fake_server():
    server = types.ModuleType("server")

    class FakeQueue:
        def __init__(self):
            self.running = []
            self.queued = []
            self.history = {}

        def get_current_queue(self):
            return list(self.running), list(self.queued)

        def get_history(self, prompt_id=None):
            if prompt_id is None:
                return dict(self.history)
            if prompt_id in self.history:
                return {prompt_id: self.history[prompt_id]}
            return {}

    class FakePromptServer:
        def __init__(self):
            self.prompt_queue = FakeQueue()
            self.broadcasts = []

        def send_sync(self, event, data, sid=None):
            self.broadcasts.append((event, data, sid))

    server.PromptServer = types.SimpleNamespace(instance=FakePromptServer())
    return server


def _load_editor_api(models_dir, output_dir, node_mappings=None, file_lists=None):
    """Import editor_api in an isolated package namespace with stubbed ComfyUI globals."""
    pkg_name = "_h3_sprite_fix_api_pkg"
    file_lists = file_lists or {}

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.models_dir = str(models_dir)
    folder_paths.folder_names_and_paths = {
        k: ([os.path.join(str(models_dir), k)], {".safetensors"})
        for k in (
            "diffusion_models", "unet_gguf", "loras", "latent_upscale_models",
            "text_encoders", "clip", "vae", "checkpoints", "model_patches", "background_removal",
        )
    }
    folder_paths.filename_list_cache = {}
    folder_paths.get_output_directory = lambda: str(output_dir)
    folder_paths.get_input_directory = lambda: str(output_dir)
    folder_paths.get_user_directory = lambda: str(output_dir)
    folder_paths.get_folder_paths = lambda kind: [os.path.join(str(models_dir), kind)]
    folder_paths.get_filename_list = lambda kind: list(file_lists.get(kind, []))

    comfy_nodes = types.ModuleType("nodes")
    comfy_nodes.NODE_CLASS_MAPPINGS = dict(node_mappings or {})
    comfy_nodes.NODE_DISPLAY_NAME_MAPPINGS = {}

    server = _make_fake_server()

    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(REPO_ROOT)]
    patched = {
        "folder_paths": folder_paths,
        "nodes": comfy_nodes,
        "server": server,
        pkg_name: pkg,
        f"{pkg_name}.fetch": fetch,
        f"{pkg_name}.project": store,
        f"{pkg_name}.sprite": sprite,
    }
    for sub in (
        "archive", "cuts", "listen", "lips", "scene", "detail", "media",
        "vosr2", "looks", "patches", "policy", "hold_cache", "hybrid",
    ):
        patched[f"{pkg_name}.{sub}"] = importlib.import_module(sub)

    with mock.patch.dict(sys.modules, patched):
        spec = importlib.util.spec_from_file_location(f"{pkg_name}.editor_api", REPO_ROOT / "editor_api.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"{pkg_name}.editor_api"] = mod
        spec.loader.exec_module(mod)
    return mod, folder_paths, comfy_nodes, server


class _FakeRequest:
    def __init__(self, name="demo", body=None, query=None):
        self.match_info = {"name": name}
        self._body = body
        self.can_read_body = body is not None
        self.query = query or {}

    async def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


def _encode_png_b64(img, data_url=True):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return ("data:image/png;base64," + b64) if data_url else b64


def _make_character_rgba(size=(256, 384), color=(210, 60, 45, 255), box=(92, 110, 164, 300)):
    im = Image.new("RGBA", size, (0, 0, 0, 0))
    arr = np.asarray(im, dtype=np.uint8).copy()
    x0, y0, x1, y1 = box
    arr[y0:y1, x0:x1] = color
    return Image.fromarray(arr, "RGBA")


def _make_character_rgb(size=(256, 384), fg=(45, 190, 90), bg=(255, 255, 255), box=(92, 110, 164, 300)):
    arr = np.full((size[1], size[0], 3), bg, dtype=np.uint8)
    x0, y0, x1, y1 = box
    arr[y0:y1, x0:x1] = fg
    return Image.fromarray(arr, "RGB")


class SpriteFixApiTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.models_dir = os.path.join(self.tmp.name, "models")
        self.output_dir = os.path.join(self.tmp.name, "output")
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        self.file_lists = {
            "diffusion_models": [store.PROFILES["hyperflow"]["unet"][0]],
            "text_encoders": [store.MODEL_CHOICES["clip"][0]],
            "vae": [store.MODEL_CHOICES["vae"][0], store.MODEL_CHOICES["audio_vae"][0]],
            "background_removal": [],
            "loras": [],
        }
        self.node_mappings = {}
        self.api, self.folder_paths, self.comfy_nodes, self.server = _load_editor_api(
            self.models_dir, self.output_dir, self.node_mappings, self.file_lists,
        )
        self.folder = "hero_sprites"
        self.clips_dir = os.path.join(self.output_dir, store.ROOT, self.folder, "clips")
        os.makedirs(self.clips_dir, exist_ok=True)
        self._create_sprite_project()

    def tearDown(self):
        self.tmp.cleanup()

    def _run_with_stubs(self):
        return mock.patch.dict(
            sys.modules,
            {"folder_paths": self.folder_paths, "nodes": self.comfy_nodes, "server": self.server},
        )

    def _create_sprite_project(self, with_cache=True, with_video=True, cutout="background key"):
        sub = f"{store.ROOT}/{self.folder}/clips"
        self.clip_id = "c1"
        self.take_id = "t1"
        self.video_file = "c1_t1_00001_.mp4"
        self.sheet_stem = "c1_t1"
        self.cache_name = self.sheet_stem + ".frames"
        self.cache_dir = os.path.join(self.clips_dir, self.cache_name)
        self.layout = {"start": 8, "height": 190.0, "center": 128.0, "ground": 300.0}
        poses = 7
        self.pose_indices = [8 + 2 * i for i in range(poses)]

        if with_cache:
            frames = {
                idx: _make_character_rgba(color=(180 + i * 5, 60, 50 + i * 10, 255))
                for i, idx in enumerate(self.pose_indices)
            }
            sprite.save_frame_cache(self.cache_dir, frames)
        elif os.path.isdir(self.cache_dir):
            shutil.rmtree(self.cache_dir)

        if with_video:
            with open(os.path.join(self.clips_dir, self.video_file), "wb") as f:
                f.write(b"fake-mp4-data")

        sheet = {
            "poses": poses,
            "source_poses": poses,
            "fps": 12,
            "tile": 64,
            "tile_size": [64, 64],
            "pivot": [32, 59],
            "layout": dict(self.layout),
            "canvas": "portrait",
            "frames": 39,
            "cutout": cutout,
            "illustrated": self.sheet_stem + ".illustrated.png",
            "indexed": self.sheet_stem + ".indexed.png",
        }
        if with_cache:
            sheet["frame_cache"] = self.cache_name
            sprite.repack_take_from_cache(self.cache_dir, sheet, self.clips_dir, self.sheet_stem, total_frames=39)

        proj = store.new_project("Hero Sprites")
        proj["revision"] = 1
        proj["clips"] = [{
            "id": self.clip_id,
            "name": "Hero · run · S",
            "action": "Run in place steadily.",
            "cast": [],
            "sprite": {"direction": "S", "camera": "rpg", "poses": poses, "canvas": "portrait"},
            "active": self.take_id,
            "takes": [{
                "id": self.take_id,
                "file": self.video_file,
                "subfolder": sub,
                "frames": 39,
                "size": [256, 384],
                "quality": "draft",
                "seed": 42,
                "sprite": sheet,
            }],
        }]
        with self._run_with_stubs():
            self.api.write_project(self.folder, proj)

    async def test_routes_registered(self):
        routes = {(m, p): h for m, p, h in self.api.ROUTES}
        expected = {
            ("GET", "/h3take/api/project/{name}/sprite-pose-info"): "sprite_pose_info",
            ("POST", "/h3take/api/project/{name}/sprite-fix-alpha"): "sprite_fix_alpha",
            ("POST", "/h3take/api/project/{name}/sprite-nudge-pose"): "sprite_nudge_pose",
            ("POST", "/h3take/api/project/{name}/sprite-fix-ai"): "sprite_fix_ai",
            ("POST", "/h3take/api/project/{name}/sprite-revert-pose"): "sprite_revert_pose",
        }
        for key, handler_name in expected.items():
            self.assertIn(key, routes)
            self.assertEqual(routes[key], handler_name)
            self.assertTrue(callable(getattr(self.api, handler_name, None)))

    async def test_pose_info_and_validation(self):
        with self._run_with_stubs():
            req = _FakeRequest(
                self.folder,
                query={"clip": self.clip_id, "take": self.take_id, "pose": "2"},
            )
            resp = await self.api.sprite_pose_info(req)
            self.assertEqual(resp.status, 200)
            info = json.loads(resp.text)
            self.assertEqual(info["pose"], 2)
            self.assertEqual(info["source_index"], 12)
            self.assertEqual(info["orig_index"], 12)
            self.assertEqual(info["total_frames"], 39)
            self.assertIsNone(info["patched"])
            self.assertEqual(info["patched_poses"], {})
            self.assertTrue(info["has_raw_video"])
            self.assertFalse(info["has_hold_latents"])
            self.assertEqual(info["tile_size"], [64, 64])

            # Out-of-range pose -> 400
            bad_pose = await self.api.sprite_pose_info(_FakeRequest(
                self.folder, query={"clip": self.clip_id, "take": self.take_id, "pose": "99"},
            ))
            self.assertEqual(bad_pose.status, 400)

            # Non-integer pose -> 400
            nan_pose = await self.api.sprite_pose_info(_FakeRequest(
                self.folder, query={"clip": self.clip_id, "take": self.take_id, "pose": "abc"},
            ))
            self.assertEqual(nan_pose.status, 400)

            # Missing take -> 404
            missing = await self.api.sprite_pose_info(_FakeRequest(
                self.folder, query={"clip": self.clip_id, "take": "nope", "pose": "0"},
            ))
            self.assertEqual(missing.status, 404)

            # Hold latents flag reflects take_has_latents
            with mock.patch.object(self.api, "take_has_latents", return_value=True):
                resp2 = await self.api.sprite_pose_info(req)
            self.assertTrue(json.loads(resp2.text)["has_hold_latents"])

    async def test_fix_alpha_erase_restore_and_sized_sprite_rev(self):
        with self._run_with_stubs():
            # Create a stale 128px variant before fixing (simulating rev=0 HD cache)
            proj_before = self.api.read_project(self.folder)
            take_before = proj_before["clips"][0]["takes"][0]
            var0 = await self.api.sized_sprite(self.folder, take_before, 128)
            stale_path = os.path.join(self.clips_dir, var0["illustrated"])
            self.assertTrue(os.path.isfile(stale_path))

            # Validation: missing both masks -> 400
            empty_resp = await self.api.sprite_fix_alpha(_FakeRequest(
                self.folder, body={"clip": self.clip_id, "take": self.take_id, "pose": 1},
            ))
            self.assertEqual(empty_resp.status, 400)

            # Validation: bool pose -> 400
            bool_resp = await self.api.sprite_fix_alpha(_FakeRequest(
                self.folder,
                body={"clip": self.clip_id, "take": self.take_id, "pose": True,
                      "erase_mask": _encode_png_b64(Image.new("L", (64, 64), 255))},
            ))
            self.assertEqual(bool_resp.status, 400)

            # Erase entire upper-left quadrant of pose 1 (frame 10)
            erase = Image.new("L", (64, 64), 0)
            erase_arr = np.zeros((64, 64), dtype=np.uint8)
            erase_arr[12:40, 20:44] = 255
            erase = Image.fromarray(erase_arr, "L")

            resp = await self.api.sprite_fix_alpha(_FakeRequest(
                self.folder,
                body={
                    "clip": self.clip_id,
                    "take": self.take_id,
                    "pose": 1,
                    "erase_mask": _encode_png_b64(erase, data_url=True),
                    "restore_mask": None,
                },
            ))
            self.assertEqual(resp.status, 200)
            payload = json.loads(resp.text)
            self.assertEqual(payload["pose"], 1)
            self.assertEqual(payload["source_index"], 10)
            self.assertEqual(payload["sprite"]["rev"], 1)
            self.assertIn("1", payload["sprite"]["patched_poses"])
            self.assertEqual(payload["sprite"]["patched_poses"]["1"]["method"], "alpha")
            self.assertEqual(payload["revision"], 2)

            # Stale rev=0 variant was purged; sized_sprite generates rev=1 variant cleanly
            self.assertFalse(os.path.isfile(stale_path))
            proj_after = self.api.read_project(self.folder)
            take_after = proj_after["clips"][0]["takes"][0]
            self.assertEqual(take_after["sprite"]["rev"], 1)
            var1 = await self.api.sized_sprite(self.folder, take_after, 128)
            self.assertIn(".r1.", var1["illustrated"])
            self.assertTrue(os.path.isfile(os.path.join(self.clips_dir, var1["illustrated"])))

            # Verify erased region actually became transparent in frame_cache f0010.png
            with Image.open(os.path.join(self.cache_dir, "f0010.png")) as im:
                alpha_after_erase = np.asarray(im.convert("RGBA"))[:, :, 3]
            self.assertEqual(int(alpha_after_erase[180, 128]), 0)

            # Restore region from mocked raw video frame
            restore_arr = np.zeros((64, 64), dtype=np.uint8)
            restore_arr[12:40, 20:44] = 255
            restore_img = Image.fromarray(restore_arr, "L")
            raw_rgb = _make_character_rgb(fg=(30, 220, 110))

            with mock.patch.object(self.api, "_extract_single_frame", new=mock.AsyncMock(return_value=raw_rgb)) as ex_mock:
                resp_restore = await self.api.sprite_fix_alpha(_FakeRequest(
                    self.folder,
                    body={
                        "clip": self.clip_id,
                        "take": self.take_id,
                        "pose": 1,
                        "erase_mask": None,
                        "restore_mask": _encode_png_b64(restore_img, data_url=False),
                    },
                ))
            self.assertEqual(resp_restore.status, 200)
            ex_mock.assert_awaited_once()
            p2 = json.loads(resp_restore.text)
            self.assertEqual(p2["sprite"]["rev"], 2)
            with Image.open(os.path.join(self.cache_dir, "f0010.png")) as im:
                restored_arr = np.asarray(im.convert("RGBA"))
            self.assertGreater(int(restored_arr[180, 128, 3]), 200)
            self.assertGreater(int(restored_arr[180, 128, 1]), 150)

    async def test_fix_alpha_lazily_builds_frame_cache_when_missing(self):
        # Re-create project without frame_cache on disk
        self._create_sprite_project(with_cache=False, with_video=True)
        self.assertFalse(os.path.isdir(self.cache_dir))

        async def fake_run(args):
            # Populate extracted f_0001.png .. f_0039.png into temp directory
            pattern = args[-1]
            tmp_dir = os.path.dirname(pattern)
            frame = _make_character_rgb(fg=(200, 70, 50))
            for i in range(1, 40):
                frame.save(os.path.join(tmp_dir, f"f_{i:04d}.png"), format="PNG")
            return ""

        erase_arr = np.zeros((64, 64), dtype=np.uint8)
        erase_arr[20:30, 24:40] = 255
        with self._run_with_stubs(), mock.patch.object(self.api, "run", side_effect=fake_run):
            resp = await self.api.sprite_fix_alpha(_FakeRequest(
                self.folder,
                body={
                    "clip": self.clip_id,
                    "take": self.take_id,
                    "pose": 0,
                    "erase_mask": _encode_png_b64(Image.fromarray(erase_arr, "L")),
                },
            ))
        self.assertEqual(resp.status, 200)
        data = json.loads(resp.text)
        self.assertEqual(data["sprite"]["frame_cache"], self.cache_name)
        self.assertTrue(os.path.isfile(os.path.join(self.cache_dir, "f0008.png")))

    async def test_nudge_pose_and_revert(self):
        raw_next = _make_character_rgb(fg=(40, 120, 230))
        with self._run_with_stubs():
            # Invalid delta -> 400
            bad = await self.api.sprite_nudge_pose(_FakeRequest(
                self.folder, body={"clip": self.clip_id, "take": self.take_id, "pose": 2, "delta": "1"},
            ))
            self.assertEqual(bad.status, 400)

            # Nudge pose 2 (frame 12) by +1 -> frame 13
            with mock.patch.object(self.api, "_extract_single_frame", new=mock.AsyncMock(return_value=raw_next)) as ex_mock:
                resp = await self.api.sprite_nudge_pose(_FakeRequest(
                    self.folder, body={"clip": self.clip_id, "take": self.take_id, "pose": 2, "delta": 1},
                ))
            self.assertEqual(resp.status, 200)
            ex_mock.assert_awaited_once()
            self.assertEqual(ex_mock.await_args.args[1], 13)
            out = json.loads(resp.text)
            self.assertEqual(out["pose"], 2)
            self.assertEqual(out["source_index"], 13)
            self.assertEqual(out["sprite"]["rev"], 1)
            self.assertEqual(out["sprite"]["layout"]["indices"][2], 13)
            self.assertTrue(os.path.isfile(os.path.join(self.cache_dir, "f0013.png")))

            # Clamp source_index to [0, total_frames - 1]
            with mock.patch.object(self.api, "_extract_single_frame", new=mock.AsyncMock(return_value=raw_next)) as ex_clamp:
                resp_clamp = await self.api.sprite_nudge_pose(_FakeRequest(
                    self.folder, body={"clip": self.clip_id, "take": self.take_id, "pose": 2, "source_index": 999},
                ))
            self.assertEqual(resp_clamp.status, 200)
            self.assertEqual(ex_clamp.await_args.args[1], 38)
            self.assertEqual(json.loads(resp_clamp.text)["source_index"], 38)

            # Pose info reports nudged source_index + orig_index
            info_resp = await self.api.sprite_pose_info(_FakeRequest(
                self.folder, query={"clip": self.clip_id, "take": self.take_id, "pose": "2"},
            ))
            info = json.loads(info_resp.text)
            self.assertEqual(info["source_index"], 38)
            self.assertEqual(info["orig_index"], 12)
            self.assertIsNotNone(info["patched"])

            # Revert pose 2 back to original frame 12
            rev_resp = await self.api.sprite_revert_pose(_FakeRequest(
                self.folder, body={"clip": self.clip_id, "take": self.take_id, "pose": 2},
            ))
            self.assertEqual(rev_resp.status, 200)
            rev_out = json.loads(rev_resp.text)
            self.assertEqual(rev_out["pose"], 2)
            self.assertEqual(rev_out["sprite"]["rev"], 3)
            self.assertNotIn("patched_poses", rev_out["sprite"])

            info_after = json.loads((await self.api.sprite_pose_info(_FakeRequest(
                self.folder, query={"clip": self.clip_id, "take": self.take_id, "pose": "2"},
            ))).text)
            self.assertEqual(info_after["source_index"], 12)
            self.assertIsNone(info_after["patched"])

    async def test_nudge_pose_birefnet_and_fallback(self):
        self._create_sprite_project(with_cache=True, with_video=True, cutout="foreground model")
        self.file_lists["background_removal"] = ["BiRefNet.safetensors"]
        self.node_mappings["LoadBackgroundRemovalModel"] = object()
        self.comfy_nodes.NODE_CLASS_MAPPINGS["LoadBackgroundRemovalModel"] = object()

        raw_rgb = _make_character_rgb(fg=(20, 180, 220))
        biref_rgba = _make_character_rgba(color=(20, 180, 220, 255))
        saved_name = "biref_nudge_out.png"
        saved_sub = f"{store.ROOT}/{self.folder}/clips"

        async def fake_queue(graph, client_id=None):
            self.assertEqual(graph["2"]["class_type"], "LoadBackgroundRemovalModel")
            biref_rgba.save(os.path.join(self.clips_dir, saved_name), format="PNG")
            self.server.PromptServer.instance.prompt_queue.history["job-biref"] = {
                "outputs": {"5": {"images": [{"filename": saved_name, "subfolder": saved_sub}]}},
            }
            return "job-biref"

        with self._run_with_stubs(), \
             mock.patch.object(self.api, "_extract_single_frame", new=mock.AsyncMock(return_value=raw_rgb)), \
             mock.patch.object(self.api, "queue_prompt", side_effect=fake_queue):
            resp = await self.api.sprite_nudge_pose(_FakeRequest(
                self.folder, body={"clip": self.clip_id, "take": self.take_id, "pose": 0, "delta": 1},
            ))
        self.assertEqual(resp.status, 200)
        self.assertEqual(json.loads(resp.text)["source_index"], 9)
        # Temp output was cleaned up
        self.assertFalse(os.path.isfile(os.path.join(self.clips_dir, saved_name)))

        # Fallback to with_matte when ComfyUI BiRefNet fails
        with self._run_with_stubs(), \
             mock.patch.object(self.api, "_extract_single_frame", new=mock.AsyncMock(return_value=raw_rgb)), \
             mock.patch.object(self.api, "queue_prompt", side_effect=self.api.Refused("comfy offline")):
            resp_fb = await self.api.sprite_nudge_pose(_FakeRequest(
                self.folder, body={"clip": self.clip_id, "take": self.take_id, "pose": 1, "delta": -1},
            ))
        self.assertEqual(resp_fb.status, 200)
        self.assertEqual(json.loads(resp_fb.text)["source_index"], 9)

    async def test_fix_ai_masked_and_unmasked(self):
        # Capture original frame 12 (pose 2) before AI repair
        with Image.open(os.path.join(self.cache_dir, "f0012.png")) as im:
            orig_f12 = np.asarray(im.convert("RGBA")).copy()

        # Paint mask over center of tile only
        mask_arr = np.zeros((64, 64), dtype=np.uint8)
        mask_arr[24:40, 24:40] = 255
        mask_b64 = _encode_png_b64(Image.fromarray(mask_arr, "L"), data_url=True)

        repaired_rgb = _make_character_rgb(fg=(20, 230, 210), bg=(255, 255, 255))
        saved_name = "c1_t1_fix_p2_00001_.png"
        saved_sub = f"{store.ROOT}/{self.folder}/clips"
        seen_graphs = []

        async def fake_queue(graph, client_id=None):
            seen_graphs.append(graph)
            # Verify frame-space mask PNG was created and wired to H3LoadStill
            if "7" in graph:
                mask_rel = graph["7"]["inputs"]["file"]
                mask_abs = os.path.join(self.output_dir, *mask_rel.split("/"))
                self.assertTrue(os.path.isfile(mask_abs))
                with Image.open(mask_abs) as m_im:
                    self.assertEqual(m_im.size, (256, 384))
            repaired_rgb.save(os.path.join(self.clips_dir, saved_name), format="PNG")
            pid = f"ai-job-{len(seen_graphs)}"
            self.server.PromptServer.instance.prompt_queue.history[pid] = {
                "outputs": {"25": {"images": [{"filename": saved_name, "subfolder": saved_sub}]}},
            }
            return pid

        with self._run_with_stubs(), \
             mock.patch.object(self.api, "installed_patches", return_value={}), \
             mock.patch.object(self.api, "queue_prompt", side_effect=fake_queue):
            # Validation: bad strength -> 400
            bad_str = await self.api.sprite_fix_ai(_FakeRequest(
                self.folder, body={"clip": self.clip_id, "take": self.take_id, "pose": 2, "strength": "extreme"},
            ))
            self.assertEqual(bad_str.status, 400)

            # Masked AI repair
            resp = await self.api.sprite_fix_ai(_FakeRequest(
                self.folder,
                body={
                    "clip": self.clip_id,
                    "take": self.take_id,
                    "pose": 2,
                    "prompt": "fix extra grey limb",
                    "mask": mask_b64,
                    "strength": "heavy",
                    "seed": 777,
                },
            ))
            self.assertEqual(resp.status, 200)
            out = json.loads(resp.text)
            self.assertEqual(out["pose"], 2)
            self.assertEqual(out["seed"], 777)
            self.assertEqual(out["sprite"]["rev"], 1)
            self.assertEqual(out["sprite"]["patched_poses"]["2"]["method"], "ai")
            self.assertEqual(out["revision"], 2)

            # Temp output & mask files were cleaned up
            self.assertFalse(os.path.isfile(os.path.join(self.clips_dir, saved_name)))
            assets_dir = os.path.join(self.output_dir, store.ROOT, self.folder, "assets")
            self.assertEqual(os.listdir(assets_dir) if os.path.isdir(assets_dir) else [], [])

            # Pixels far outside the mask remain bit-exact; pixels inside changed to repaired cyan
            with Image.open(os.path.join(self.cache_dir, "f0012.png")) as im:
                patched_f12 = np.asarray(im.convert("RGBA"))
            np.testing.assert_array_equal(patched_f12[115:125, 95:105], orig_f12[115:125, 95:105])
            self.assertGreater(int(patched_f12[200, 128, 1]), 150)

            # Unmasked AI repair (mask=None -> SolidMask)
            resp2 = await self.api.sprite_fix_ai(_FakeRequest(
                self.folder,
                body={
                    "clip": self.clip_id,
                    "take": self.take_id,
                    "pose": 2,
                    "prompt": "sharpen silhouette",
                    "mask": None,
                    "strength": "full",
                    "seed": 888,
                },
            ))
            self.assertEqual(resp2.status, 200)
            self.assertEqual(seen_graphs[-1]["8"]["class_type"], "SolidMask")
            self.assertEqual(json.loads(resp2.text)["seed"], 888)
            self.assertEqual(json.loads(resp2.text)["sprite"]["rev"], 2)

            # ComfyUI refusal -> 502
            with mock.patch.object(self.api, "queue_prompt", side_effect=self.api.Refused("queue rejected")):
                err_resp = await self.api.sprite_fix_ai(_FakeRequest(
                    self.folder,
                    body={"clip": self.clip_id, "take": self.take_id, "pose": 2, "prompt": "fix", "strength": "full"},
                ))
            self.assertEqual(err_resp.status, 502)


if __name__ == "__main__":
    unittest.main()
