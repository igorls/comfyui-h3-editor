import json
import os
import struct
import tempfile
import unittest

import hybrid
import project as store


def _make_safetensors(path, header_dict):
    raw = json.dumps(header_dict).encode("utf-8")
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(raw)))
        handle.write(raw)


def _sample_header(prefix="model.diffusion_model.", quant=True):
    entries = {}
    for b in (0, 25, 45, 49):
        entries[f"{prefix}blocks.{b}.adaLN_modulation.1.weight"] = {"dtype": "BF16", "shape": [1536, 512], "data_offsets": [0, 0]}
        entries[f"{prefix}blocks.{b}.adaLN_modulation.1.bias"] = {"dtype": "BF16", "shape": [1536], "data_offsets": [0, 0]}
        entries[f"{prefix}blocks.{b}.attn.qkv_proj.weight"] = {"dtype": "F8_E4M3" if quant else "BF16", "shape": [4608, 1536], "data_offsets": [0, 0]}
        if quant:
            entries[f"{prefix}blocks.{b}.attn.qkv_proj.weight_scale"] = {"dtype": "F32", "shape": [1], "data_offsets": [0, 0]}
            entries[f"{prefix}blocks.{b}.attn.qkv_proj.comfy_quant"] = {"dtype": "U8", "shape": [32], "data_offsets": [0, 0]}
        entries[f"{prefix}blocks.{b}.mlp.fc1.weight"] = {"dtype": "BF16", "shape": [4096, 1536], "data_offsets": [0, 0]}
    entries[f"{prefix}final_layer.linear.weight"] = {"dtype": "BF16", "shape": [64, 1536], "data_offsets": [0, 0]}
    return entries


class HybridParsingTest(unittest.TestCase):
    def test_parse_and_format_blocks(self):
        self.assertEqual(hybrid.parse_blocks("45-49"), (45, 46, 47, 48, 49))
        self.assertEqual(hybrid.parse_blocks("0-2, 45-46, 49"), (0, 1, 2, 45, 46, 49))
        self.assertEqual(hybrid.parse_blocks("all"), tuple(range(50)))
        self.assertEqual(hybrid.format_blocks((0, 1, 2, 45, 46, 49)), "0-2, 45-46, 49")

    def test_rejects_invalid_blocks(self):
        for bad in ("", "   ", "49-45", "50", "0-50", "abc", "1-2-3"):
            with self.assertRaises(ValueError):
                hybrid.parse_blocks(bad)


class HybridPlanTest(unittest.TestCase):
    def test_scope_filtering_and_prefix_normalization(self):
        base = _sample_header(prefix="model.diffusion_model.", quant=True)
        overlay = _sample_header(prefix="", quant=True)

        plan_adaln = hybrid.plan_swap(base, overlay, "45-49", scope="adaln")
        self.assertEqual(plan_adaln.blocks_matched, (45, 49))
        self.assertEqual(plan_adaln.families_swapped, 2)
        self.assertEqual(len(plan_adaln.swaps), 4)  # weight + bias for blocks 45 & 49
        self.assertTrue(all(b.startswith("model.diffusion_model.") and not o.startswith("model.diffusion_model.") for b, o in plan_adaln.swaps))

        plan_attn = hybrid.plan_swap(base, overlay, "45-49", scope="attn")
        self.assertEqual(plan_attn.families_swapped, 2)
        self.assertEqual(len(plan_attn.swaps), 6)  # weight + weight_scale + comfy_quant x 2 blocks

        plan_all = hybrid.plan_swap(base, overlay, "45-49", scope="all")
        self.assertEqual(plan_all.families_swapped, 6)  # adaln + attn + mlp x 2 blocks

    def test_skips_mismatched_quantization_family_atomically(self):
        base = _sample_header(quant=True)
        overlay = _sample_header(quant=False)
        plan = hybrid.plan_swap(base, overlay, "45-49", scope="all")
        # attn family is skipped atomically (never partially swapped), while adaln and mlp still swap
        self.assertTrue(any("attn.qkv_proj" in s for s in plan.skipped))
        self.assertFalse(any("attn.qkv_proj" in b for b, _ in plan.swaps))
        self.assertEqual(plan.families_swapped, 4)

    def test_raises_when_nothing_compatible(self):
        base = _sample_header(quant=True)
        overlay = _sample_header(quant=False)
        with self.assertRaises(ValueError):
            hybrid.plan_swap(base, overlay, "45-49", scope="attn")

    def test_read_safetensors_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.safetensors")
            _make_safetensors(path, {"__metadata__": {"format": "pt"}, **_sample_header()})
            header = hybrid.read_safetensors_header(path)
            self.assertNotIn("__metadata__", header)
            self.assertIn("model.diffusion_model.blocks.45.adaLN_modulation.1.weight", header)

    def test_rejects_hyperflow_checkpoint_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "hf.safetensors")
            _make_safetensors(path, {"__metadata__": {"hyperflow": '{"format": 1}'}, **_sample_header()})
            with self.assertRaisesRegex(ValueError, "HyperFlow"):
                hybrid.read_safetensors_header(path)


class ProjectHybridAndTakeMetadataTest(unittest.TestCase):
    def _project_with_clip(self):
        proj = store.new_project("Hybrid Test")
        clip, _ = store.add_clip(proj, {"name": "Shot 1", "action": "A person smiling in warm light."})
        return proj, clip

    def test_hybrid_project_config_and_graph_wiring(self):
        proj, clip = self._project_with_clip()
        store.set_project(proj, {
            "hybrid": {
                "on": True,
                "base": "minimax_h3_fl2va_fp8.safetensors",
                "overlay": "minimax_h3_ref2va_fp8.safetensors",
                "blocks": "45-49",
                "scope": "adaln",
                "checkpoint": "ref2va",
                "weight_dtype": "default",
            }
        })
        self.assertIn("hybrid", store.describe(proj))
        job, graph = store.plan_render(proj, "proj", clip["id"], "draft", seed=42)
        self.assertEqual(graph["1"]["class_type"], "H3HybridLoader")
        self.assertEqual(graph["1"]["inputs"]["base_unet"], "minimax_h3_fl2va_fp8.safetensors")
        self.assertEqual(graph["1"]["inputs"]["overlay_unet"], "minimax_h3_ref2va_fp8.safetensors")
        self.assertEqual(graph["1"]["inputs"]["blocks"], "45-49")
        self.assertEqual(graph["10"]["inputs"]["checkpoint"], "ref2va")

        models = job["models"]
        self.assertEqual(models["unet"], "minimax_h3_fl2va_fp8.safetensors")
        self.assertEqual(models["hybrid"]["overlay"], "minimax_h3_ref2va_fp8.safetensors")
        self.assertEqual(models["hybrid"]["blocks"], "45-49")
        self.assertEqual(models["hybrid"]["scope"], "adaln")

    def test_take_records_extended_generation_parameters(self):
        proj, clip = self._project_with_clip()
        job, _graph = store.plan_render(proj, "proj", clip["id"], "final", seed=12345)
        models = job["models"]
        self.assertIn("audio_vae", models)
        self.assertIn("shift", models)
        self.assertIn("shift_audio", models)
        self.assertIn("sample_size", models)
        self.assertTrue("scheduler" in models or "sigmas" in models)
        self.assertIn("refine", models)
        self.assertTrue("steps" in models["refine"] or "from_step" in models["refine"])
        self.assertIn("size", models["refine"])

    def test_hybrid_validation_rejects_same_model_or_gguf(self):
        proj, _ = self._project_with_clip()
        with self.assertRaises(ValueError):
            store.set_project(proj, {"hybrid": {"on": True, "base": "a.safetensors", "overlay": "a.safetensors"}})
        with self.assertRaises(ValueError):
            store.set_project(proj, {"hybrid": {"on": True, "base": "a.gguf", "overlay": "b.safetensors"}})


if __name__ == "__main__":
    unittest.main()
