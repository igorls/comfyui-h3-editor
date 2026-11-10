"""Needs torch and ComfyUI on the path, and gguf for the GGUF tests. Run inside the container with PYTHONPATH set
to ComfyUI. See README."""

import gc
import json
import os
import struct
import tempfile
import unittest
from unittest import mock

try:
    import torch
    from safetensors.torch import save_file
    from torch import nn

    import hyperflow
except ImportError:
    hyperflow = None

try:
    import gguf
    import numpy as np

    import ggml_dequant
    import ggml_ops
except ImportError:
    gguf = None

# Every type the torch dequantizer knows, checked against gguf's own numpy reference.
TYPES = ("BF16", "Q8_0", "Q5_1", "Q5_0", "Q4_1", "Q4_0", "Q6_K", "Q5_K", "Q4_K", "Q3_K", "Q2_K", "IQ4_NL", "IQ4_XS")

# A three-block model small enough to build and run on the CPU.
DIMS = dict(hidden_size=64, num_layers=3, token_refiner_num_layers=1, num_attention_heads=4, attention_head_dim=16,
            ffn_hidden_size=96, latents_dim=4, audio_latents_dim=8, text_dim=32, timestep_input_dim=16,
            time_embed_hidden_size=64, time_embed_dim=16, rope_inv_freq_len=4)
# The convrot group has to divide every layer's width: 16 here, where the published files use 256.
GROUP = 16
GATE = 0.5
MARKED = {"format": 1, "variant": "fl2va", "weights": "int8_convrot", "blocks": 3}
INT8 = "int8_tensorwise"


def metadata(info=MARKED):
    return {"format": "pt", "hyperflow": json.dumps(info)}


def tiny_checkpoint(path):
    """A HyperFlow INT8 file with the three kinds of block the published ones hold: all INT8, low-rank with an
    INT8 fc2, and INT8 with a low-rank fc2."""
    import comfy.ops

    stock = hyperflow.packed_dit.MiniMaxH3Model(**DIMS, dtype=torch.bfloat16, operations=comfy.ops.disable_weight_init)
    generator = torch.Generator().manual_seed(3)

    def random(*shape):
        return torch.randn(shape, generator=generator) * 0.05

    state = {name: random(*value.shape).to(value.dtype) for name, value in stock.state_dict().items()}
    quant = json.dumps({"format": INT8, "convrot": True, "convrot_groupsize": GROUP}).encode()

    def int8(name):
        weight = state.pop(name + ".weight").float()
        scale = weight.abs().amax(dim=1, keepdim=True) / 127.0
        state[name + ".weight"] = torch.round(weight / scale).clamp(-127, 127).to(torch.int8)
        state[name + ".weight_scale"] = scale
        state[name + ".comfy_quant"] = torch.tensor(list(quant), dtype=torch.uint8)

    def low_rank(name, rank, out_features, in_features):
        state[name + ".down.weight"] = random(rank, in_features).to(torch.bfloat16)
        state[name + ".up.weight"] = random(out_features, rank).to(torch.bfloat16)

    for layer in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"):
        int8("blocks.0." + layer)
    hidden = state.pop("blocks.1.attn.qkv_proj.weight").shape[1]
    for part, rank in zip("qkv", (8, 8, 12)):
        low_rank(f"blocks.1.attn.{part}_proj", rank, hidden, hidden)
    for layer, rank in (("attn.out_proj", 8), ("mlp.fc1", 16)):
        low_rank("blocks.1." + layer, rank, *state.pop(f"blocks.1.{layer}.weight").shape)
    int8("blocks.1.mlp.fc2")
    for layer in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1"):
        int8("blocks.2." + layer)
    low_rank("blocks.2.mlp.fc2", 8, *state.pop("blocks.2.mlp.fc2.weight").shape)
    state["hyperflow_gate"] = torch.tensor([GATE])
    state["hyperflow_sigmas"] = hyperflow.step_sigmas(6)
    for leaf in ("proj_in.weight", "proj_in.bias", "proj_out.weight", "proj_out.bias"):
        state["endpoint_time_embedder." + leaf] = state["time_embedder." + leaf] + 0.01
    save_file(state, path, metadata=metadata())


@unittest.skipIf(hyperflow is None or gguf is None, "needs torch, gguf and ComfyUI")
class DequantTest(unittest.TestCase):
    def test_torch_dequant_matches_the_reference_for_every_type(self):
        generator = np.random.default_rng(3)
        for name in TYPES:
            qtype = gguf.GGMLQuantizationType[name]
            block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
            with self.subTest(qtype=name):
                rows, blocks = 4, 2
                raw = generator.integers(0, 256, size=(rows, blocks * type_size), dtype=np.uint8)
                # Random bytes can make an infinite or undefined scale, so only finite rows are compared.
                expected = gguf.quants.dequantize(raw, qtype)
                finite = np.isfinite(expected).all(axis=1)
                tensor = ggml_ops.GGMLTensor(torch.from_numpy(raw.copy()), tensor_type=qtype,
                                             tensor_shape=torch.Size((rows, blocks * block_size)))
                # At run time the kernels compute in float16, as upstream's do; here float32, to compare the math.
                found = ggml_dequant.dequantize_tensor(tensor, dtype=torch.float32, dequant_dtype=torch.float32).numpy()
                np.testing.assert_allclose(found[finite], expected[finite], rtol=1e-5, atol=1e-5)


@unittest.skipIf(hyperflow is None, "needs torch and ComfyUI")
class HyperFlowTest(unittest.TestCase):
    def test_the_step_schedules_run_from_one_to_zero(self):
        for steps in (4, 6, 8):
            sigmas = hyperflow.step_sigmas(steps)
            self.assertEqual(sigmas.shape[0], steps + 1)
            self.assertEqual((float(sigmas[0]), float(sigmas[-1])), (1.0, 0.0))
            self.assertTrue(bool((sigmas[1:] < sigmas[:-1]).all()))

    @unittest.skipIf(gguf is None, "needs gguf")
    def test_factorized_parts_leave_dense_placeholders(self):
        q8 = gguf.GGMLQuantizationType.Q8_0

        def weight(*shape):
            return ggml_ops.GGMLTensor(torch.empty(0, dtype=torch.uint8), tensor_type=q8, tensor_shape=torch.Size(shape))

        state = {
            "blocks.2.attn.q_proj.weight": weight(64, 32),
            "blocks.2.attn.k_proj.down.weight": weight(8, 32),
            "blocks.2.attn.k_proj.up.weight": weight(64, 8),
            "blocks.2.attn.v_proj.weight": weight(64, 32),
            "blocks.2.mlp.fc2.down.weight": weight(4, 128),
            "blocks.2.mlp.fc2.up.weight": weight(32, 4),
            "blocks.2.mlp.fc1.weight": weight(256, 32),
            "blocks.5.attn.out_proj.weight": weight(32, 32),
        }
        plan = hyperflow.FactorizedPlan(state, hyperflow.gguf_placeholder)
        self.assertEqual(sorted(plan.parts[2]), ["attn.k_proj", "attn.q_proj", "attn.v_proj", "mlp.fc2"])
        self.assertEqual(tuple(state["blocks.2.attn.qkv_proj.weight"].tensor_shape), (192, 32))
        self.assertEqual(tuple(state["blocks.2.mlp.fc2.weight"].tensor_shape), (32, 128))
        # A dense layer outside the factorized ones stays as it is.
        self.assertEqual(tuple(state["blocks.2.mlp.fc1.weight"].tensor_shape), (256, 32))
        self.assertNotIn(5, plan.parts)

    def test_the_endpoint_is_the_next_sigma_of_the_running_schedule(self):
        schedule = hyperflow.step_sigmas(6)
        self.assertAlmostEqual(float(hyperflow.step_endpoint(schedule[2], schedule)), float(schedule[3]), places=6)
        self.assertEqual(float(hyperflow.step_endpoint(schedule[-1], schedule)), 0.0)
        self.assertEqual(float(hyperflow.step_endpoint(torch.tensor(0.5), None)), 0.5)

    def test_a_missing_gguf_package_is_said_not_raised(self):
        with mock.patch("importlib.util.find_spec", return_value=None):
            message = hyperflow.gguf_missing()
        self.assertIn("pip install -r requirements.txt", message)
        self.assertIn("INT8 safetensors files load without it", message)
        with mock.patch("importlib.util.find_spec", return_value=object()):
            self.assertIsNone(hyperflow.gguf_missing())


@unittest.skipIf(hyperflow is None, "needs torch and ComfyUI")
class SafetensorsTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = folder.name

    def file(self, name, info=None, tensors=None):
        path = os.path.join(self.folder, name)
        save_file(tensors or {"hyperflow_gate": torch.ones(1)}, path, metadata=None if info is None else metadata(info))
        return path

    def raw(self, name, data):
        path = os.path.join(self.folder, name)
        with open(path, "wb") as file:
            file.write(data)
        return path

    def test_a_hyperflow_file_is_told_by_its_metadata(self):
        self.assertEqual(hyperflow.hyperflow_info(self.file("marked.safetensors", MARKED)), MARKED)
        self.assertEqual(hyperflow.checked_info(self.file("again.safetensors", MARKED))["variant"], "fl2va")
        plain = self.file("plain.safetensors")
        self.assertIsNone(hyperflow.hyperflow_info(plain))
        with self.assertRaisesRegex(RuntimeError, "not a HyperFlow checkpoint"):
            hyperflow.checked_info(plain)
        with self.assertRaisesRegex(RuntimeError, "not a HyperFlow checkpoint"):
            hyperflow.load(plain)

    def test_a_newer_layout_asks_for_an_update(self):
        newer = self.file("newer.safetensors", {**MARKED, "format": 2})
        self.assertEqual(hyperflow.hyperflow_info(newer)["format"], 2)
        with self.assertRaisesRegex(RuntimeError, "layout 2.*Update the pack"):
            hyperflow.load(newer)

    def test_a_file_that_is_not_safetensors_is_no_hyperflow_file(self):
        def header(text):
            return struct.pack("<Q", len(text)) + text

        broken = {
            "short": b"hello",
            "huge": b"\xff" * 16,
            "cut": struct.pack("<Q", 400) + b"{",
            "text": header(b"not json at all"),
            "list": header(b"[1, 2]"),
            "metadata_list": header(b'{"__metadata__": ["hyperflow"]}'),
            "metadata_text": header(b'{"__metadata__": {"hyperflow": "not json"}}'),
            "info_number": header(b'{"__metadata__": {"hyperflow": "7"}}'),
        }
        for name, data in broken.items():
            with self.subTest(file=name):
                self.assertIsNone(hyperflow.hyperflow_info(self.raw(name + ".safetensors", data)))
        self.assertIsNone(hyperflow.hyperflow_info(os.path.join(self.folder, "missing.safetensors")))

    def test_a_rewritten_file_is_read_again(self):
        path = self.file("model.safetensors")
        self.assertIsNone(hyperflow.hyperflow_info(path))
        self.file("model.safetensors", MARKED, {"hyperflow_gate": torch.ones(3)})
        self.assertEqual(hyperflow.hyperflow_info(path), MARKED)

    def test_the_loader_lists_ggufs_and_marked_safetensors_only(self):
        self.file("marked-int8_convrot.safetensors", MARKED)
        self.file("plain_h3.safetensors")
        self.raw("broken.safetensors", b"hello")
        self.raw("any-Q4_K_M.gguf", b"GGUF")
        self.raw("notes.txt", b"hello")
        folders = {"diffusion_models": ([self.folder], set())}
        paths = hyperflow.folder_paths
        with mock.patch.object(paths, "folder_names_and_paths", folders), \
                mock.patch.object(paths, "get_filename_list", lambda key: sorted(os.listdir(self.folder))), \
                mock.patch.object(paths, "get_full_path", lambda key, name: os.path.join(self.folder, name)):
            self.assertEqual(hyperflow.files(), ["any-Q4_K_M.gguf", "marked-int8_convrot.safetensors"])
            self.assertEqual(hyperflow.model_path("plain_h3.safetensors"), os.path.join(self.folder, "plain_h3.safetensors"))

    def test_a_file_is_found_under_another_folder_or_separator(self):
        listed = ["hyperflow/model-int8_convrot.safetensors", "other.safetensors"]
        folders = {"diffusion_models": ([self.folder], set())}
        paths = hyperflow.folder_paths

        def full_path(key, name):
            return "/models/" + name if name in listed else None

        with mock.patch.object(paths, "folder_names_and_paths", folders), \
                mock.patch.object(paths, "get_filename_list", lambda key: listed), \
                mock.patch.object(paths, "get_full_path", full_path):
            found = "/models/hyperflow/model-int8_convrot.safetensors"
            for name in ("hyperflow/model-int8_convrot.safetensors", "hyperflow\\model-int8_convrot.safetensors",
                         "model-int8_convrot.safetensors", "elsewhere/model-int8_convrot.safetensors"):
                with self.subTest(name=name):
                    self.assertEqual(hyperflow.model_path(name), found)
            for name in ("missing.safetensors", "", None):
                with self.subTest(name=name):
                    with self.assertRaises(FileNotFoundError):
                        hyperflow.model_path(name)

    def test_a_marked_file_without_the_step_endpoint_tensors_does_not_load(self):
        with self.assertRaisesRegex(RuntimeError, "no step-endpoint tensors"):
            hyperflow.load(self.file("half.safetensors", MARKED))

    def test_int8_layers_stay_and_low_rank_ones_leave_placeholders(self):
        state = {}

        def int8(name, out_features, in_features):
            state[name + ".weight"] = torch.zeros(out_features, in_features, dtype=torch.int8)
            state[name + ".weight_scale"] = torch.ones(out_features, 1)
            state[name + ".comfy_quant"] = torch.zeros(8, dtype=torch.uint8)

        def low_rank(name, rank, out_features, in_features):
            state[name + ".down.weight"] = torch.zeros(rank, in_features, dtype=torch.bfloat16)
            state[name + ".up.weight"] = torch.zeros(out_features, rank, dtype=torch.bfloat16)

        dense = {"attn.qkv_proj": (96, 32), "attn.out_proj": (32, 32), "mlp.fc1": (128, 32), "mlp.fc2": (32, 64)}
        for layer, shape in dense.items():
            int8("blocks.0." + layer, *shape)
        # Like block 2 of the published fl2va file: everything low-rank but fc2.
        for part, rank in zip("qkv", (4, 4, 6)):
            low_rank(f"blocks.1.attn.{part}_proj", rank, 32, 32)
        low_rank("blocks.1.attn.out_proj", 4, *dense["attn.out_proj"])
        low_rank("blocks.1.mlp.fc1", 8, *dense["mlp.fc1"])
        int8("blocks.1.mlp.fc2", *dense["mlp.fc2"])
        # Like its block 46: everything INT8 but fc2.
        for layer in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1"):
            int8("blocks.2." + layer, *dense[layer])
        low_rank("blocks.2.mlp.fc2", 4, *dense["mlp.fc2"])
        stored = dict(state)

        plan = hyperflow.FactorizedPlan(state, hyperflow.dense_placeholder)
        self.assertNotIn(0, plan.parts)
        self.assertEqual(sorted(plan.parts[1]), ["attn.k_proj", "attn.out_proj", "attn.q_proj", "attn.v_proj", "mlp.fc1"])
        self.assertEqual(sorted(plan.parts[2]), ["mlp.fc2"])
        self.assertEqual(sorted(plan.parts[2]["mlp.fc2"]), ["down", "up"])
        self.assertEqual([name for name in state if ".down." in name or ".up." in name or "_proj.q" in name], [])
        for name, tensor in stored.items():
            if tensor.dtype != torch.bfloat16:
                self.assertIs(state[name], tensor, name)
        # What ComfyUI detects the model from: one dense weight per layer, in every block.
        for block in (1, 2):
            for layer, shape in dense.items():
                self.assertEqual(tuple(state[f"blocks.{block}.{layer}.weight"].shape), shape, (block, layer))
        for name in ("blocks.1.attn.qkv_proj.weight", "blocks.1.attn.out_proj.weight", "blocks.1.mlp.fc1.weight",
                     "blocks.2.mlp.fc2.weight"):
            self.assertEqual(state[name].dtype, torch.bfloat16, name)

    def test_a_placeholder_has_the_dense_shape_and_no_weight_behind_it(self):
        placeholder = hyperflow.dense_placeholder(None, (21504, 5376))
        self.assertEqual((tuple(placeholder.shape), placeholder.dtype), ((21504, 5376), torch.bfloat16))
        self.assertEqual(placeholder.untyped_storage().nbytes(), 2)

    def test_float32_tensors_are_put_back_after_a_cast(self):
        root = nn.Module()
        root.proj = nn.Linear(4, 2)
        root.norm = nn.Linear(4, 2)
        state = {
            "proj.weight": torch.randn(2, 4),
            "proj.bias": torch.randn(2),
            "norm.weight": torch.randn(2, 4).to(torch.bfloat16),
            "blocks.0.mlp.fc2.weight_scale": torch.ones(2, 1),
            "gone.weight": torch.randn(2, 4),
            "hyperflow_other": torch.ones(1),
        }
        kept = hyperflow.float32_tensors(state)
        self.assertEqual(sorted(kept), ["gone.weight", "hyperflow_other", "proj.bias", "proj.weight"])
        root.to(torch.bfloat16)
        self.assertEqual(hyperflow.keep_float32(root, kept), 2)
        self.assertEqual((root.proj.weight.dtype, root.proj.bias.dtype), (torch.float32, torch.float32))
        self.assertTrue(torch.equal(root.proj.weight, state["proj.weight"]))
        self.assertFalse(root.proj.weight.requires_grad)
        # ComfyUI casts a weight to this dtype when it moves the model, so it has to say float32 too.
        self.assertEqual((root.proj.weight_comfy_model_dtype, root.proj.bias_comfy_model_dtype), (torch.float32,) * 2)
        self.assertEqual(root.norm.weight.dtype, torch.bfloat16)
        self.assertEqual(hyperflow.keep_float32(root, kept), 0)

    def test_int8_work_runs_in_row_chunks_with_the_same_result(self):
        rows = []

        class Counted(nn.Linear):
            def forward(self, x):
                rows.append(x.reshape(-1, x.shape[-1]).shape[0])
                return super().forward(x)

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(8, 16, bias=False)
                self.fc2 = Counted(16, 4, bias=False)

            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x)))

        root = nn.Module()
        root.plain = Counted(8, 4, bias=False)
        root.int8 = Counted(8, 4, bias=False)
        root.mlp = MLP()
        root.int8.quant_format = root.mlp.fc2.quant_format = INT8
        x = torch.randn(2, 5, 8)
        whole = {name: root.get_submodule(name)(x) for name in ("plain", "int8", "mlp")}

        with mock.patch.object(hyperflow, "ROW_ELEMENTS", 32):
            # The MLP counts beside its fc2: ComfyUI runs fc2 inside the MLP's own fused call.
            self.assertEqual(hyperflow.limit_int8_rows(root), 3)
        self.assertEqual([name for name, module in root.named_modules() if "forward" in vars(module)],
                         ["int8", "mlp", "mlp.fc2"])
        for name, expected, calls in (("plain", whole["plain"], [10]), ("int8", whole["int8"], [4, 4, 2]),
                                      ("mlp", whole["mlp"], [2] * 5)):
            del rows[:]
            found = root.get_submodule(name)(x)
            self.assertEqual(rows, calls, name)
            self.assertEqual(found.shape, expected.shape, name)
            self.assertTrue(torch.allclose(found, expected, atol=1e-6), name)

    def test_a_call_within_the_limit_is_not_split(self):
        calls = []

        def forward(x, scale=1.0):
            calls.append(tuple(x.shape))
            return x[..., :2] * scale

        with mock.patch.object(hyperflow, "ROW_ELEMENTS", 40):
            chunked = hyperflow._row_chunks(forward, 8)
        x = torch.randn(5, 8)
        self.assertTrue(torch.equal(chunked(x, scale=2.0), x[:, :2] * 2.0))
        self.assertEqual(calls, [(5, 8)])
        del calls[:]
        x = torch.randn(3, 2, 8)
        self.assertTrue(torch.equal(chunked(x, scale=2.0), x[..., :2] * 2.0))
        self.assertEqual(calls, [(5, 8), (1, 8)])


@unittest.skipIf(hyperflow is None, "needs torch and ComfyUI")
class TinyCheckpointTest(unittest.TestCase):
    """A synthetic INT8 checkpoint through ComfyUI's own loader, on the CPU."""

    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.path = os.path.join(cls.folder.name, "tiny_hyperflow-int8_convrot.safetensors")
        tiny_checkpoint(cls.path)
        cls.patcher = hyperflow.load(cls.path)
        cls.model = cls.patcher.model.diffusion_model

    @classmethod
    def tearDownClass(cls):
        cls.patcher = cls.model = None
        gc.collect()
        cls.folder.cleanup()

    def layer(self, name):
        return self.model.get_submodule(name)

    def test_the_blocks_are_rebuilt_as_the_file_stores_them(self):
        model = self.model
        self.assertEqual((type(model).__name__, type(model.time_embedder).__name__), ("HyperFlowModel", "TwoTimeEmbedder"))
        self.assertEqual(model.dtype, torch.bfloat16)
        stock_mlp = hyperflow.packed_dit.MLP
        first, second, third = model.blocks
        for linear in (first.attn.qkv_proj, first.attn.out_proj, first.mlp.fc1, first.mlp.fc2):
            self.assertEqual(linear.quant_format, INT8)
            # ComfyUI wraps the weight and reports the dtype it computes in; what it holds stays int8.
            self.assertEqual((linear.weight.dtype, linear.weight._qdata.dtype), (torch.bfloat16, torch.int8))
        self.assertIs(type(first.mlp), stock_mlp)

        self.assertIsInstance(second.attn.qkv_proj, hyperflow.SplitQKV)
        self.assertEqual([part.down.weight.shape[0] for part in second.attn.qkv_proj.children()], [8, 8, 12])
        self.assertIsInstance(second.attn.out_proj, hyperflow.LowRankLinear)
        self.assertIsInstance(second.mlp.fc1, hyperflow.LowRankLinear)
        self.assertEqual(second.mlp.fc2.quant_format, INT8)
        # An INT8 fc2 keeps ComfyUI's own MLP, which runs it in one fused call.
        self.assertIs(type(second.mlp), stock_mlp)

        for linear in (third.attn.qkv_proj, third.attn.out_proj, third.mlp.fc1):
            self.assertEqual(linear.quant_format, INT8)
        self.assertIsInstance(third.mlp.fc2, hyperflow.LowRankLinear)
        self.assertIs(type(third.mlp), hyperflow.FactorizedMLP)
        for name in ("blocks.1.attn.out_proj.down", "blocks.1.attn.qkv_proj.q_proj.up", "blocks.2.mlp.fc2.down"):
            self.assertEqual(self.layer(name).weight.dtype, torch.bfloat16, name)

    def test_full_precision_tensors_stay_float32(self):
        for name in ("video_patch_proj", "audio_patch_proj", "time_embedder.proj_in", "time_embedder.proj_out",
                     "endpoint_time_embedder.proj_in", "endpoint_time_embedder.proj_out", "final_layer.video_out",
                     "final_layer.audio_out"):
            layer = self.layer(name)
            self.assertEqual((layer.weight.dtype, layer.weight_comfy_model_dtype), (torch.float32, torch.float32), name)
        self.assertEqual(self.model.condition_proj.weight.dtype, torch.bfloat16)

    def test_the_schedule_and_the_gate_come_from_the_file(self):
        self.assertTrue(torch.equal(self.model.hyperflow_sigmas, hyperflow.step_sigmas(6)))
        self.assertEqual(self.model.time_embedder.__dict__["endpoint_state"].gate, GATE)
        self.assertEqual(self.patcher.size, 0)
        self.assertEqual([hyperflow.sigmas(self.patcher, steps).shape[0] for steps in (4, 6, 8)], [5, 7, 9])
        with self.assertRaisesRegex(ValueError, r"\[4, 6, 8\] steps, got 5"):
            hyperflow.sigmas(self.patcher, 5)

    def test_every_int8_module_runs_in_row_chunks(self):
        wrapped = [name for name, module in self.model.named_modules() if "forward" in vars(module)]
        self.assertEqual(sorted(wrapped), sorted([
            "blocks.0.attn.qkv_proj", "blocks.0.attn.out_proj", "blocks.0.mlp", "blocks.0.mlp.fc1", "blocks.0.mlp.fc2",
            "blocks.1.mlp", "blocks.1.mlp.fc2",
            "blocks.2.attn.qkv_proj", "blocks.2.attn.out_proj", "blocks.2.mlp.fc1",
        ]))

    def test_int8_and_low_rank_layers_run(self):
        x = torch.randn(2, 5, DIMS["hidden_size"], generator=torch.Generator().manual_seed(5)).to(torch.bfloat16)
        widths = {"blocks.0.attn.qkv_proj": 192, "blocks.0.attn.out_proj": 64, "blocks.0.mlp": 64,
                  "blocks.1.attn.qkv_proj": 192, "blocks.1.attn.out_proj": 64, "blocks.1.mlp": 64,
                  "blocks.2.attn.qkv_proj": 192, "blocks.2.attn.out_proj": 64, "blocks.2.mlp": 64}
        for name, width in widths.items():
            with self.subTest(layer=name):
                found = self.layer(name)(x)
                self.assertEqual((tuple(found.shape), found.dtype), ((2, 5, width), torch.bfloat16))
                self.assertTrue(bool(torch.isfinite(found.float()).all()))
                self.assertGreater(float(found.float().abs().max()), 0.0)

    def test_row_chunks_give_the_whole_call_s_output(self):
        with mock.patch.object(hyperflow, "ROW_ELEMENTS", 256):
            patcher = hyperflow.load(self.path)
        model = patcher.model.diffusion_model
        x = torch.randn(2, 5, DIMS["hidden_size"], generator=torch.Generator().manual_seed(7)).to(torch.bfloat16)
        # 256 elements are one to four rows of these layers, so ten rows make several chunks.
        for name in ("blocks.0.attn.qkv_proj", "blocks.0.mlp", "blocks.1.mlp", "blocks.2.attn.out_proj"):
            with self.subTest(layer=name):
                layer = model.get_submodule(name)
                self.assertIn("forward", vars(layer))
                whole = type(layer).forward(layer, x)
                self.assertTrue(torch.equal(layer(x), whole))
        del patcher, model
        gc.collect()

    def test_the_time_embedding_moves_toward_the_step_endpoint_by_the_gate(self):
        embedder = self.model.time_embedder
        state = embedder.__dict__["endpoint_state"]
        self.addCleanup(state.set_endpoints, {})
        t = torch.tensor([0.25])
        plain = hyperflow.packed_dit.TimeEmbedder.forward(embedder, t)
        # With no endpoint set for this time, r = t.
        same = self.model.endpoint_time_embedder(t).to(plain.dtype)
        self.assertTrue(torch.allclose(embedder(t), plain + GATE * (same - plain), atol=1e-6))
        state.set_endpoints({0.25: 0.5})
        end = self.model.endpoint_time_embedder(torch.tensor([0.5])).to(plain.dtype)
        self.assertFalse(torch.allclose(end, same, atol=1e-6))
        self.assertTrue(torch.allclose(embedder(t), plain + GATE * (end - plain), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
