"""Needs torch and ComfyUI on the path. Run inside the container. See README."""

import os
import tempfile
import unittest

try:
    import torch
    from safetensors.torch import save_file

    import comfy.ldm.minimax.model as minimax_model
    import comfy.model_patcher
    import comfy.ops
    import comfy.patcher_extension
    from comfy.quant_ops import QuantizedTensor

    import h3lora
    import hyperflow
except ImportError:
    torch = None


class _DummyBase(torch.nn.Module if torch is not None else object):
    def __init__(self, dm):
        super().__init__()
        self.diffusion_model = dm


def _inject(patcher):
    for items in patcher.injections.values():
        for item in items:
            item.inject(patcher)


def _eject(patcher):
    for items in patcher.injections.values():
        for item in items:
            item.eject(patcher)


@unittest.skipIf(torch is None, "needs torch and ComfyUI")
class LoraFileTest(unittest.TestCase):
    def test_keys_load_with_or_without_the_diffusion_model_prefix(self):
        tensors = {"diffusion_model.blocks.3.mlp.fc1.lora_A.weight": torch.zeros(1, 4),
                   "blocks.3.mlp.fc1.lora_B.weight": torch.zeros(4, 1),
                   "diffusion_model.blocks.3.mlp.fc1.alpha": torch.tensor(1.0)}
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "l.safetensors")
            save_file(tensors, path)
            lora = h3lora.read_lora(path)
        self.assertEqual(sorted(lora), ["blocks.3.mlp.fc1.alpha", "blocks.3.mlp.fc1.lora_A.weight",
                                        "blocks.3.mlp.fc1.lora_B.weight"])
        self.assertEqual(h3lora.module_names(lora), ["blocks.3.mlp.fc1"])


@unittest.skipIf(torch is None, "needs torch and ComfyUI")
class StackedLoraTest(unittest.TestCase):
    def test_several_loras_on_one_layer_add_up(self):
        torch.manual_seed(0)
        linear = torch.nn.Linear(6, 5)
        x = torch.randn(3, 6)
        parts = [(torch.randn(5, 2), torch.randn(2, 6), 0.5), (torch.randn(5, 1), torch.randn(1, 6), -1.5)]
        adapter = h3lora.StackedLoRA("diffusion_model.l.weight", parts)
        out = adapter.bypass_forward(linear.forward, x)
        expected = linear(x) + sum(scale * (x @ down.T @ up.T) for up, down, scale in parts)
        self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    def test_sigma_windows_filter_entries_per_step(self):
        torch.manual_seed(2)
        linear = torch.nn.Linear(6, 5)
        x = torch.randn(3, 6)
        state = {"sigma": None}
        up_full, down_full = torch.randn(5, 1), torch.randn(1, 6)
        up_late, down_late = torch.randn(5, 1), torch.randn(1, 6)
        adapter = h3lora.StackedLoRA(
            "diffusion_model.l.weight",
            [(up_full, down_full, 1.0, (1.0, 0.0)), (up_late, down_late, 1.5, (0.68, 0.0))],
            state=state,
        )

        state["sigma"] = 0.92
        early = adapter.bypass_forward(linear.forward, x)
        self.assertTrue(torch.allclose(early, linear(x) + (x @ down_full.T @ up_full.T), atol=1e-5))

        state["sigma"] = 0.675
        late = adapter.bypass_forward(linear.forward, x)
        expected_late = linear(x) + (x @ down_full.T @ up_full.T) + 1.5 * (x @ down_late.T @ up_late.T)
        self.assertTrue(torch.allclose(late, expected_late, atol=1e-5))


class _Adaln(torch.nn.Module if torch is not None else object):
    """The shape of ComfyUI's AdalnProj on a pruned base: an 8-wide curve input, no silu."""

    def __init__(self):
        super().__init__()
        self.expand, self.modalities, self.hidden, self.apply_silu = 2, 1, 3, False
        self.linear = torch.nn.Linear(8, 6)


@unittest.skipIf(torch is None, "needs torch and ComfyUI")
class AdalnTest(unittest.TestCase):
    def test_the_update_comes_from_the_grid_row_matching_the_curve_point(self):
        torch.manual_seed(1)
        base = _Adaln()
        grid = h3lora.silu_grid("cpu")
        table = torch.randn(grid.shape[0], 8)
        a, b = torch.randn(4, grid.shape[1]), torch.randn(6, 4)
        rows = torch.tensor([10, 700])
        t_emb = table[rows]
        chunks = h3lora.adaln_forward(base, [(a, b)], table)(t_emb)
        plain = base.linear(t_emb)
        expected = plain + (b @ (a @ grid[rows].T)).T
        self.assertTrue(torch.allclose(torch.cat(chunks, dim=-1), expected.view(2, 6), atol=1e-3))


@unittest.skipIf(torch is None or not torch.cuda.is_available(), "needs CUDA INT8 kernels and ComfyUI")
class Int8FusedFc2BypassTest(unittest.TestCase):
    def test_small_rank1_fc2_lora_survives_at_runtime_instead_of_rounding_to_zero_in_int8(self):
        torch.manual_seed(4)
        device = torch.device("cuda")
        ops = comfy.ops.manual_cast

        class _TinyDM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList(
                    [torch.nn.ModuleDict({"mlp": minimax_model.MLP(256, 512, dtype=torch.bfloat16, device=device, operations=ops)})]
                    + [torch.nn.Identity() for _ in range(49)]
                )

            def forward(self, x, timestep, **kwargs):
                wrappers = comfy.patcher_extension.get_all_wrappers(
                    comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, kwargs.get("transformer_options", {})
                )
                return comfy.patcher_extension.WrapperExecutor.new_class_executor(
                    self._forward, self, wrappers
                ).execute(x, timestep, **kwargs)

            def _forward(self, x, timestep, **kwargs):
                return self.blocks[0]["mlp"](x)

        dm = _TinyDM()
        # Realistic H3 weight scale (~0.02 -> INT8 step ~0.0008, half-step ~0.0004).
        w1 = QuantizedTensor.from_float(
            torch.randn(1024, 256, device=device, dtype=torch.bfloat16) * 0.02,
            "TensorWiseINT8Layout", is_weight=True, per_channel=True, convrot=True,
        )
        w2 = QuantizedTensor.from_float(
            torch.randn(256, 512, device=device, dtype=torch.bfloat16) * 0.02,
            "TensorWiseINT8Layout", is_weight=True, per_channel=True, convrot=True,
        )
        dm.blocks[0]["mlp"].fc1.weight = torch.nn.Parameter(w1, requires_grad=False)
        dm.blocks[0]["mlp"].fc2.weight = torch.nn.Parameter(w2, requires_grad=False)

        patcher = comfy.model_patcher.ModelPatcher(_DummyBase(dm), device, torch.device("cpu"))
        # Rank-1 delta dw = 1.5 * 0.014 * 0.014 = 0.000294 < half an INT8 step (~0.0004), matching MMH3-RealismSlider-V1.
        gen = torch.Generator().manual_seed(7)
        down = (torch.rand((1, 512), generator=gen) * 2 - 1).to(torch.bfloat16) * 0.014
        up = (torch.rand((256, 1), generator=gen) * 2 - 1).to(torch.bfloat16) * 0.014
        with tempfile.TemporaryDirectory() as folder:
            lora_path = os.path.join(folder, "tiny_fc2.safetensors")
            save_file({
                "blocks.0.mlp.fc2.lora_A.weight": down,
                "blocks.0.mlp.fc2.lora_B.weight": up,
            }, lora_path)
            merged = h3lora.apply(patcher, lora_path, "tiny_fc2", 1.5, mode="merge")
            gated = h3lora.apply(patcher, lora_path, "tiny_fc2", 1.5, mode="auto", start_sigma=0.68, end_sigma=0.0)

        # In merge mode, re-quantizing into INT8 wipes out sub-step rank-1 updates; auto keeps 0 merge patches.
        self.assertEqual(len(merged.patches), 1)
        self.assertEqual(len(gated.patches), 0)
        self.assertIsNone(patcher.get_attachment(h3lora.STACK))

        x = torch.randn(1, 16, 256, device=device, dtype=torch.bfloat16)
        base_out = dm(x, torch.tensor([500.0], device=device)).clone()
        opts = {"patches_replace": {}, "wrappers": gated.wrappers}

        _inject(gated)
        try:
            # At sigma=0.95 (> 0.68), the fused INT8 kernel runs untouched and matches base bit-for-bit.
            early_out = dm(x, torch.tensor([950.0], device=device), transformer_options=opts).clone()
            self.assertTrue(torch.equal(early_out, base_out))

            # At sigma=0.50 (<= 0.68), the bf16 runtime bypass fires cleanly.
            late_out = dm(x, torch.tensor([500.0], device=device), transformer_options=opts).clone()
        finally:
            _eject(gated)

        after_eject = dm(x, torch.tensor([500.0], device=device)).clone()
        self.assertTrue(torch.equal(after_eject, base_out))

        h = dm.blocks[0]["mlp"].fc1(x)
        gate, hidden = h.chunk(2, dim=-1)
        act = torch.nn.functional.silu(gate) * hidden
        expected_delta = 1.5 * torch.nn.functional.linear(
            torch.nn.functional.linear(act, down.to(device)), up.to(device)
        )
        cos = torch.nn.functional.cosine_similarity(
            (late_out - base_out).float().flatten(), expected_delta.float().flatten(), dim=0
        ).item()
        self.assertGreater(cos, 0.94)
        expected_out = (base_out.float() + expected_delta.float()).to(torch.bfloat16)
        self.assertTrue(torch.allclose(late_out, expected_out, atol=1e-3))


@unittest.skipIf(torch is None, "needs torch and ComfyUI")
class FactorizedHyperFlowAttachmentTest(unittest.TestCase):
    def test_lowrank_and_split_qkv_layers_attach_at_runtime_including_trainer_keys(self):
        import test_hyperflow
        DIMS, tiny_checkpoint = test_hyperflow.DIMS, test_hyperflow.tiny_checkpoint

        with tempfile.TemporaryDirectory() as folder:
            ckpt = os.path.join(folder, "hf_tiny.safetensors")
            tiny_checkpoint(ckpt)
            patcher = hyperflow.load(ckpt)

            hidden = DIMS["hidden_size"]
            ffn = DIMS["ffn_hidden_size"]
            lora_path = os.path.join(folder, "factorized_lora.safetensors")
            save_file({
                # SplitQKV of LowRankLinears in block 1 (trainer lora_unet_ key format):
                "lora_unet_blocks_1_attn_qkv_proj.lora_down.weight": torch.full((2, hidden), 0.1, dtype=torch.float32),
                "lora_unet_blocks_1_attn_qkv_proj.lora_up.weight": torch.full((hidden * 3, 2), 0.1, dtype=torch.float32),
                # LowRankLinear mlp.fc2 in block 2 (mirrors fl2va blocks.46.mlp.fc2):
                "diffusion_model.blocks.2.mlp.fc2.lora_A.weight": torch.full((1, ffn), 0.2, dtype=torch.float32),
                "diffusion_model.blocks.2.mlp.fc2.lora_B.weight": torch.full((hidden, 1), 0.2, dtype=torch.float32),
            }, lora_path)

            patched = h3lora.apply(patcher, lora_path, "factorized_lora", 1.5, mode="auto")

        stack = patched.get_attachment(h3lora.STACK)
        self.assertEqual(len(stack), 1)
        self.assertEqual(sorted(stack[0]["bypass"]), [
            "diffusion_model.blocks.1.attn.qkv_proj.weight",
            "diffusion_model.blocks.2.mlp.fc2.weight",
        ])

        dm = patched.model.diffusion_model
        qkv = dm.blocks[1].attn.qkv_proj
        fc2 = dm.blocks[2].mlp.fc2
        x_qkv = torch.randn(2, 4, hidden, dtype=torch.bfloat16)
        x_fc2 = torch.randn(2, 4, ffn, dtype=torch.bfloat16)
        base_qkv = qkv(x_qkv).clone()
        base_fc2 = fc2(x_fc2).clone()

        _inject(patched)
        try:
            out_qkv = qkv(x_qkv).clone()
            out_fc2 = fc2(x_fc2).clone()
        finally:
            _eject(patched)

        down_q = torch.full((2, hidden), 0.1, dtype=torch.bfloat16)
        up_q = torch.full((hidden * 3, 2), 0.1, dtype=torch.bfloat16)
        exp_qkv = base_qkv + 1.5 * torch.nn.functional.linear(torch.nn.functional.linear(x_qkv, down_q), up_q)
        self.assertTrue(torch.allclose(out_qkv, exp_qkv, atol=2e-2))

        down_f = torch.full((1, ffn), 0.2, dtype=torch.bfloat16)
        up_f = torch.full((hidden, 1), 0.2, dtype=torch.bfloat16)
        exp_fc2 = base_fc2 + 1.5 * torch.nn.functional.linear(torch.nn.functional.linear(x_fc2, down_f), up_f)
        self.assertTrue(torch.allclose(out_fc2, exp_fc2, atol=2e-2))


if __name__ == "__main__":
    unittest.main()
