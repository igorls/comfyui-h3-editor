"""Needs torch and ComfyUI on the path. Run inside the container. See README."""

import os
import tempfile
import unittest

try:
    import torch
    from safetensors.torch import save_file

    import h3lora
except ImportError:
    torch = None


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


if __name__ == "__main__":
    unittest.main()
