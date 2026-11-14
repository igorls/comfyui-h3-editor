"""Needs torch and safetensors. Run inside the container. See README."""

import os
import tempfile
import unittest

try:
    import torch
    from safetensors.torch import save_file

    import clipproj
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "needs torch")
class ProjectionTest(unittest.TestCase):
    def write(self, tensors, meta):
        folder = tempfile.mkdtemp()
        path = os.path.join(folder, "p.safetensors")
        save_file(tensors, path, metadata=meta)
        return path

    def test_a_projection_standardises_maps_and_restores_the_scale(self):
        torch.manual_seed(0)
        d_in, d_out = 4, 6
        tensors = {"W": torch.randn(d_in, d_out).half(), "mean_in": torch.randn(d_in), "std_in": torch.rand(d_in) + 0.5,
                   "mean_out": torch.randn(d_out), "std_out": torch.rand(d_out) + 0.5, "sink_out": torch.full((d_out,), 7.0)}
        path = self.write(tensors, {"tap": "24", "cos_test": "0.9"})
        proj = clipproj.load_projection(path)
        self.assertEqual((proj["tap"], proj["cos_test"], proj["W"].dtype), (24, 0.9, torch.float32))
        self.assertEqual(clipproj.projection_width(path), d_in)
        h = torch.randn(1, 3, d_in)
        cond = clipproj.project(h, proj, {})
        want = ((h - tensors["mean_in"]) / tensors["std_in"]) @ tensors["W"].float() * tensors["std_out"] + tensors["mean_out"]
        # Token 0, the attention sink, takes the measured vector; the rest are projected.
        self.assertTrue(torch.allclose(cond[0, 1:], want[0, 1:], atol=1e-5))
        self.assertTrue(torch.equal(cond[0, 0], tensors["sink_out"]))
        with self.assertRaises(ValueError):
            clipproj.project(torch.randn(1, 3, d_in + 1), proj, {})

    def test_a_residual_network_adds_to_the_matrix_or_stands_alone(self):
        torch.manual_seed(1)
        d_in, d_out, hidden = 4, 6, 8
        stats = {"mean_in": torch.zeros(d_in), "std_in": torch.ones(d_in), "mean_out": torch.zeros(d_out), "std_out": torch.ones(d_out)}
        mlp = {"mlp.0.weight": torch.randn(hidden, d_in), "mlp.0.bias": torch.randn(hidden), "mlp.2.weight": torch.randn(d_out, hidden)}
        w = torch.randn(d_in, d_out)
        h = torch.randn(1, 2, d_in)
        net = torch.nn.functional.gelu(h @ mlp["mlp.0.weight"].T + mlp["mlp.0.bias"]) @ mlp["mlp.2.weight"].T
        both = clipproj.load_projection(self.write({"W": w, **stats, **mlp}, {"tap": "24"}))
        self.assertTrue(torch.allclose(clipproj.project(h, both, {}), h @ w + net, atol=1e-5))
        alone = clipproj.load_projection(self.write({**stats, **mlp}, {"tap": "24"}))
        self.assertTrue(torch.allclose(clipproj.project(h, alone, {}), net, atol=1e-5))
        with self.assertRaises(ValueError):
            clipproj.load_projection(self.write(dict(stats), {"tap": "24"}))
        with self.assertRaises(KeyError):
            clipproj.load_projection(self.write({"W": w}, {"tap": "24"}))

    def test_vision_blocks_are_tagged_with_their_start_and_end_tokens(self):
        tags = clipproj.token_tags(10, [{"type": "image", "index": 3, "size": 4}])
        self.assertEqual(tags.tolist(), [1, 1, 0, 0, 0, 0, 0, 0, 1, 1])


if __name__ == "__main__":
    unittest.main()
