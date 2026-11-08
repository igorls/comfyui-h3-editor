"""Needs torch, gguf and ComfyUI on the path. Run inside the container with PYTHONPATH set to ComfyUI. See README."""

import unittest

try:
    import gguf
    import numpy as np
    import torch

    import ggml_dequant
    import ggml_ops
    import hyperflow
except ImportError:
    hyperflow = None

# Every type the torch dequantizer knows, checked against gguf's own numpy reference.
TYPES = ("BF16", "Q8_0", "Q5_1", "Q5_0", "Q4_1", "Q4_0", "Q6_K", "Q5_K", "Q4_K", "Q3_K", "Q2_K", "IQ4_NL", "IQ4_XS")


@unittest.skipIf(hyperflow is None, "needs torch, gguf and ComfyUI")
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


@unittest.skipIf(hyperflow is None, "needs torch, gguf and ComfyUI")
class HyperFlowTest(unittest.TestCase):
    def test_the_step_schedules_run_from_one_to_zero(self):
        for steps in (4, 6, 8):
            sigmas = hyperflow.step_sigmas(steps)
            self.assertEqual(sigmas.shape[0], steps + 1)
            self.assertEqual((float(sigmas[0]), float(sigmas[-1])), (1.0, 0.0))
            self.assertTrue(bool((sigmas[1:] < sigmas[:-1]).all()))

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
        plan = hyperflow.FactorizedPlan(state)
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


if __name__ == "__main__":
    unittest.main()
