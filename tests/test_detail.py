import unittest

try:
    import numpy as np

    import detail
except ImportError:  # numpy ships with ComfyUI, not with a bare Python
    np = None


def sharp(seed=1, h=270, w=480):
    """A picture with detail down to the single pixel."""
    return np.random.default_rng(seed).random((h, w)).astype(np.float32)


def enlarged(y, n=3):
    """The same picture made n times smaller and smoothly enlarged back, as a draft is in a cut of finals."""
    h, w = (y.shape[0] // n) * n, (y.shape[1] // n) * n
    small = y[:h, :w].reshape(h // n, n, w // n, n).mean(axis=(1, 3))
    rows = np.linspace(0, small.shape[0] - 1, h)
    cols = np.linspace(0, small.shape[1] - 1, w)
    wide = np.stack([np.interp(cols, np.arange(small.shape[1]), row) for row in small])
    return np.stack([np.interp(rows, np.arange(small.shape[0]), wide[:, c]) for c in range(w)], axis=1).astype(np.float32)


@unittest.skipIf(np is None, "numpy is not installed")
class DetailTest(unittest.TestCase):
    def test_an_enlarged_picture_has_far_less_fine_detail(self):
        full = detail.fine(sharp())
        soft = detail.fine(enlarged(sharp()))
        self.assertGreater(full, 1.5 * soft)

    def test_contrast_does_not_change_the_measure(self):
        y = sharp()
        self.assertAlmostEqual(detail.fine(y), detail.fine(0.3 * y + 0.2), places=3)

    def test_a_flat_picture_is_not_measured(self):
        self.assertIsNone(detail.fine(np.full((90, 160), 0.4, dtype=np.float32)))

    def test_a_soft_stretch_is_a_drop(self):
        samples = [(k / 2.0, 0.45) for k in range(40)]
        for k in range(10, 16):
            samples[k] = (k / 2.0, 0.17)
        found = detail.drops(samples)
        self.assertEqual(len(found["drops"]), 1)
        self.assertEqual((found["drops"][0]["from"], found["drops"][0]["to"]), (5.0, 8.0))
        self.assertLess(found["drops"][0]["part"], 0.5)

    def test_one_soft_sample_is_not_a_drop(self):
        samples = [(k / 2.0, 0.45) for k in range(40)]
        samples[7] = (3.5, 0.1)
        self.assertEqual(detail.drops(samples)["drops"], [])

    def test_an_empty_picture_does_not_end_a_drop(self):
        samples = [(k / 2.0, 0.45) for k in range(40)]
        for k in range(10, 16):
            samples[k] = (k / 2.0, None if k == 12 else 0.17)
        self.assertEqual(len(detail.drops(samples)["drops"]), 1)

    def test_a_draft_among_finals_is_soft(self):
        found = detail.soft_clips([("a", 0.45, 120), ("b", 0.18, 120), ("c", 0.47, 120), ("d", None, 60)])
        self.assertEqual([row["soft"] for row in found["clips"]], [False, True, False, False])
        self.assertAlmostEqual(found["level"], 0.45)
        self.assertIsNone(found["clips"][3]["part"])

    def test_a_cut_of_drafts_has_no_soft_clip(self):
        found = detail.soft_clips([("a", 0.18, 120), ("b", 0.2, 120), ("c", 0.17, 120)])
        self.assertFalse(any(row["soft"] for row in found["clips"]))

    def test_the_level_follows_the_longer_clips(self):
        found = detail.soft_clips([("a", 0.2, 30), ("b", 0.5, 400), ("c", 0.2, 30)])
        self.assertAlmostEqual(found["level"], 0.5)
        self.assertEqual([row["soft"] for row in found["clips"]], [True, False, True])

    def test_a_file_that_is_soft_throughout_has_no_drops(self):
        found = detail.drops([(k / 2.0, 0.18) for k in range(40)])
        self.assertEqual(found["drops"], [])
        self.assertAlmostEqual(found["level"], 0.18)


if __name__ == "__main__":
    unittest.main()
