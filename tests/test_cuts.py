import unittest

try:
    import numpy as np

    import cuts
except ImportError:  # numpy ships with ComfyUI, not with a bare Python
    np = None


def picture(seed, shift=0):
    """A soft random picture, optionally moved sideways."""
    rng = np.random.default_rng(seed)
    coarse = rng.random((9, 16, 3))
    full = np.kron(coarse, np.ones((6, 6, 1)))
    return np.roll(full, shift, axis=1)


def drift(base, count, start=0):
    """Frames of one shot with a little change from each to the next."""
    return np.stack([np.clip(base + 0.004 * (start + n), 0, 1) for n in range(count)])


@unittest.skipIf(np is None, "numpy is not installed")
class CutScoreTest(unittest.TestCase):
    def judged(self, before, after):
        return cuts.judge(cuts.measure(before, after))

    def test_a_shot_that_carries_on_is_continuous(self):
        base = picture(1)
        result = self.judged(drift(base, 5), drift(base, 5, start=5))
        self.assertEqual(result["kind"], "continuous")
        self.assertGreaterEqual(result["score"], 90)

    def test_another_picture_is_a_new_shot(self):
        result = self.judged(drift(picture(1), 5), drift(picture(2), 5))
        self.assertEqual(result["kind"], "new shot")
        self.assertGreaterEqual(result["score"], 70)

    def test_the_same_shot_with_a_part_that_pops_is_a_jump(self):
        base = picture(1)
        popped = base.copy()
        popped[18:36, 36:60] = 1 - popped[18:36, 36:60]      # the subject changes, the room does not
        result = self.judged(drift(base, 5), drift(popped, 5))
        self.assertEqual(result["kind"], "jump")
        self.assertLess(result["score"], 40)

    def test_two_stills_of_the_same_picture_are_continuous(self):
        base = picture(3)
        result = self.judged(np.stack([base] * 5), np.stack([base] * 5))
        self.assertEqual(result["kind"], "continuous")
        self.assertIn("still on both sides", result["notes"])

    def test_the_same_frames_always_give_the_same_numbers(self):
        a, b = drift(picture(4), 5), drift(picture(5), 5)
        self.assertEqual(cuts.measure(a, b), cuts.measure(a, b))


if __name__ == "__main__":
    unittest.main()
