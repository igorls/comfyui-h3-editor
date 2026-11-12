import unittest

try:
    import numpy as np

    import cuts
except ImportError:  # numpy ships with ComfyUI, not with a bare Python
    np = None


def picture(seed):
    """A soft random picture at the scorer's size."""
    coarse = np.random.default_rng(seed).random((9, 16, 3))
    return np.kron(coarse, np.ones((6, 6, 1)))


def shot(seed, count):
    """Frames of one shot with a little change from each to the next."""
    base = picture(seed)
    return [np.clip(base + 0.002 * n, 0, 1) for n in range(count)]


@unittest.skipIf(np is None, "numpy is not installed")
class ScanTest(unittest.TestCase):
    def test_a_cut_between_two_shots_is_found_where_it_is(self):
        found = cuts.find_cuts(np.stack(shot(1, 20) + shot(2, 20)).astype(np.float32))
        self.assertEqual([(c["at"], c["kind"]) for c in found], [(20, "new shot")])

    def test_one_shot_has_no_cuts(self):
        self.assertEqual(cuts.find_cuts(np.stack(shot(1, 40)).astype(np.float32)), [])

    def test_two_frames_of_another_picture_are_a_stray_frame(self):
        found = cuts.find_cuts(np.stack(shot(1, 20) + shot(2, 2) + shot(3, 20)).astype(np.float32))
        self.assertEqual(len(found), 1)
        self.assertEqual((found[0]["at"], found[0]["kind"], found[0]["score"], found[0]["frames"]), (20, "stray frame", 0, 2))

    def test_a_fade_to_black_is_not_a_cut(self):
        frames = shot(1, 20) + [picture(1) * (1 - n / 10.0) for n in range(11)] + [picture(2) * (n / 10.0) for n in range(11)] + shot(2, 20)
        self.assertEqual(cuts.find_cuts(np.stack(frames).astype(np.float32)), [])

    def test_a_take_that_fades_out_is_reported_where_the_fade_starts(self):
        frames = shot(1, 30) + [picture(1) * (1 - n / 10.0) for n in range(11)] + [picture(2) * (n / 10.0) for n in range(11)] + shot(2, 20)
        found = cuts.find_inside(np.stack(frames).astype(np.float32))
        self.assertEqual(found[0]["kind"], "fade")
        self.assertTrue(26 <= found[0]["at"] <= 34, found[0]["at"])
        self.assertEqual(found[0]["score"], 0)

    def test_a_take_that_cuts_to_another_picture_scores_nothing(self):
        found = cuts.find_inside(np.stack(shot(1, 30) + shot(2, 30)).astype(np.float32))
        self.assertEqual([(row["at"], row["kind"], row["score"]) for row in found], [(30, "cut", 0)])

    def test_a_small_pop_inside_a_take_is_not_reported(self):
        base = picture(1)
        moved = base.copy()
        moved[18:30, 36:54] += 0.08                      # a gesture: a small part of the frame changes a little
        frames = shot(1, 20) + [np.clip(moved + 0.002 * n, 0, 1) for n in range(20)]
        self.assertEqual(cuts.find_inside(np.stack(frames).astype(np.float32)), [])

    def test_one_shot_is_clean_inside(self):
        self.assertEqual(cuts.find_inside(np.stack(shot(1, 60)).astype(np.float32)), [])


if __name__ == "__main__":
    unittest.main()
