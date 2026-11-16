"""Needs numpy. The network's weights are not in the pack, so what is tested here runs without them: the sound's
features, the judgement of a measure, and that nothing is measured when the files are missing."""

import math
import os
import tempfile
import unittest

try:
    import numpy as np
    import lips
except ImportError:
    np = None


@unittest.skipIf(np is None, "needs numpy")
class FeaturesTest(unittest.TestCase):
    def signal(self, seconds=2.0):
        return np.random.default_rng(3).standard_normal(int(lips.RATE * seconds)) * 3000

    def test_ten_milliseconds_a_column_and_thirteen_cepstra(self):
        found = lips.mfcc(self.signal(2.0))
        # 400-sample windows every 160 samples, as python_speech_features cuts them.
        self.assertEqual(found.shape, (1 + int(math.ceil((2 * lips.RATE - 400) / 160.0)), 13))
        self.assertTrue(np.isfinite(found).all())

    def test_the_first_cepstrum_is_the_log_energy_of_each_window(self):
        signal = self.signal(1.0)
        emphasised = np.append(signal[:1], signal[1:] - 0.97 * signal[:-1])
        first = emphasised[:400]
        energy = (np.abs(np.fft.rfft(first, 512)) ** 2 / 512).sum()
        self.assertAlmostEqual(lips.mfcc(signal)[0, 0], math.log(energy), places=6)

    def test_the_same_sound_gives_the_same_numbers(self):
        self.assertTrue(np.array_equal(lips.mfcc(self.signal()), lips.mfcc(self.signal())))

    def test_a_louder_sound_only_moves_the_energy(self):
        quiet, loud = lips.mfcc(self.signal()), lips.mfcc(self.signal() * 4)
        self.assertTrue(np.allclose(quiet[:, 1:], loud[:, 1:], atol=1e-6))
        self.assertTrue(np.allclose(loud[:, 0] - quiet[:, 0], math.log(16), atol=1e-6))


@unittest.skipIf(np is None, "needs numpy")
class JudgeTest(unittest.TestCase):
    def test_lips_on_the_sound_are_in_sync(self):
        found = lips.judge({"offset": -1, "conf": 3.5})
        self.assertEqual((found["kind"], found["notes"]), ("in sync", []))
        self.assertGreaterEqual(found["score"], 90)

    def test_lips_before_the_sound_are_early_and_say_by_how_much(self):
        found = lips.judge({"offset": -8, "conf": 2.3})
        self.assertEqual(found["kind"], "early")
        self.assertIn("0.33 s before the sound", found["notes"][0])
        self.assertLess(found["score"], 50)

    def test_lips_after_the_sound_are_late(self):
        self.assertEqual(lips.judge({"offset": 6, "conf": 3.0})["kind"], "late")

    def test_no_clear_match_is_no_match_whatever_the_offset(self):
        for offset in (0, -14, 9):
            found = lips.judge({"offset": offset, "conf": 0.4})
            self.assertEqual(found["kind"], "no match")
            self.assertLess(found["score"], 30)
            self.assertIn("another seed", found["notes"][0])

    def test_a_surer_match_scores_higher(self):
        self.assertGreater(lips.judge({"offset": 0, "conf": 4.0})["score"], lips.judge({"offset": 0, "conf": 2.0})["score"])


@unittest.skipIf(np is None, "needs numpy")
class InstalledTest(unittest.TestCase):
    def test_without_its_files_nothing_is_measured(self):
        before = os.environ.get("H3_SYNCNET_DIR")
        with tempfile.TemporaryDirectory() as empty:
            os.environ["H3_SYNCNET_DIR"] = empty
            try:
                self.assertEqual(lips.missing(), [lips.MODEL_FILE, lips.FACE_FILE])
                self.assertIsNone(lips.measure(os.path.join(empty, "none.mp4"), 0, 100))
            finally:
                if before is None:
                    os.environ.pop("H3_SYNCNET_DIR", None)
                else:
                    os.environ["H3_SYNCNET_DIR"] = before


if __name__ == "__main__":
    unittest.main()
