"""Needs numpy. No ffmpeg: the waves are made here."""

import unittest

try:
    import numpy as np
    import cuts
    import listen
except ImportError:
    np = None


def tone(seconds, hz=440.0, gain=0.1):
    t = np.arange(int(seconds * listen.RATE)) / listen.RATE
    return gain * np.sin(2 * np.pi * hz * t)


def noise(seconds, gain=0.05, seed=1):
    return gain * np.random.default_rng(seed).standard_normal(int(seconds * listen.RATE))


@unittest.skipIf(np is None, "needs numpy")
class LoudnessTest(unittest.TestCase):
    def test_a_full_scale_tone_reads_what_the_standard_says(self):
        # ITU-R BS.1770: a 997 Hz sine at full scale in one channel is -3.01 LKFS.
        self.assertAlmostEqual(listen.loudness(tone(2, 997.0, 1.0)), -3.01, places=1)

    def test_ten_times_quieter_is_twenty_lu_less(self):
        self.assertAlmostEqual(listen.loudness(tone(2, 997.0, 1.0)) - listen.loudness(tone(2, 997.0, 0.1)), 20.0, places=1)

    def test_a_low_rumble_counts_for_less_than_a_voice(self):
        self.assertLess(listen.loudness(tone(2, 40.0)), listen.loudness(tone(2, 1000.0)) - 5)

    def test_no_sound_is_far_below_silence(self):
        self.assertLess(listen.loudness(np.zeros(listen.RATE)), listen.SILENCE)
        self.assertLess(listen.loudness(np.zeros(0)), listen.SILENCE)


@unittest.skipIf(np is None, "needs numpy")
class CutTest(unittest.TestCase):
    def judged(self, before, after):
        return listen.judge(listen.measure(before, after))

    def test_a_sound_that_carries_on_is_continuous(self):
        wave = noise(4)
        found = self.judged(wave[:2 * listen.RATE], wave[2 * listen.RATE:])
        self.assertEqual(found["kind"], "continuous")
        self.assertGreaterEqual(found["score"], 90)

    def test_a_swell_chopped_off_is_a_level_jump(self):
        found = self.judged(noise(2, 0.3), noise(2, 0.005, seed=2))
        self.assertEqual(found["kind"], "level jump")
        self.assertEqual(found["score"], 0)
        self.assertIn("36 dB quieter after the cut", found["notes"])

    def test_a_small_step_costs_a_little(self):
        found = self.judged(noise(2, 0.05), noise(2, 0.1, seed=2))
        self.assertEqual(found["kind"], "level jump")
        self.assertTrue(60 < found["score"] < 90, found)

    def test_a_sound_that_moves_by_itself_may_step_as_much_at_a_cut(self):
        # A level that already swings 8 LU from one stretch to the next: a 6 LU step at the cut is nothing new.
        size = int(listen.SIDE * listen.RATE)
        swinging = np.concatenate([noise(listen.SIDE, 0.02 if n % 2 else 0.05, seed=n)[:size] for n in range(5)])
        steady = self.judged(noise(2, 0.05), noise(2, 0.1, seed=2))
        moving = self.judged(swinging, np.concatenate([noise(listen.SIDE, 0.1, seed=9)[:size], swinging]))
        self.assertGreater(moving["score"], steady["score"])

    def test_another_sound_at_the_same_level_is_a_new_sound(self):
        low, high = tone(2, 120.0, 0.3), tone(2, 3000.0, 0.02)
        m = listen.measure(low, high * 10 ** ((listen.loudness(low) - listen.loudness(high)) / 20))
        found = listen.judge(m)
        self.assertLess(abs(m["step"]), 1.0)
        self.assertEqual(found["kind"], "new sound")
        self.assertGreaterEqual(found["score"], 80)

    def test_sound_into_silence_is_a_dropout(self):
        found = self.judged(noise(2), np.zeros(2 * listen.RATE))
        self.assertEqual((found["kind"], found["notes"][0]), ("dropout", "the sound stops at the cut"))
        self.assertLess(found["score"], 30)

    def test_two_silences_are_fine(self):
        self.assertEqual(self.judged(np.zeros(listen.RATE), np.zeros(listen.RATE))["kind"], "silent")

    def test_a_step_in_the_wave_is_a_click(self):
        before, after = tone(2, 220.0), tone(2, 220.0) + 0.5
        found = self.judged(before, after)
        self.assertIn("the wave steps at the cut: a click", found["notes"])


@unittest.skipIf(np is None, "needs numpy")
class InsideTest(unittest.TestCase):
    def test_a_steady_take_has_nothing_to_report(self):
        self.assertEqual(listen.find_inside(noise(5)), [])

    def test_a_sound_that_starts_at_once_is_ordinary(self):
        self.assertEqual(listen.find_inside(np.concatenate([noise(3, 0.01), noise(2, 0.08, seed=2)])), [])

    def test_a_sound_cut_off_is_found_where_it_is(self):
        rows = listen.find_inside(np.concatenate([noise(3, 0.2), noise(2, 0.004, seed=2)]))
        self.assertEqual([r["kind"] for r in rows], ["chop", "fade"])
        self.assertTrue(abs(rows[0]["at"] - 72) <= 2, rows)
        self.assertIn("cut off", rows[0]["notes"][0])

    def test_a_sound_that_dies_away_is_not_a_chop(self):
        decay = np.concatenate([np.ones(2 * listen.RATE), np.exp(-np.arange(3 * listen.RATE) / (0.25 * listen.RATE))])
        rows = listen.find_inside(noise(5, 0.2) * decay)
        self.assertEqual([r["kind"] for r in rows if r["kind"] != "fade"], [])

    def test_a_pause_between_two_words_is_not_a_chop(self):
        word = noise(0.6, 0.2)
        speech = np.concatenate([word, noise(0.15, 0.002, seed=2), word, noise(0.15, 0.002, seed=3), word, noise(1.0, 0.2, seed=4)])
        self.assertEqual(listen.find_inside(speech), [])

    def test_a_hole_in_the_sound_is_a_dropout(self):
        rows = listen.find_inside(np.concatenate([noise(2), np.zeros(listen.RATE), noise(2, seed=2)]))
        self.assertEqual(rows[0]["kind"], "dropout")

    def test_a_sound_that_runs_out_before_the_take_ends_is_said_so(self):
        # A song that dies away and then nothing: the track ran out before the picture did.
        decay = np.exp(-np.arange(int(0.6 * listen.RATE)) / (0.15 * listen.RATE))
        wave = np.concatenate([noise(3, 0.2), noise(0.6, 0.2, seed=2) * decay, np.zeros(int(1.2 * listen.RATE))])
        rows = listen.find_inside(wave)
        self.assertEqual([r["kind"] for r in rows], ["ends silent"])
        self.assertTrue(abs(rows[0]["at"] - 3.6 * 24) <= 12, rows)
        self.assertRegex(rows[0]["notes"][0], r"the sound stops 1\.[0-9] s before the take ends")

    def test_a_short_silence_at_the_very_end_is_left_alone(self):
        self.assertEqual(listen.find_inside(np.concatenate([noise(4), np.zeros(int(0.2 * listen.RATE))])), [])

    def test_a_take_that_ends_far_louder_than_it_began_is_a_swell(self):
        ramp = np.linspace(0.003, 0.3, 6 * listen.RATE)
        rows = listen.find_inside(np.random.default_rng(3).standard_normal(6 * listen.RATE) * ramp)
        self.assertEqual([r["kind"] for r in rows], ["swell"])

    def test_a_silent_take_is_left_alone(self):
        self.assertEqual(listen.find_inside(np.zeros(5 * listen.RATE)), [])


@unittest.skipIf(np is None, "needs numpy")
class ArrivalTest(unittest.TestCase):
    def setUp(self):
        self.pin = np.random.default_rng(4).random((cuts.H, cuts.W, 3)) * 0.5 + 0.2
        self.clean = {"kind": "continuous", "score": 96, "notes": []}

    def test_a_take_that_lands_on_its_pin_arrives(self):
        found = cuts.judge_arrival(self.pin * 0.97, self.pin, self.clean)
        self.assertEqual((found["kind"], found["score"], found["notes"]), ("arrives", 96, []))

    def test_a_take_still_fading_up_is_short_of_its_pin(self):
        found = cuts.judge_arrival(self.pin * 0.76, self.pin, self.clean)
        self.assertEqual(found["kind"], "short of its pin")
        self.assertIn("ends 24% darker than the frame it is pinned to", found["notes"])
        self.assertLess(found["score"], 60)

    def test_a_jump_into_the_next_clip_is_not_an_arrival(self):
        found = cuts.judge_arrival(self.pin, self.pin, {"kind": "jump", "score": 71, "notes": ["same shot, the subject pops"]})
        self.assertEqual((found["kind"], found["score"]), ("short of its pin", 71))
        self.assertIn("same shot, the subject pops", found["notes"][0])


if __name__ == "__main__":
    unittest.main()
