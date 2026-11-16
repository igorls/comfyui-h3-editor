"""Needs numpy and OpenCV. MODNet's file is not in the pack, so what is tested here runs without it: the comparison of
two backgrounds with the person's masks given, the judgement, and that nothing is measured when the file is missing."""

import os
import tempfile
import unittest

try:
    import cv2
    import numpy as np
    import scene
except ImportError:
    np = None


def room(seed=5):
    """A portrait picture of a made-up room: blocks of colour at the scale of furniture, a little grain."""
    rng = np.random.default_rng(seed)
    picture = np.full((scene.H, scene.W, 3), 40, np.uint8)
    for _ in range(40):
        x, y = int(rng.integers(0, scene.W - 30)), int(rng.integers(0, scene.H - 30))
        w, h = int(rng.integers(12, 60)), int(rng.integers(12, 60))
        picture[y:y + h, x:x + w] = rng.integers(0, 255, 3)
    return np.clip(picture + rng.normal(0, 2, picture.shape), 0, 255).astype(np.uint8)


def nobody():
    return np.zeros((scene.H, scene.W), bool)


def someone(left=60):
    """A person standing in the middle of the frame, as a mask."""
    mask = nobody()
    mask[90:, left:left + 100] = True
    return mask


@unittest.skipIf(np is None, "needs numpy and OpenCV")
class ChangeTest(unittest.TestCase):
    def test_the_same_room_has_not_changed(self):
        self.assertEqual(scene.change(room(), nobody(), room(), nobody()), 0.0)

    def test_another_room_has_changed(self):
        self.assertGreater(scene.change(room(5), nobody(), room(6), nobody()), 0.5)

    def test_a_person_moving_across_the_room_is_not_a_change(self):
        a, b = room(), room()
        a[someone(40)] = (200, 160, 140)
        b[someone(80)] = (200, 160, 140)
        self.assertEqual(scene.change(a, someone(40), b, someone(80)), 0.0)

    def test_a_camera_that_pans_is_followed(self):
        wide = cv2.resize(room(), (scene.W + 40, scene.H), interpolation=cv2.INTER_LINEAR)
        a, b = wide[:, :scene.W], wide[:, 12:12 + scene.W]
        self.assertLessEqual(scene.change(a.copy(), nobody(), b.copy(), nobody()), 0.1)

    def test_part_of_the_room_turning_into_something_else_is_a_change(self):
        a, b = room(), room()
        b[:scene.H // 2] = room(9)[:scene.H // 2]
        found = scene.change(a, nobody(), b, nobody())
        self.assertGreater(found, 0.25)
        self.assertLess(found, 0.75)

    def test_too_little_background_is_not_judged(self):
        everyone = np.ones((scene.H, scene.W), bool)
        self.assertIsNone(scene.change(room(), everyone, room(6), everyone))


@unittest.skipIf(np is None, "needs numpy and OpenCV")
class FramingTest(unittest.TestCase):
    def test_frames_a_push_has_taken_far_apart_are_not_compared(self):
        zoom = {0: 1.0, 24: 1.05, 48: 1.2, 72: 2.0}
        self.assertTrue(scene._framed_alike(zoom, 0, 24))
        self.assertFalse(scene._framed_alike(zoom, 0, 48))
        self.assertFalse(scene._framed_alike(zoom, 72, 0))

    def test_an_unknown_zoom_does_not_stop_a_comparison(self):
        self.assertTrue(scene._framed_alike({}, 0, 48))


@unittest.skipIf(np is None, "needs numpy and OpenCV")
class JudgeTest(unittest.TestCase):
    def test_a_background_that_holds_holds(self):
        found = scene.judge([{"at": 48, "near": 0.0, "start": 0.0}, {"at": 72, "near": 0.1, "start": 0.0}])
        self.assertEqual(found["kind"], "holds")
        self.assertEqual(found["notes"], [])
        self.assertGreaterEqual(found["score"], 80)

    def test_the_worst_second_decides_and_says_where(self):
        found = scene.judge([{"at": 48, "near": 0.0, "start": 0.0}, {"at": 96, "near": 0.1, "start": 0.44},
                             {"at": 120, "near": 0.3, "start": None}])
        self.assertEqual(found["kind"], "changes")
        self.assertEqual((found["share"], found["frame"]), (0.44, 96))
        self.assertIn("44%", found["notes"][0])

    def test_more_change_scores_lower(self):
        some = scene.judge([{"at": 48, "near": 0.25, "start": None}])["score"]
        most = scene.judge([{"at": 48, "near": 0.9, "start": None}])["score"]
        self.assertLess(most, some)

    def test_nothing_seen_is_no_judgement(self):
        self.assertIsNone(scene.judge([{"at": 48, "near": None, "start": None}]))
        self.assertIsNone(scene.judge([]))


@unittest.skipIf(np is None, "needs numpy and OpenCV")
class UnreadableTest(unittest.TestCase):
    def setUp(self):
        self.before = (os.environ.get("H3_MATTING_DIR"), dict(scene._NET))
        scene._NET.clear()

    def tearDown(self):
        before, net = self.before
        if before is None:
            os.environ.pop("H3_MATTING_DIR", None)
        else:
            os.environ["H3_MATTING_DIR"] = before
        scene._NET.clear()
        scene._NET.update(net)

    def test_a_file_nothing_can_read_turns_the_check_off_and_says_why(self):
        import builtins
        real = builtins.__import__

        def no_onnxruntime(name, *args, **kwargs):
            if name == "onnxruntime":
                raise ImportError("not installed")
            return real(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as folder:
            os.environ["H3_MATTING_DIR"] = folder
            with open(os.path.join(folder, scene.MODEL_FILE), "wb") as f:
                f.write(b"not a network")
            builtins.__import__ = no_onnxruntime
            try:
                with self.assertRaises(RuntimeError):
                    scene.network()
            finally:
                builtins.__import__ = real
            self.assertIn("cannot read", scene.broken())
            with self.assertRaises(RuntimeError):     # and stays off, without trying again
                scene.person(room())

    def test_nothing_is_wrong_before_anything_is_loaded(self):
        self.assertIsNone(scene.broken())


@unittest.skipIf(np is None, "needs numpy and OpenCV")
class InstalledTest(unittest.TestCase):
    def test_without_its_file_nothing_is_measured(self):
        before = os.environ.get("H3_MATTING_DIR")
        with tempfile.TemporaryDirectory() as empty:
            os.environ["H3_MATTING_DIR"] = empty
            try:
                self.assertEqual(scene.missing(), [scene.MODEL_FILE])
                self.assertIsNone(scene.measure(os.path.join(empty, "none.mp4"), 0, 200))
            finally:
                if before is None:
                    os.environ.pop("H3_MATTING_DIR", None)
                else:
                    os.environ["H3_MATTING_DIR"] = before


if __name__ == "__main__":
    unittest.main()
