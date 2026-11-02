"""Hit-event semantics independent of the robot dynamics test."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from strike import HitDetector


class HitDetectionTests(unittest.TestCase):
    def setUp(self):
        self.detector = HitDetector(dict(hit_force_N=.5, hit_impulse_Ns=.002,
            release_force_N=.05, rearm_clearance_m=.005, rearm_seconds=.04))
        self.time = 0.

    def samples(self, force, gap, count):
        for _ in range(count):
            self.time += .001
            self.detector.update(self.time, force, gap, .001)

    def test_force_and_impulse_both_required(self):
        self.samples(.1, 0, 100)
        self.assertEqual(len(self.detector.events), 0)
        self.samples(0, .01, 50)
        self.samples(1., 0, 1)
        self.assertEqual(len(self.detector.events), 0)
        self.samples(1., 0, 1)
        self.assertEqual(len(self.detector.events), 1)

    def test_sustained_contact_and_chatter_count_once(self):
        self.samples(1., 0, 100)
        for _ in range(20):
            self.samples(0, .001, 2)
            self.samples(1., 0, 2)
        self.assertEqual(len(self.detector.events), 1)
        self.assertFalse(self.detector.armed)

    def test_requires_clearance_and_quiet_time_to_rearm(self):
        self.samples(1., 0, 10)
        self.samples(0, .01, 20)
        self.samples(1., 0, 10)
        self.assertEqual(len(self.detector.events), 1)
        self.samples(0, .001, 100)
        self.assertFalse(self.detector.armed)
        self.samples(0, .01, 50)
        self.assertTrue(self.detector.armed)
        self.samples(1., 0, 10)
        self.assertEqual(len(self.detector.events), 2)


if __name__ == '__main__':
    unittest.main()
