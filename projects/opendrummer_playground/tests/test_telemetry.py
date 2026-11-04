"""Exercise telemetry against actual contact dynamics and display-only updates."""
import unittest

import mujoco
import numpy as np

from openarm_drummer.scene import build_scene
from openarm_drummer.strike import StrikeSession
from openarm_drummer.telemetry import DrumTelemetry


class TelemetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, robot, scene = build_scene()
        cls.data = mujoco.MjData(cls.model)
        cls.session = StrikeSession(cls.model, robot, scene, 2)
        cls.session.reset(cls.data)
        cls.wall_time = 0.
        cls.monitor = DrumTelemetry(cls.model, scene, clock=lambda: cls.wall_time)
        while not cls.session.finished:
            cls.session.step(cls.data)
            cls.monitor.sample(cls.data)

    def test_matches_physics_detector_and_isolates_drums(self):
        self.assertEqual(self.session.report()['status'], 'PASS')
        self.assertEqual(len(self.monitor.events), 2)
        self.assertEqual(set(self.monitor.latest), {'snare'})
        for measured, expected in zip(self.monitor.events, self.session.detector.events):
            for field in ('peak_force_N', 'normal_impulse_Ns', 'contact_duration_s'):
                self.assertAlmostEqual(measured[field], expected[field], places=10)
            self.assertEqual(measured['contact_parts'], ['left_stick_tip_collision'])

    def test_strike_and_rebound_command_only_joint7(self):
        held = np.r_[0:6, 7:14]
        rows = [row for row in self.session.rows
                if row['state'] in ('downstroke', 'contact', 'retract', 'recovery')]
        commands = np.array([row['commanded_joint_positions_rad'] for row in rows])
        np.testing.assert_allclose(commands[:, held],
            np.broadcast_to(self.session.impact_pose[held], commands[:, held].shape), atol=1e-12)
        self.assertGreater(np.ptp(commands[:, 6]), .3)
        self.assertLess(self.session.metrics['max_held_joint_deviation_rad'], .01)
        self.assertLess(self.session.metrics['max_hit_radial_offset_m'], .005)
        # The positioning phase may move the arm before targets are held.
        self.assertGreater(np.max(np.abs(self.session.staging[:6]-self.session.approach[:6])), .1)

    def test_frozen_frames_do_not_integrate_twice(self):
        before = [event.copy() for event in self.monitor.events]
        for _ in range(10):
            self.monitor.sample(self.data)
        self.assertEqual(before, self.monitor.events)

    def test_flash_fades_without_erasing_readout_or_changing_physics(self):
        type(self).wall_time = 0.
        state = self.data.qpos.copy()
        self.monitor.update_colors()
        site = self.monitor.flash_sites['snare']
        self.assertGreater(self.model.site_rgba[site, 3], 0)
        type(self).wall_time = 1.
        self.monitor.update_colors()
        self.assertEqual(self.model.site_rgba[site, 3], 0)
        self.assertEqual(len(self.monitor.events), 2)
        np.testing.assert_array_equal(state, self.data.qpos)
        self.assertEqual(self.monitor.draw_panel(330, 720).shape, (720, 330, 3))


if __name__ == '__main__':
    unittest.main()
