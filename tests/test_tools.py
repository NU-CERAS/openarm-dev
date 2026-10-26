"""Behavior checks: actual MuJoCo, CLI persistence and hardware command boundaries."""
import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openarm_tools.cli import main
from openarm_tools.robot.config import load, validate_positions
from openarm_tools.motion.trajectory import trajectory
from openarm_tools.calibration import offsets
from openarm_tools.backends.mujoco import Simulation


class ToolsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.cfg = load()

    def command(self, *args, expected=0):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as out:
            rc = main([*args, '--data-dir', str(self.path)])
        self.assertEqual(rc, expected, out.getvalue())

    def test_real_models_and_names(self):
        for version in ('v0.3', 'v1.0'):
            for side in ('left', 'right'):
                cfg = load(robot=version, arm=side)
                sim = Simulation(cfg, self.path / 'unused.json')
                self.assertEqual(len(sim.read()), 7)
                sim.step(sim.read())
                self.assertTrue(all(math.isfinite(x) for x in sim.read()))

    def test_trajectory_speed_and_endpoints(self):
        start = [j['home'] for j in self.cfg['joints']]
        target = start.copy()
        target[0] += 0.1
        pts = list(trajectory(self.cfg, start, target, 1))
        self.assertEqual(pts[0][1], start)
        self.assertEqual(pts[-1][1], target)
        for (t1, q1), (t2, q2) in zip(pts, pts[1:]):
            self.assertLessEqual(abs(q2[0] - q1[0]) / (t2 - t1), self.cfg['max_speed'])
        with self.assertRaises(ValueError):
            list(trajectory(self.cfg, start, target, .1))
        for value in (math.nan, math.inf, 999):
            invalid = start.copy()
            invalid[0] = value
            with self.assertRaises(ValueError):
                validate_positions(self.cfg, invalid)

    def test_preview_does_not_write_state(self):
        self.command('home')
        self.assertFalse(list(self.path.rglob('state.json')))
        self.command('jog', 'openarm_left_joint1', '100', expected=1)
        self.assertFalse(list(self.path.rglob('state.json')))

    def test_sim_zero_pose_and_motion_roundtrip(self):
        self.command('sim', '--headless', '--seconds', '.02')
        self.command('status')
        self.command('zero', '--execute')
        self.command('pose', 'save', 'start')
        self.command('pose', 'save', 'start', expected=1)
        self.command('pose', 'move', 'start', '--execute')
        self.command('jog', 'openarm_left_joint1', '.02', '--execute')
        self.command('home', '--execute')
        state = json.loads(next(self.path.rglob('state.json')).read_text())
        self.assertTrue(all(math.isfinite(x) for x in state['qpos']))

    def test_config_changes_invalidate_calibration(self):
        path = self.path / 'zero.json'
        offsets.save(path, self.cfg, [j['home'] for j in self.cfg['joints']])
        changed = dict(self.cfg, _fingerprint='different')
        with self.assertRaises(ValueError):
            offsets.read(path, changed)

    def test_names_and_can_guard(self):
        self.command('status', '--session', '../escape', expected=1)
        self.command('pose', 'save', '../escape', expected=1)
        self.command('status', '--backend', 'can', expected=1)

    def test_can_preview_and_support_guard_never_move(self):
        class FakeHardware:
            moved = 0
            def __init__(self, cfg, interface):
                self.cfg = cfg
            def read(self):
                return [j['home'] for j in self.cfg['joints']]
            def move(self, points):
                FakeHardware.moved += 1
            def close(self):
                pass
        with patch('openarm_tools.backends.can.Hardware', FakeHardware):
            self.command('home', '--backend', 'can', '--interface', 'can0')
            self.command('home', '--backend', 'can', '--interface', 'can0', '--execute', expected=1)
        self.assertEqual(FakeHardware.moved, 0)

class HardwareBoundaryTests(unittest.TestCase):
    def hardware(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        from openarm_tools.backends.can import Hardware
        hw = Hardware.__new__(Hardware)
        hw.cfg = load()
        hw.arm = MagicMock()
        hw.component = MagicMock()
        hw.motors = [MagicMock() for _ in hw.cfg['joints']]
        for motor in hw.motors:
            motor.is_enabled.return_value = False
        hw.enabled = False
        hw.oa = SimpleNamespace(MITParam=lambda *args: args)
        hw.read = lambda: [j['home'] for j in hw.cfg['joints']]
        return hw

    def test_unreviewed_hardware_never_enables(self):
        hw = self.hardware()
        with self.assertRaises(ValueError):
            hw.move([(0, hw.read())])
        hw.arm.enable_all.assert_not_called()

    def test_partial_enable_failure_disables(self):
        hw = self.hardware()
        hw.cfg['hardware_reviewed'] = True
        hw.arm.enable_all.side_effect = RuntimeError('CAN write failed')
        with self.assertRaises(RuntimeError):
            hw.move([(0, hw.read())])
        hw.arm.disable_all.assert_called_once()

    def test_external_enabled_motor_refused(self):
        hw = self.hardware()
        hw.cfg['hardware_reviewed'] = True
        hw.motors[0].is_enabled.return_value = True
        with self.assertRaises(RuntimeError):
            hw.move([(0, hw.read())])
        hw.arm.enable_all.assert_not_called()

    def test_stale_feedback_refused(self):
        from openarm_tools.backends.can import Hardware
        hw = self.hardware()
        hw.component.get_link_stats.return_value.is_stale.return_value = True
        with self.assertRaisesRegex(RuntimeError, 'Stale feedback'):
            Hardware.read(hw)


if __name__ == '__main__':
    unittest.main()
