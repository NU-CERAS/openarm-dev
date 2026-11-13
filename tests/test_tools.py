"""Behavior checks: actual MuJoCo, CLI persistence and hardware command boundaries."""
import contextlib
import io
import json
import math
import tempfile
import time
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
                sim = Simulation({side: cfg}, self.path / 'unused.json')
                self.assertEqual(len(sim.read(side)), 7)
                sim.step({side: sim.read(side)})
                self.assertTrue(all(math.isfinite(x) for x in sim.read(side)))

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

    def test_arm_selection_required_and_invalid_moves_rejected(self):
        self.command('home', expected=1)
        self.command('jog', '1', '100', '--sim', 'can0', expected=1)
        self.assertFalse(list(self.path.rglob('state.json')))

    def test_sim_zero_pose_and_motion_roundtrip(self):
        self.command('sim', '--headless', '--seconds', '.02')
        self.command('status', '--sim', 'can0', 'can1')
        self.command('zero', '--sim', 'can0')
        self.command('pose', 'save', 'start', '--sim', 'can0', 'can1')
        self.command('pose', 'save', 'start', '--sim', 'can0', expected=1)
        self.command('pose', 'move', 'start', '--sim', 'can0', 'can1')
        self.command('jog', 'openarm_left_joint1', '.02', '--sim', 'can0')
        self.command('jog', 'openarm_left_joint1', '.02', '--sim', 'can1', expected=1)
        self.command('jog', '2', '-.05', '--sim', 'can0', 'can1')
        self.command('home', '--sim', 'can0', 'can1', '--duration', '10')
        states = list(self.path.rglob('state.json'))
        self.assertEqual(len(states), 1)
        state = json.loads(states[0].read_text())
        self.assertTrue(all(math.isfinite(x) for x in state['qpos']))

    def test_config_changes_invalidate_calibration(self):
        path = self.path / 'zero.json'
        offsets.save(path, self.cfg, [j['home'] for j in self.cfg['joints']])
        changed = dict(self.cfg, _fingerprint='different')
        with self.assertRaises(ValueError):
            offsets.read(path, changed)

    def test_names_and_config_mapping(self):
        self.command('status', '--sim', 'can0', '--session', '../escape', expected=1)
        self.command('pose', 'save', '../escape', '--sim', 'can0', expected=1)
        right = 'robots/v1.0/config/right.json'
        self.command('status', '--sim', 'can0', '--config', f'can0={right}', expected=1)
        self.command('status', '--sim', 'can1', '--config', f'can1={right}')

    def fake_hardware(self, fail=None):
        class FakeHardware:
            moves = []
            def __init__(self, cfg, interface):
                self.cfg, self.interface = cfg, interface
            def read(self):
                return [j['home'] for j in self.cfg['joints']]
            def status(self):
                return []
            def move(self, points, stop=None, hold=0):
                if self.interface == fail:
                    raise RuntimeError('motor fault')
                for _ in range(200):
                    if stop.is_set():
                        raise RuntimeError('Stopped.')
                    time.sleep(0.001)
                FakeHardware.moves.append(self.interface)
            def close(self):
                pass
        return FakeHardware

    def test_sim_and_real_move_together(self):
        fake = self.fake_hardware()
        with patch('openarm_tools.backends.can.Hardware', fake):
            self.command('jog', '7', '.05', '--sim', 'can0', 'can1', '--real', 'can0', 'can1')
        self.assertEqual(sorted(fake.moves), ['can0', 'can1'])
        self.assertEqual(len(list(self.path.rglob('state.json'))), 1)

    def test_one_arm_failure_stops_the_other(self):
        fake = self.fake_hardware(fail='can0')
        with patch('openarm_tools.backends.can.Hardware', fake):
            self.command('jog', '7', '.05', '--real', 'can0', 'can1', expected=1)
        self.assertEqual(fake.moves, [])

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
        hw.component.get_motors.return_value = hw.motors
        for motor in hw.motors:
            motor.is_enabled.return_value = False
        hw.component.get_motors.return_value = hw.motors
        hw.enabled = False
        hw.oa = SimpleNamespace(MITParam=lambda *args: args)
        hw.read = lambda: [j['home'] for j in hw.cfg['joints']]
        hw.load_gravity_model()
        return hw

    def test_gravity_matches_model_and_is_signed_scaled_capped(self):
        hw = self.hardware()
        bent = [0, 0, 0, 1.57, 0, 0, 0]
        self.assertAlmostEqual(hw.gravity(bent)[3], 4.88, delta=0.3)
        self.assertLess(max(abs(x) for x in hw.gravity([0.0] * 7)), 0.2)
        hw.cfg['gravity_scale'] = 1.0
        self.assertAlmostEqual(hw.feedforward(bent)[3], hw.gravity(bent)[3])
        hw.cfg['joints'][3]['sign'] = -1
        self.assertAlmostEqual(hw.feedforward(bent)[3], -hw.gravity(bent)[3])
        hw.cfg['joints'][3]['tau_max'] = 1.0
        self.assertEqual(hw.feedforward(bent)[3], -1.0)
        hw.cfg['gravity_scale'] = 0.0
        self.assertEqual(hw.feedforward(bent), [0.0] * 7)

    def test_motor_command_carries_feedforward(self):
        hw = self.hardware()
        hw.cfg['gravity_scale'] = 1.0
        bent = [0, 0, 0, 1.57, 0, 0, 0]
        hw.command(bent, bent)
        params = hw.component.mit_control_all.call_args[0][0]
        self.assertAlmostEqual(params[3][4], hw.gravity(bent)[3])

    def test_hold_keeps_commanding_then_stops_cleanly(self):
        import threading
        hw = self.hardware()
        stop = threading.Event()
        threading.Timer(0.8, stop.set).start()
        started = time.monotonic()
        hw.move([(0, hw.read())], stop=stop, hold=math.inf)
        self.assertGreater(time.monotonic() - started, 0.7)
        self.assertGreater(hw.component.mit_control_all.call_count, 30)
        hw.arm.disable_all.assert_called()

    def test_partial_enable_failure_disables(self):
        hw = self.hardware()
        hw.arm.enable_all.side_effect = RuntimeError('CAN write failed')
        with self.assertRaises(RuntimeError):
            hw.move([(0, hw.read())])
        hw.arm.disable_all.assert_called_once()

    def test_disable_retries_until_motors_report_disabled(self):
        hw = self.hardware()
        hw.enabled = True
        hw.motors[2].is_enabled.side_effect = [True, True, False]
        hw.close()
        self.assertEqual(hw.arm.disable_all.call_count, 3)
        self.assertFalse(hw.enabled)

    def test_unconfirmed_disable_is_reported(self):
        hw = self.hardware()
        hw.enabled = True
        hw.motors[0].is_enabled.return_value = True
        with self.assertRaisesRegex(RuntimeError, 'emergency stop'):
            hw.close()

    def test_external_enabled_motor_refused(self):
        hw = self.hardware()
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
