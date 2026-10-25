"""Headless planning and fail-closed execution tests. No hardware is accessed."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np

import ik_move as ik


class PlanningTests(unittest.TestCase):
    def setUp(self):
        self.cfg = ik.load_mapping(ik.ROOT / 'examples/ik_mapping.example.json')
        self.model_path = ik.ROOT / 'models/bimanual.xml'
        self.planner = ik.Planner(self.model_path, self.cfg)
        self.start = {s: np.zeros(7) for s in ik.SIDES}

    def test_requested_right_lift_and_entire_path(self):
        self.planner.forward(self.start)
        origin = self.planner.pose('right')[0]
        goal, error, *_ = self.planner.solve(self.start, 'right', origin + [0, 0, .02])
        self.assertLess(error, .001)
        self.assertLessEqual(np.max(np.abs(goal)), .6)
        self.assertGreaterEqual(self.planner.validate_path(self.start, 'right', goal), 101)
        self.assertLess(np.linalg.norm(self.planner.pose('right')[0] - origin - [0, 0, .02]), .001)

    def test_left_move_and_fixed_orientation(self):
        self.planner.forward(self.start)
        origin, rotation = self.planner.pose('left')
        goal, error, *_ = self.planner.solve(self.start, 'left', origin + [0, .01, 0], keep_orientation=True)
        self.planner.validate_path(self.start, 'left', goal)
        self.assertLess(error, .001)
        self.assertLess(np.linalg.norm(ik.rotation_error(rotation, self.planner.pose('left')[1])), .035)

    def test_far_and_nonfinite_targets_rejected(self):
        self.planner.forward(self.start)
        origin = self.planner.pose('right')[0]
        for target in (origin + [1, 0, 0], [np.nan, 0, 0]):
            with self.assertRaises(ik.Refused):
                self.planner.solve(self.start, 'right', target)

    def test_joint_limits_and_large_path_rejected(self):
        bad = {s: self.start[s].copy() for s in ik.SIDES}
        bad['right'][1] = .1
        with self.assertRaises(ik.Refused):
            self.planner.check(bad)
        with self.assertRaises(ik.Refused):
            self.planner.validate_path(self.start, 'right', np.ones(7))

    def test_limit_error_names_both_joints_and_exact_excess(self):
        self.start['right'][1] = .00514992
        self.start['left'][1] = -.01659419
        with self.assertRaises(ik.Refused) as failure:
            self.planner.check(self.start)
        message = str(failure.exception)
        self.assertIn('right J2', message)
        self.assertIn('left J2', message)
        self.assertIn('0.295 deg', message)
        self.assertIn('0.951 deg', message)

    def test_real_collision_geometry_rejects_obstacle(self):
        self.planner.forward(self.start)
        xyz = self.planner.pose('right')[0]
        tree = ET.parse(self.model_path)
        for mesh in tree.getroot().findall('./asset/mesh'):
            mesh.set('file', str((self.model_path.parent / mesh.get('file')).resolve()))
        ET.SubElement(tree.getroot().find('worldbody'), 'geom',
                      name='test_obstacle', type='sphere', size='.04', pos=' '.join(map(str, xyz)))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'obstacle.xml'
            tree.write(path)
            blocked = ik.Planner(path, self.cfg)
            with self.assertRaisesRegex(ik.Refused, 'collision/clearance'):
                blocked.check(self.start)

    def test_quintic_velocity_and_acceleration_bounds(self):
        distance = .55366
        duration = max(3, 1.875 * distance / .08, np.sqrt(5.774 * distance / .15))
        t = np.linspace(0, duration, 10001)
        q = distance * ik.blend(t / duration)
        v = np.gradient(q, t)
        a = np.gradient(v, t)
        self.assertLessEqual(np.max(np.abs(v)), .080001)
        self.assertLessEqual(np.max(np.abs(a)), .150001)
        self.assertAlmostEqual(q[0], 0)
        self.assertAlmostEqual(q[-1], distance)

    def test_dry_run_does_not_construct_hardware(self):
        with patch.object(ik, 'MotorBus', side_effect=AssertionError('No CAN in offline mode')):
            with contextlib.redirect_stdout(io.StringIO()):
                ik.main(['--offline', '--arm', 'right', '--delta', '0', '0', '.02'])

    def test_unverified_mapping_rejected_before_hardware(self):
        with patch.object(ik, 'MotorBus', side_effect=AssertionError('Must refuse first')):
            with self.assertRaisesRegex(ik.Refused, 'verified'):
                ik.main(['--live', '--mapping', str(ik.ROOT / 'examples/ik_mapping.example.json')])


class FakeBus:
    def __init__(self, fail_after_enable=False):
        self.specials, self.commands = [], []
        self.fail_after_enable = fail_after_enable
        self.status = 0
        self.timeout = 1000

    def snapshot(self):
        if self.fail_after_enable and self.specials:
            raise OSError('simulated missing bus')
        a = np.zeros((8, 4))
        a[:7, 3] = self.status
        return a

    def register(self, motor, rid):
        return self.timeout if rid == 9 else 1

    def command(self, q):
        self.commands.append(q.copy())

    def special(self, motor, code):
        self.specials.append((motor, code))


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = ik.load_mapping(ik.ROOT / 'examples/ik_mapping.example.json')
        self.start = {s: np.zeros(7) for s in ik.SIDES}
        self.buses = {s: FakeBus() for s in ik.SIDES}
        self.planner = ik.Planner(ik.ROOT / 'models/bimanual.xml', self.cfg)

    def test_existing_controller_rejected_without_any_write(self):
        self.buses['right'].status = 1
        with self.assertRaisesRegex(ik.Refused, 'already enabled'):
            ik.execute(self.planner, self.start, 'right', np.zeros(7), 3, self.buses, self.cfg)
        self.assertEqual(self.buses['right'].specials, [])
        self.assertEqual(self.buses['right'].commands, [])

    def test_zero_watchdog_rejected_without_any_write(self):
        self.buses['right'].timeout = 0
        with self.assertRaisesRegex(ik.Refused, 'watchdog'):
            ik.execute(self.planner, self.start, 'right', np.zeros(7), 3, self.buses, self.cfg)
        self.assertEqual(self.buses['right'].specials, [])
        self.assertEqual(self.buses['right'].commands, [])

    def test_arm_moves_during_preflight_no_enable(self):
        original = self.buses['right'].snapshot
        calls = 0
        def moved():
            nonlocal calls
            calls += 1
            state = original()
            if calls >= 2:
                state[0, 0] = .02
            return state
        self.buses['right'].snapshot = moved
        with self.assertRaisesRegex(ik.Refused, 'moved after planning'):
            ik.execute(self.planner, self.start, 'right', np.zeros(7), 3, self.buses, self.cfg)
        self.assertEqual(self.buses['right'].specials, [])
        self.assertEqual(self.buses['right'].commands, [])

    def test_link_failure_disables_only_selected_joints(self):
        self.buses['right'].fail_after_enable = True
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(OSError, 'missing bus'):
                ik.execute(self.planner, self.start, 'right', np.zeros(7), 3, self.buses, self.cfg)
        self.assertEqual(self.buses['right'].specials,
                         [(i, 0xFC) for i in range(1, 8)] + [(i, 0xFD) for i in range(1, 8)])
        self.assertEqual(self.buses['left'].specials, [])
        self.assertEqual(self.buses['left'].commands, [])

    def test_command_roundtrip_and_invalid_inputs(self):
        d = ik.mit_packet(.12, 20, 1, [12.5, 45, 54])
        self.assertAlmostEqual(((d[0] << 8 | d[1]) / 65535 * 2 - 1) * 12.5, .12, delta=.0004)
        self.assertAlmostEqual((d[3] & 15) * 256 + d[4], int(20 / 500 * 4095))
        for q in (np.nan, 20):
            with self.assertRaises(ik.Refused):
                ik.mit_packet(q, 20, 1, [12.5, 45, 54])

    def test_tracking_diagnostic_identifies_actual_failing_joint(self):
        actual, target, sample = np.zeros(7), np.zeros(7), np.zeros((8, 4))
        target[3] = .09
        sample[3, 2] = .123
        detail = ik.tracking_detail('right', actual, target, sample)
        self.assertIn('right J4', detail)
        self.assertIn('-0.09000', detail)
        self.assertIn('target +0.09000', detail)
        self.assertIn('torque +0.123', detail)


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.cfg = ik.load_mapping(ik.ROOT / 'examples/ik_mapping.example.json')
        self.history = {s: np.zeros((20, 8, 4)) for s in ik.SIDES}
        self.history['right'][:, 1, 0] = .00514992
        self.history['left'][:, 1, 0] = -.01659419

    def test_confirmed_reference_offsets_fix_boundary_without_widening(self):
        offsets, _ = ik.reference_offsets(self.history, self.cfg)
        for side in ik.SIDES:
            self.cfg[side]['offset_rad'] = np.array(offsets[side])
        mapped = ik.model_angles({s: self.history[s][0] for s in ik.SIDES}, self.cfg)
        planner = ik.Planner(ik.ROOT / 'models/bimanual.xml', self.cfg)
        planner.check(mapped)
        for side in ik.SIDES:
            np.testing.assert_allclose(mapped[side], np.zeros(7), atol=1e-12)
        self.assertEqual(planner.limits['right'][1, 1], 0)
        self.assertEqual(planner.limits['left'][1, 0], 0)

    def test_reject_enabled_moving_and_large_offset(self):
        for kind in ('enabled', 'moving', 'large', 'nan'):
            history = {s: self.history[s].copy() for s in ik.SIDES}
            if kind == 'enabled': history['left'][5, 0, 3] = 1
            if kind == 'moving': history['left'][5, 0, 0] = .01
            if kind == 'large': history['left'][:, 0, 0] = .2
            if kind == 'nan': history['left'][0, 0, 0] = np.nan
            with self.assertRaises(ik.Refused):
                ik.reference_offsets(history, self.cfg)

    def test_capture_preserves_original_and_never_sends_commands(self):
        buses = {s: FakeBus() for s in ik.SIDES}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'mapping.json'
            original = (ik.ROOT / 'examples/ik_mapping.example.json').read_text()
            path.write_text(original)
            with patch.object(ik.time, 'sleep'), contextlib.redirect_stdout(io.StringIO()):
                ik.capture_reference_zero(buses, self.cfg, path)
            backups = list(path.parent.glob('mapping.json.before-reference-*'))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_text(), original)
            self.assertFalse(ik.load_mapping(path)['verified'])
        for bus in buses.values():
            self.assertEqual(bus.specials, [])
            self.assertEqual(bus.commands, [])


if __name__ == '__main__':
    unittest.main()
