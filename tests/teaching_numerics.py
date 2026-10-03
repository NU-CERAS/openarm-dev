"""Independent checks for the default teaching arm. Run inside the container."""
import unittest
import numpy as np
import pinocchio as pin
from pinocchio_teaching_examples.common import load_arm, hand_position
from pinocchio_teaching_examples.demos import solve_ik


class TeachingNumerics(unittest.TestCase):
    def setUp(self):
        self.model, self.data, self.frame_id = load_arm()

    def test_forward_kinematics_against_trigonometry(self):
        for q in ([0, 0], [-np.pi / 2, 0], [0.4, -0.7], [-1.2, 0.8]):
            shoulder, elbow = q
            expected = [0.6 * np.cos(shoulder) + 0.4 * np.cos(shoulder + elbow),
                        0, -0.6 * np.sin(shoulder) - 0.4 * np.sin(shoulder + elbow)]
            np.testing.assert_allclose(
                hand_position(self.model, self.data, self.frame_id, np.array(q)),
                expected, atol=1e-12)

    def test_jacobian_against_finite_difference(self):
        for angles in ([0.4, -0.7], [0, 0], [-1.2, 0.8]):
            q = np.array(angles, dtype=float)
            J = pin.computeFrameJacobian(
                self.model, self.data, q, self.frame_id, pin.LOCAL_WORLD_ALIGNED)[:3, :]
            numeric = np.zeros((3, 2))
            for axis in range(2):
                step = np.zeros(2)
                step[axis] = 1e-6
                numeric[:, axis] = (
                    hand_position(self.model, self.data, self.frame_id, q + step)
                    - hand_position(self.model, self.data, self.frame_id, q - step)
                ) / 2e-6
            np.testing.assert_allclose(J, numeric, atol=1e-9)

    def test_gravity_against_mass_times_lever_arm(self):
        for angles in ([0, 0], [-np.pi / 2, 0], [0.4, -0.7]):
            q = np.array(angles, dtype=float)
            expected = -9.81 * np.array([
                (1.0 * 0.3 + 0.6 * 0.6) * np.cos(q[0])
                + 0.6 * 0.2 * np.cos(q.sum()),
                0.6 * 0.2 * np.cos(q.sum()),
            ])
            np.testing.assert_allclose(
                pin.computeGeneralizedGravity(self.model, self.data, q), expected, atol=1e-10)

    def test_ik_reaches_targets_and_obeys_limits(self):
        for target in ([0.7, 0, -0.35], [0.6, 0, 0.2], [0.4, 0, -0.5]):
            target = np.array(target)
            q, converged, _ = solve_ik(self.model, self.data, self.frame_id, target)
            self.assertTrue(converged)
            self.assertTrue(np.all(q >= self.model.lowerPositionLimit))
            self.assertTrue(np.all(q <= self.model.upperPositionLimit))
            self.assertLess(np.linalg.norm(
                hand_position(self.model, self.data, self.frame_id, q) - target), 1e-5)

    def test_ik_reports_failure(self):
        _, converged, _ = solve_ik(
            self.model, self.data, self.frame_id, np.array([2, 0, 0]), max_iterations=20)
        self.assertFalse(converged)


if __name__ == '__main__':
    unittest.main(verbosity=2)
