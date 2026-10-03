"""Lab 06 exercise. Run: python3 labs/exercises/forward_kinematics_starter.py"""
import numpy as np
from pinocchio_teaching_examples.common import load_arm, hand_position


def your_forward_kinematics(q, length1=0.6, length2=0.4):
    shoulder, elbow = q
    # TODO: return [x, 0, z]. Each link contributes length*cos(angle) to X.
    # Positive rotation about Y contributes -length*sin(angle) to Z.
    # Link 2's angle relative to the base is shoulder + elbow.
    raise NotImplementedError('Fill in the two-link forward kinematics formula.')


if __name__ == '__main__':
    model, data, frame_id = load_arm()
    for q in [np.array([0.0, 0.0]), np.array([-np.pi / 2, 0.0]), np.array([0.4, -0.7])]:
        expected = hand_position(model, data, frame_id, q)
        actual = your_forward_kinematics(q)
        np.testing.assert_allclose(actual, expected, atol=1e-9)
        print(f'q={q}: your position={actual}; matches Pinocchio')
