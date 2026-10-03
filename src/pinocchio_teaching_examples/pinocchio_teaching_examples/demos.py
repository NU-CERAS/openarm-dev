"""Small numerical demonstrations. Angles are radians; lengths are meters."""
import argparse
import numpy as np
import pinocchio as pin
from .common import load_arm, hand_position, analytical_position


def configuration_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--q', nargs=2, type=float, default=[0.4, -0.7],
                        metavar=('SHOULDER', 'ELBOW'), help='Joint angles in radians')
    args = parser.parse_args()
    q = np.array(args.q)
    if not np.all(np.isfinite(q)) or np.any(np.abs(q) > np.pi):
        parser.error('Joint angles must be finite and within [-pi, pi].')
    return q


def model_info():
    model, data, frame_id = load_arm()
    print(f'Robot: {model.name}; nq={model.nq}, nv={model.nv}')
    print('nq counts configuration coordinates; nv counts velocity coordinates.')
    for joint_id in range(1, model.njoints):
        print(f'Joint {joint_id}: {model.names[joint_id]}, q index={model.joints[joint_id].idx_q}')
    print('Frames:', ', '.join(frame.name for frame in model.frames))
    print('Neutral q:', pin.neutral(model))
    print('Neutral hand position [m]:', hand_position(model, data, frame_id, pin.neutral(model)))


def forward_kinematics():
    q = configuration_parser('Calculate the hand position from joint angles.')
    model, data, frame_id = load_arm()
    calculated = hand_position(model, data, frame_id, q)
    expected = analytical_position(q)
    print('q [rad]:', q)
    print('Pinocchio hand XYZ [m]:', calculated)
    print('Trigonometry XYZ [m]:  ', expected)
    print(f'Difference [m]: {np.linalg.norm(calculated - expected):.3e}')


def jacobian():
    q = configuration_parser('Predict a small hand displacement with the Jacobian.')
    model, data, frame_id = load_arm()
    before = hand_position(model, data, frame_id, q)
    # First 3 rows are linear motion at the hand origin, expressed in world axes.
    J = pin.computeFrameJacobian(model, data, q, frame_id, pin.LOCAL_WORLD_ALIGNED)[:3, :]
    dq = np.array([0.001, -0.002])
    after = hand_position(model, data, frame_id, pin.integrate(model, q, dq))
    print('Translational Jacobian [m/rad]:\n', J)
    print('Joint change [rad]:', dq)
    print('Predicted hand change [m]:', J @ dq)
    print('Actual hand change [m]:   ', after - before)
    print(f'Linear approximation error [m]: {np.linalg.norm(after - before - J @ dq):.3e}')


def gravity():
    model, data, frame_id = load_arm()
    print('Gravity acceleration [m/s^2]:', model.gravity.linear)
    print('Holding torques [N m], ordered shoulder then elbow:')
    for label, q in [('horizontal', [0.0, 0.0]),
                     ('upward', [-np.pi / 2, 0.0]), ('bent', [0.4, -0.7])]:
        torques = pin.computeGeneralizedGravity(model, data, np.array(q))
        print(f'{label:12s} q={q}: {torques}')
    print('Negative torque holds the horizontal arm against gravity for our +Y axes.')
    print('These are ideal model predictions: friction and motor gearing are omitted.')


def solve_ik(model, data, frame_id, target, max_iterations=200):
    # A bent seed avoids beginning at the fully extended singular configuration.
    q = np.array([0.2, -0.5])
    for iteration in range(max_iterations):
        position = hand_position(model, data, frame_id, q)
        error = target - position
        if np.linalg.norm(error) < 1e-5:
            return q, True, iteration
        J = pin.computeFrameJacobian(model, data, q, frame_id, pin.LOCAL_WORLD_ALIGNED)[:3, :]
        # Damping makes the matrix invertible near singular configurations.
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), error)
        q = pin.integrate(model, q, 0.5 * dq)
        q = np.clip(q, model.lowerPositionLimit, model.upperPositionLimit)
    return q, False, max_iterations


def inverse_kinematics():
    parser = argparse.ArgumentParser(description='Find angles for a target in the XZ plane.')
    parser.add_argument('--target', nargs=2, type=float, default=[0.7, -0.35],
                        metavar=('X', 'Z'), help='Target coordinates in meters')
    args = parser.parse_args()
    target = np.array([args.target[0], 0.0, args.target[1]])
    if not np.all(np.isfinite(target)):
        parser.error('Target must be finite.')
    radius = np.linalg.norm(target)
    if not 0.2 - 1e-9 <= radius <= 1.0 + 1e-9:
        parser.error('Target is outside the arm reach: 0.2 <= sqrt(x*x + z*z) <= 1.0 m.')
    model, data, frame_id = load_arm()
    q, converged, iterations = solve_ik(model, data, frame_id, target)
    reached = hand_position(model, data, frame_id, q)
    print('Target XYZ [m]:', target)
    print('Joint angles [rad]:', q)
    print('Reached XYZ [m]:', reached)
    print(f'Iterations: {iterations}; error [m]: {np.linalg.norm(target - reached):.3e}')
    if not converged:
        raise SystemExit('Did not converge: try another target or initial configuration.')
    print('Converged. Set these angles in the RViz joint sliders to compare.')
