# Lab 08 — Gravity and holding torque

[Learning guide](README.md) · Next: [Inverse kinematics](09_inverse_kinematics.md)

**Goal:** relate mass, center of mass, leverage, and joint torque.
Pinocchio's generalized gravity result is the ideal joint torque needed to hold
this fixed-base model motionless against gravity, without friction or other loads.

## Run the demonstration

```bash
ros2 run pinocchio_teaching_examples gravity
```

**Expected:** the horizontal arm needs approximately `[-7.6518, -1.1772]` N m
at the shoulder and elbow. The upward arm needs nearly zero gravity torque.
Negative values follow the chosen +Y joint axes; magnitude indicates the effort.

For the horizontal pose, use gravity magnitude 9.81 m/s²:

```text
elbow magnitude    = 9.81 × (0.6 × 0.2) = 1.1772 N m
shoulder magnitude = 9.81 × (1.0 × 0.3 + 0.6 × (0.6 + 0.2)) = 7.6518 N m
```

The distances are from each joint to each link's center of mass, not to the hand.
The centers of mass lie halfway along the uniform boxes.

## Try it yourself

Double link 2's mass in the Xacro macro invocation. Predict both horizontal
holding torques before rebuilding and rerunning the example. The new magnitudes
should be 11.1834 N m and 2.3544 N m. **Restore the mass to 0.6 kg afterward.**

Ask whether doubling the mass should change forward kinematics. Run that example
to check: kinematics depends on geometry and joint angles, not mass.

## Explain what happened

1. Why does the shoulder hold the weight of both links?
2. Why does holding the arm vertically require little gravity torque?
3. Which URDF fields affect gravity torques, and which affect acceleration torques?
4. Why is a torque at a joint not automatically the torque at a motor shaft?

This model does not include motor gearing, friction, payloads, or a physical controller.
