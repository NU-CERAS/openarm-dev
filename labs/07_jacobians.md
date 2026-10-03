# Lab 07 — The Jacobian

[Learning guide](README.md) · Next: [Gravity](08_gravity.md)

**Goal:** understand a local linear approximation of robot motion.
A translational Jacobian tells us how small changes in joint angles change the
hand position near the current pose:

```text
small hand displacement ≈ J × small joint-angle change
```

Here J has three rows (X, Y, Z) and two columns (shoulder, elbow).
It also maps joint velocities in rad/s to hand velocity in m/s.

## Run the demonstration

```bash
ros2 run pinocchio_teaching_examples jacobian --q 0.4 -0.7
ros2 run pinocchio_teaching_examples jacobian --q 0 0
```

**Expected:** predicted and actual displacements nearly agree. The Y row is zero.
At the fully extended pose, both joints initially move the hand along Z, so the
Jacobian cannot produce arbitrary XZ motion to first order: this is a singular pose.

Read `jacobian()` in [demos.py](../src/pinocchio_teaching_examples/pinocchio_teaching_examples/demos.py).
`LOCAL_WORLD_ALIGNED` gives motion at the hand origin using world-aligned axes.
For Pinocchio's spatial Jacobian the first three rows are linear velocity and the
last three are angular velocity. We select the first three.

## Try it yourself

Increase `dq` by 10 times in the example. Rebuild and rerun. Compare the prediction
error. Try changing only the shoulder, then only the elbow; relate the result to
the corresponding Jacobian column.

Do not expect a single Jacobian to predict a large movement accurately: it changes
with the robot configuration. Restore the small `dq` after the exercise.

## Explain what happened

1. What does each Jacobian column mean physically?
2. Why do larger changes usually produce a worse linear approximation?
3. Why is the Y row zero?
4. Is a zero X derivative at full extension the same as saying X can never change?
