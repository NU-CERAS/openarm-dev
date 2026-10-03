# Lab 09 — Inverse kinematics

[Learning guide](README.md) · Next: [ROS + Pinocchio](10_ros_and_pinocchio.md)

**Goal:** find angles for a desired position and understand why a numerical solver
needs an initial guess, a stopping condition, and a failure result.
Forward kinematics starts with angles; inverse kinematics starts with a target.

## Run the demonstration

```bash
ros2 run pinocchio_teaching_examples inverse_kinematics
ros2 run pinocchio_teaching_examples inverse_kinematics --target 0.6 0.2
ros2 run pinocchio_teaching_examples inverse_kinematics --target 2.0 0.0
```

Arguments are X and Z in meters; Y is fixed to zero. **Expected:** the first two
commands converge with position error below 0.00001 m. The third is rejected as
out of reach and exits unsuccessfully. Set the returned angles in the teaching
viewer's sliders and inspect the hand position.

Read `solve_ik()` in [demos.py](../src/pinocchio_teaching_examples/pinocchio_teaching_examples/demos.py).
Each iteration computes the error, uses a damped Jacobian to choose an angle
change, and updates the configuration with `pin.integrate`. Damping helps near
singularities; the smaller step reduces overshooting. Joint limits are enforced.

With unrestricted planar revolute joints, this arm's radial reach is between
|0.6 - 0.4| = 0.2 m and 0.6 + 0.4 = 1.0 m. Joint limits, obstacles, orientation
requirements, and other robots can further restrict reachable targets.

## Try it yourself

Try five targets well inside that reach. Record target, angle solution, iteration
count, and final error. Change the initial bent pose to a different one, rebuild,
and rerun. You may find another elbow configuration for the same hand position.

Try a boundary target such as `[1.0, 0.0]`. Reachability alone does not guarantee
that this numerical solver converges quickly from every starting configuration.
Restore the original initial pose after the exercise.

## Explain what happened

1. Why can one target have more than one angle solution?
2. Why does the solver start bent rather than fully extended?
3. What does the stopping tolerance measure?
4. Does reaching a target prove the path avoids collisions or controls real motors?

**Finish:** stop the viewer if you started it.
