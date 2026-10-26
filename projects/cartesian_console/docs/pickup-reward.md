# Pickup reward

`pickup_milestones_v1` is shared by the native MuJoCo and MuJoCo Warp pickup environments. The control interval is 0.02 seconds and the task deadline is six seconds.

| Term | Value |
| --- | --- |
| Progress | `0.99 * potential(next) - potential(current)` |
| First sustained 2 cm lift | +1 |
| First sustained 5 cm lift | +2 |
| Successful lift and hold | +50 |
| Time | -0.01 per control step |
| Grasping without height progress for 0.5 seconds | Additional -0.02 per step |
| Deadline failure | -5 |
| Unsafe contact, dropped object, or lost grasp after lifting | -10 |
| Action change | `-0.002 * mean((action - previous_action)**2)` |

Potential is `0.5 * exp(-(distance / 0.05)**2) + 0.5 * grasped + 2 * grasped * clip(lift / 0.10, 0, 1)`. Terminal potential is zero. A fixed contact configuration does not generate a positive stream of progress rewards.

Milestones require two-finger contact, no contact with external supports or other robot geometry, and three consecutive qualified control observations. Each milestone is awarded once per episode. Lowering and lifting again does not reset award flags.

Success requires a lift exceeding 0.10 m, two-finger contact, no external support, and stick linear speed below 0.12 m/s for 0.3 seconds. Grasp loss lasting 0.10 seconds after an airborne lift of at least 0.02 m ends the task as failure. Unsafe contact or falling below the table also ends it as failure. The six-second deadline is a task failure with zero terminal potential; it is not bootstrapped as a time-limit truncation.

## Teacher observations

The actor receives 57 values: the 42 robot/stick state and action-history values, ten reward-progress values, finger contact count, external support indicator, and drum position relative to TCP. Reward progress includes milestone flags, qualification timers, hold time, maximum lift, stagnation time, grasp-loss time, lifted-state flag, and remaining task time.

## Monitoring

The trainer logs maximum lift, 2 cm and 5 cm milestone rates, grasp fraction, timeout rate, Gaussian action clipping fraction, and episode sums for each reward term. Evaluation records these measurements for every episode alongside the execution video and state trace. W&B receives both training and evaluation diagnostics.

Every checkpoint includes the policy and normalization state. Evaluation freezes normalization statistics and reports original rewards.
