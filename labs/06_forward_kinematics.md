# Lab 06 — Forward kinematics

[Learning guide](README.md) · Next: [Jacobians](07_jacobians.md)

**Goal:** calculate hand position from joint angles and check it independently.
Forward kinematics maps a configuration to the poses of the robot's frames.

For this arm, with shoulder angle q1, elbow angle q2, and lengths L1 and L2:

```text
x = L1*cos(q1) + L2*cos(q1 + q2)
y = 0
z = -L1*sin(q1) - L2*sin(q1 + q2)
```

The minus sign follows our +Y rotation axes. The elbow angle is relative to link 1,
so link 2's orientation relative to the base is q1 + q2.

## Run the demonstration

```bash
ros2 run pinocchio_teaching_examples forward_kinematics --q 0 0
ros2 run pinocchio_teaching_examples forward_kinematics --q -1.57079632679 0
ros2 run pinocchio_teaching_examples forward_kinematics --q 0.4 -0.7
```

**Expected:** approximately `[1, 0, 0]`, `[0, 0, 1]`, and
`[0.9348, 0, -0.1154]` m, respectively. The Pinocchio and trigonometry results
agree to floating-point precision with the unmodified robot dimensions.

The example calls `forwardKinematics`, then `updateFramePlacements`, then reads
`data.oMf[frame_id]`. Its translation is the hand's position in the base/world frame.
The rotation also exists, even though this example prints only translation.

## Try it yourself

Complete [forward_kinematics_starter.py](exercises/forward_kinematics_starter.py):

```bash
python3 labs/exercises/forward_kinematics_starter.py
```

The checks compare your formula against Pinocchio at three poses. Start with a
prediction on paper. Keep lengths at 0.6 and 0.4 m for these checks.

Launch the teaching viewer, enter the same angles with the sliders, and compare
`ros2 run tf2_ros tf2_echo base_link tool0` with your calculation. Slider precision
may make a small difference.

## Explain what happened

1. Why does link 2 use q1 + q2?
2. Why is Y always zero in this model?
3. What frame is the printed position expressed in?
4. Would the same joint angles give the same position on an arm with different lengths?

**Finish:** stop the viewer and TF command if you started them.
