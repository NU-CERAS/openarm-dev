# Lab 02 — URDF and RViz

[Learning guide](README.md) · Next: [Xacro](03_xacro.md)

**Goal:** connect a URDF tree, joint angles, and the visible robot.
A link is a rigid body. A joint specifies how a child link moves relative to its
parent. A URDF forms a tree; it does not define arbitrary closed loops.

## Run the demonstration

Stop other robot viewers first. In Terminal A:

```bash
ros2 launch teaching_arm_description display.launch.py
```

Open the browser desktop and move the shoulder and elbow sliders. In Terminal B:

```bash
ros2 topic echo /joint_states --once
ros2 interface show sensor_msgs/msg/JointState
ros2 run tf2_ros tf2_echo base_link tool0
```

**Expected:** at zero angles the arm extends 1.0 m along +X. The hand transform's
translation is approximately `[1, 0, 0]`. Moving the shoulder moves both links;
moving the elbow moves only link 2. Positive shoulder rotation moves toward -Z.
The TF command continues printing; stop it with Ctrl+C.

Open [the robot description](../src/teaching_arm_description/urdf/two_link_arm.urdf.xacro).
Although the file uses Xacro macros, the expanded result is a normal URDF.
Find `base_link`, `link1`, `link2`, `tool0`, and the three joints. `tool_joint` is
fixed, so it does not add a controllable angle.

The sliders publish `/joint_states`. `robot_state_publisher` combines those
angles with the URDF and publishes transforms. RViz uses the geometry and transforms
to draw the robot. The message names identify which angle belongs to which joint.

## Try it yourself

Predict the hand position for shoulder = -1.5708 rad and elbow = 0, then set that
pose with the sliders. It should point upward, near `[0, 0, 1]` m. Expand RViz's
Frames display and inspect the local axes.

Sketch the parent-child tree on paper. Label each joint's axis and origin.
Change link 2's color in the Xacro file; rebuild, source, and restart the launch.

## Explain what happened

1. Why does moving the shoulder also move the elbow's coordinate frame?
2. How is a joint origin different from the visual geometry's origin?
3. What information is in `/joint_states` that is absent from a single `Float64`?
4. Why is a robot that looks correct in RViz not necessarily physically accurate?

**Finish:** Ctrl+C in Terminal A stops the launch and its nodes.
