# Lab 10 — Connect ROS and Pinocchio

[Learning guide](README.md)

**Goal:** turn incoming joint-state messages into a calculated hand position and
check it against ROS transforms.

## Run the demonstration

Stop other robot launches. Terminal A:

```bash
ros2 launch teaching_arm_description display.launch.py
```

Terminal B:

```bash
ros2 run pinocchio_teaching_examples hand_position
```

Terminal C:

```bash
ros2 topic echo /teaching/hand_position --once
ros2 run tf2_ros tf2_echo base_link tool0
```

**Expected:** at zero angles, both report `[1, 0, 0]` m. Hold the sliders still,
then compare values at several poses. Moving while comparing can produce different
samples because the commands do not necessarily observe the same timestamp.
The calculated message identifies `base_link` as its frame and copies the input timestamp.

The flow is:

```text
joint sliders → /joint_states → hand_position node → /teaching/hand_position
                         └──→ robot_state_publisher → /tf → RViz
```

Read [hand_position.py](../src/pinocchio_teaching_examples/pinocchio_teaching_examples/hand_position.py).
The callback maps joint names to the model's configuration indices, computes
kinematics, and publishes `geometry_msgs/PointStamped`. It does not assume that
joint positions always arrive in a particular order. Messages missing either
joint or containing invalid values are ignored.

## Try it yourself

Stop the TF command, then add a **PointStamped** display in RViz. Select topic
`/teaching/hand_position`. The point should follow the `tool0` frame as you move
the sliders. Increase the display's history length to leave a short trace.

Inspect the graph using `ros2 node info /hand_position` and
`ros2 topic info /joint_states --verbose`. Explain which nodes publish and subscribe.

Optional extension: publish the gravity holding torques on a new topic. Document
the torque units and joint order. Treat these as calculated values, not motor commands.

## Explain what happened

1. Which part handles communication and which part calculates robot geometry?
2. Why does the callback match joint names instead of trusting array order?
3. Why must a position message identify its reference frame?
4. What mismatch would you expect if ROS and Pinocchio loaded different link lengths?

**Finish:** stop the bridge node and viewer with Ctrl+C. You can now repeat these
ideas with OpenArm, after inspecting its joint names, frame names, and model conventions.
The teaching bridge is specific to our two-link robot, not a generic OpenArm node.
