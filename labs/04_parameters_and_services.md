# Lab 04 — Parameters and services

[Learning guide](README.md) · Next: [Load a Pinocchio model](05_pinocchio_model.md)

**Goal:** distinguish continuous topic messages, node configuration, and a
request/response operation.

## Run the demonstration

Stop the slider-based viewer. In Terminal A:

```bash
ros2 launch teaching_arm_description display.launch.py use_gui:=false
```

This starts [joint_motion.py](../src/ros_teaching_examples/ros_teaching_examples/joint_motion.py)
in place of the sliders. It publishes shoulder and elbow positions at 20 Hz.
`amplitude` is in radians; `frequency_hz` is the motion's cycles per second,
not the message publishing rate.

Terminal B:

```bash
ros2 param list /joint_motion
ros2 param get /joint_motion amplitude
ros2 param set /joint_motion amplitude 1.0
ros2 param set /joint_motion frequency_hz 0.1
ros2 topic hz /joint_states
```

**Expected:** the arm swings farther and more slowly. Messages still arrive at
approximately 20 Hz. Stop the rate command before continuing.

```bash
ros2 service list -t
ros2 service call /reset_motion std_srvs/srv/Trigger '{}'
ros2 param set /joint_motion amplitude -1.0
```

Reset returns `success: true` and restarts the sine wave's phase. Motion resumes
on the next timer tick; reset does not permanently stop the arm. Negative amplitude
is rejected, and the previous value remains in effect.

## Try it yourself

Set `frequency_hz` to 0.0 to freeze the current pose. Call reset while frozen:
the next tick publishes zero angles. Restore frequency to 0.2.

Modify the node so the elbow moves at the same angle as the shoulder, instead of
-0.5 times that angle. Rebuild and relaunch; explain the changed hand path.

## Explain what happened

1. Why does reset fit a service better than a stream of messages?
2. What is the difference between motion frequency and publishing frequency?
3. Why should a node validate parameters?
4. Is this node issuing motor commands, simulating dynamics, or publishing a pose?

**Finish:** Ctrl+C in Terminal A.
