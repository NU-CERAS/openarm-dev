# Lab 01 — Nodes and topics

[Learning guide](README.md) · Next: [URDF and RViz](02_urdf_and_rviz.md)

**Goal:** explain how two independent programs communicate through a typed ROS topic.
A node is a running ROS participant. A publisher sends messages on a topic; a
subscriber's callback handles arriving messages. A topic is not a saved message history.

## Run the demonstration

Terminal A:

```bash
ros2 run ros_teaching_examples angle_publisher
```

Terminal B:

```bash
ros2 run ros_teaching_examples angle_subscriber
```

Terminal C:

```bash
ros2 node list
ros2 topic list
ros2 topic info /teaching/angle
ros2 interface show std_msgs/msg/Float64
ros2 topic echo /teaching/angle --once
ros2 topic hz /teaching/angle
```

**Expected:** the publisher sends an angle between -1 and 1 radians, roughly twice
per second. The subscriber prints the same angle converted to degrees. Stop the
rate command with Ctrl+C. Stop Terminal A: the subscriber stays alive but receives
no new messages. Restart A: messages resume.

Read [the publisher](../src/ros_teaching_examples/ros_teaching_examples/angle_publisher.py)
and [the subscriber](../src/ros_teaching_examples/ros_teaching_examples/angle_subscriber.py).
Find the timer, message type, topic name, and callback. `rclpy.spin()` lets the
executor process callbacks; it is not a loop you write yourself.

## Try it yourself

Stop the reference subscriber. Complete
[subscriber_starter.py](exercises/subscriber_starter.py), then run:

```bash
python3 labs/exercises/subscriber_starter.py
```

Create a `Float64` subscription to `teaching/angle` with depth 10 and log the
angle in degrees. The starter intentionally raises `NotImplementedError` until
completed. Test that 1 radian converts to about 57.3 degrees.

Then change the publisher timer period from 0.5 to 0.25 seconds. Rebuild with `cb`
and source with `cs` before restarting it. Measure the new rate.

## Explain what happened

1. Does the publisher need to know the subscriber's process ID?
2. Why must both ends agree on a message type?
3. What happens when two subscribers listen to the same topic?
4. How does timer period relate to publishing frequency?

**Finish:** stop your nodes with Ctrl+C.
