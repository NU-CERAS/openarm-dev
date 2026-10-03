# Lab 05 — From URDF to a Pinocchio model

[Learning guide](README.md) · Next: [Forward kinematics](06_forward_kinematics.md)

**Goal:** understand the model, workspace data, joint coordinates, and frames.
These examples do not start ROS nodes. We use `ros2 run` to find the installed
Python programs; their calculations run without publishers or subscribers.

## Run the demonstration

```bash
ros2 run pinocchio_teaching_examples model_info
```

**Expected:** `nq=2`, `nv=2`, shoulder and elbow joint names, a `tool0` frame,
and a neutral hand position of `[1, 0, 0]` m.

Read [common.py](../src/pinocchio_teaching_examples/pinocchio_teaching_examples/common.py)
and `model_info()` in [demos.py](../src/pinocchio_teaching_examples/pinocchio_teaching_examples/demos.py).
The loader expands Xacro and passes XML to `pin.buildModelFromXML`.

- `model` stores the structure, geometry needed for kinematics, masses, and inertias.
  These examples load the kinematic/dynamic model, not visual mesh geometry.
- `data` stores intermediate calculations and algorithm results.
- `q` contains configuration coordinates, here two joint angles.
- `nq` counts configuration coordinates; `nv` counts velocity coordinates.
  They agree for this arm, but can differ for floating bases and other joint types.
- Frames include link and tool coordinate systems. Fixed joints can add frames
  without adding an independent angle.

## Try it yourself

Print each moving joint's lower and upper position limits in `model_info()`.
Rebuild and rerun. Check those numbers against the URDF.
Identify the hand frame and its parent joint. Explain why there are more frames
than movable joints. Do not assume a frame ID and a joint ID mean the same thing.

## Explain what happened

1. Why does this model have two angles instead of three?
2. What would adding a free-floating base change?
3. Why does Pinocchio separate `model` from `data`?
4. What is the role of Xacro before Pinocchio sees the robot?
