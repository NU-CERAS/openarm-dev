# Lab 03 — Xacro and reusable geometry

[Learning guide](README.md) · Next: [Parameters and services](04_parameters_and_services.md)

**Goal:** explain how Xacro generates URDF from parameters and macros.
URDF is the robot description format; Xacro is a preprocessing tool that reduces
repetition. ROS and Pinocchio receive the expanded URDF, not the macro definitions.

## Run the demonstration

```bash
xacro src/teaching_arm_description/urdf/two_link_arm.urdf.xacro > /tmp/teaching_arm.urdf
check_urdf /tmp/teaching_arm.urdf
```

**Expected:** the checker reports a tree rooted at `base_link` with two moving
links and a fixed tool frame. Open `/tmp/teaching_arm.urdf` and compare it with
[the Xacro source](../src/teaching_arm_description/urdf/two_link_arm.urdf.xacro).
The `arm_link` macro expands to visual, collision, and inertial definitions.

```bash
xacro src/teaching_arm_description/urdf/two_link_arm.urdf.xacro length2:=0.2 > /tmp/short_arm.urdf
check_urdf /tmp/short_arm.urdf
```

This changes only the generated temporary file; it does not change the running viewer.
Look for the shorter second link, its center of mass, and the tool joint origin.

## Try it yourself

Temporarily change the `length2` default in the source from 0.4 to 0.2. Rebuild,
source, and relaunch the teaching viewer. At zero angles, predict and verify that
the hand is at `[0.8, 0, 0]` m. **Restore length2 to 0.4 and relaunch before later labs.**

Compare the visual, collision, and inertial blocks. The box inertia formula uses
mass and dimensions; the inertial origin places its center of mass halfway along
the link. Changing length while keeping inertia constant would describe a different
mass distribution.

## Explain what happened

1. Which repeated definitions does the macro eliminate?
2. Why must the elbow and tool origins agree with link lengths?
3. Why do the visual origin and center-of-mass origin both use half the link length?
4. Can visual geometry and collision geometry intentionally be different?

**Finish:** restore the original model dimensions and stop the viewer.
