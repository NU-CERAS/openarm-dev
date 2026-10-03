# Start here: robotics learning labs

Learn how ROS moves information, how a URDF describes a robot, and how Pinocchio
calculates its motion and forces. No physical robot is needed. Basic Python
(functions, classes, lists) and basic sine/cosine are helpful; later labs explain
the matrix operations they introduce.

## Before your first lab

1. Follow the [project setup](../README.md#setup-everyone) and open the container.
2. If you already had the container before these lessons were added, run:

   ```bash
   cb
   cs
   ```

3. Open [the browser desktop](http://localhost:6080/vnc.html), click Connect,
   and enter `openarm`.
4. Run all commands below **inside a container terminal**, from
   `/workspaces/openarm-dev`. Open a fresh terminal for each long-running node;
   use `cs` in terminals that were open before a build.
5. Stop the existing OpenArm viewer before starting the teaching arm. Use Ctrl+C
   in the terminal that launched it. If it was started in the background during
   setup, find it with `pgrep -af 'ros2 launch openarm_description'` and run
   `kill -INT PID` using the displayed launch process ID. Wait for its children
   to exit. One robot viewer and one joint-state publisher should run at a time.

## Choose your path

| Lab | Topic | Approximate time |
|---|---|---|
| [01 — Nodes and topics](01_nodes_and_topics.md) | Send and receive an angle | 25 min |
| [02 — URDF and RViz](02_urdf_and_rviz.md) | Links, joints, frames, joint states | 35 min |
| [03 — Xacro](03_xacro.md) | Reusable robot geometry | 25 min |
| [04 — Parameters and services](04_parameters_and_services.md) | Change and reset motion | 30 min |
| [05 — Load a model](05_pinocchio_model.md) | URDF to a mathematical model | 20 min |
| [06 — Forward kinematics](06_forward_kinematics.md) | Angles to hand position | 35 min |
| [07 — Jacobians](07_jacobians.md) | Small angle changes to small movements | 35 min |
| [08 — Gravity](08_gravity.md) | Mass, leverage, and holding torque | 30 min |
| [09 — Inverse kinematics](09_inverse_kinematics.md) | Position to angles | 40 min |
| [10 — ROS + Pinocchio](10_ros_and_pinocchio.md) | Calculate position from live messages | 35 min |

For a short introduction, do **01 → 02 → 05 → 06 → 08 → 10**.
Labs 07 and 09 are extensions for students ready for matrices and numerical methods.

## Where the examples live

- [Robot description](../src/teaching_arm_description/): Xacro, launch file, RViz settings.
- [ROS examples](../src/ros_teaching_examples/ros_teaching_examples/): small Python nodes.
- [Pinocchio examples](../src/pinocchio_teaching_examples/pinocchio_teaching_examples/):
  standalone calculations and one ROS integration node.
- [Starter exercises](exercises/): deliberately incomplete files for students to finish.
  The runnable package examples serve as references; try the exercise first.

These folders are tracked by Git. `src/vendor/`, `build/`, `install/`, and `log/`
are generated locally. Do not edit vendor code for these labs.

## Shared robot conventions

The shoulder and elbow rotate about **+Y**, and the arm moves in the **XZ plane**.
At zero angles it points along +X; a positive shoulder angle bends it toward -Z.
Lengths are 0.6 m and 0.4 m; masses are 1.0 kg and 0.6 kg. Angles use radians.
`base_link` is the reference frame; `tool0` is the hand at the end of link 2.
The base is fixed, and Pinocchio's default gravity points along -Z.

RViz displays geometry and transforms. It does not simulate forces or drive motors.
The joint motion example publishes state messages for visualization only.

## If something goes wrong

- Package not found: run `cb`, then `cs` in every terminal you use.
- Blank robot or flickering motion: stop other robot launches and joint-state
  publishers; check `ros2 topic info /joint_states --verbose`.
- GUI missing: check port 6080 and reconnect the browser desktop.
- `NotImplementedError` in an exercise: that marks the code you need to complete.
- Edited URDF but nothing changed: run `cb`, `cs`, stop the launch, and relaunch.
- A node waits silently: start its publisher in another terminal. Confirm the
  topic name and message type with `ros2 topic info`.

## For instructors

Each lab includes a demonstration, an exercise, expected observations, and questions.
Have students predict before running, then explain any mismatch. Ask for a screenshot
or terminal result plus answers to the questions, rather than code alone. Keep the
same robot dimensions through Labs 05–10 so the numerical reference values remain valid.
Students can commit their package changes and exercise solutions to their own clone.

To check the unchanged reference examples after a build:

```bash
python3 tests/teaching_numerics.py
python3 tests/teaching_ros.py
```

The numerical checks compare against trigonometry, finite differences, and lever
arms. The ROS checks start temporary nodes in domain 77 and stop them afterward.
Use domain 77 only for this check while it runs. Deliberately changing the teaching
model's lengths or masses will change the numerical expectations.

Further reading: [ROS 2 Jazzy tutorials](https://docs.ros.org/en/jazzy/Tutorials.html),
[robot_state_publisher](https://github.com/ros/robot_state_publisher),
[Pinocchio examples](https://github.com/stack-of-tasks/pinocchio/tree/master/examples).
