# Motion planning, drumming RL, and robot execution

## Installation and cuRobo reach

The validated workstation uses an RTX 5080, NVIDIA driver 580.65.06, Python 3.12, and PyTorch 2.14.1 with CUDA 13.0. cuRobo is pinned to commit `78fd485fa82d9b9a063fb4985e371814587e666a`. Installation changes the conda environment only.

```bash
conda activate mujoco
bash projects/cartesian_console/scripts/setup_curobo.sh
openarm-reach --planner curobo
```

The setup script installs the CUDA dependencies and exports the robot model from `robots/v1.0/models/scene.xml`. `yourdfpy` writes the URDF; `trimesh` exports the compiled convex collision meshes; cuRobo's RobotBuilder fits collision spheres. Assets are generated under repository `runs/cartesian_console/planning/`. The MJCF self-collision exclusions and adjacent-link exclusions are included. Fixed base collision spheres are clipped at its floor mounting plane.

cuRobo handles IK, trajectory optimization, and graph seeding. For position-only clicks, mink selects a reachable goal orientation. cuRobo then plans the full pose with the table, ground, and configured static obstacles. The inactive arm and grippers remain locked at their measured state in the planning model. The complete robot geometry remains available for collision checking.

The returned trajectory is reordered by joint names, retimed to the demo's velocity and acceleration bounds, checked against MuJoCo collision meshes, and executed through the physical position servos. The GUI defaults to cuRobo. `--planner mink` explicitly selects the local IK and joint-interpolation controller. The VLM CLI also accepts `--planner` and defaults to cuRobo.

IK or collision-feasibility failures return a rejected reach result containing the arm, target coordinates, and reason. Joint commands remain unchanged, and the GUI accepts another target. Invalid configuration and execution errors propagate directly.

Current cuRobo integration covers reach with empty grippers. Planning transport of a grasped free object requires an attached-object collision configuration; that request currently raises an explicit error. Pickup and strike preparation use the existing physical IK reference controllers. Contact-rich striking uses its own reference plus residual controller.

## Drumming training preparation

The task design follows DexDrummer's separation of planned arm motion and residual RL, with explicit grasp, trajectory, and impact rewards. This implementation targets OpenArm's parallel gripper and a single drum. It does not reproduce DexDrummer's dexterous-hand policy or its bimanual song benchmark. Reference: [DexDrummer project](https://dexdrummer.github.io/) and [official source](https://github.com/hc-fang/dexdrummer).

| Environment | Observation | Action | Initial state |
| --- | --- | --- | --- |
| `OpenArm-StickPickup-v0` | 42 | Seven joint-target increments and gripper opening | IK approach above a free stick |
| `OpenArm-DrumStrike-v0` | 39 | Seven joint-target increments and gripper opening | Physical pickup and transport above drum |
| `OpenArm-ResidualDrumStrike-v0` | 51 | Seven bounded joint residuals and a gripper correction | Physical pickup, transport, and a nominal strike reference |

Residual control runs at 50 Hz. Each arm target equals the nominal trajectory plus a bounded residual, with each command increment limited to 0.025 rad. The default residual bound is 0.08 rad. The gripper correction allows 0 to 0.0066 m opening around the closed reference. The reference is generated from IK contact and recovery poses with a velocity condition at the desired impact time. It provides a baseline for learning contact corrections; nominal execution is not a trained policy.

The 51-value observation contains the base joint/gripper and object-relative state, requested impact timing/speed, previous low-level action, next reference joint position, reference tip error, episode phase, and curriculum stage. Simulation currently supplies exact object state. Physical deployment requires a synchronized object-state estimator in the robot frame.

Curriculum stages are explicit configuration choices:

- `0`: grasp retention, trajectory tracking, effort, and action smoothness.
- `1`: the same terms plus measured impact-event rewards.
- `2`: the same terms plus impact timing/speed and recovery task rewards.

The default physical randomization changes stick mass and inertia together by up to 10%, and stick friction by up to 10%. Drum position, desired strike time/speed, and initial stick position are also sampled. Reset validates actual two-finger contact and preparation error. No attachment constraint is used for grasping. Drum response remains a rigid-surface contact model requiring hardware calibration.

Environment validation runs without learning:

```bash
openarm-train-drum --config projects/cartesian_console/configs/rl/pickup.yaml --check-env
openarm-train-drum --config projects/cartesian_console/configs/rl/residual_strike.yaml --check-env
```

Learning requires an explicit `--train`. Curriculum stages are selected with `--curriculum-stage 0`, `1`, or `2`; `--resume PATH --train` continues from a checkpoint. `--checkpoint PATH` performs evaluation. See the [project README](../README.md) for ACT pickup results and pipeline status. The strike policy has no demonstrated trained performance.

## Shared sim-to-real interfaces

`ArmCommand` specifies seven named joint positions in radians and gripper displacement in meters. `JointTrajectory` specifies time, joint position, and joint velocity. Joint names use `openarm_{left,right}_joint1` through `joint7` in that order. Gripper displacement follows the existing controllers: `0` closed and `0.044` open.

| Backend | Read | Write |
| --- | --- | --- |
| `MujocoArmInterface` | Actual simulated joint position, velocity, and gripper position | Position targets for the existing MuJoCo actuators |
| `Ros2ArmInterface` | `/joint_states`, mapped by joint name | `/{side}_forward_position_controller/commands` and `/{side}_gripper_controller/commands` |

The ROS 2 message types and topic names match `src/ros/openarm_mujoco_sim/scripts/simple_motion.py` and `src/ros/openarm_mujoco_sim/config/controllers.yaml`. The backend uses the robot's existing ros2_control hardware implementation. It does not activate the CAN driver.

`DrumObservation.vector`, `residual_observation`, and `residual_strike_command` are shared state/action conversion functions. They let a physical controller use the same observation order and residual interpretation as simulation. Required external inputs include calibrated TCP pose, stick pose and velocity, tip position/velocity, and drum surface pose. Missing estimates must be handled by the application before inference; simulation state is not substituted for physical measurements.

The video recorder exports real cuRobo trajectories. Inspect or replay one through the shared simulation backend:

```bash
openarm-replay .work/demo/right_reach_trajectory.json
openarm-replay .work/demo/right_reach_trajectory.json --backend sim --execute
```

The default command only prints the trajectory and controller messages. ROS execution is an explicit `--backend ros2 --execute` operation from an environment providing ROS 2 Python packages. It checks initial joint agreement, fresh feedback, tracking error, and final joint error. Physical execution requires calibrated joint conventions and camera/base/TCP transforms, verified controller limits, and the robot's normal stop mechanism. The ROS 2 Python runtime is not installed on this workstation. No physical motor execution was performed.

## Demonstration MP4

```bash
openarm-record-demo --output .work/demo/openarm_demo.mp4
```

The video uses actual rendered MuJoCo states and English captions. It shows calibrated pixel reach with cuRobo, obstacle-aware reach, physical reference pickup, and nominal striking with zero arm residual. The obstacle scenario has collision-free endpoints and a blocked direct joint-interpolation path. It includes the official midpoint camera and a separate presentation camera. On-screen task status and contact metrics are measured, including any unmet timing/speed objective.

The MP4 is accompanied by segment metadata, a per-frame state trace, and exported reach trajectories. It contains no learned-policy or hardware-execution claims. The tested export is 1920x1080, 30 fps, 29.9 seconds, and uses H.264 with a fast-start MP4 container.

## Verified checks

- Exported URDF forward kinematics agree with MuJoCo for both TCPs across ten joint configurations.
- cuRobo reach executes through MuJoCo dynamics and converges below the 10 mm success threshold.
- With an obstacle blocking direct interpolation, cuRobo finds and physically executes a path to the target.
- Exported cuRobo trajectories replay through `MujocoArmInterface`.
- Pickup establishes both finger contacts and lifts the free stick above 100 mm.
- The nominal strike reference produces real drum contacts and retains its grasp; the full timing/speed task remains a learning objective.
- The residual environment passes Gymnasium API checks without learning.
- The MP4 fully decodes and its frame count matches the recorded state trace.

cuRobo installation and robot-model references: [installation](https://nvlabs.github.io/curobo/latest/getting-started/installation.html), [motion planning](https://nvlabs.github.io/curobo/latest/getting-started/motion_planning.html). OpenArm model attribution is preserved in `robots/README.md` and `robots/v1.0/LICENSE`.
