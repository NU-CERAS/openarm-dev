# Cartesian Console

Camera-guided Cartesian reach, electronic-drum simulation, physical stick pickup,
and oracle ACT / residual PPO skill learning for OpenArm v1.0.

## Installation

Run from the repository root in the `mujoco` environment:

```bash
conda activate mujoco
mkdir -p runs/cartesian_console/cache
export TMPDIR="$PWD/runs/cartesian_console/cache"
uv pip install --python "$CONDA_PREFIX/bin/python" -e . \
  -e './projects/cartesian_console[dev,agent,rl,gpu-sim,tracking,imitation,video,assets]'
```

For a new conda environment, run `conda env create -f environment.yml` from this
project directory. Install CUDA PyTorch and cuRobo with
`bash projects/cartesian_console/scripts/setup_curobo.sh` from the repository root.
The script generates planning assets from `robots/v1.0/models/scene.xml`.

## Interactive demos

```bash
openarm-reach --planner mink
openarm-reach --planner curobo
openarm-reach --scene pickup
openarm-reach --scene strike
openarm-reach --scene kit --planner curobo \
  --drum-asset runs/cartesian_console/drumset/converted/manifest.json
```

Reach displays the calibrated head camera, accepts a pixel selection and executes
a joint trajectory. Camera position, viewing direction and field of view are
editable in the English GUI. cuRobo plans against scene obstacles; `mink` uses
IK with joint interpolation and collision checks. Pickup and strike buttons run
physical reference controllers. Opening a demo starts no training.

The electronic kit requires the separately downloaded Sketchfab archive:

```bash
openarm-import-drum /home/jixinyan/Downloads/electronic_drum_set.zip
```

See [camera reach](docs/simulation-demo.md),
[electronic kit](docs/electronic-drum-scene.md), and
[planning and execution](docs/planning-rl-sim2real.md).

## Package boundaries

| Location | Responsibility |
| --- | --- |
| `src/openarm_sim/` | Task scenes, camera geometry, reach, RL, ACT, VLM tools and ROS adapter |
| `configs/rl/` | Task training configurations |
| `tests/` | Real MuJoCo and CUDA integration checks |
| Repository `src/openarm_tools/` | Shared robot configuration, commissioning and backend commands |
| Repository `robots/` | Canonical models, meshes and per-arm configurations |
| Repository `runs/cartesian_console/` | Generated planning assets, datasets, checkpoints, videos and test output |

`openarm_sim.paths` obtains the checkout through `openarm_tools.robot.config.root`;
`OPENARM_ROOT` can select an explicit checkout. Both arm joint orders and the
model path come from the shared v1.0 configuration. The distribution remains
`openarm-sim`, with existing imports, console commands, dataset schema and
checkpoint formats. Robot geometry remains in the shared `robots/` directory.

The shared `openarm` CLI retains its independent commissioning sessions. Task
simulation owns one synchronized bimanual MuJoCo state. Their saved state files
are separate. Task code does not modify shared calibration or start CAN motors.

## Learning and execution status

The pipeline provides reference collection, oracle ACT training, frozen-ACT
residual PPO, camera/proprioception student supervision and W&B tracking.
The oracle teacher can access object state. The student receives robot feedback,
action history and RGB images.

The existing pickup experiment achieved 58/64 successes with ACT and 57/64 with
ACT plus PPO residual on 64 distinct held-out starting states from the same
distribution. This experiment shows no PPO improvement. The camera student has
only completed a short execution check and has no demonstrated pickup ability.
These results belong to existing checkpoints; migration checks do not repeat
the training experiment. Existing files outside this checkout can be supplied
through explicit CLI paths.

`ArmCommand` carries seven joint targets in radians and gripper displacement in
meters. `JointTrajectory` preserves named-joint timing and velocities.
`MujocoArmInterface` and `Ros2ArmInterface` consume these commands; ROS topics
match `src/ros/openarm_mujoco_sim/config/controllers.yaml`. Real execution also
requires calibrated robot limits, feedback, scene perception and camera/base/TCP
transforms. ROS runtime and physical motor execution have not been validated on
this workstation. Current VLM tools cover observation and reach.

## Verification

From this project directory after installation:

```bash
export TMPDIR="$(pwd)/../../runs/cartesian_console/cache"
MUJOCO_GL=egl python -m pytest -q tests
openarm-reach --planner mink --smoke-test
openarm-reach --scene pickup --smoke-test
openarm-reach --scene strike --smoke-test
```

cuRobo checks require generated planning assets. Kit checks require the imported
asset. GPU/ACT checks execute real MuJoCo Warp physics and require CUDA.
