# OpenArm camera reach demo

Install using the [project README](../README.md). Run in the `mujoco` conda environment:

```bash
conda activate mujoco
openarm-reach
```

The window displays a calibrated perspective camera at robot-head height. Select an arm and click a location on the tabletop. The selected pixel defines a ray that intersects the known table plane. The TCP goal is 0.14 m above that surface.

The camera controls expose `x`, `y`, `z`, `look_x`, `look_y`, `look_z`, and vertical `fovy`. Press the apply button after editing values. Positions are in meters in the simulation world frame. Pixel coordinates use a top-left origin, with `u` increasing rightward and `v` downward.

Current scene settings:

| Setting | Value |
| --- | --- |
| Robot | Repository OpenArm v1.0 model |
| Camera position | `(0.038, 0, 0.808)` m |
| Camera viewing direction | `(0.469472, 0, -0.882948)` |
| Vertical field of view | `52` degrees |
| Image | `1280 × 720`, native display size, 4x MSAA |
| Table surface | `z = 0.30` m |
| Reach TCP height | `z = 0.44` m |
| Success | TCP error below 10 mm and low joint speed for 0.2 s |

The camera uses the midpoint of official `camera_head_left` and `camera_head_right`, located at `(0.223, ±0.0315, 1.45)` m in the Cell frame. Their midpoint is `(0.223, 0, 1.45)`. To preserve the pose relative to the robot mounting, the demo subtracts the official mounting position `(0.185, 0, 1.34)` and adds the local mounting position `(0, 0, 0.698)`, giving `(0.038, 0, 0.808)` m. The official orientation is retained. Rendering displays the full image without resizing or compression. Source: [official Cell model](https://github.com/enactic/openarm_mujoco/blob/a3c8a57ec15d385e3c1dbd3fd090ee486b7f258c/v2/cell/cell.xml).

The GUI defaults to cuRobo motion planning. From the repository root, install planning dependencies and generate collision assets with `bash projects/cartesian_console/scripts/setup_curobo.sh`. cuRobo plans a collision-aware trajectory, which is retimed, checked against MuJoCo collision meshes, and executed through the position actuators. Model-based gravity feedforward retains actuator effort limits. Motion is evaluated from the actual simulated TCP state. `--planner mink` selects the IK/interpolation controller explicitly.

Image clicks specify position. A reachable orientation is selected using mink, and cuRobo plans the resulting pose. Full installation, RL, execution-interface, and recording details are in [planning-rl-sim2real.md](planning-rl-sim2real.md).

## Drum environment preview

```bash
openarm-reach --scene pickup
openarm-reach --scene strike
```

The imported Sketchfab electronic kit has a separate [electronic drum scene](electronic-drum-scene.md), including conversion, pad collisions, camera controls, and recording commands.

Both scenes expose the same camera controls. Pickup starts with a free stick on the table and the gripper above it. Press **Reference pickup (controller)** to lower the gripper, close it, lift the stick, and hold it. This deterministic controller supports contact inspection. The preview performs no learning. `--scene drum` also selects pickup.

The strike scene exposes **Reference strike (zero arm residual)**. It executes the nominal strike reference, displays measured impacts and grasp retention, and reports the timing/speed task result. Reset starts a new prepared episode.

The stick is a free body with a 0.30 m capsule centerline, 0.006 m radius, and 0.04 kg mass. Grasping uses physical contacts with both fingers. The drum is a rigid circular contact surface. Contact parameters require calibration against the intended stick, gripper pads, drum, and mounting.

Both Gymnasium environments use eight actions at 50 Hz: seven joint-target increments and one gripper opening command.

| Environment | Observation | Task |
| --- | --- | --- |
| `OpenArm-StickPickup-v0` | 42 values | Establish two-finger contact, lift the stick by more than 0.10 m, and hold it |
| `OpenArm-DrumStrike-v0` | 39 values | Strike at the requested time and speed, retain the grasp, and recover |

Strike reset executes the physical reference pickup and an IK transport to the preparation pose, then checks the grasp and tip position. Hit events use actual stick/drum contacts and measured forces. Neither environment contains an attachment constraint between the gripper and stick.

The implementation uses OpenArm's parallel gripper. DexDrummer's dexterous-hand controller, residual policy, and rewards have not been reproduced here.

`OpenArm-ResidualDrumStrike-v0` adds a nominal striking trajectory, bounded arm residuals, contact curriculum, and physical randomization. It has 51 observation values and eight actions. Training and evaluation are explicit CLI operations through `openarm-train-drum`. The default operation validates the environment without learning; training requires `--train`. No training starts from the GUI, and no trained pickup or strike policy is included.

## Installation

The package uses shared robot assets from this checkout and should be installed in editable mode. From the project directory:

```bash
mkdir -p ../../runs/cartesian_console/cache
TMPDIR="$PWD/../../runs/cartesian_console/cache" conda env create -f environment.yml
conda activate mujoco
```

For an existing environment:

```bash
python -m pip install -e ../.. -e '.[dev,agent]'
```

RL uses optional `rl` dependencies and PyTorch. CUDA PyTorch is installed in the current environment for cuRobo. RL training performance has not been measured.

## Verification

```bash
python -m pytest -q
openarm-reach --smoke-test
openarm-reach --scene drum --smoke-test
openarm-reach --scene strike --smoke-test
```

Tests use real MuJoCo rendering, IK, actuators, and contacts. The reach GUI check generates a mouse-click event and checks TCP convergence. The pickup GUI check invokes the reference-controller button and checks physical lift and finger contacts. No visual inspection is performed by the assistant.

## VLM tool interface

`SimulationTools` exposes `observe`, `get_state`, and `reach_pixel`. Observation includes the image, a `frame_id`, camera calibration, and measured robot state. A pixel command must reference the current frame. Tool arguments are checked with JSON Schema, and each execution returns its measured result.

The `openarm-agent` entry point connects to an OpenAI-compatible Chat Completions service supporting images and function tools. It requires an explicit endpoint, model, and `OPENARM_API_KEY`. Runtime traces are written under repository `runs/cartesian_console/agent/` by default. This remote model loop has not been tested because those service settings are not configured.
