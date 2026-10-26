# Start Cartesian Console from a fresh checkout

The primary tested platform is an Ubuntu 24.04 x86_64 desktop. The default
tabletop demo uses CPU OMPL planning, MuJoCo physics and a local Tk interface.
It requires no camera, ROS, CUDA package, trained checkpoint or downloaded drum
asset. Run these commands in Bash.

Use a graphical desktop login with systemd user services available. The shared
compute guard uses systemd-run --user to enforce memory and CPU limits.

## 1. Python environment

The desktop needs Git, curl, OpenGL/Mesa libraries and DejaVu fonts. On Ubuntu:

~~~bash
sudo apt-get update
sudo apt-get install git curl libgl1 libglx-mesa0 libgl1-mesa-dri fonts-dejavu-core
~~~

Install Miniforge using its [official installation instructions](https://github.com/conda-forge/miniforge).
For a fresh Linux x86_64 machine:

~~~bash
curl -L -o Miniforge3.sh \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda create -n openarm-console --override-channels -c conda-forge python=3.12 tk pip
conda activate openarm-console
~~~

If Miniforge is already installed, activate its shell integration and create
the new environment. Keep your existing installation.

## 2. Get the branch and install

While this contribution is under review:

~~~bash
git clone --single-branch --branch feat/jixin \
  https://github.com/jixinyan/openarm-dev.git
cd openarm-dev
mkdir -p runs/cartesian_console/cache
export TMPDIR="$PWD/runs/cartesian_console/cache"
python -m pip install -e . -e './projects/cartesian_console[dev,video]'
~~~

The CPU demo uses the robot models committed under robots/v1.0/; it needs no
vendor submodule installation. The shared tools and independent project package
must both be installed in editable mode.

## 3. Open the interface

~~~bash
bash projects/cartesian_console/scripts/run_sync.sh
~~~

The script uses the active environment's python3. OPENARM_PYTHON selects an
explicit interpreter. It starts the independent MuJoCo robot, feedback transport,
live twin, motion planner and interface together. Closing the window releases
the process and its compute slot.

1. Select an arm.
2. Drag its TCP's red X, green Y or blue Z arrow. Release to preview.
3. Inspect the preview, then choose **Move robot**.
4. **Esc** cancels target dragging; **Stop** cancels robot motion.
5. Save a reached position under **Saved positions**, then load it to replan.

Coordinates are centimeters from the torso's top-center point. X is forward,
Y is left and Z is up. For the right arm, (40, -25, -30) cm is the default
target. The fixed origin and axes are marked in the viewport. It remains fixed
while either arm moves.

## 4. CPU software rendering and checks

Select Mesa software rendering explicitly when needed:

~~~bash
__GLX_VENDOR_LIBRARY_NAME=mesa LIBGL_ALWAYS_SOFTWARE=1 MUJOCO_GL=glfw \
  bash projects/cartesian_console/scripts/run_sync.sh
~~~

Close the GUI before starting tests; both use the shared compute slot.

~~~bash
CUDA_VISIBLE_DEVICES="" __GLX_VENDOR_LIBRARY_NAME=mesa LIBGL_ALWAYS_SOFTWARE=1 \
  MUJOCO_GL=glfw python -m openarm_sim.compute_guard \
  python -m pytest -q projects/cartesian_console/tests/test_gizmo.py \
  projects/cartesian_console/tests/test_sync.py \
  projects/cartesian_console/tests/test_reach.py \
  projects/cartesian_console/tests/test_ompl.py
~~~

An automatic check opens the actual interface, selects a tabletop point, plans,
executes in MuJoCo, saves/loads a position and closes:

~~~bash
bash projects/cartesian_console/scripts/run_sync.sh --smoke-test \
  --targets runs/cartesian_console/smoke/targets.json
~~~

## 5. Record the feature showcase

For the isolated GUI recorder, install Xvfb and xauth:

~~~bash
sudo apt-get install xvfb xauth
openarm-showcase --output runs/cartesian_console/showcase
~~~

The command opens an authenticated private display, performs actual Tk mouse
and button actions, records both-arm execution and position persistence, and
appends native OMPL obstacle avoidance, reference pickup and reference strike
episodes. It writes openarm_showcase.mp4, reports and traces. The output
directory must be new. Use --xvfb /path/to/Xvfb for an explicit executable.

The pickup/strike scenes demonstrate physical reference controllers and learning
environment interfaces. They start no training and execute no physical hardware.
The recorder captures only its private display's application window.

## 6. Optional capabilities

| Capability | Setup / command |
| --- | --- |
| cuRobo CUDA planning | Install uv, then run scripts/setup_curobo.sh; select --planner curobo |
| Native pickup preview | openarm-reach --scene pickup |
| Native strike preview | openarm-reach --scene strike |
| RL / ACT / distillation | Install rl,gpu-sim,tracking,imitation extras and use the training commands in the README |
| W&B | Install tracking, then authenticate with wandb login |
| Electronic drum asset | Install assets, then import your independently downloaded archive |
| VLM tool calls | Install agent and configure an OpenAI-compatible image/tool endpoint |
| Physical robot | Fill the reviewed CAN connection configuration described below |

The optional cuRobo setup script is run from the repository root:

~~~bash
python -m pip install uv
bash projects/cartesian_console/scripts/setup_curobo.sh
openarm-sync --planner curobo
~~~

Training data, policies, API credentials and imported third-party assets are
supplied separately. The source branch contains training/evaluation tools.

## 7. Physical robot connection

Use the [synchronization guide](sync-console.md) to fill the
configs/sync_hardware.example.json template. Provide both real CAN interface
names and reviewed v1.0 motor configurations. The adapter commands seven arm
joints; the grippers remain unchanged.

~~~bash
bash projects/cartesian_console/scripts/run_sync.sh \
  --backend can --hardware-config runs/cartesian_console/sync/hardware.json \
  --supported
~~~

The shared hardware backend releases motor torque at trajectory completion,
cancellation and errors; mechanically support both arms. Current CAN feedforward
torque is zero. Physical gains, zero/sign conventions and tracking must be
commissioned on the actual hardware. Physical CAN motor execution remains
unverified in this contribution.

Use direct observation to adjust sim coordinates through trials. Collision
checks cover the configured sim geometry; the interface has no object perception.
