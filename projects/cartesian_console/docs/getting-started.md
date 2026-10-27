# Start Cartesian Console from a fresh checkout

The primary tested platform is an Ubuntu 24.04 x86_64 desktop. The default
tabletop demo uses CPU OMPL planning, MuJoCo physics and a local Tk interface.
It requires no camera, ROS, CUDA package, trained checkpoint or downloaded drum
asset. Run these commands in Bash.

Use a graphical desktop login with systemd user services available. The shared
compute guard uses systemd-run --user to enforce memory and CPU limits.
The guard requires at least 8 GiB of available memory throughout execution.

## 1. Get the branch

Install Git if needed:

~~~bash
sudo apt-get update
sudo apt-get install -y git
~~~

While this contribution is under review:

~~~bash
git clone --single-branch --branch feat/jixin \
  https://github.com/jixinyan/openarm-dev.git
cd openarm-dev
~~~

## 2. Install the environment

~~~bash
bash projects/cartesian_console/scripts/setup.sh
~~~

This command installs missing Ubuntu system packages using sudo, creates a
private Python 3.12 environment at `runs/cartesian_console/env/`, installs the
exact CPU package versions from `requirements-ubuntu-cpu.txt`, installs both
editable project packages, and checks dependencies, actual OMPL reaches,
feedback synchronization, Tk and rendering. Repeating setup reuses the
environment. Existing hardware configuration files are preserved.

The CPU demo uses committed models under `robots/v1.0/`. The installation needs
Internet access. Conda, ROS and CUDA are optional. The script requires Ubuntu
24.04 x86_64; macOS is not supported by this installer.

Administrators who have already installed the system packages may use
`--skip-system-packages`. `--python /path/to/python3.12` selects another Python
3.12 interpreter with Tk. `OPENARM_ENV` selects an explicit environment directory;
use the same value for setup and subsequent launcher commands. For an existing
manually installed environment, set `OPENARM_PYTHON` to its interpreter.

| Script | Purpose |
| --- | --- |
| `scripts/setup.sh` | Install the CPU environment and perform native checks |
| `scripts/setup.sh --hardware` | Also install the pinned CAN binding and create hardware templates |
| `scripts/check_setup.sh` | Check dependencies, graphics and two feedback-driven reaches |
| `scripts/check_setup.sh --gui` | Also check both actual demo windows and automatic sync |
| `scripts/run_demo.sh` | Open the console and independent MuJoCo test robot |
| `scripts/run_robot.sh --supported` | Connect the configured physical robot and open the console |

Script paths in the table are relative to `projects/cartesian_console/`.

## 3. Open the interface

~~~bash
bash projects/cartesian_console/scripts/run_demo.sh
~~~

The launchers select the installed private environment automatically. The demo
opens the console and a separate-process MuJoCo robot window. Closing the console
closes the robot process and releases the compute slot.

1. Select an arm.
2. Drag its TCP's red X, green Y or blue Z arrow. Release to preview the target pose.
3. Choose **Execute action**. The preview holds the target during motion and
   automatically synchronizes its pose, TCP arrows and coordinates from measured
   feedback at completion. Drag again from that synchronized pose to repeat.
4. **Esc** cancels target dragging; **Stop** cancels robot motion.
5. Save a reached position under **Saved positions**; **Preview saved position**
   plans that target from current feedback. Choose **Execute action** to execute it.

The additional MuJoCo window shows the independent demo robot. It receives
joint commands only after **Execute action**. The console renders its own
preview model and requires no camera.

Coordinates are centimeters from the torso's top-center point. X is forward,
Y is left and Z is up. For the right arm, (40, -25, -30) cm is a reachable
example. At startup the target fields show the current measured TCP; the offline
demo initializes both seven-joint arms at zero. Physical startup reads both
actual arm states using the reviewed joint zero/sign mappings. The fixed origin
and axes are marked in the viewport. It remains fixed
while either arm moves.

## 4. CPU software rendering and checks

Select Mesa software rendering explicitly when needed:

~~~bash
OPENARM_SOFTWARE_RENDERING=1 bash projects/cartesian_console/scripts/run_demo.sh
~~~

Close the GUI before starting tests; both use the shared compute slot.

~~~bash
bash projects/cartesian_console/scripts/check_setup.sh
bash projects/cartesian_console/scripts/check_setup.sh --gui
~~~

The `--gui` check opens the actual interface, drags a TCP axis, previews and
explicitly executes in MuJoCo, saves/reuses a named position and closes both windows.

## 5. Record the feature showcase

For the isolated GUI recorder, install Xvfb and xauth:

~~~bash
sudo apt-get install xvfb xauth
source runs/cartesian_console/env/bin/activate
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

### Learned-policy recording

Install the learning and video extras in the environment with your compatible
CUDA PyTorch installation:

~~~bash
python -m pip install -e './projects/cartesian_console[rl,gpu-sim,imitation,video,tracking]'
openarm-record-policy --checkpoint /path/to/pickup_student.pt \
  --bank /path/to/heldout_resets.npz \
  --strike-checkpoint /path/to/trained_strike_checkpoint \
  --console-video runs/cartesian_console/showcase/console.mp4 \
  --episodes 3 --output runs/cartesian_console/policy_demo
~~~

This recording runs a visual ACT pickup student distilled from an ACT+PPO
teacher, followed by a trained oracle residual-PPO strike actor. The pickup
camera panel shows the exact RGB supplied to the student, with all site markers
hidden. Its remaining inputs are seven arm positions, gripper position and
controller elapsed time. Strike uses 51 oracle values and a nominal reference;
its camera panel is an observer view. Each strike trial starts after the
reference reset establishes a physical grasp and preparation pose.

The strike checkpoint directory must contain policy.zip and vecnormalize.pkl,
and its saved PPO training-step count must be positive. Pickup resets use
consecutive indices starting at zero; strike uses consecutive evaluation seeds.
The recorder retains every requested result, including failures, in
demo_report.json and produces openarm_visual_demo.mp4. No training is started
by the recording command. Use --console-start and --console-duration to select
a segment from the actual interface recording.

The [policy demonstration release](https://github.com/jixinyan/openarm-dev/releases/tag/cartesian-policy-demo-20261026)
supplies the recorded policies and pickup reset bank. Download them under runs/:

~~~bash
mkdir -p runs/cartesian_console/policies
curl -fL -o runs/cartesian_console/policies/pickup_student.pt \
  https://github.com/jixinyan/openarm-dev/releases/download/cartesian-policy-demo-20261026/pickup_student.pt
curl -fL -o runs/cartesian_console/policies/pickup_resets.npz \
  https://github.com/jixinyan/openarm-dev/releases/download/cartesian-policy-demo-20261026/pickup_resets.npz
curl -fL -o runs/cartesian_console/policies/oracle_strike.zip \
  https://github.com/jixinyan/openarm-dev/releases/download/cartesian-policy-demo-20261026/oracle_strike.zip
python -m zipfile -e runs/cartesian_console/policies/oracle_strike.zip \
  runs/cartesian_console/policies/oracle_strike
openarm-record-policy \
  --checkpoint runs/cartesian_console/policies/pickup_student.pt \
  --bank runs/cartesian_console/policies/pickup_resets.npz \
  --strike-checkpoint runs/cartesian_console/policies/oracle_strike \
  --episodes 3 --output runs/cartesian_console/policy_replay
~~~

This records both policies without the optional TCP console clip. These small
demonstration samples describe the recorded resets; use separate evaluation
banks and larger seed sets to assess task robustness.

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

Install the optional binding and prepare the connection templates:

~~~bash
bash projects/cartesian_console/scripts/setup.sh --hardware
~~~

Use the [synchronization guide](sync-console.md) to fill the
configs/sync_hardware.example.json template. Provide both real CAN interface
names and reviewed v1.0 motor configurations. The adapter commands seven arm
joints; the grippers remain unchanged.

~~~bash
bash projects/cartesian_console/scripts/run_robot.sh --supported
~~~

The shared hardware backend releases motor torque at trajectory completion,
cancellation and errors; mechanically support both arms. Current CAN feedforward
torque is zero. Physical gains, zero/sign conventions and tracking must be
commissioned on the actual hardware. Physical CAN motor execution remains
unverified in this contribution.

Use direct observation to adjust sim coordinates through trials. Collision
checks cover the configured sim geometry; the interface has no object perception.
