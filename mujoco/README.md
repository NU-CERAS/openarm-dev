# OpenARM MuJoCo

Standalone MuJoCo simulations and controller examples for OpenARM. Runs locally, independently of the repository's ROS 2/Docker setup; integration can be added later.

## Structure

- `models/` — OpenARM v0.3 MJCF models (`single_arm.xml`, `bimanual.xml`) and required `meshes/` assets
- `examples/` — Python controllers (`actuator_motion.py`, `sim_to_real.py`, `zero_arms.py`; see each script for details)

Use lowercase snake_case for new files and folders, with descriptive names instead of example numbers.

## Setup

Install Python and create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install mujoco numpy
```

## Run

From the repository root, for example:

**macOS** (use `mjpython` for the passive viewer):

```bash
mjpython mujoco/examples/actuator_motion.py
```

**Linux / Windows:**

```bash
python mujoco/examples/actuator_motion.py
```

**VS Code:** Select the virtual environment as the Python interpreter, then use **Tasks: Run Task → Run MuJoCo** if the task is configured. Open the desired script before running the task.

Keep the `models/meshes/` directory with the model XML files; the models refer to these assets by relative path.

## Notes

- Controllers run directly in Python, without ROS 2 or Docker.
- The `v0.3` model assets originate from the upstream OpenARM MuJoCo project. Retain their copyright and license notices (Apache 2.0) when redistributing them.

## Inverse kinematics

See [the checked IK move guide](examples/IK_README.md) for planning, previewing,
and executing a small Cartesian move. `examples/ik_move.py` defaults to planning
only; real-arm motion requires `--live` and successful model and hardware checks.
