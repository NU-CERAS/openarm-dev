# OpenARM MuJoCo

Standalone MuJoCo simulations and controller examples for OpenARM. Runs locally, independently of the repository's ROS 2/Docker setup; integration can be added later.

## Structure

- `v0.3/` — OpenARM MJCF models and required `meshes/` assets
- `Robot Controller Examples/` — Python controllers (see each script for details)

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
mjpython "mujoco/Robot Controller Examples/example1: basic actuator motion.py"
```

**Linux / Windows:**

```bash
python "mujoco/Robot Controller Examples/example1: basic actuator motion.py"
```

**VS Code:** Select the virtual environment as the Python interpreter, then use **Tasks: Run Task → Run MuJoCo** if the task is configured. Open the desired script before running the task.

Keep the `v0.3/meshes/` directory with the model XML files; the models refer to these assets by relative path.

## Notes

- Controllers run directly in Python, without ROS 2 or Docker.
- The `v0.3` model assets originate from the upstream OpenARM MuJoCo project. Retain their copyright and license notices (Apache 2.0) when redistributing them.
