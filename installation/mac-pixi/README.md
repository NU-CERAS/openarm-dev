# Installing ROS 2 Jazzy + opening MuJoCo on macOS (via pixi / RoboStack)

Works on Apple Silicon and Intel Macs. ROS 2 is installed from RoboStack (conda-forge-based packages) using **pixi**, which creates an isolated, reproducible environment per project. No venv, Homebrew ROS, or source build needed.

---

## 1. Pick a project location

This could be anywhere, but feel free to put any experimentation in this repo, under autonomy-dev.

Note: put the project in a plain path, **not** in an iCloud-synced folder (e.g. `~/Documents` with Desktop & Documents sync on).

```bash
mkdir -p ~/dev && cd ~/dev
```

## 2. Use zsh (recommended)

macOS ships bash **3.2** (2007). `pixi shell` activation scripts fail under it (symptom: `Did not detect successful shell initialization`, prompt becomes `bash-3.2$`). Switch to zsh:

```bash
chsh -s /bin/zsh
```

Close and reopen the terminal.

> If you stay on bash: avoid `pixi shell` and use `pixi run <command>` for everything.

## 3. Install pixi

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

Make sure `~/.pixi/bin` is on your PATH, then reload:

```bash
# zsh
echo 'export PATH="$HOME/.pixi/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# bash (macOS login shells read .bash_profile, not .bashrc)
echo 'export PATH="$HOME/.pixi/bin:$PATH"' >> ~/.bash_profile
source ~/.bash_profile
```

Verify:

```bash
pixi --version
```

## 4. Create the project with ROS 2 Jazzy + MuJoCo

```bash
pixi init ros_mujoco -c robostack-jazzy -c conda-forge
cd ros_mujoco
pixi add ros-jazzy-desktop ros-dev-tools colcon-common-extensions mujoco python
```

## 5. Verify the install

All `pixi` commands must be run **inside the folder containing `pixi.toml`**.

```bash
pixi run printenv ROS_DISTRO          # -> jazzy
pixi run ros2 run demo_nodes_cpp talker   # Ctrl+C to stop
pixi run python -c "import mujoco; print(mujoco.__version__)"
pixi run python -m mujoco.viewer      # opens an empty viewer window
```

Optional: test pub/sub across terminals by running `pixi run ros2 run demo_nodes_py listener` in a second terminal while the talker runs.

## 6. Minimal MuJoCo → ROS 2 bridge

Save as `sim_node.py` in `ros_mujoco/`. It simulates a pendulum and publishes `sensor_msgs/JointState`.

```python
import mujoco, rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

XML = """<mujoco><worldbody><body>
  <joint name="hinge" type="hinge" axis="0 1 0"/>
  <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.05"/>
</body></worldbody></mujoco>"""

class SimNode(Node):
    def __init__(self):
        super().__init__('mujoco_sim')
        self.m = mujoco.MjModel.from_xml_string(XML)
        self.d = mujoco.MjData(self.m)
        self.d.qpos[0] = 1.0  # start tilted so it swings
        self.pub = self.create_publisher(JointState, 'joint_states', 10)
        self.create_timer(self.m.opt.timestep, self.step)

    def step(self):
        mujoco.mj_step(self.m, self.d)
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['hinge']
        msg.position = [float(self.d.qpos[0])]
        msg.velocity = [float(self.d.qvel[0])]
        self.pub.publish(msg)

rclpy.init()
rclpy.spin(SimNode())
```

Run and check:

```bash
# terminal 1
pixi run python sim_node.py

# terminal 2
pixi run ros2 topic echo /joint_states
```

Oscillating `position` values = working bridge.

## 7. Next steps

1. Load a real robot: `mujoco.MjModel.from_xml_path("robot.xml")`.
2. Add a subscriber that writes commands into `self.d.ctrl` so ROS can drive actuators.
3. When it outgrows one file, make a ROS 2 package under `src/` and build with `colcon build`.
4. Viewer + ROS together: on macOS, `mujoco.viewer.launch_passive()` must run under `mjpython` (e.g. `pixi run mjpython sim_node.py`), not `python`.

## 8. Repo setup

Commit:

- `pixi.toml`
- `pixi.lock` (makes the env reproducible: `pixi install` recreates it exactly)
- `src/` and your scripts

`.gitignore`:

```
.pixi/
build/
install/
log/
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `pixi: command not found` | `~/.pixi/bin` not on PATH, or written to a file your shell doesn't read | Step 3; check `ls ~/.pixi/bin` exists, reinstall if not |
| `bash-3.2$` prompt / shell init warning after `pixi shell` | macOS bash 3.2 too old | `exit`, switch to zsh, or use `pixi run` |
| `could not find pixi.toml` | Running from the wrong directory | `cd` into the project folder |
| Env breaks after moving the folder | `.pixi/` contains absolute paths | `rm -rf .pixi && pixi install` |
| `colcon build` fails with weird path errors | Spaces or `;` in path | Move project to e.g. `~/dev` |
| Viewer crashes / errors with `launch_passive` | macOS requires main-thread GUI | Use `mjpython` instead of `python` |
| Should I make a venv? | No | pixi already isolates the env; a venv conflicts with ROS's Python |
