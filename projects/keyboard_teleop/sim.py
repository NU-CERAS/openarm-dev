#!/usr/bin/env python3
"""Jog both OpenArm v1.0 arms with keys pressed in the MuJoCo viewer window.

  1-7          select joint 1-7          8   select gripper
  Up / Down    move the selected joint; hold to keep moving (gripper: open / close)
  Left/Right   select the left / right arm
  - / =        slower / faster
  Backspace    home both arms

macOS: mjpython projects/keyboard_teleop/sim.py
Linux: python projects/keyboard_teleop/sim.py
"""
import argparse
import collections
import sys
import time
import glfw
import numpy as np
import mujoco
import mujoco.viewer
from openarm_tools.robot.config import load, root

ARMS = ('left', 'right')
# Seven arm joints followed by the gripper; index 7 is the gripper.
NAMES = [f'joint {i}' for i in range(1, 8)] + ['gripper']
STEPS = [0.01, 0.02, 0.05, 0.1, 0.2]  # radians per tap
HOLD_DELAY = 0.25  # seconds a key is held before continuous motion starts
HOLD_RATE = 10  # steps per second while held
GRIPPER_SCALE = 0.05  # metres of finger travel per radian of step
CONTROLS = ('1-7  /  8\nUp / Down\nLeft / Right\n-  /  =\nBackspace',
            'select joint / gripper\nmove (hold for continuous)\nselect arm\nslower / faster\nhome both arms')
# GLFW key codes, as passed to key_callback.
KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_BACKSPACE = (
    glfw.KEY_UP, glfw.KEY_DOWN, glfw.KEY_LEFT, glfw.KEY_RIGHT, glfw.KEY_BACKSPACE)


class Teleop:
    def __init__(self, robot):
        cfgs = {arm: load(robot=robot, arm=arm) for arm in ARMS}
        self.model = mujoco.MjModel.from_xml_path(str(root() / cfgs['left']['model']))
        self.data = mujoco.MjData(self.model)
        self.actuators = {arm: [self.model.actuator(f'{arm}_joint{i}_ctrl').id for i in range(1, 8)]
                          + [self.model.actuator(f'{arm}_finger1_ctrl').id] for arm in ARMS}
        # Arm targets obey the reviewed config limits; the gripper uses the model's control range.
        self.limits = {arm: [tuple(j['limits']) for j in cfgs[arm]['joints']]
                       + [tuple(self.model.actuator_ctrlrange[self.actuators[arm][7]])] for arm in ARMS}
        self.arm, self.index, self.step_index = 'left', 0, 2
        self.held = None  # (key, press time) of the arrow key last pressed
        self.home()

    def home(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        self.start = time.monotonic()

    def jog(self, sign, steps=1.0):
        aid = self.actuators[self.arm][self.index]
        lo, hi = self.limits[self.arm][self.index]
        delta = sign * steps * STEPS[self.step_index] * (GRIPPER_SCALE if self.index == 7 else 1)
        self.data.ctrl[aid] = float(np.clip(self.data.ctrl[aid] + delta, lo, hi))

    def press(self, key):
        if ord('1') <= key <= ord('8'):
            self.index = key - ord('1')
        elif key in (KEY_UP, KEY_DOWN):
            self.jog(1 if key == KEY_UP else -1)
            self.held = (key, time.monotonic())
        elif key in (KEY_LEFT, KEY_RIGHT):
            self.arm = 'left' if key == KEY_LEFT else 'right'
        elif key == ord('-'):
            self.step_index = max(self.step_index - 1, 0)
        elif key == ord('='):
            self.step_index = min(self.step_index + 1, len(STEPS) - 1)
        elif key == KEY_BACKSPACE:
            self.home()

    def hold(self, is_down, dt):
        """Keep moving while the last arrow key pressed stays down, after the initial tap step."""
        if self.held is None:
            return
        key, pressed = self.held
        if not is_down(key):
            self.held = None
        elif time.monotonic() - pressed > HOLD_DELAY:
            self.jog(1 if key == KEY_UP else -1, HOLD_RATE * dt)

    def overlay(self):
        aid = self.actuators[self.arm][self.index]
        qpos = self.data.qpos[self.model.jnt_qposadr[self.model.actuator_trnid[aid, 0]]]
        lo, hi = self.limits[self.arm][self.index]
        unit = 'm' if self.index == 7 else 'rad'
        labels = 'Arm\nJoint\nTarget\nActual\nLimits\nSpeed'
        values = (f'{self.arm}\n{NAMES[self.index]}\n{self.data.ctrl[aid]:+.3f} {unit}\n'
                  f'{qpos:+.3f} {unit}\n[{lo:+.2f}, {hi:+.2f}]\n'
                  f'{STEPS[self.step_index]:.2f} rad/tap, {STEPS[self.step_index] * HOLD_RATE:.1f} rad/s held')
        return labels, values

    def tick(self):
        """Step physics up to wall time, dropping (not replaying) any lag above 50 ms."""
        lag = time.monotonic() - self.start - self.data.time
        if lag > 0.05:
            self.start += lag - 0.05
        while self.data.time < time.monotonic() - self.start:
            mujoco.mj_step(self.model, self.data)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--robot', default='v1.0', choices=['v1.0'])
    args = parser.parse_args(argv)
    if sys.platform == 'darwin' and getattr(mujoco.viewer, '_MJPYTHON', None) is None:
        raise SystemExit('On macOS run this with mjpython instead of python.')

    teleop = Teleop(args.robot)
    keys, window = collections.deque(), []

    def on_key(key):
        # Runs on the viewer's render thread, where its GLFW window is current. The viewer
        # shares the glfw module's library, so the window lets us poll held keys later.
        if not window:
            window.append(glfw.get_current_context())
        keys.append(key)

    def is_down(key):
        return bool(window and window[0]) and glfw.get_key(window[0], key) == glfw.PRESS

    print(__doc__)
    with mujoco.viewer.launch_passive(teleop.model, teleop.data, key_callback=on_key) as viewer:
        # The viewer also binds 0-5 to geom-group visibility; pin groups so joint selection never hides the robot.
        geomgroup, shown, last = viewer.opt.geomgroup.copy(), None, time.monotonic()
        while viewer.is_running():
            now = time.monotonic()
            with viewer.lock():
                # Apply keys here, under the lock, rather than on the render thread.
                while keys:
                    teleop.press(keys.popleft())
                teleop.hold(is_down, now - last)
                viewer.opt.geomgroup[:] = geomgroup
                teleop.tick()
                text = teleop.overlay()
            last = now
            viewer.sync()
            # set_texts blocks until the render thread draws, which needs the lock: never call it while holding it.
            if text != shown:
                viewer.set_texts([(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPRIGHT, *CONTROLS),
                                  (mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, *text)])
                shown = text
            time.sleep(0.002)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
