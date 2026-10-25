"""Launch the existing ROS-control integration in an activated workspace."""
import shutil
import subprocess


def launch(headless=False):
    if not shutil.which('ros2'):
        raise RuntimeError('Activate ROS Jazzy and source the built workspace first.')
    subprocess.run(['ros2', 'launch', 'openarm_mujoco_sim', 'openarm_mujoco.launch.py',
                    f'headless:={str(headless).lower()}'], check=True)
