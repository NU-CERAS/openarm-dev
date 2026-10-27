import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import rclpy


def main():
    parser = argparse.ArgumentParser(description="Install Cartesian Console into the activated root ROS environment")
    parser.add_argument("--hardware", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    project = root / "projects/cartesian_console"
    cache = root / "runs/cartesian_console/cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(cache)
    os.environ["PIP_CACHE_DIR"] = str(cache / "pip")
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("The root console profile requires Python 3.12")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(project / "requirements-ubuntu-cpu.txt")], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-build-isolation", "--no-deps", "-e", str(root),
                    "-e", f"{project}[dev,video]"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    if args.hardware:
        import openarm_can

        print(f"CAN binding: {openarm_can.__file__}", flush=True)
        destination = root / "runs/cartesian_console/sync"
        destination.mkdir(parents=True, exist_ok=True)
        for arm in ("left", "right"):
            path = destination / f"{arm}.json"
            if not path.exists():
                shutil.copyfile(root / f"robots/v1.0/config/{arm}.json", path)
        path = destination / "hardware.json"
        if not path.exists():
            values = json.loads((project / "configs/sync_hardware.example.json").read_text())
            for arm in ("left", "right"):
                values["arms"][arm]["config"] = f"{arm}.json"
            path.write_text(json.dumps(values, indent=2) + "\n")
    print(f"Console installed with ROS: {rclpy.__file__}", flush=True)
    print("Ready: bash projects/cartesian_console/scripts/run_demo.sh", flush=True)


if __name__ == "__main__":
    main()
