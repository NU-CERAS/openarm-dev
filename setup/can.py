import os
from pathlib import Path
import subprocess
import sys

import yaml


def main():
    root = Path(__file__).resolve().parents[1]
    pin = yaml.safe_load((root / "dependencies/upstream.repos").read_text())["repositories"]["openarm_can"]
    source = root / "src/vendor/openarm_can"
    if not source.exists():
        subprocess.run(["git", "init", str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "remote", "add", "origin", pin["url"]], check=True)
        subprocess.run(["git", "-C", str(source), "fetch", "--depth", "1", "origin", pin["version"]], check=True)
        subprocess.run(["git", "-C", str(source), "switch", "--detach", "FETCH_HEAD"], check=True)
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if head != pin["version"]:
        raise ValueError(f"CAN source differs from dependencies/upstream.repos: {source}")
    changes = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True)
    if changes.strip():
        raise ValueError(f"CAN source has local changes; review them before installation: {source}")
    os.environ["CMAKE_BUILD_PARALLEL_LEVEL"] = "4"
    subprocess.run([sys.executable, "-m", "pip", "install", str(source / "python")], check=True)
    import openarm_can

    print(f"Pinned CAN binding ready: {openarm_can.__file__}", flush=True)


if __name__ == "__main__":
    main()
