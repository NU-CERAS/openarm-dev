from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[3]
pin = yaml.safe_load((ROOT / "dependencies/upstream.repos").read_text())["repositories"]["openarm_can"]
source = ROOT / "runs/cartesian_console/dependencies/openarm_can"
if not source.exists():
    source.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "remote", "add", "origin", pin["url"]], check=True)
    subprocess.run(["git", "-C", str(source), "fetch", "--depth", "1", "origin", pin["version"]], check=True)
    subprocess.run(["git", "-C", str(source), "switch", "--detach", "FETCH_HEAD"], check=True)
head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
if head != pin["version"]:
    raise ValueError(f"CAN source differs from dependencies/upstream.repos: {source}")
subprocess.run([sys.executable, "-m", "pip", "install", str(source / "python")], check=True)
subprocess.run([sys.executable, "-c", "import openarm_can; print('CAN binding import passed')"], check=True)
