import argparse
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parents[1]


def run(command: list[str]) -> None:
    print(f"$ {shlex.join(command)}", flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Install the Ubuntu 24.04 CPU console environment")
    parser.add_argument("--python", default="/usr/bin/python3", help="Python 3.12 interpreter used to create the environment")
    parser.add_argument("--hardware", action="store_true", help="Also install the pinned CAN binding and create connection templates")
    parser.add_argument("--skip-system-packages", action="store_true", help="Use system dependencies already installed by the administrator")
    args = parser.parse_args()
    release = platform.freedesktop_os_release()
    if release.get("ID") != "ubuntu" or release.get("VERSION_ID") != "24.04" or platform.machine() != "x86_64":
        raise RuntimeError("This installer supports Ubuntu 24.04 x86_64")
    cache = ROOT / "runs/cartesian_console/cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(cache)
    os.environ["PIP_CACHE_DIR"] = str(cache / "pip")
    packages = ["python3-venv", "python3-tk", "libgl1", "libglx-mesa0", "libgl1-mesa-dri",
                "fonts-dejavu-core", "git", "curl", "ca-certificates"]
    if args.hardware:
        packages += ["build-essential", "cmake", "libcli11-dev"]
    if not args.skip_system_packages:
        missing = []
        for package in packages:
            result = subprocess.run(["dpkg-query", "-W", "-f=${Status}", package], capture_output=True, text=True)
            if result.returncode != 0 or result.stdout.strip() != "install ok installed":
                missing.append(package)
        if missing:
            prefix = [] if os.geteuid() == 0 else ["sudo"]
            run([*prefix, "apt-get", "update"])
            run([*prefix, "apt-get", "install", "-y", *missing])
    base_python = shutil.which(args.python)
    if base_python is None:
        raise FileNotFoundError(f"Python interpreter not found: {args.python}")
    run([base_python, "-c", "import sys, tkinter; assert sys.version_info[:2] == (3, 12), 'Python 3.12 is required'"])
    environment = Path(os.environ.get("OPENARM_ENV", ROOT / "runs/cartesian_console/env"))
    environment = (environment if environment.is_absolute() else ROOT / environment).resolve()
    if environment.exists() and not (environment / "pyvenv.cfg").is_file():
        raise ValueError(f"Environment destination already exists without pyvenv.cfg: {environment}")
    if not environment.exists():
        run([base_python, "-m", "venv", str(environment)])
    python = str(environment / "bin/python")
    run([python, "-c", "import sys, tkinter; assert sys.version_info[:2] == (3, 12), 'Environment must use Python 3.12'"])
    run([python, "-m", "pip", "install", "--disable-pip-version-check", "--only-binary=:all:",
         "-r", str(PROJECT / "requirements-ubuntu-cpu.txt")])
    run([python, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "-e", str(ROOT),
         "-e", f"{PROJECT}[dev,video]"])
    run([python, "-m", "pip", "check"])
    if args.hardware:
        os.environ["CMAKE_BUILD_PARALLEL_LEVEL"] = "4"
        run([python, str(PROJECT / "scripts/install_can.py")])
        target = ROOT / "runs/cartesian_console/sync"
        target.mkdir(parents=True, exist_ok=True)
        sources = {"hardware.json": PROJECT / "configs/sync_hardware.example.json",
                   "left.json": ROOT / "robots/v1.0/config/left.json", "right.json": ROOT / "robots/v1.0/config/right.json"}
        for name, source in sources.items():
            destination = target / name
            if not destination.exists():
                if name == "hardware.json":
                    configuration = json.loads(source.read_text())
                    for arm in ("left", "right"):
                        configuration["arms"][arm]["config"] = f"{arm}.json"
                    destination.write_text(json.dumps(configuration, indent=2) + "\n")
                else:
                    shutil.copyfile(source, destination)
        print(f"Fill both CAN interfaces and review the arm configurations under {target}", flush=True)
    os.environ["OPENARM_PYTHON"] = python
    run(["bash", str(PROJECT / "scripts/check_setup.sh"), *(["--hardware"] if args.hardware else [])])
    with (environment / "installed-packages.txt").open("w") as output:
        subprocess.run([python, "-m", "pip", "freeze", "--exclude-editable"], check=True, stdout=output)
    report = {"python": python, "base_python": base_python, "hardware_setup_requested": args.hardware,
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    (environment / "setup_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Ready: bash projects/cartesian_console/scripts/run_demo.sh", flush=True)


if __name__ == "__main__":
    main()
