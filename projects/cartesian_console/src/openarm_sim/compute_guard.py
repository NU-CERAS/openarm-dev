import argparse
import fcntl
from functools import wraps
import json
import os
import signal
import subprocess
import sys
import time
import uuid

import psutil

from .paths import RUNS


GIB = 1024**3


def check_resources():
    if psutil.virtual_memory().available < 8 * GIB:
        raise MemoryError("Stopping this compute task to preserve 8 GiB of available system memory")
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        free, _ = torch.cuda.mem_get_info()
        if free < 4 * GIB:
            raise MemoryError("Stopping this compute task to preserve 4 GiB of free GPU memory")


def budgeted(module):
    def decorate(function):
        @wraps(function)
        def run(*args, **kwargs):
            if "--help" in sys.argv or "-h" in sys.argv:
                return function(*args, **kwargs)
            if os.environ.get("OPENARM_COMPUTE_GUARDED") != "1":
                os.execv(sys.executable, [sys.executable, "-m", "openarm_sim.compute_guard",
                                         sys.executable, "-m", module, *sys.argv[1:]])
            torch = sys.modules.get("torch")
            if torch is not None:
                torch.set_num_threads(1)
                if torch.cuda.is_available():
                    torch.cuda.set_per_process_memory_fraction(0.5)
            check_resources()
            return function(*args, **kwargs)
        return run
    return decorate


def interrupt(signum, frame):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description="Run one compute task within desktop resource limits")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command:
        parser.error("A command is required")
    RUNS.mkdir(parents=True, exist_ok=True)
    cache = RUNS / "cache"
    cache.mkdir(exist_ok=True)
    signal.signal(signal.SIGTERM, interrupt)
    with (RUNS / "compute.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        check_resources()
        memory = min(14 * GIB, psutil.virtual_memory().available - 8 * GIB)
        cores = min(8, max(1, len(os.sched_getaffinity(0)) // 2))
        unit = f"openarm-compute-{uuid.uuid4().hex[:8]}"
        environment = dict(os.environ, OPENARM_COMPUTE_GUARDED="1", OMP_NUM_THREADS="1",
                           OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
                           TMPDIR=str(cache))
        command = ["systemd-run", "--user", "--scope", "--quiet", "--unit", unit,
                   "-p", f"MemoryMax={memory}", "-p", f"MemoryHigh={min(memory, 12 * GIB)}",
                   "-p", "MemorySwapMax=0", "-p", f"CPUQuota={cores * 100}%",
                   "/usr/bin/nice", "-n", "10", *args.command]
        state = {"active": True, "pid": os.getpid(), "unit": unit, "command": args.command,
                 "memory_limit_bytes": memory, "cpu_cores": cores, "system_memory_reserve_bytes": 8 * GIB,
                 "gpu_memory_reserve_bytes": 4 * GIB}
        status = RUNS / "active_compute.json"
        status.write_text(json.dumps(state, indent=2) + "\n")
        process = subprocess.Popen(command, env=environment)
        try:
            while process.poll() is None:
                check_resources()
                time.sleep(1)
            code = process.returncode
        finally:
            if process.poll() is None:
                subprocess.run(["systemctl", "--user", "stop", unit], check=True)
                process.wait()
            state.update(active=False, exit_code=process.returncode)
            status.write_text(json.dumps(state, indent=2) + "\n")
        if code:
            raise subprocess.CalledProcessError(code, args.command)


if __name__ == "__main__":
    main()
