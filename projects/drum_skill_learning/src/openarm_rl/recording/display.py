import os
from pathlib import Path
import secrets
import socket
import subprocess
import time


class NativeDisplay:
    def __init__(self, executable, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        number = None
        for candidate in range(220, 320):
            with socket.socket() as connection:
                if connection.connect_ex(("127.0.0.1", 6000 + candidate)) != 0:
                    number = candidate
                    break
        if number is None:
            raise RuntimeError("No free private display port")
        self.display = f"127.0.0.1:{number}.0"
        authority = directory / "display.auth"
        authority.touch(mode=0o600)
        subprocess.run(["xauth", "-f", str(authority), "add", self.display, "MIT-MAGIC-COOKIE-1", secrets.token_hex(16)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.log = (directory / "display.log").open("w")
        self.process = subprocess.Popen([str(executable), f":{number}", "-screen", "0", "1720x1080x24",
                                         "-nolock", "-nolisten", "unix", "-listen", "tcp", "-auth", str(authority)],
                                        stdout=self.log, stderr=subprocess.STDOUT)
        self.previous = {key: os.environ.get(key) for key in ("DISPLAY", "XAUTHORITY", "MUJOCO_GL", "__GLX_VENDOR_LIBRARY_NAME", "LIBGL_ALWAYS_SOFTWARE")}
        deadline = time.monotonic() + 10
        while True:
            if self.process.poll() is not None:
                raise RuntimeError("Private display exited")
            with socket.socket() as connection:
                if connection.connect_ex(("127.0.0.1", 6000 + number)) == 0:
                    break
            if time.monotonic() > deadline:
                self.close()
                raise TimeoutError("Private display did not start")
            time.sleep(.1)
        os.environ.update(DISPLAY=self.display, XAUTHORITY=str(authority), MUJOCO_GL="glfw",
                          __GLX_VENDOR_LIBRARY_NAME="mesa", LIBGL_ALWAYS_SOFTWARE="1")

    def close(self):
        self.process.terminate()
        self.process.wait(timeout=5)
        self.log.close()
        for key, value in self.previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
