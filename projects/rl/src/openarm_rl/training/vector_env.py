import time

from stable_baselines3.common.vec_env import SubprocVecEnv


class TrainingSubprocVecEnv(SubprocVecEnv):
    def step_wait(self):
        deadline = time.monotonic() + 30
        while True:
            failed = [(index, p.exitcode) for index, p in enumerate(self.processes) if p.exitcode is not None]
            if failed:
                raise RuntimeError(f"Training environment workers exited: {failed}; inspect the worker traceback")
            if all(remote.poll() for remote in self.remotes):
                return super().step_wait()
            if time.monotonic() >= deadline:
                raise TimeoutError("Training environments exceeded the 30-second response deadline")
            time.sleep(0.002)

    def close(self):
        if self.closed:
            return
        try:
            if not self.waiting and all(p.is_alive() for p in self.processes):
                for remote in self.remotes:
                    remote.send(("close", None))
                for process in self.processes:
                    process.join(timeout=3)
        finally:
            for process in self.processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=3)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=3)
            for remote in self.remotes:
                remote.close()
            self.waiting, self.closed = False, True
