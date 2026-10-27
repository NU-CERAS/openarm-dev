import argparse
import time

from openarm_tools.ros_execution import RosExecutionClient

from ..simulation import ReachSimulation
from .plant_window import MujocoPlantWindow
from .session import Feedback, apply_feedback


def main():
    parser = argparse.ArgumentParser(description="Display measured ROS robot joints in a host MuJoCo window")
    parser.add_argument("--namespace", default="")
    args = parser.parse_args()
    client = RosExecutionClient(args.namespace)
    sim = None
    window = None
    try:
        client.wait_ready()
        sim = ReachSimulation(worktable=True, tabletop_mount=True)
        while window is None or not window.closed:
            snapshot = client.read()
            states = {arm: snapshot.arm(arm) for arm in ("left", "right")}
            apply_feedback(sim, Feedback({arm: state[0] for arm, state in states.items()},
                                        {arm: state[1] for arm, state in states.items()}, snapshot.received_at))
            if window is None:
                window = MujocoPlantWindow(sim, feedback_only=True)
            window.sync()
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        if window is not None:
            window.close()
        if sim is not None:
            sim.close()
        client.close()


if __name__ == "__main__":
    main()
