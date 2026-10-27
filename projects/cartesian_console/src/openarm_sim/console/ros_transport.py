from contextlib import ExitStack
import time

import numpy as np

from openarm_tools.ros_execution import ExecutionCancelled, RosExecutionClient, trajectory_message

from .session import Feedback, MotionCancelled, check_start


class RosPlant:
    def __init__(self, namespace=""):
        self.client = RosExecutionClient(namespace)
        with ExitStack() as resources:
            resources.callback(self.client.close)
            self.client.wait_ready()
            resources.pop_all()

    def convert(self, snapshot):
        states = {arm: snapshot.arm(arm) for arm in ("left", "right")}
        return Feedback({arm: state[0] for arm, state in states.items()},
                        {arm: state[1] for arm, state in states.items()}, snapshot.received_at)

    def read(self):
        return self.convert(self.client.read())

    def idle(self):
        return self.read()

    def execute(self, plan, cancel, publish):
        check_start(plan, self.read())
        trajectory = plan.trajectory
        message = trajectory_message(trajectory.arm, trajectory.times, trajectory.positions, trajectory.velocities)
        worst_error = 0.0
        started = time.monotonic()

        def feedback(snapshot):
            nonlocal worst_error
            measured = self.convert(snapshot)
            for other in ("left", "right"):
                if other != trajectory.arm and np.max(np.abs(measured.positions[other] - plan.start.positions[other])) > 0.03:
                    raise RuntimeError("The other arm moved after the trajectory was planned")
            desired = trajectory.sample(time.monotonic() - started)
            worst_error = max(worst_error, float(np.max(np.abs(desired - measured.positions[trajectory.arm]))))
            publish(measured)

        try:
            self.client.execute(trajectory.arm, message, cancel, feedback)
        except ExecutionCancelled:
            raise MotionCancelled() from None
        final = self.read()
        error = float(np.max(np.abs(final.positions[trajectory.arm] - trajectory.positions[-1])))
        if error > 0.03:
            raise RuntimeError("ROS action succeeded without matching measured final joints")
        return {"max_tracking_error_rad": worst_error, "final_joint_error_rad": error, "transport": "ros2"}

    def close(self):
        self.client.close()
