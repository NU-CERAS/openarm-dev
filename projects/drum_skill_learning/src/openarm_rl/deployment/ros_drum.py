import json
import hashlib
from contextlib import ExitStack
from pathlib import Path
import threading
import time

import numpy as np
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rclpy.parameter_client import AsyncParameterClient

from openarm_drummer.model import HoldController
from openarm_tools.ros_execution import RosExecutionClient, arm_joint_names, duration, trajectory_message
from openarm_tools.recording.storage import lock

from .drum_actor import DrumActor
from ..skills.runtime import prepare_transfers


class JointMapping:
    def __init__(self, document, limits, physical):
        if document["schema"] != "openarm.drum_joint_mapping.v1":
            raise ValueError("Unsupported joint mapping")
        if physical and (not document["hardware_reviewed"] or not document["stream_watchdog_verified"]):
            raise ValueError("Physical mapping and controller watchdog must be reviewed")
        if document["command_interfaces"] != ["position", "velocity"]:
            raise ValueError("The policy requires position and velocity command interfaces")
        expected = [f"{side}j{i}" for side in ("l", "r") for i in range(1, 8)]
        if document["joint_order"] != expected:
            raise ValueError("Mapping joint order must match the OpenDrummer model")
        self.sign = np.asarray(document["sign"], dtype=float)
        self.zero = np.asarray(document["zero_rad"], dtype=float)
        if self.sign.shape != (14,) or self.zero.shape != (14,) or not np.isfinite(self.zero).all() or not np.isin(self.sign, [-1, 1]).all():
            raise ValueError("Mapping requires fourteen direction and zero values")
        self.document = document
        if physical:
            for name in ("velocity_rad_s", "acceleration_rad_s2", "position_gains", "velocity_gains"):
                if not np.allclose(document[name], limits[name]):
                    raise ValueError(f"Reviewed hardware {name} differs from the trained controller")

    def measured(self, snapshot):
        states = [snapshot.arm(arm) for arm in ("left", "right")]
        q = np.concatenate([state[0] for state in states])
        dq = np.concatenate([state[1] for state in states])
        return self.sign * (q - self.zero), self.sign * dq

    def command(self, q, dq):
        return self.zero + self.sign * q, self.sign * dq


class RosDrumTool:
    def __init__(self, namespace, mapping_file, physical=False):
        self.client = RosExecutionClient(namespace, feedback_timeout=.1, background=True)
        with ExitStack() as resources:
            resources.callback(self.client.close)
            self.namespace = self.client.node.get_namespace()
            self.client.wait_ready()
            self.mapping_document = json.loads(Path(mapping_file).read_text())
            self.physical = physical
            self.parameters = (AsyncParameterClient(self.client.node, self.namespace.rstrip("/") + "/openarm_execution_server")
                               if not physical else None)
            self.publishers = {arm: self.client.node.create_publisher(JointTrajectory,
                               f"{arm}_joint_trajectory_controller/joint_trajectory", 1) for arm in ("left", "right")}
            self.cancelled = threading.Event()
            self.reservation = threading.Lock()
            resources.pop_all()

    def execute(self, checkpoint, speed=.6, execute=False):
        if not np.isfinite(speed) or not .35 <= speed <= .85:
            raise ValueError("Requested speed is outside the trained range")
        if not self.reservation.acquire(blocking=False):
            raise RuntimeError("Another robot skill is executing")
        actor = None
        mapping = None
        streaming = False
        ownership = ExitStack()
        try:
            if execute:
                ownership.enter_context(lock(f"ros-command:{self.namespace}"))
            actor = DrumActor(checkpoint)
            self.client.feedback_timeout = min(.1, actor.limits["feedback_timeout_s"])
            mapping = JointMapping(self.mapping_document, actor.limits, self.physical)
            if not self.physical:
                parameters = self.parameters
                if not parameters.wait_for_services(timeout_sec=2):
                    raise RuntimeError("Execution backend model identity service is unavailable")
                values = self.client.wait_future(parameters.get_parameters(["robot_model", "layout_sha256"]), time.monotonic() + 2).values
                if values[0].string_value != "opendrummer" or values[1].string_value != actor.manifest["scene_identity"].get("layout", ""):
                    raise ValueError("Execution backend robot or annotated layout differs from the policy")
            elif (self.mapping_document["robot_source_sha256"] != actor.manifest["scene_identity"]["robot"]
                  or self.mapping_document["import_config_sha256"] != actor.manifest["scene_identity"]["import"]
                  or self.mapping_document["layout_sha256"] != actor.manifest["scene_identity"].get("layout", "")):
                raise ValueError("Reviewed physical model/layout identity differs from the policy")
            snapshot = self.client.read()
            current, dq = mapping.measured(snapshot)
            from openarm_drummer.model import CONFIG

            if hashlib.sha256(CONFIG.read_bytes()).hexdigest() != actor.manifest["scene_identity"]["import"]:
                raise ValueError("Current controller configuration differs from the bundle")
            report = {"config": json.loads(CONFIG.read_text()), "limits": {}}
            for index, name in enumerate([f"{side}j{i}" for side in ("l", "r") for i in range(1, 8)]):
                report["limits"][name] = {"lower": actor.limits["bounds_rad"][index][0], "upper": actor.limits["bounds_rad"][index][1],
                                           "velocity": actor.limits["velocity_rad_s"][index] / .8,
                                           "effort": float(np.max(np.abs(actor.model.actuator(name + "_motor").ctrlrange)))}
            controller = HoldController(actor.model, report)
            transfers = prepare_transfers(actor.model, controller, current, actor.reference.positions[0])
            preview = {"tool": actor.skill.name, "arm": actor.skill.arm, "execute": execute,
                       "measured_joints_rad": current.tolist(), "ready_joints_rad": actor.reference.positions[0].tolist(),
                       "preparation_duration_s": float(sum(t.ends[-1] for _, t in transfers)), "strike_duration_s": actor.reference.duration}
            if not execute:
                return preview
            self.cancelled.clear()
            for arm, trajectory in transfers:
                times = np.arange(0, trajectory.ends[-1], .02)
                times = np.r_[times, trajectory.ends[-1]]
                samples = [trajectory.sample(t) for t in times]
                positions = np.array([mapping.command(q, v)[0] for q, v in samples])
                velocities = np.array([mapping.command(q, v)[1] for q, v in samples])
                selected = slice(0, 7) if arm == "left" else slice(7, 14)
                message = trajectory_message(arm, times, positions[:, selected], velocities[:, selected])
                self.client.execute(arm, message, self.cancelled, lambda _: None)
            deadline = time.monotonic() + 2
            while True:
                snapshot = self.client.read()
                q, dq = mapping.measured(snapshot)
                if np.max(np.abs(q - actor.reference.positions[0])) < .03 and np.max(np.abs(dq)) < .08:
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError("Measured robot did not settle at the fixed ready pose")
                time.sleep(.01)
            started = time.monotonic()
            actor.begin(q, dq, snapshot.received_at, started, speed)
            tick = 0
            periods, feedback_ages = [], []
            previous_tick = None
            streaming = True
            while actor.running:
                if self.cancelled.is_set():
                    raise InterruptedError("ROS drum skill cancelled")
                snapshot = self.client.read(wait=False)
                q, dq = mapping.measured(snapshot)
                now = time.monotonic()
                if previous_tick is not None:
                    periods.append(now - previous_tick)
                previous_tick = now
                feedback_ages.append(now - snapshot.received_at)
                command = actor.step(q, dq, snapshot.received_at, now)
                if command is None:
                    break
                self.publish(mapping, command, actor.output_velocity)
                tick += 1
                time.sleep(max(0., started + tick * .01 - time.monotonic()))
            time.sleep(.05)
            snapshot = self.client.read()
            final, dq = mapping.measured(snapshot)
            recovered = np.max(np.abs(final - actor.reference.positions[-1])) < .03 and np.max(np.abs(dq)) < .08
            return {**preview, "encoder_verified": True, "recovered": bool(recovered),
                    "hit_verification": "operator_required" if self.physical else "not_observed_by_actor",
                    "command_count": tick, "max_control_period_s": float(max(periods, default=0)),
                    "max_feedback_age_s": float(max(feedback_ages, default=0)),
                    "final_joints_rad": final.tolist(), "final_feedback_age_s": time.monotonic() - snapshot.received_at}
        finally:
            try:
                if actor is not None:
                    actor.cancel()
                if streaming and mapping is not None:
                    snapshot = self.client.read()
                    q, _ = mapping.measured(snapshot)
                    hold = actor.servo.position_target(q, q)
                    bounds = np.asarray(actor.limits["bounds_rad"])
                    hold = np.clip(hold, bounds[:, 0], bounds[:, 1])
                    self.publish(mapping, hold, np.zeros(14))
                    time.sleep(.12)
            finally:
                try:
                    ownership.close()
                finally:
                    self.reservation.release()

    def publish(self, mapping, q, dq):
        q, dq = mapping.command(q, dq)
        for arm, selected in (("left", slice(0, 7)), ("right", slice(7, 14))):
            message = JointTrajectory(joint_names=arm_joint_names(arm))
            message.points.append(JointTrajectoryPoint(positions=q[selected].tolist(), velocities=dq[selected].tolist(),
                                                       time_from_start=duration(.01)))
            self.publishers[arm].publish(message)

    def close(self):
        self.cancelled.set()
        self.client.close()
