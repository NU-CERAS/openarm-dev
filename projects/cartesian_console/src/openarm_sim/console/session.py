from contextlib import ExitStack
from dataclasses import dataclass
import json
from pathlib import Path
import queue
import threading
import time

import mujoco
import numpy as np

from openarm_tools.recording.storage import lock, write_json
from openarm_tools.robot.config import load, validate_positions

from ..execution import JointTrajectory, joint_names
from ..paths import BODY_REFERENCE
from ..simulation import ARMS, ReachSimulation


@dataclass
class Feedback:
    positions: dict[str, np.ndarray]
    velocities: dict[str, np.ndarray]
    received_at: float


@dataclass
class Preview:
    trajectory: JointTrajectory
    target: np.ndarray
    start: Feedback


@dataclass(frozen=True)
class BodyReference:
    site: str
    origin_m: np.ndarray
    rotation: np.ndarray

    def to_world(self, coordinates_cm):
        values = np.asarray(coordinates_cm, dtype=float)
        if values.shape != (3,) or not np.isfinite(values).all():
            raise ValueError("Body-relative coordinates require three finite values in centimeters")
        return self.origin_m + self.rotation @ (values / 100.0)

    def from_world(self, position_m):
        values = np.asarray(position_m, dtype=float)
        if values.shape != (3,) or not np.isfinite(values).all():
            raise ValueError("World coordinates require three finite values in meters")
        return self.rotation.T @ (values - self.origin_m) * 100.0


class MotionCancelled(Exception):
    pass


def apply_feedback(sim: ReachSimulation, feedback: Feedback):
    if set(feedback.positions) != set(ARMS):
        raise ValueError("Both arm positions are required for a bimanual twin")
    for arm in ARMS:
        positions = np.asarray(feedback.positions[arm], dtype=float)
        velocities = np.asarray(feedback.velocities[arm], dtype=float)
        if positions.shape != (7,) or velocities.shape != (7,) or not np.isfinite([positions, velocities]).all():
            raise ValueError("Feedback requires seven finite positions and velocities per arm")
        sim.data.qpos[sim.qpos_ids[arm]] = positions
        sim.data.qvel[sim.model.jnt_dofadr[sim.joint_ids[arm]]] = velocities
    mujoco.mj_forward(sim.model, sim.data)


def check_start(plan: Preview, feedback: Feedback, tolerance=0.03):
    if time.monotonic() - feedback.received_at > 0.25:
        raise RuntimeError("Robot feedback is stale")
    for arm in ARMS:
        if np.max(np.abs(feedback.positions[arm] - plan.start.positions[arm])) > tolerance:
            raise RuntimeError("Robot moved after preview; create a new preview")


class MujocoPlant:
    def __init__(self, scene=None, realtime=True):
        self.sim = ReachSimulation(drum_asset=scene, worktable=scene is None, tabletop_mount=scene is None)
        self.sim.reset_zero()
        self.realtime = realtime

    def read(self):
        return Feedback({arm: self.sim.data.qpos[self.sim.qpos_ids[arm]].copy() for arm in ARMS},
                        {arm: self.sim.data.qvel[self.sim.model.jnt_dofadr[self.sim.joint_ids[arm]]].copy() for arm in ARMS},
                        time.monotonic())

    def idle(self):
        for _ in range(round(0.02 / self.sim.model.opt.timestep)):
            self.sim.step()
        return self.read()

    def execute(self, plan, cancel, publish):
        check_start(plan, self.read())
        trajectory = plan.trajectory
        worst_error = 0.0
        started = time.monotonic()
        for step in range(int(np.ceil((trajectory.duration + 1.0) / 0.02)) + 1):
            if cancel.is_set():
                self.sim._desired[self.sim.actuator_ids[trajectory.arm]] = self.read().positions[trajectory.arm]
                raise MotionCancelled()
            elapsed = step * 0.02
            desired = trajectory.sample(elapsed)
            self.sim._desired[self.sim.actuator_ids[trajectory.arm]] = desired
            for _ in range(round(0.02 / self.sim.model.opt.timestep)):
                self.sim.step()
            measured = self.read()
            error = float(np.max(np.abs(desired - measured.positions[trajectory.arm])))
            worst_error = max(worst_error, error)
            if error > 0.25:
                raise RuntimeError("MuJoCo plant tracking error exceeds 0.25 rad")
            publish(measured)
            if self.realtime:
                cancel.wait(max(0.0, started + (step + 1) * 0.02 - time.monotonic()))
        final_error = float(np.max(np.abs(self.read().positions[trajectory.arm] - trajectory.positions[-1])))
        if final_error > 0.03:
            raise RuntimeError("MuJoCo plant did not reach the requested joint target")
        return {"max_tracking_error_rad": worst_error, "final_joint_error_rad": final_error}

    def close(self):
        self.sim.close()


def hardware_configuration(path: Path):
    payload = json.loads(path.read_text())
    if payload["schema"] != "openarm.sync_hardware.v1" or set(payload["arms"]) != set(ARMS):
        raise ValueError("Hardware configuration requires schema openarm.sync_hardware.v1 and both arms")
    configurations = {}
    for arm in ARMS:
        item = payload["arms"][arm]
        if not item["interface"] or not item["config"]:
            raise ValueError(f"Fill the {arm} interface and reviewed config placeholders in {path}")
        config_path = Path(item["config"]).expanduser()
        if not config_path.is_absolute():
            config_path = path.resolve().parent / config_path
        cfg = load(config_path)
        if cfg["arm"] != arm or cfg["robot"] != "v1.0" or [j["name"] for j in cfg["joints"]] != joint_names(arm):
            raise ValueError("Hardware configuration does not match the v1.0 model joint order")
        if not cfg["hardware_reviewed"]:
            raise ValueError(f"Review motor mapping, limits and gains in {config_path}")
        if not (Path("/sys/class/net") / item["interface"]).is_dir():
            raise ValueError(f"SocketCAN interface does not exist: {item['interface']}")
        configurations[arm] = (item["interface"], cfg)
    if len({item[0] for item in configurations.values()}) != 2:
        raise ValueError("This adapter requires a separate CAN interface for each arm")
    return configurations


class CanPlant:
    def __init__(self, configuration: Path):
        from openarm_tools.backends.can import Hardware

        self.configurations = hardware_configuration(configuration)
        self.resources = ExitStack()
        self.hardware = {}
        try:
            for arm, (interface, cfg) in sorted(self.configurations.items()):
                self.resources.enter_context(lock(f"can:{interface}"))
                device = Hardware(cfg, interface)
                self.resources.callback(device.close)
                self.hardware[arm] = device
        except BaseException:
            self.resources.close()
            raise

    def read(self):
        positions, velocities = {}, {}
        for arm, device in self.hardware.items():
            values = device.status()
            positions[arm] = np.array([item["position"] for item in values])
            velocities[arm] = np.array([item["velocity"] for item in values])
        return Feedback(positions, velocities, time.monotonic())

    def idle(self):
        return self.read()

    def execute(self, plan, cancel, publish):
        check_start(plan, self.read())
        trajectory = plan.trajectory
        cfg = self.configurations[trajectory.arm][1]
        worst_error = 0.0
        started = None

        def feedback_callback(actual):
            nonlocal worst_error
            if cancel.is_set():
                raise MotionCancelled()
            device = self.hardware[trajectory.arm]
            positions = {trajectory.arm: np.array(actual)}
            velocities = {trajectory.arm: np.array([j["sign"] * m.get_velocity()
                                                   for j, m in zip(cfg["joints"], device.motors)])}
            for other in ARMS:
                if other == trajectory.arm:
                    continue
                status = self.hardware[other].status()
                positions[other] = np.array([item["position"] for item in status])
                velocities[other] = np.array([item["velocity"] for item in status])
                if np.max(np.abs(positions[other] - plan.start.positions[other])) > 0.03:
                    raise RuntimeError("The other arm moved during execution")
            elapsed = 0.0 if started is None else time.monotonic() - started
            error = float(np.max(np.abs(trajectory.sample(elapsed) - actual)))
            worst_error = max(worst_error, error)
            if error > cfg["max_tracking_error"]:
                raise RuntimeError("Measured arm tracking error exceeds configured limit")
            publish(Feedback(positions, velocities, time.monotonic()))

        def points():
            nonlocal started
            started = time.monotonic()
            for elapsed in np.arange(0, trajectory.duration + 0.02, 0.02):
                if cancel.is_set():
                    raise MotionCancelled()
                command = trajectory.sample(elapsed)
                validate_positions(cfg, command.tolist())
                yield float(min(elapsed, trajectory.duration)), command.tolist()

        self.hardware[trajectory.arm].move(points(), on_feedback=feedback_callback)
        measured = self.read()
        publish(measured)
        final_error = float(np.max(np.abs(measured.positions[trajectory.arm] - trajectory.positions[-1])))
        if final_error > 0.03:
            raise RuntimeError("Physical arm did not reach the requested joint target")
        return {"max_tracking_error_rad": worst_error,
                "final_joint_error_rad": final_error,
                "motors_released": True}

    def close(self):
        self.resources.close()


class RobotWorker:
    def __init__(self, backend="demo", configuration=None, scene=None, realtime=True):
        self.commands = queue.Queue(maxsize=1)
        self.cancel_event = threading.Event()
        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.mutex = threading.Condition()
        self.feedback = None
        self.error = None
        self.state = "connecting"
        self.report = None
        self.completed_plan = None
        self.backend = backend
        self.configuration = configuration
        self.scene = scene
        self.realtime = realtime
        self.thread = threading.Thread(target=self._run, name="openarm-robot-transport", daemon=False)
        self.thread.start()

    def _publish(self, feedback):
        with self.mutex:
            self.feedback = feedback
            self.mutex.notify_all()

    def _run(self):
        plant = None
        try:
            plant = MujocoPlant(self.scene, self.realtime) if self.backend == "demo" else CanPlant(self.configuration)
            self._publish(plant.read())
            self.state = "connected"
            self.ready.set()
            while not self.stop_event.is_set():
                try:
                    plan = self.commands.get(timeout=0.02)
                except queue.Empty:
                    self._publish(plant.idle())
                    continue
                try:
                    self.report = plant.execute(plan, self.cancel_event, self._publish)
                    self.completed_plan = plan
                    self.state = "completed"
                except MotionCancelled:
                    self._publish(plant.read())
                    self.state = "cancelled"
        except BaseException as error:
            self.error = error
            self.state = "failed"
            self.ready.set()
        finally:
            if plant is not None:
                plant.close()

    def read(self):
        with self.mutex:
            if not self.stop_event.is_set() and self.error is None:
                self.mutex.wait_for(lambda: self.error is not None or
                                    (self.feedback is not None and time.monotonic() - self.feedback.received_at <= 0.25),
                                    timeout=0.25)
            result = self.feedback
        if self.error is not None:
            raise RuntimeError("Robot transport failed") from self.error
        if result is None or time.monotonic() - result.received_at > 0.25:
            raise RuntimeError("Robot feedback is missing or stale")
        return result

    def execute(self, plan):
        if self.state == "executing":
            raise RuntimeError("A trajectory is already executing")
        check_start(plan, self.read())
        self.cancel_event.clear()
        self.report = None
        self.completed_plan = None
        self.state = "executing"
        self.commands.put_nowait(plan)

    def cancel(self):
        self.cancel_event.set()

    def close(self):
        self.cancel_event.set()
        self.stop_event.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError("Robot transport did not stop within five seconds")


class SyncSession:
    def __init__(self, worker: RobotWorker, planner="ompl", scene=None, speed=0.2, acceleration=0.5):
        if not 0 < speed <= 0.6 or not 0 < acceleration <= 1.5:
            raise ValueError("Invalid trajectory speed or acceleration limit")
        self.worker = worker
        self.twin = ReachSimulation(drum_asset=scene, worktable=scene is None, tabletop_mount=scene is None)
        self.preview = ReachSimulation(planner=planner, drum_asset=scene, worktable=scene is None, tabletop_mount=scene is None)
        site = self.twin.model.site(BODY_REFERENCE["site"]).id
        if self.twin.model.site_bodyid[site] != self.twin.model.body(BODY_REFERENCE["body"]).id:
            raise ValueError("The body reference must be attached to the fixed chassis")
        self.reference = BodyReference(BODY_REFERENCE["site"], self.twin.data.site_xpos[site].copy(),
                                       self.twin.data.site_xmat[site].reshape(3, 3).copy())
        self.plan = None
        self.rejection_reason = ""
        self.speed = speed
        self.acceleration = acceleration

    def sync(self):
        feedback = self.worker.read()
        apply_feedback(self.twin, feedback)
        return feedback

    def prepare(self, arm: str, target) -> Preview | None:
        if self.worker.state == "executing":
            raise RuntimeError("Cancel or finish execution before planning")
        self.plan = None
        self.rejection_reason = ""
        self.preview.cancel()
        start = self.sync()
        apply_feedback(self.preview, start)
        result = self.preview.start_reach(arm, np.asarray(target, dtype=float))
        if not result.accepted:
            self.rejection_reason = result.reason
            return None
        trajectory = self.preview.trajectory
        if trajectory is None:
            initial = start.positions[arm]
            final = self.preview._goal_q[self.preview.actuator_ids[arm]]
            duration = max(1.2, 1.875 * float(np.max(np.abs(final - initial))) / self.speed,
                           np.sqrt(5.8 * float(np.max(np.abs(final - initial))) / self.acceleration))
            times = np.linspace(0, duration, 101)
            phase = times / duration
            blend = 10 * phase**3 - 15 * phase**4 + 6 * phase**5
            derivative = (30 * phase**2 - 60 * phase**3 + 30 * phase**4) / duration
            trajectory = JointTrajectory(arm, times, initial + blend[:, None] * (final - initial),
                                         derivative[:, None] * (final - initial), "mink")
        trajectory = trajectory.retime(self.speed, self.acceleration)
        if self.worker.backend == "can":
            configurations = hardware_configuration(self.worker.configuration)
            cfg = configurations[arm][1]
            trajectory = trajectory.retime(min(self.speed, cfg["max_speed"]), self.acceleration)
            for positions in trajectory.positions:
                validate_positions(cfg, positions.tolist())
        self.preview.cancel()
        self.plan = Preview(trajectory, np.asarray(target, dtype=float).copy(), start)
        self.show_preview(0)
        return self.plan

    def prepare_relative(self, arm: str, coordinates_cm) -> Preview | None:
        return self.prepare(arm, self.reference.to_world(coordinates_cm))

    def show_preview(self, elapsed):
        if self.plan is None:
            return
        apply_feedback(self.preview, self.plan.start)
        arm = self.plan.trajectory.arm
        self.preview.data.qpos[self.preview.qpos_ids[arm]] = self.plan.trajectory.sample(elapsed)
        mujoco.mj_forward(self.preview.model, self.preview.data)

    def execute(self):
        if self.plan is None:
            raise RuntimeError("Create a preview before execution")
        self.worker.execute(self.plan)

    def save(self, path: Path, name: str):
        if self.plan is None or not name.strip():
            raise ValueError("A preview and target name are required")
        if self.worker.state == "executing":
            raise RuntimeError("Wait for execution to finish before saving observed joints")
        feedback = self.sync()
        payload = json.loads(path.read_text()) if path.exists() else {"schema": "openarm.sync_targets.v1", "targets": {}}
        if payload["schema"] != "openarm.sync_targets.v1":
            raise ValueError("Unsupported target file schema")
        payload["targets"][name.strip()] = {
            "arm": self.plan.trajectory.arm, "sim_target_m": self.plan.target.tolist(),
            "reference_site": self.reference.site,
            "reference_origin_m": self.reference.origin_m.tolist(),
            "reference_rotation": self.reference.rotation.tolist(),
            "target_relative_cm": self.reference.from_world(self.plan.target).tolist(),
            "planned_joint_positions_rad": self.plan.trajectory.positions[-1].tolist(),
            "observed_joint_positions_rad": feedback.positions[self.plan.trajectory.arm].tolist(),
            "observed_sim_tcp_m": self.twin.tcp_position(self.plan.trajectory.arm).tolist(),
            "execution_state": "completed" if self.worker.completed_plan is self.plan else "not_executed",
            "backend": self.worker.backend, "saved_at_unix": time.time(),
        }
        write_json(path, payload)

    def load(self, path: Path, name: str):
        payload = json.loads(path.read_text())
        if payload["schema"] != "openarm.sync_targets.v1":
            raise ValueError("Unsupported target file schema")
        item = payload["targets"][name]
        return item["arm"], np.array(item["sim_target_m"], dtype=float)

    def close(self):
        self.worker.close()
        self.twin.close()
        self.preview.close()
