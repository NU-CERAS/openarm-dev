import json
from pathlib import Path
import threading
import time

import numpy as np

from openarm_tools.robot.config import motor_position, validate_positions

from ..console.session import CanPlant, hardware_configuration
from ..execution import ArmCommand, ArmState, joint_names


class GripperCalibration:
    def __init__(self, payload: dict):
        if payload["schema"] != "openarm.policy_gripper.v1" or payload["hardware_reviewed"] is not True:
            raise ValueError("Fill and review the gripper calibration placeholders before connecting")
        self.config = payload
        self.positions = np.asarray(payload["model_positions_m"], dtype=float)
        self.angles = np.asarray(payload["motor_positions_rad"], dtype=float)
        if (self.positions.ndim != 1 or len(self.positions) < 2 or self.angles.shape != self.positions.shape
                or not np.isfinite([self.positions, self.angles]).all() or np.any(np.diff(self.positions) <= 0)
                or not (np.all(np.diff(self.angles) > 0) or np.all(np.diff(self.angles) < 0))):
            raise ValueError("Gripper calibration requires matching strictly monotonic position/angle tables")
        if self.positions[0] != 0 or self.positions[-1] != 0.044:
            raise ValueError("The gripper calibration must cover the model's [0, 0.044] meter coordinates")
        if not isinstance(payload["motor_type"], str) or not payload["motor_type"]:
            raise ValueError("Supply the actual gripper motor type")
        for key in ("send_id", "recv_id"):
            if type(payload[key]) is not int or not 0 < payload[key] <= 0x7ff:
                raise ValueError("Supply valid gripper CAN IDs")
        if payload["send_id"] == payload["recv_id"]:
            raise ValueError("Gripper command and feedback CAN IDs must differ")
        for key in ("kp", "max_speed_rad_s", "max_tracking_error_rad", "max_temperature_c", "feedback_timeout_s", "command_timeout_s"):
            value = payload[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"Supply a positive gripper {key}")
        if not isinstance(payload["kd"], (int, float)) or not np.isfinite(payload["kd"]) or payload["kd"] < 0:
            raise ValueError("Supply a nonnegative gripper kd")

    def to_motor(self, position: float) -> float:
        if not np.isfinite(position) or not self.positions[0] <= position <= self.positions[-1]:
            raise ValueError("Gripper model coordinate exceeds its calibrated range")
        return float(np.interp(position, self.positions, self.angles))

    def to_model(self, angle: float) -> float:
        if not np.isfinite(angle) or not np.min(self.angles) <= angle <= np.max(self.angles):
            raise ValueError("Gripper motor feedback exceeds its calibrated range")
        order = np.argsort(self.angles)
        return float(np.interp(angle, self.angles[order], self.positions[order]))


class CanPolicyInterface:
    def __init__(self, hardware_config: Path, gripper_config: Path, arm: str):
        joint_names(arm)
        self.calibration = GripperCalibration(json.loads(gripper_config.read_text()))
        configurations = hardware_configuration(hardware_config)
        cfg = configurations[arm][1]
        used_ids = {j[key] for j in cfg["joints"] for key in ("send_id", "recv_id")}
        if used_ids.intersection(self.calibration.config[key] for key in ("send_id", "recv_id")):
            raise ValueError("Gripper CAN IDs conflict with an arm motor")
        self.arm_name, self.cfg = arm, cfg
        self.limits = np.asarray([j["limits"] for j in cfg["joints"]], dtype=float)
        self.mutex = threading.RLock()
        self.stop = threading.Event()
        self.watchdog = None
        self.fault = None
        self.closed = False
        self.last_command = None
        self.last_motor_target = None
        self.last_write = None
        self.plant = CanPlant(hardware_config)
        self.device = self.plant.hardware[arm]
        complete = False
        try:
            g = self.calibration.config
            self.device.arm.init_gripper_motor(getattr(self.device.oa.MotorType, g["motor_type"]), g["send_id"], g["recv_id"])
            self.gripper = self.device.arm.get_gripper()
            self.device.arm.set_callback_mode_all(self.device.oa.CallbackMode.STATE)
            deadline = time.monotonic() + 1.0
            while True:
                self.device.arm.refresh_all()
                self.device.arm.recv_all(10000)
                if not self.gripper.get_link_stats(0).is_stale(g["feedback_timeout_s"]):
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("Gripper did not provide startup feedback")
            initial = self.read()
            self.last_command = ArmCommand(initial.position, initial.gripper)
            self.last_motor_target = self.calibration.to_motor(initial.gripper)
            self.other_positions = {name: value.copy() for name, value in self.last_feedback.positions.items() if name != arm}
            complete = True
        finally:
            if not complete:
                self.close()

    def read(self) -> ArmState:
        with self.mutex:
            if self.closed or self.fault is not None:
                raise RuntimeError(f"CAN policy interface is closed: {self.fault}")
            started = time.monotonic()
            feedback = self.plant.read()
            self.last_feedback = feedback
            motor = self.gripper.get_motors()[0]
            g = self.calibration.config
            if self.gripper.get_link_stats(0).is_stale(g["feedback_timeout_s"]):
                raise RuntimeError("Gripper feedback is stale")
            if motor.has_error() or max(motor.get_state_tmos(), motor.get_state_trotor()) > g["max_temperature_c"]:
                raise RuntimeError("Gripper reports a motor fault or excessive temperature")
            if abs(motor.get_velocity()) > g["max_speed_rad_s"]:
                raise RuntimeError("Measured gripper speed exceeds its reviewed limit")
            self.last_gripper_motor = motor
            return ArmState(feedback.positions[self.arm_name], feedback.velocities[self.arm_name],
                            self.calibration.to_model(motor.get_position()), started)

    def write(self, command: ArmCommand) -> None:
        with self.mutex:
            state = self.read()
            for arm, original in self.other_positions.items():
                if np.max(np.abs(self.last_feedback.positions[arm] - original)) > 0.03:
                    raise RuntimeError("The other arm moved during policy execution")
            validate_positions(self.cfg, command.position.tolist())
            if np.max(np.abs(command.position - state.position)) > self.cfg["max_tracking_error"]:
                raise RuntimeError("CAN policy target exceeds the reviewed tracking-error limit")
            if np.max(np.abs(command.position - self.last_command.position)) / 0.02 > self.cfg["max_speed"] + 1e-8:
                raise RuntimeError("CAN policy target exceeds the reviewed joint speed")
            motor_target = self.calibration.to_motor(command.gripper)
            g = self.calibration.config
            if abs(motor_target - self.last_motor_target) / 0.02 > g["max_speed_rad_s"] + 1e-8:
                raise RuntimeError("Gripper target exceeds its reviewed speed")
            if abs(motor_target - self.last_gripper_motor.get_position()) > g["max_tracking_error_rad"]:
                raise RuntimeError("Gripper target exceeds its reviewed tracking-error limit")
            if not self.device.enabled:
                if any(m.is_enabled() for device in self.plant.hardware.values() for m in device.motors) or self.last_gripper_motor.is_enabled():
                    raise RuntimeError("Stop other controllers before enabling policy execution")
                self.device.enabled = True
                self.device.arm.enable_all()
                self.last_write = time.monotonic()
                self.watchdog = threading.Thread(target=self._watchdog, name="openarm-policy-watchdog", daemon=False)
                self.watchdog.start()
            self.device.component.mit_control_all([
                self.device.oa.MITParam(j["kp"], j["kd"], motor_position(j, float(value)), 0, 0)
                for j, value in zip(self.cfg["joints"], command.position)])
            self.gripper.mit_control_all([self.device.oa.MITParam(g["kp"], g["kd"], motor_target, 0, 0)])
            self.device.arm.recv_all(10000)
            self.last_command = ArmCommand(command.position.copy(), command.gripper)
            self.last_motor_target = motor_target
            self.last_write = time.monotonic()

    def _watchdog(self) -> None:
        while not self.stop.wait(0.01):
            with self.mutex:
                if time.monotonic() - self.last_write > self.calibration.config["command_timeout_s"]:
                    self.fault = "Policy command timeout"
                    self.close()
                    return

    def close(self) -> None:
        with self.mutex:
            if not self.closed:
                self.stop.set()
                self.plant.close()
                self.closed = True
        if self.watchdog is not None and self.watchdog is not threading.current_thread():
            self.watchdog.join(timeout=2)
            if self.watchdog.is_alive():
                raise RuntimeError("CAN policy watchdog did not stop")
