"""Linux SocketCAN adapter for the pinned OpenArm CAN Python bindings."""
import math
import platform
import time
from openarm_tools.robot.config import model_position, motor_position, root, validate_positions


class Hardware:
    def __init__(self, cfg, interface):
        if platform.system() != 'Linux':
            raise RuntimeError('SocketCAN requires Linux. Use simulation on macOS.')
        import openarm_can as oa
        self.oa, self.cfg, self.enabled = oa, cfg, False
        self.arm = oa.OpenArm(interface, False)
        self.arm.init_arm_motors([getattr(oa.MotorType, j['motor_type']) for j in cfg['joints']],
                                 [j['send_id'] for j in cfg['joints']],
                                 [j['recv_id'] for j in cfg['joints']])
        self.arm.set_callback_mode_all(oa.CallbackMode.STATE)
        self.component = self.arm.get_arm()
        self.motors = self.component.get_motors()
        self.load_gravity_model()
        # Establish initial feedback, without enabling motors.
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            self.arm.refresh_all()
            # recv_all returns as soon as any frame arrives; without a pause this loop
            # outruns the USB adapter, overflows the kernel TX queue and latches the bus unhealthy.
            time.sleep(0.005)
            self.arm.recv_all(10000)
            if all(not self.component.get_link_stats(i).is_stale(cfg['feedback_timeout'])
                   for i in range(len(self.motors))):
                break
        self.read()

    def read(self):
        self.arm.refresh_all()
        time.sleep(0.002)  # Let the adapter drain before the next burst of writes.
        self.arm.recv_all(10000)
        self.motors = self.component.get_motors()
        if not self.arm.is_bus_healthy() or not self.arm.is_link_running():
            raise RuntimeError('CAN bus is unhealthy or has no carrier.')
        positions = []
        for i, (j, motor) in enumerate(zip(self.cfg['joints'], self.motors)):
            if self.component.get_link_stats(i).is_stale(self.cfg['feedback_timeout']):
                raise RuntimeError(f'Stale feedback from motor {j["send_id"]}.')
            if motor.has_error():
                raise RuntimeError(f'{j["name"]}: {self.oa.motor_error_to_string(motor.get_error_code())}')
            if max(motor.get_state_tmos(), motor.get_state_trotor()) > self.cfg['max_temperature']:
                raise RuntimeError(f'{j["name"]}: temperature exceeds configured limit.')
            positions.append(model_position(j, motor.get_position()))
        return positions

    def load_gravity_model(self):
        import mujoco
        self.mujoco = mujoco
        self.gmodel = mujoco.MjModel.from_xml_path(str(root() / self.cfg['model']))
        self.gdata = mujoco.MjData(self.gmodel)
        ids = [self.gmodel.joint(j['name']).id for j in self.cfg['joints']]
        self.gqadr = [int(self.gmodel.jnt_qposadr[i]) for i in ids]
        self.gvadr = [int(self.gmodel.jnt_dofadr[i]) for i in ids]

    def gravity(self, positions):
        """Model joint torques (Nm) that hold the arm still against gravity at these positions."""
        self.gdata.qpos[self.gqadr] = positions
        self.gdata.qvel[:] = 0
        self.mujoco.mj_forward(self.gmodel, self.gdata)
        return [float(self.gdata.qfrc_bias[v]) for v in self.gvadr]

    def feedforward(self, positions):
        """Motor-frame gravity torques, scaled by gravity_scale and capped per joint at tau_max."""
        scale = self.cfg.get('gravity_scale', 0.0)
        if not scale:
            return [0.0] * len(self.cfg['joints'])
        return [max(-j.get('tau_max', 0.0), min(j.get('tau_max', 0.0), j['sign'] * scale * tau))
                for j, tau in zip(self.cfg['joints'], self.gravity(positions))]

    def command(self, target, actual):
        self.component.mit_control_all([
            self.oa.MITParam(j['kp'], j['kd'], motor_position(j, value), 0, tau)
            for j, value, tau in zip(self.cfg['joints'], target, self.feedforward(actual))])

    def status(self):
        positions = self.read()
        return [{'joint': j['name'], 'motor_id': j['send_id'], 'position': p,
                 'velocity': j['sign'] * m.get_velocity(), 'torque': m.get_torque(),
                 'gravity_feedforward': tau,
                 'enabled': m.is_enabled(), 'temperature_mos': m.get_state_tmos(),
                 'temperature_rotor': m.get_state_trotor()}
                for j, m, p, tau in zip(self.cfg['joints'], self.motors, positions, self.feedforward(positions))]

    def check_tracking(self, actual, expected, what):
        errors = [abs(a - b) for a, b in zip(actual, expected)]
        worst = max(range(len(errors)), key=errors.__getitem__)
        if errors[worst] > self.cfg['max_tracking_error']:
            raise RuntimeError(f'{what} exceeded configured limit: {self.cfg["joints"][worst]["name"]} is at '
                               f'{actual[worst]:+.3f} rad, expected {expected[worst]:+.3f} '
                               f'(error {errors[worst]:.3f} > {self.cfg["max_tracking_error"]}).')

    def move(self, points, *, on_feedback=None, stop=None, hold=0):
        """Follow points, settle, then keep holding the final target for hold seconds (math.inf: until stopped)."""
        if any(m.is_enabled() for m in self.motors):
            raise RuntimeError('Motors are already enabled; stop the existing controller first.')
        previous = self.read()
        validate_positions(self.cfg, previous)
        if on_feedback is not None:
            on_feedback(previous)
        self.enabled = True  # Cleanup even if enabling only partially succeeds.
        try:
            self.arm.enable_all()
            time.sleep(0.02)  # Let the enable frames drain before polling feedback.
            started = time.monotonic()
            for t, q in points:
                if stop is not None and stop.is_set():
                    raise RuntimeError('Stopped.')
                wait = started + t - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                elif -wait > self.cfg['feedback_timeout']:
                    raise RuntimeError('Control deadline missed.')
                actual = self.read()
                validate_positions(self.cfg, actual)
                if on_feedback is not None:
                    on_feedback(actual)
                self.check_tracking(actual, previous, 'Tracking error')
                self.command(q, actual)
                self.arm.recv_all(10000)
                previous = q
            # Observe settling (0.5 s), then optionally keep holding, while commanding the final target.
            settle_end = time.monotonic() + 0.5
            hold_end = settle_end + hold
            reported = False
            while time.monotonic() < hold_end:
                if stop is not None and stop.is_set():
                    if time.monotonic() >= settle_end and hold:
                        break  # Stopping a hold is the normal way to end it.
                    raise RuntimeError('Stopped.')
                actual = self.read()
                validate_positions(self.cfg, actual)
                if on_feedback is not None:
                    on_feedback(actual)
                self.check_tracking(actual, previous, 'Final tracking error')
                self.command(previous, actual)
                if hold and not reported and time.monotonic() >= settle_end:
                    reported = True
                    until = 'Ctrl-C' if math.isinf(hold) else f'{hold:g} s'
                    print(f'Reached target; measured {[round(a, 3) for a in actual]}. Holding until {until}.', flush=True)
                time.sleep(0.01)
            print('Final measured joint positions:', self.read())
        finally:
            self.close()

    def close(self):
        if not self.enabled:
            return
        # A full TX queue silently drops disable frames, so retry until every motor reports disabled.
        for _ in range(20):
            self.arm.disable_all()
            time.sleep(0.02)
            self.arm.refresh_all()
            time.sleep(0.01)
            self.arm.recv_all(10000)
            if not any(m.is_enabled() for m in self.component.get_motors()):
                self.enabled = False
                return
        raise RuntimeError('Could not confirm motors disabled. Press the emergency stop or cut arm power.')
