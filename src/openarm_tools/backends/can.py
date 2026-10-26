"""Linux SocketCAN adapter for the pinned OpenArm CAN Python bindings."""
import platform
import time
from openarm_tools.robot.config import finite, validate_positions


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
        # Establish initial feedback, without enabling motors.
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            self.arm.refresh_all()
            self.arm.recv_all(10000)
            if all(not self.component.get_link_stats(i).is_stale(cfg['feedback_timeout'])
                   for i in range(len(self.motors))):
                break
        self.read()

    def read(self):
        self.arm.refresh_all()
        self.arm.recv_all(10000)
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
            positions.append(finite(j['sign'] * motor.get_position()))
        return positions

    def status(self):
        positions = self.read()
        return [{'joint': j['name'], 'motor_id': j['send_id'], 'position': p,
                 'velocity': j['sign'] * m.get_velocity(), 'torque': m.get_torque(),
                 'enabled': m.is_enabled(), 'temperature_mos': m.get_state_tmos(),
                 'temperature_rotor': m.get_state_trotor()}
                for j, m, p in zip(self.cfg['joints'], self.motors, positions)]

    def move(self, points):
        if not self.cfg.get('hardware_reviewed', False):
            raise ValueError('Review the physical robot mapping, limits and gains in a custom config; set hardware_reviewed=true.')
        if any(m.is_enabled() for m in self.motors):
            raise RuntimeError('Motors are already enabled; stop the existing controller first.')
        previous = self.read()
        validate_positions(self.cfg, previous)
        self.enabled = True  # Cleanup even if enabling only partially succeeds.
        try:
            self.arm.enable_all()
            started = time.monotonic()
            for t, q in points:
                wait = started + t - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                elif -wait > self.cfg['feedback_timeout']:
                    raise RuntimeError('Control deadline missed.')
                actual = self.read()
                validate_positions(self.cfg, actual)
                if max(abs(a - b) for a, b in zip(actual, previous)) > self.cfg['max_tracking_error']:
                    raise RuntimeError('Tracking error exceeded configured limit.')
                self.component.mit_control_all([
                    self.oa.MITParam(j['kp'], j['kd'], j['sign'] * value, 0, 0)
                    for j, value in zip(self.cfg['joints'], q)])
                self.arm.recv_all(10000)
                previous = q
            # Observe settling while still commanding the final target.
            for _ in range(50):
                actual = self.read()
                validate_positions(self.cfg, actual)
                if max(abs(a - b) for a, b in zip(actual, previous)) > self.cfg['max_tracking_error']:
                    raise RuntimeError('Final tracking error exceeded configured limit.')
                self.component.mit_control_all([
                    self.oa.MITParam(j['kp'], j['kd'], j['sign'] * value, 0, 0)
                    for j, value in zip(self.cfg['joints'], previous)])
                time.sleep(0.01)
            print('Final measured joint positions:', self.read())
        finally:
            self.close()

    def close(self):
        if self.enabled:
            self.arm.disable_all()
            self.arm.recv_all(10000)
            self.enabled = False
