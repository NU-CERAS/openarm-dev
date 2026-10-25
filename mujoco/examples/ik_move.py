#!/usr/bin/env python3
"""Plan a small Cartesian move, validate it in MuJoCo, optionally execute over CAN.

Default: feedback/parameter reads only. --offline never opens CAN. --live requires
a verified mapping file. Model collision checks do not certify physical safety:
tables, people, cables and payloads are absent unless added to the model.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import select
import signal
import socket
import struct
import sys
import tempfile
import time

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SIDES = ('right', 'left')
PERIOD = 0.02
MAX_TRACKING = 0.08
TORQUE_LIMIT = np.array([6., 6., 4., 4., 2., 1.5, 1.5])
KP = np.array([20., 20., 12., 12., 8., 6., 6.])
KD = np.array([1., 1., .7, .7, .4, .3, .3])


class Refused(RuntimeError):
    """A required planning or execution check failed."""


def vector(value, size, name):
    a = np.asarray(value, dtype=float)
    if a.shape != (size,) or not np.isfinite(a).all():
        raise Refused(f'{name} must contain {size} finite numbers')
    return a


def blend(t):
    u = np.clip(t, 0, 1)
    return u**3 * (10 + u * (-15 + 6 * u))


def rotation_error(target, actual):
    r = target @ actual.T
    theta = math.acos(np.clip((np.trace(r) - 1) / 2, -1, 1))
    v = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    if theta < 1e-7:
        return v / 2
    if math.pi - theta < .001:
        raise Refused('Orientation is too far from the starting orientation')
    return v * theta / (2 * math.sin(theta))


def load_mapping(path, live=False):
    cfg = json.loads(Path(path).read_text())
    if live and cfg.get('verified') is not True:
        raise Refused('Live motion needs verified motor/model zeros and signs. See ik_mapping.example.json.')
    for side in SIDES:
        entry = cfg[side]
        entry['sign'] = vector(entry['sign'], 7, side + ' signs')
        entry['offset_rad'] = vector(entry['offset_rad'], 7, side + ' offsets')
        entry['jaw_qpos_m'] = vector(entry['jaw_qpos_m'], 2, side + ' jaw positions')
        if not np.isin(entry['sign'], [-1, 1]).all():
            raise Refused('Each joint sign must be +1 or -1')
        if np.any(entry['jaw_qpos_m'] < -.0451) or np.any(entry['jaw_qpos_m'] > 0):
            raise Refused('Jaw model positions must be between -0.0451 and 0 meters')
    if cfg['left']['interface'] == cfg['right']['interface']:
        raise Refused('The two arms must use different interfaces')
    return cfg


class Planner:
    def __init__(self, model, cfg, clearance=.003):
        self.model = mujoco.MjModel.from_xml_path(str(model))
        self.data = mujoco.MjData(self.model)
        self.clearance = clearance
        active = (self.model.geom_contype != 0) | (self.model.geom_conaffinity != 0)
        self.model.geom_margin[active] = np.maximum(self.model.geom_margin[active], clearance)
        self.ids, self.qadr, self.dadr = {}, {}, {}
        self.limits = {}
        for side in SIDES:
            ids = np.array([self.model.joint(f'{side}_rev{i}').id for i in range(1, 8)])
            self.ids[side] = ids
            self.qadr[side] = self.model.jnt_qposadr[ids]
            self.dadr[side] = self.model.jnt_dofadr[ids]
            self.limits[side] = self.model.jnt_range[ids].copy()
            for name, value in zip(('left_pris1', 'right_pris2'), cfg[side]['jaw_qpos_m']):
                self.data.joint(f'{side}_{name}').qpos[0] = value

    def forward(self, arms):
        for side in SIDES:
            self.data.qpos[self.qadr[side]] = vector(arms[side], 7, side + ' model angles')
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def check(self, arms):
        violations = []
        for side in SIDES:
            q = vector(arms[side], 7, side + ' model angles')
            bounds = self.limits[side]
            # Allow only encoder quantization at an exact mechanical boundary.
            for j in np.flatnonzero((q < bounds[:, 0] - .0005) | (q > bounds[:, 1] + .0005)):
                excess = max(bounds[j, 0] - q[j], q[j] - bounds[j, 1])
                violations.append(f'{side} J{j + 1}: {q[j]:+.6f} rad ({math.degrees(q[j]):+.3f} deg), '
                                  f'allowed [{bounds[j, 0]:+.6f}, {bounds[j, 1]:+.6f}] rad, '
                                  f'outside by {math.degrees(excess):.3f} deg')
        if violations:
            raise Refused('Angle outside model joint limits:\n  ' + '\n  '.join(violations)
                          + '\nCheck physical pose and mapping. If BOTH arms are physically at the '
                          'reference zero pose, --capture-reference-zero records software offsets; '
                          'it must not be used to hide a different pose or direction mismatch.')
        self.forward(arms)
        for c in self.data.contact:
            if c.dist < self.clearance:
                a, b = self.model.geom(c.geom1).name, self.model.geom(c.geom2).name
                raise Refused(f'Modeled collision/clearance: {a} / {b}, {c.dist * 1000:.2f} mm')

    def pose(self, side):
        body = self.data.body(side + '_gripper_center')
        return body.xpos.copy(), body.xmat.reshape(3, 3).copy()

    def solve(self, start, side, target, max_move=.6, keep_orientation=False):
        self.check(start)
        origin, orientation = self.pose(side)
        target = vector(target, 3, 'target')
        if np.linalg.norm(target - origin) > .050001:
            raise Refused('First-test targets are limited to 5 cm from the current gripper pose')
        q0 = start[side].copy()
        limits = self.limits[side]
        # Keep a 0.02 rad margin where available; do not push a boundary-start inward abruptly.
        lo = np.maximum(limits[:, 0], np.minimum(limits[:, 0] + .02, q0))
        hi = np.minimum(limits[:, 1], np.maximum(limits[:, 1] - .02, q0))
        lo, hi = np.maximum(lo, q0 - max_move), np.minimum(hi, q0 + max_move)
        q = np.clip(q0, lo, hi)
        arms = {s: start[s].copy() for s in SIDES}
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros_like(jacp)
        body_id = self.model.body(side + '_gripper_center').id
        rotation_weight = .2 if keep_orientation else .03

        def error(candidate):
            arms[side] = candidate
            self.forward(arms)
            p, r = self.pose(side)
            return np.r_[target - p, rotation_weight * rotation_error(orientation, r)]

        # The straight zero pose is near singular. Try both elbow branches in
        # computation only; every candidate stays inside the same movement bounds.
        seeds = [q.copy()]
        for elbow in (.15, -.15, .30):
            seed = q0.copy()
            seed[3] += elbow
            seeds.append(np.clip(seed, lo, hi))
        candidates = []
        for seed in seeds:
            q = seed
            for _ in range(400):
                e = error(q)
                rotation_limit = .035 if keep_orientation else math.radians(15)
                if np.linalg.norm(e[:3]) <= .001 and np.linalg.norm(e[3:]) <= rotation_weight * rotation_limit:
                    try:
                        self.validate_path(start, side, q)
                    except Refused:
                        break
                    candidates.append((np.linalg.norm(q - q0), q.copy(), float(np.linalg.norm(e[:3]))))
                    break
                mujoco.mj_jacBody(self.model, self.data, jacp, jacr, body_id)
                j = np.vstack((jacp[:, self.dadr[side]], rotation_weight * jacr[:, self.dadr[side]]))
                gradient = j.T @ e
                free = ~(((q <= lo + 1e-8) & (gradient < 0)) | ((q >= hi - 1e-8) & (gradient > 0)))
                jf = j[:, free]
                dq = np.zeros(7)
                dq[free] = jf.T @ np.linalg.solve(jf @ jf.T + .000025 * np.eye(len(e)), e)
                dq *= min(1., .025 / max(np.max(np.abs(dq)), 1e-12))
                accepted = False
                for direction in (dq, gradient):
                    for factor in (1., .5, .25, .1, .05):
                        trial = np.clip(q + factor * direction, lo, hi)
                        trial_error = error(trial)
                        if np.linalg.norm(trial_error) < np.linalg.norm(e) - 1e-10:
                            q, accepted = trial, True
                            break
                    if accepted:
                        break
                if not accepted:
                    break
        if candidates:
            _, best, residual = min(candidates, key=lambda item: item[0])
            return best, residual, origin, orientation
        raise Refused('IK did not converge within the small-move bounds. Try a smaller/different target.')

    def validate_path(self, start, side, goal):
        goal = vector(goal, 7, 'IK goal')
        if np.max(np.abs(goal - start[side])) > .600001:
            raise Refused('Path exceeds the 0.6 rad per-joint first-test movement limit')
        self.forward(start)
        orientation = self.pose(side)[1]
        count = max(101, int(np.ceil(np.max(np.abs(goal - start[side])) / .002)) + 1)
        for alpha in np.linspace(0, 1, count):
            arms = {s: start[s].copy() for s in SIDES}
            arms[side] += alpha * (goal - start[side])
            self.check(arms)
            if np.linalg.norm(rotation_error(orientation, self.pose(side)[1])) > math.radians(15):
                raise Refused('Path would rotate the gripper by more than 15 degrees')
        return count


class MotorBus:
    """Classic-CAN requests; strict matching and fresh feedback on every snapshot."""
    def __init__(self, interface):
        self.interface = interface
        self.sock = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        try:
            self.sock.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FD_FRAMES, 1)
            self.sock.settimeout(.03)
            self.sock.bind((interface,))
            self.limits = np.array([[self.register(i, rid) for rid in (21, 22, 23)]
                                    for i in range(1, 9)])
            if not np.isfinite(self.limits).all() or np.any(self.limits <= 0):
                raise Refused(f'{interface}: invalid motor scaling parameters')
        except BaseException:
            self.sock.close()
            raise

    def send(self, cid, payload):
        self.sock.send(struct.pack('=IB3x8s', cid, 8, bytes(payload)))

    def drain(self):
        for _ in range(512):
            if not select.select([self.sock], [], [], 0)[0]:
                return
            self.sock.recv(72)
        raise Refused(f'{self.interface}: excessive bus traffic / another controller')

    def read(self, deadline):
        if not select.select([self.sock], [], [], max(0, deadline - time.monotonic()))[0]:
            return None
        frame = self.sock.recv(72)
        cid, n = struct.unpack_from('=IB', frame)
        return cid, frame[8:8+n]

    def register(self, motor, rid):
        self.drain()
        self.send(0x7FF, [motor, 0, 0x33, rid, 0, 0, 0, 0])
        deadline = time.monotonic() + .15
        while time.monotonic() < deadline:
            result = self.read(deadline)
            if result is None:
                break
            cid, data = result
            if cid == motor + 0x10 and len(data) == 8 and data[:4] == bytes([motor, 0, 0x33, rid]):
                return struct.unpack('<I' if rid in (9, 10) else '<f', data[4:])[0]
        raise Refused(f'{self.interface}: motor {motor} did not answer register {rid}')

    def snapshot(self):
        self.drain()
        for i in range(1, 9):
            self.send(0x7FF, [i, 0, 0xCC, 0, 0, 0, 0, 0])
            time.sleep(.0002)
        frames = {}
        deadline = time.monotonic() + .08
        while len(frames) < 8 and time.monotonic() < deadline:
            result = self.read(deadline)
            if result is None:
                break
            cid, d = result
            if 0x11 <= cid <= 0x18 and len(d) == 8 and d[0] & 15 == cid - 0x10:
                frames[cid - 0x10] = d
        if len(frames) != 8:
            raise Refused(f'{self.interface}: missing fresh feedback from IDs {sorted(set(range(1, 9)) - frames.keys())}')
        rows = []
        for i in range(1, 9):
            d = frames[i]
            raw = np.array([d[1] << 8 | d[2], d[3] << 4 | d[4] >> 4, (d[4] & 15) << 8 | d[5]])
            physical = (raw / [65535, 4095, 4095] * 2 - 1) * self.limits[i-1]
            status = d[0] >> 4
            if status not in (0, 1) or max(d[6:8]) >= 60:
                raise Refused(f'{self.interface}: motor {i} fault/status {status}, temperatures {d[6:8].hex()}')
            rows.append([*physical, status])
        return np.array(rows)

    def special(self, motor, code):
        if code not in (0xFC, 0xFD):
            raise Refused('Only enable/disable are permitted; zero and flash writes are not implemented')
        self.send(motor, [255] * 7 + [code])

    def command(self, q):
        for i in range(7):
            self.send(i + 1, mit_packet(q[i], KP[i], KD[i], self.limits[i]))


def mit_packet(q, kp, kd, limits):
    """Encode a position hold with zero desired velocity and zero feedforward torque."""
    pmax, vmax, tmax = limits
    if not np.isfinite([q, kp, kd, *limits]).all() or not -pmax <= q <= pmax:
        raise Refused('Invalid or out-of-range motor command')
    def encode(x, lo, hi, bits):
        if not lo <= x <= hi:
            raise Refused('Command exceeds protocol limits')
        return int((x - lo) / (hi - lo) * ((1 << bits) - 1))
    p, v, k, d, t = (encode(q, -pmax, pmax, 16), encode(0, -vmax, vmax, 12),
                      encode(kp, 0, 500, 12), encode(kd, 0, 5, 12), encode(0, -tmax, tmax, 12))
    return bytes([p >> 8, p & 255, v >> 4, (v & 15) << 4 | k >> 8,
                  k & 255, d >> 4, (d & 15) << 4 | t >> 8, t & 255])


def model_angles(samples, cfg):
    return {s: samples[s][:7, 0] * cfg[s]['sign'] + cfg[s]['offset_rad'] for s in SIDES}


def reference_offsets(history, cfg):
    """Compute small corrections only from stationary, disabled reference-pose samples."""
    offsets, summary = {}, {}
    for side in SIDES:
        a = np.asarray(history[side], dtype=float)
        if a.ndim != 3 or a.shape[1:] != (8, 4) or len(a) < 10 or not np.isfinite(a).all():
            raise Refused(f'{side}: invalid reference samples')
        if np.any(a[:, :, 3] != 0):
            raise Refused(f'{side}: disable the existing controller before capturing a reference')
        if np.max(np.abs(a[:, :, 1])) > .1 or np.max(np.ptp(a[:, :, 0], axis=0)) > .003:
            raise Refused(f'{side}: arm/gripper moved during reference capture; support it and retry')
        zero = np.median(a[:, :7, 0], axis=0)
        old_model = zero * cfg[side]['sign'] + cfg[side]['offset_rad']
        if np.max(np.abs(old_model)) > math.radians(5):
            raise Refused(f'{side}: reference correction exceeds 5 degrees; inspect pose/signs instead')
        offsets[side] = (-cfg[side]['sign'] * zero).tolist()
        summary[side] = dict(motor_reference_rad=zero.tolist(),
                             old_reference_model_deg=np.degrees(old_model).tolist(),
                             new_offset_rad=offsets[side])
    return offsets, summary


def capture_reference_zero(buses, cfg, path):
    path = Path(path)
    original = path.read_text()
    history = {s: [] for s in SIDES}
    print('Reading the confirmed physical reference pose. Keep BOTH arms supported and still.', flush=True)
    for _ in range(20):
        for side in SIDES:
            history[side].append(buses[side].snapshot())
        time.sleep(.03)
    offsets, summary = reference_offsets(history, cfg)
    updated = json.loads(original)
    for side in SIDES:
        updated[side]['offset_rad'] = offsets[side]
    updated['reference_capture'] = dict(utc=datetime.now(timezone.utc).isoformat(),
                                        method='Operator-confirmed physical model zero; 20 stationary samples',
                                        readings=summary)
    # Preserve sign verification and all unrelated settings. Do not automatically
    # mark an unverified mapping as verified merely because a zero was captured.
    updated['note'] = ('q_model = sign * q_motor + offset_rad. Software reference offsets were '
                       'captured at the operator-confirmed model zero. Motor zeros were not changed. '
                       'Reverify signs, jaw geometry and this mapping after reassembly or motor zeroing.')
    if path.read_text() != original:
        raise Refused('Mapping changed during sampling; no calibration was saved')
    backup = path.with_name(path.name + '.before-reference-' + str(time.time_ns()))
    with backup.open('x') as out:
        out.write(original)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=path.name + '.', delete=False) as out:
            temporary = Path(out.name)
            out.write(json.dumps(updated, indent=2) + '\n')
        os.chmod(temporary, path.stat().st_mode & 0o777)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(json.dumps(summary, indent=2))
    print(f'Software reference saved: {path}\nPrevious mapping: {backup}\n'
          'No enable, motion, disable, motor-zero or flash commands sent.')


def preflight(samples, buses, selected):
    for side in SIDES:
        if np.any(samples[side][:, 3] != 0):
            raise Refused(f'{side}: motors already enabled; stop the existing controller first')
        if np.max(np.abs(samples[side][:7, 1])) > .1:
            raise Refused(f'{side}: arm is moving; support it and wait')
        for i in (range(1, 8) if side == selected else ()):
            if buses[side].register(i, 10) != 1:
                raise Refused(f'{side} motor {i}: requires MIT mode; this script will not change it')
            if buses[side].register(i, 9) == 0:
                raise Refused(f'{side} motor {i}: communication watchdog is disabled')


def tracking_detail(side, actual, target, sample):
    error = actual - target
    joint = int(np.argmax(np.abs(error)))
    return (f'{side} J{joint + 1}: error {error[joint]:+.5f} rad '
            f'({math.degrees(error[joint]):+.2f} deg), target {target[joint]:+.5f}, '
            f'measured {actual[joint]:+.5f}, velocity {sample[joint, 1]:+.4f} rad/s, '
            f'torque {sample[joint, 2]:+.3f} Nm')


def execute(planner, start, side, goal, duration, buses, cfg):
    """Sole writer for selected J1-J7; other arm and grippers are read only."""
    samples = {s: buses[s].snapshot() for s in SIDES}
    preflight(samples, buses, side)
    # Parameter reads can take time. Recheck state immediately before enabling.
    samples = {s: buses[s].snapshot() for s in SIDES}
    if any(np.any(samples[s][:, 3] != 0) or np.max(np.abs(samples[s][:7, 1])) > .1 for s in SIDES):
        raise Refused('Motor state changed during preflight; rerun with both arms stationary and disabled')
    initial_grippers = {s: samples[s][7, 0] for s in SIDES}
    actual = model_angles(samples, cfg)
    if any(np.max(np.abs(actual[s] - start[s])) > .01 for s in SIDES):
        raise Refused('An arm moved after planning; rerun to generate a fresh plan')
    planner.check(actual)
    bus = buses[side]
    other = 'right' if side == 'left' else 'left'
    attempted = []
    last_target = start[side].copy()
    trace_path = Path(tempfile.gettempdir()) / f'openarm-ik-trace-{time.time_ns()}.jsonl'
    trace = trace_path.open('x', buffering=1)
    trace.write(json.dumps(dict(event='start', arm=side, start_rad=start[side].tolist(),
                                goal_rad=goal.tolist(), duration_s=duration,
                                kp=KP.tolist(), kd=KD.tolist(), max_tracking_rad=MAX_TRACKING)) + '\n')
    print('LIVE: moving, then holding the target. Ctrl+C releases this arm; support it first.', flush=True)
    print(f'Per-joint telemetry: {trace_path}', flush=True)
    try:
        # Replace any old MIT command while disabled before enabling.
        bus.command((start[side] - cfg[side]['offset_rad']) * cfg[side]['sign'])
        for i in range(1, 8):
            attempted.append(i)
            bus.special(i, 0xFC)
        begun, previous = time.monotonic(), time.monotonic()
        last_print = -1.
        while True:
            tick = time.monotonic()
            if tick - previous > .1:
                raise Refused('Control loop stalled for over 100 ms')
            previous = tick
            samples = {s: buses[s].snapshot() for s in SIDES}
            actual = model_angles(samples, cfg)
            trace.write(json.dumps(dict(event='feedback', elapsed_s=tick - begun,
                                        target_model_rad=last_target.tolist(),
                                        target_motor_rad=((last_target - cfg[side]['offset_rad']) * cfg[side]['sign']).tolist(),
                                        actual_model_rad=actual[side].tolist(),
                                        actual_motor_rad=samples[side][:7, 0].tolist(),
                                        velocity_rad_s=samples[side][:7, 1].tolist(),
                                        torque_nm=samples[side][:7, 2].tolist(),
                                        status=samples[side][:7, 3].astype(int).tolist())) + '\n')
            if tick - begun - last_print >= .5:
                print(f't={tick - begun:.2f}s ' + tracking_detail(side, actual[side], last_target, samples[side]), flush=True)
                last_print = tick - begun
            if np.any(samples[side][:7, 3] != 1):
                raise Refused('Selected motor disabled unexpectedly')
            if np.any(samples[other][:, 3] != 0):
                raise Refused('Other arm was enabled by another controller')
            if any(samples[s][7, 3] != 0 or abs(samples[s][7, 0] - initial_grippers[s]) > .02 for s in SIDES):
                raise Refused('A gripper moved or was enabled; model geometry is no longer valid')
            if np.max(np.abs(actual[other] - start[other])) > .02:
                raise Refused('Other arm moved; collision plan is no longer valid')
            if np.max(np.abs(actual[side] - last_target)) > MAX_TRACKING:
                raise Refused('Tracking error exceeded 0.08 rad: '
                              + tracking_detail(side, actual[side], last_target, samples[side]))
            if np.max(np.abs(samples[side][:7, 1])) > .5:
                raise Refused('Measured joint speed exceeded 0.5 rad/s')
            if np.any(np.abs(samples[side][:7, 2]) > TORQUE_LIMIT):
                raise Refused('Measured torque exceeded conservative first-test limits')
            planner.check(actual)
            target = start[side] + blend((tick - begun) / duration) * (goal - start[side])
            if np.max(np.abs(target - last_target)) > .01:
                raise Refused('Command step exceeded 0.01 rad')
            commanded = {s: actual[s].copy() for s in SIDES}
            commanded[side] = target
            planner.check(commanded)
            if time.monotonic() - tick > .08:
                raise Refused('Feedback/check cycle exceeded 80 ms')
            bus.command((target - cfg[side]['offset_rad']) * cfg[side]['sign'])
            last_target = target
            if tick - begun >= duration and tick - begun < duration + PERIOD * 2:
                print('Target reached by the command trajectory; holding. Ctrl+C releases.', flush=True)
            time.sleep(max(0, PERIOD - (time.monotonic() - tick)))
    except BaseException as exc:
        trace.write(json.dumps(dict(event='stop', reason=str(exc), exception=type(exc).__name__)) + '\n')
        raise
    finally:
        failures = []
        for i in attempted:
            try:
                bus.special(i, 0xFD)
            except OSError as exc:
                failures.append(f'{i}: {exc}')
        print('Disable commands sent to selected joints.' if not failures else
              'Disable transmission failed; use the hardware stop: ' + '; '.join(failures), flush=True)
        trace.close()


def preview(planner, start, side, goal, duration):
    import mujoco.viewer
    with mujoco.viewer.launch_passive(planner.model, planner.data) as viewer:
        print('Kinematic path preview (not a dynamics test). Close viewer to continue.', flush=True)
        begun = time.monotonic()
        while viewer.is_running():
            phase = ((time.monotonic() - begun) % (duration + 2)) / duration
            arms = {s: start[s].copy() for s in SIDES}
            arms[side] += blend(phase) * (goal - start[side])
            with viewer.lock():
                planner.forward(arms)
            viewer.sync()
            time.sleep(.02)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=SIDES, default='right')
    p.add_argument('--keep-orientation', action='store_true', help='Also constrain the final gripper orientation')
    target = p.add_mutually_exclusive_group()
    target.add_argument('--delta', nargs=3, type=float, metavar=('DX', 'DY', 'DZ'))
    target.add_argument('--target', nargs=3, type=float, metavar=('X', 'Y', 'Z'))
    p.add_argument('--model', type=Path, default=ROOT / 'models/bimanual.xml')
    p.add_argument('--mapping', type=Path, default=ROOT / 'examples/ik_mapping.json')
    p.add_argument('--offline', action='store_true', help='Use supplied model angles; never connect CAN')
    p.add_argument('--q-left', nargs=7, type=float, default=[0.] * 7)
    p.add_argument('--q-right', nargs=7, type=float, default=[0.] * 7)
    p.add_argument('--preview', action='store_true')
    p.add_argument('--live', action='store_true', help='Execute only after every planning/preflight check passes')
    p.add_argument('--capture-reference-zero', action='store_true',
                   help='BOTH arms must physically match model zero; save stationary readings as software offsets only')
    p.add_argument('--output', type=Path, help='Save a plan report; it cannot be replayed as a motion command')
    args = p.parse_args(argv)
    if args.offline and args.live:
        raise Refused('--offline and --live cannot be combined')
    if args.capture_reference_zero and (args.offline or args.live or args.preview or
                                        args.target is not None or args.delta is not None or args.output):
        raise Refused('--capture-reference-zero must be run separately from planning/preview/motion')
    cfg = load_mapping(args.mapping, args.live)
    buses = {}
    try:
        if args.offline:
            start = {'left': vector(args.q_left, 7, 'left start'),
                     'right': vector(args.q_right, 7, 'right start')}
        else:
            for side in SIDES:
                buses[side] = MotorBus(cfg[side]['interface'])
            if args.capture_reference_zero:
                capture_reference_zero(buses, cfg, args.mapping)
                return
            start = model_angles({s: buses[s].snapshot() for s in SIDES}, cfg)
        planner = Planner(args.model, cfg)
        planner.check(start)
        origin, _ = planner.pose(args.arm)
        target_xyz = (vector(args.target, 3, 'target') if args.target is not None else
                      origin + vector(args.delta if args.delta is not None else [0, 0, .02], 3, 'delta'))
        goal, residual, _, _ = planner.solve(start, args.arm, target_xyz, keep_orientation=args.keep_orientation)
        points = planner.validate_path(start, args.arm, goal)
        distance = float(np.max(np.abs(goal - start[args.arm])))
        duration = max(3., 1.875 * distance / .08, math.sqrt(5.774 * distance / .15))
        report = dict(arm=args.arm, frame='MuJoCo world; meters', start_xyz=origin.tolist(),
                      target_xyz=target_xyz.tolist(), start_rad=start[args.arm].tolist(),
                      goal_rad=goal.tolist(), residual_mm=residual * 1000,
                      duration_s=duration, checked_path_samples=points,
                      max_joint_move_deg=math.degrees(distance),
                      goal_deg=np.degrees(goal).tolist(),
                      clearance_mm=planner.clearance * 1000, mapping_verified=cfg.get('verified', False),
                      warning='Model-checked only. Requires verified physical mapping and a clear workspace.')
        print(json.dumps(report, indent=2), flush=True)
        if args.output:
            args.output.write_text(json.dumps(report, indent=2) + '\n')
        if args.preview:
            preview(planner, start, args.arm, goal, duration)
        if args.live:
            execute(planner, start, args.arm, goal, duration, buses, cfg)
        else:
            print('PLAN ONLY: no enable, motion, disable, zero, or flash commands sent.')
    finally:
        for bus in buses.values():
            bus.sock.close()


if __name__ == '__main__':
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    if hasattr(signal, 'SIGHUP'):
        signal.signal(signal.SIGHUP, interrupted)
    try:
        main()
    except KeyboardInterrupt:
        print('Stopped.')
    except (Refused, OSError, ValueError, KeyError) as exc:
        print(f'REFUSED: {exc}', file=sys.stderr)
        sys.exit(1)
