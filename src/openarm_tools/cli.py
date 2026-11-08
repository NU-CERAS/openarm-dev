"""Common operator commands. Choose simulated arms with --sim and physical arms with --real."""
import argparse
import importlib
import json
import math
import platform
import re
import subprocess
import sys
import threading
import time
from contextlib import ExitStack
from pathlib import Path
from openarm_tools.robot.config import load, root, validate_positions, finite
from openarm_tools.recording.storage import lock, write_json
from openarm_tools.calibration import offsets
from openarm_tools.motion.trajectory import trajectory

# Each CAN interface is wired to one arm; simulation uses the same names to select arms.
ARMS = {'can0': 'left', 'can1': 'right'}


def positive(text):
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError('Expected a positive finite number.')
    return value


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('doctor', 'sim', 'status', 'can-check', 'zero', 'home', 'jog', 'pose'):
        cmd = sub.add_parser(name)
        cmd.add_argument('--config', action='append', default=[], metavar='IFACE=PATH',
                         help='Use another arm configuration, e.g. can1=runs/configs/right-can1.json')
        cmd.add_argument('--session', default='default', help='Named local simulation session')
        cmd.add_argument('--data-dir', type=Path, default=root() / 'runs')
        if name in ('status', 'zero', 'home', 'jog', 'pose'):
            cmd.add_argument('--sim', action='extend', nargs='+', choices=ARMS, default=[], metavar='IFACE',
                             help='Simulated arms: can0 (left) and/or can1 (right)')
        if name in ('status', 'can-check', 'zero', 'home', 'jog', 'pose'):
            cmd.add_argument('--real', action='extend', nargs='+', choices=ARMS, default=[], metavar='IFACE',
                             help='Physical arms on CAN: can0 (left) and/or can1 (right)')
        if name in ('home', 'jog', 'pose'):
            cmd.add_argument('--duration', type=positive, default=5)
            cmd.add_argument('--hold', type=positive, nargs='?', const=math.inf, default=0, metavar='SECONDS',
                             help='Real arms keep holding the target afterward: for SECONDS, or until Ctrl-C if omitted')
        if name == 'doctor':
            cmd.add_argument('--profile', choices=['sim', 'ros', 'hardware'], default='sim')
        if name == 'sim':
            cmd.add_argument('--headless', action='store_true')
            cmd.add_argument('--seconds', type=positive, default=5, help='Headless run duration')
            cmd.add_argument('--reset', action='store_true')
            cmd.add_argument('--ros', action='store_true', help='Launch the v1.0 ROS-control simulation instead')
        if name == 'jog':
            cmd.add_argument('joint', help='Joint number 1-7 (applies to every selected arm) or exact joint name')
            cmd.add_argument('delta', type=float, help='Increment in radians')
        if name == 'pose':
            cmd.add_argument('action', choices=['save', 'move'])
            cmd.add_argument('name', help='Pose name; shared by sim and real for each arm')
            cmd.add_argument('--overwrite', action='store_true')
    return p


def doctor(args, cfg):
    failures = []
    modules = ['numpy', 'mujoco', 'pinocchio']
    if args.profile in ('ros', 'hardware'):
        modules.append('rclpy')
    if args.profile == 'hardware':
        modules.append('openarm_can')
    print(f'{platform.system()} {platform.machine()} | Python {platform.python_version()}')
    for name in modules:
        try:
            module = importlib.import_module(name)
            print(f'OK {name}: {getattr(module, "__version__", "imported")}')
        except Exception as exc:
            failures.append(f'{name}: {exc}')
    try:
        import mujoco
        m = mujoco.MjModel.from_xml_path(str(root() / cfg['model']))
        for j in cfg['joints']:
            m.joint(j['name'])
        d = mujoco.MjData(m)
        for _ in range(10):
            mujoco.mj_step(m, d)
        print(f'OK model: {m.nq} coordinates, {m.nu} actuators; 10 physics steps')
    except Exception as exc:
        failures.append(f'model: {exc}')
    if args.profile == 'ros':
        print('ROS imports checked. Integration build/GUI and DDS discovery require separate checks.')
    if args.profile == 'hardware':
        if platform.system() != 'Linux':
            failures.append('SocketCAN requires Linux.')
        print('No motors contacted. Run can-check --real can0 can1.')
    for failure in failures:
        print('FAIL', failure)
    return 1 if failures else 0


def safe_name(name):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', name):
        raise ValueError('Names must use 1–64 letters, digits, underscores or hyphens.')
    return name


def configs(args):
    overrides = {}
    for item in args.config:
        iface, sep, path = item.partition('=')
        if not sep or iface not in ARMS or not path:
            raise ValueError('--config expects can0=PATH or can1=PATH.')
        overrides[iface] = Path(path)
    cfgs = {}
    for iface, arm in ARMS.items():
        cfg = load(overrides.get(iface), 'v1.0', arm)
        if cfg['arm'] != arm:
            raise ValueError(f'{iface} is the {arm} arm, but {cfg["_path"]} configures the {cfg["arm"]} arm.')
        cfgs[iface] = cfg
    return cfgs


def joint_index(cfg, joint):
    names = [j['name'] for j in cfg['joints']]
    if joint in names:
        return names.index(joint)
    if re.fullmatch(r'(joint)?[1-9]', joint) and int(joint[-1]) <= len(names):
        return int(joint[-1]) - 1
    raise ValueError(f'Unknown joint {joint!r} for the {cfg["arm"]} arm. Use 1-{len(names)} or one of: {", ".join(names)}')


def simulate(args, cfgs):
    if args.ros:
        if args.config or args.reset:
            raise ValueError('--ros uses the packaged v1.0 model, not custom configs or session reset.')
        from openarm_tools.backends.ros import launch
        launch(args.headless)
        return 0
    from openarm_tools.backends.mujoco import Simulation
    folder = args.data_dir / 'sim' / safe_name(args.session) / 'v1.0'
    with lock(f'sim:{folder.resolve()}'):
        sim = Simulation(cfgs, folder / 'state.json', args.reset)
        try:
            target = {k: sim.read(k) for k in cfgs}
            if args.headless:
                steps = math.ceil(args.seconds / sim.model.opt.timestep)
                for _ in range(steps):
                    sim.step(target)
                print(f'Simulated {args.seconds:g}s; positions: {target}')
                return 0
            if platform.system() == 'Darwin':
                import mujoco.viewer
                if getattr(mujoco.viewer, '_MJPYTHON', None) is None:
                    raise ValueError('On macOS use: mjpython scripts/sim.py (with the same options).')
            import mujoco.viewer
            with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
                while viewer.is_running():
                    start = time.monotonic()
                    with viewer.lock():
                        sim.step(target)
                    viewer.sync()
                    time.sleep(max(0, sim.model.opt.timestep - (time.monotonic() - start)))
            return 0
        finally:
            sim.close()


def run_together(jobs, interrupt_ok=False):
    """Run (label, fn(stop)) jobs in parallel; a failure or Ctrl-C stops all of them.

    With interrupt_ok, Ctrl-C is the normal way to end (e.g. a hold) and is not re-raised."""
    stop, errors = threading.Event(), []

    def worker(label, fn):
        try:
            fn(stop)
        except BaseException as exc:
            errors.append(f'{label}: {exc}')
            stop.set()

    threads = [threading.Thread(target=worker, args=job) for job in jobs]
    for t in threads:
        t.start()
    try:
        while any(t.is_alive() for t in threads):
            for t in threads:
                t.join(0.05)
    except KeyboardInterrupt:
        stop.set()
        for t in threads:
            t.join()
        if not interrupt_ok or errors:
            raise
    if errors:
        raise RuntimeError('; '.join(errors))


def run(args):
    cfgs = configs(args)
    if args.command == 'doctor':
        return doctor(args, cfgs['can0'])
    if args.command == 'sim':
        return simulate(args, cfgs)
    sim_arms = list(dict.fromkeys(getattr(args, 'sim', [])))
    real_arms = list(dict.fromkeys(args.real))
    if not sim_arms and not real_arms:
        raise ValueError('Choose arms with --sim and/or --real, e.g. --sim can0 can1 or --real can1.')
    if args.command == 'pose' and args.action == 'save' and sim_arms and real_arms:
        raise ValueError('Save a pose from either --sim or --real, not both.')
    if real_arms and platform.system() != 'Linux':
        raise RuntimeError('SocketCAN requires Linux.')
    moving = args.command in ('home', 'jog') or (args.command == 'pose' and args.action == 'move')

    with ExitStack() as stack:
        # Each endpoint is one arm on one backend: (label, iface, hardware, folder, read, status).
        endpoints = []
        sim = None
        if sim_arms:
            from openarm_tools.backends.mujoco import Simulation
            folder = args.data_dir / 'sim' / safe_name(args.session) / 'v1.0'
            stack.enter_context(lock(f'sim:{folder.resolve()}'))
            sim = Simulation(cfgs, folder / 'state.json')
            for iface in sim_arms:
                endpoints.append((f'sim {iface}', iface, False, folder / ARMS[iface],
                                  lambda k=iface: sim.read(k), lambda k=iface: sim.status(k)))
        hardware = {}
        if real_arms:
            from openarm_tools.backends.can import Hardware
        for iface in real_arms:
            if args.command == 'can-check':
                subprocess.run(['ip', '-details', '-statistics', 'link', 'show', iface], check=True)
            hw = hardware[iface] = Hardware(cfgs[iface], iface)
            stack.callback(hw.close)
            endpoints.append((f'real {iface}', iface, True, args.data_dir / 'can' / iface / 'v1.0' / ARMS[iface],
                              hw.read, hw.status))

        if args.command == 'zero':
            for label, iface, is_hw, folder, read, status in endpoints:
                positions = read()
                validate_positions(cfgs[iface], positions)
                if is_hw:
                    state = status()
                    if any(s['enabled'] for s in state):
                        raise ValueError(f'{label}: disable motors before recording a zero reference.')
                    if any(abs(finite(s['velocity'])) > 0.02 for s in state):
                        raise ValueError(f'{label}: keep the arm stationary before recording a zero reference.')
                offsets.save(folder / 'zero.json', cfgs[iface], positions)
                print(f'{label}: saved software zero (does not write motor firmware): {positions}')
            return 0

        if args.command in ('status', 'can-check'):
            report = {}
            for label, iface, _, folder, _, status in endpoints:
                rows = status()
                for row, z in zip(rows, offsets.read(folder / 'zero.json', cfgs[iface])):
                    row['relative_position'] = row['position'] - z
                report[label] = rows
            print(json.dumps(report, indent=2, allow_nan=False))
            return 0

        # Plan and validate every arm before any arm moves.
        plans = []
        for label, iface, is_hw, folder, read, _ in endpoints:
            cfg = cfgs[iface]
            current = read()
            pose_path = args.data_dir / 'poses' / 'v1.0' / ARMS[iface] / f'{safe_name(getattr(args, "name", "x"))}.json'
            if args.command == 'pose' and args.action == 'save':
                validate_positions(cfg, current)
                if pose_path.exists() and not args.overwrite:
                    raise ValueError(f'{label}: pose exists; use --overwrite to replace it.')
                write_json(pose_path, {'config': cfg['_fingerprint'], 'positions': current,
                                       'coordinates': 'raw_joint_radians'})
                print(f'{label}: saved {pose_path}')
                continue
            if args.command == 'pose':
                record = json.loads(pose_path.read_text())
                if record['config'] != cfg['_fingerprint']:
                    raise ValueError(f'{label}: pose belongs to a different robot configuration.')
                target = record['positions']
            elif args.command == 'jog':
                if args.joint.startswith('openarm_') and args.joint not in [j['name'] for j in cfg['joints']]:
                    raise ValueError(f'{args.joint} is not on {iface} ({ARMS[iface]} arm).')
                target = current.copy()
                target[joint_index(cfg, args.joint)] += finite(args.delta)
            else:  # home: return to the recorded zero, or shipped home pose.
                calibration = folder / 'zero.json'
                target = offsets.read(calibration, cfg) if calibration.exists() else [j['home'] for j in cfg['joints']]
            try:
                points = list(trajectory(cfg, current, target, args.duration))
            except ValueError as exc:
                raise ValueError(f'{label}: {exc}') from exc
            print(f'{label}: target joint positions (rad): {target}')
            plans.append((iface, is_hw, points))
        if not moving:
            return 0

        jobs = []
        sim_plans = {iface: points for iface, is_hw, points in plans if not is_hw}
        if sim_plans:
            stack.callback(sim.close)  # Save the simulated result even if a move stops partway.
            jobs.append(('sim', lambda stop: sim.move(sim_plans, stop)))
        for iface, is_hw, points in plans:
            if is_hw:
                jobs.append((f'real {iface}', lambda stop, hw=hardware[iface], p=points:
                             hw.move(p, stop=stop, hold=args.hold)))
        run_together(jobs, interrupt_ok=bool(args.hold))
        return 0


def main(argv=None):
    try:
        return run(parser().parse_args(argv))
    except KeyboardInterrupt:
        print('Interrupted.', file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, OSError, ImportError, KeyError, subprocess.CalledProcessError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
