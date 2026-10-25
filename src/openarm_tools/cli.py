"""Common operator commands. Hardware execution is explicit; simulation is default."""
import argparse
import importlib
import json
import math
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from openarm_tools.robot.config import load, root, validate_positions, finite
from openarm_tools.recording.storage import lock, write_json
from openarm_tools.calibration import offsets
from openarm_tools.motion.trajectory import trajectory


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
        cmd.add_argument('--robot', choices=['v0.3', 'v1.0'], default='v1.0')
        cmd.add_argument('--arm', choices=['left', 'right'], default='left')
        cmd.add_argument('--config', type=Path, help='Reviewed per-arm JSON configuration')
        cmd.add_argument('--backend', choices=['sim', 'can'], default='sim')
        cmd.add_argument('--interface', help='Required for CAN, e.g. can0')
        cmd.add_argument('--session', default='default', help='Named local simulation session')
        cmd.add_argument('--data-dir', type=Path, default=root() / 'runs')
        if name in ('zero', 'home', 'jog', 'pose'):
            cmd.add_argument('--execute', action='store_true', help='Execute changes; otherwise preview only')
        if name in ('home', 'jog', 'pose'):
            cmd.add_argument('--duration', type=positive, default=5)
            cmd.add_argument('--supported', action='store_true', help='Physical arm is supported for motor release at exit')
        if name == 'doctor':
            cmd.add_argument('--profile', choices=['sim', 'ros', 'hardware'], default='sim')
        if name == 'sim':
            cmd.add_argument('--headless', action='store_true')
            cmd.add_argument('--seconds', type=positive, default=5, help='Headless run duration')
            cmd.add_argument('--reset', action='store_true')
            cmd.add_argument('--ros', action='store_true', help='Launch the v1.0 ROS-control simulation instead')
        if name == 'jog':
            cmd.add_argument('joint', help='Exact configured joint name')
            cmd.add_argument('delta', type=float, help='Increment in radians')
        if name == 'pose':
            cmd.add_argument('action', choices=['save', 'move'])
            cmd.add_argument('name', help='Pose name; saved under the selected session')
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
        print('No motors contacted. Run can-check with an explicit interface.')
    for failure in failures:
        print('FAIL', failure)
    return 1 if failures else 0


def safe_name(name):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', name):
        raise ValueError('Names must use 1–64 letters, digits, underscores or hyphens.')
    return name


def run(args):
    cfg = load(args.config, args.robot, args.arm)
    if args.command == 'doctor':
        return doctor(args, cfg)
    if args.command == 'sim' and args.ros:
        if args.backend != 'sim' or args.config or args.robot != 'v1.0' or args.reset:
            raise ValueError('--ros uses the packaged v1.0 model, not custom configs or session reset.')
        from openarm_tools.backends.ros import launch
        launch(args.headless)
        return 0
    hardware = args.backend == 'can'
    if args.command == 'can-check':
        hardware = True
    if hardware and not args.interface:
        raise ValueError('CAN commands require --interface can0 (or the actual interface).')
    if args.command == 'sim' and hardware:
        raise ValueError('sim cannot use the CAN backend.')
    identity = safe_name(args.interface) if hardware else safe_name(args.session)
    # Each selected arm/session owns an independent full-model snapshot.
    folder = args.data_dir / ('can' if hardware else 'sim') / identity / cfg['robot'] / cfg['arm']
    resource = f'can:{identity}' if hardware else f'sim:{folder.resolve()}'
    with lock(resource):
        if args.command == 'can-check':
            if platform.system() != 'Linux':
                raise RuntimeError('SocketCAN requires Linux.')
            subprocess.run(['ip', '-details', '-statistics', 'link', 'show', args.interface], check=True)
        if hardware:
            from openarm_tools.backends.can import Hardware
            backend = Hardware(cfg, args.interface)
        else:
            from openarm_tools.backends.mujoco import Simulation
            backend = Simulation(cfg, folder / 'state.json', getattr(args, 'reset', False))
        try:
            calibration = folder / 'zero.json'
            if args.command == 'zero':
                positions = backend.read()
                validate_positions(cfg, positions)
                if hardware:
                    state = backend.status()
                    if any(s['enabled'] for s in state):
                        raise ValueError('Disable motors before recording a zero reference.')
                    if any(abs(finite(s['velocity'])) > 0.02 for s in state):
                        raise ValueError('Keep the arm stationary before recording a zero reference.')
                print('Software zero reference (does not write motor firmware):', positions)
                if args.execute:
                    offsets.save(calibration, cfg, positions)
                    print('Saved', calibration)
                else:
                    print('Preview only; add --execute to save.')
                return 0
            zero = offsets.read(calibration, cfg)
            if args.command in ('status', 'can-check'):
                rows = backend.status()
                for row, z in zip(rows, zero):
                    row['relative_position'] = row['position'] - z
                print(json.dumps(rows, indent=2, allow_nan=False))
                return 0
            if args.command == 'sim':
                target = backend.read()
                if args.headless:
                    steps = math.ceil(args.seconds / backend.model.opt.timestep)
                    for _ in range(steps):
                        backend.step(target)
                    print(f'Simulated {args.seconds:g}s; positions: {backend.read()}')
                else:
                    if platform.system() == 'Darwin':
                        import mujoco.viewer
                        if getattr(mujoco.viewer, '_MJPYTHON', None) is None:
                            raise ValueError('On macOS use: mjpython scripts/sim.py (with the same options).')
                    import mujoco.viewer
                    with mujoco.viewer.launch_passive(backend.model, backend.data) as viewer:
                        while viewer.is_running():
                            start = time.monotonic()
                            with viewer.lock():
                                backend.step(target)
                            viewer.sync()
                            time.sleep(max(0, backend.model.opt.timestep - (time.monotonic() - start)))
                return 0
            current = backend.read()
            if args.command == 'pose':
                path = folder / 'poses' / f'{safe_name(args.name)}.json'
                if args.action == 'save':
                    validate_positions(cfg, current)
                    if path.exists() and not args.overwrite:
                        raise ValueError('Pose exists; use --overwrite to replace it.')
                    write_json(path, {'config': cfg['_fingerprint'], 'positions': current,
                                      'coordinates': 'raw_joint_radians'})
                    print('Saved', path)
                    return 0
                record = json.loads(path.read_text())
                if record['config'] != cfg['_fingerprint']:
                    raise ValueError('Pose belongs to a different robot configuration.')
                target = record['positions']
            elif args.command == 'jog':
                names = [j['name'] for j in cfg['joints']]
                if args.joint not in names:
                    raise ValueError(f'Unknown joint. Choose one of: {", ".join(names)}')
                target = current.copy()
                target[names.index(args.joint)] += finite(args.delta)
            else:  # home: return to the recorded zero, or shipped home pose.
                target = zero if calibration.exists() else [j['home'] for j in cfg['joints']]
            # Fully validate before enabling any motor.
            points = list(trajectory(cfg, current, target, args.duration))
            print('Target joint positions (rad):', target)
            if not args.execute:
                print('Preview only; add --execute to move.')
                return 0
            if hardware and not args.supported:
                raise ValueError('Hardware moves release motors at exit. Support the arm and pass --supported.')
            backend.move(points)
            return 0
        finally:
            # Preview/read commands must not change the simulated session.
            if hardware or args.command == 'sim' or getattr(args, 'execute', False):
                backend.close()


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
