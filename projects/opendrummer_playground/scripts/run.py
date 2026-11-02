"""View the combined playground or the standalone robot import."""
import argparse
import json
from pathlib import Path
import struct
import time
import zlib

import mujoco
import mujoco.viewer
import numpy as np

from model import JOINTS, OUTPUT, HoldController, build
from scene import build_scene, solve_approach


def save_png(path, pixels):
    """Write renderer RGB output without an additional imaging dependency."""
    def chunk(kind, payload):
        return struct.pack('!I', len(payload))+kind+payload+struct.pack('!I', zlib.crc32(kind+payload)&0xffffffff)
    height, width, _ = pixels.shape
    data = b''.join(b'\0'+row.tobytes() for row in pixels)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR', struct.pack('!2I5B', width, height, 8, 2, 0, 0, 0))
                     +chunk(b'IDAT', zlib.compress(data))+chunk(b'IEND', b''))


def frame_camera(model, data, camera, robot_only=False):
    points = []
    for gid in range(model.ngeom):
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = model.geom_dataid[gid]
        vertices = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid]+model.mesh_vertnum[mid]]
        points.append(vertices @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid])
    vertices = np.concatenate(points)
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    camera.lookat[:] = (low+high)/2
    camera.distance = max(1.0, float(np.linalg.norm(high-low))*(1.5 if robot_only else 1.15))
    camera.azimuth = 135 if robot_only else 35
    camera.elevation = -15 if robot_only else -30


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exercise', action='store_true', help='Small sequential motions of all 14 joints')
    parser.add_argument('--robot-only', action='store_true', help='Step-one standalone robot view')
    parser.add_argument('--preview-target', help='Static approach pose for a named surface; does not execute a trajectory')
    parser.add_argument('--arm', choices=['left', 'right'], default='left')
    parser.add_argument('--labels', action='store_true', help='Show site names, including target and tip names')
    parser.add_argument('--show-collisions', action='store_true', help='Show group-4 collision proxies and contact points')
    parser.add_argument('--collision-only', action='store_true', help='Clear collision debug view with render meshes hidden')
    parser.add_argument('--contact-test', action='store_true', help='One slow snare touch and retraction, then freeze; uses contact.json')
    parser.add_argument('--strike', action='store_true', help='Contact-triggered left-arm snare strike and recovery')
    parser.add_argument('--teach', action='store_true', help='Native MuJoCo pose teaching mode')
    parser.add_argument('--preset', metavar='DRUM', help='With --strike, play the taught preset for --arm and this drum')
    parser.add_argument('--strikes', type=int, default=1, help='Number of isolated strike cycles (1–100), with --strike')
    parser.add_argument('--no-hit-panel', action='store_true', help='Hide the right hit sidebar; retain drum flashes and logs')
    parser.add_argument('--snapshot', type=Path, help='Save a PNG of the initial scene and exit')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--seconds', type=float, default=10, help='Duration for headless runs')
    args = parser.parse_args()
    if args.teach and (args.strike or args.robot_only or args.exercise or args.preview_target or args.contact_test or args.headless or args.snapshot or args.preset):
        parser.error('--teach is a separate interactive mode; do not combine it with demo or headless options')
    if args.preset and not args.strike:
        parser.error('--preset requires --strike')
    if args.collision_only:
        args.show_collisions = True
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        parser.error('--seconds must be positive and finite')
    if args.exercise and not args.robot_only:
        parser.error('--exercise requires --robot-only; kit motion planning is a later step')
    if args.preview_target and args.robot_only:
        parser.error('--preview-target requires the combined scene')
    if args.contact_test and (args.robot_only or args.preview_target or args.exercise):
        parser.error('--contact-test cannot be combined with robot-only, preview-target, or exercise')
    if args.strike and (args.contact_test or args.robot_only or args.preview_target or args.exercise):
        parser.error('--strike cannot be combined with contact-test, robot-only, preview-target, or exercise')
    if not 1 <= args.strikes <= 100 or (args.strikes != 1 and not args.strike):
        parser.error('--strikes must be 1–100 and requires --strike')
    if args.strike and not args.preset and args.arm != 'left':
        parser.error('The validated strike routine currently supports the left arm only')
    scene_report = None
    if args.robot_only:
        model, report = build()
    else:
        model, report, scene_report = build_scene()
        print('Surface targets: '+', '.join(s['name'] for s in scene_report['surfaces']))
        print('Primitive contact geometry enabled. '+scene_report['config']['scale_note'])
    if args.teach:
        from teach import run_teaching
        run_teaching(model, report, scene_report)
        return
    data = mujoco.MjData(model)
    controller = HoldController(model, report)
    if args.preview_target:
        names = [s['name'] for s in scene_report['surfaces'] if s['kind'] != 'kick']
        if args.preview_target not in names:
            parser.error('Choose an arm target: '+', '.join(names))
        approach = solve_approach(model, report, scene_report, args.arm, args.preview_target)
        print(json.dumps(approach, indent=2))
        (OUTPUT/f'approach-{args.arm}-{args.preview_target}.json').write_text(json.dumps(approach, indent=2)+'\n')
        if not approach['reachable']:
            raise RuntimeError('No acceptable approach found for this arm and placement')
        controller.target = np.array(approach['joint_positions'])
    controller.reset(data)
    routine = None
    if args.contact_test:
        from contact_check import ContactCheck
        routine = ContactCheck(model, report, scene_report)
        routine.reset(data)
        print(f'Slow contact check: {routine.side} stick / {routine.target}; {routine.duration:.1f}s')
    elif args.strike:
        if args.preset:
            from taught import PoseStore, TaughtStrike
            preset = PoseStore(model, report, scene_report).get(args.arm, args.preset)
            routine = TaughtStrike(model, report, scene_report, preset, args.strikes)
        else:
            from strike import StrikeSession
            routine = StrikeSession(model, report, scene_report, args.strikes)
        routine.reset(data)
        print(f'{args.strikes} {routine.side}-arm {routine.target} strike(s).', flush=True)
    rest = controller.target.copy()
    last_joint = None
    telemetry = None
    if scene_report is not None:
        from telemetry import DrumTelemetry
        telemetry = DrumTelemetry(model, scene_report)
    panel_enabled = telemetry is not None and not args.no_hit_panel

    def advance():
        nonlocal last_joint
        if routine is not None:
            if not routine.finished:
                previous_state = getattr(routine, 'state', None)
                routine.step(data)
                if args.strike and routine.state != previous_state:
                    print(f'{data.time:.3f}s | strike {routine.cycle}: {routine.state} | hits: {len(routine.detector.events)}', flush=True)
                if routine.finished:
                    routine.save()
                    result = routine.report()
                    if args.strike:
                        result = {key: result[key] for key in ('status', 'failure', 'completed_strikes', 'hit_count', 'metrics', 'hits')}
                    print(json.dumps(result, indent=2), flush=True)
                    print('Routine finished; scene frozen. Close the viewer to exit.', flush=True)
            return
        goal = rest.copy()
        if args.exercise:
            index = int(data.time/4) % 14
            phase = data.time % 4
            goal[index] += 0.15*np.sin(np.pi*phase/4)**2
            if index != last_joint:
                print(f'Exercising {JOINTS[index]} (+0.15 rad excursion)', flush=True)
                last_joint = index
        controller.step(data, goal)

    def step():
        advance()
        if telemetry is not None:
            telemetry.sample(data)

    def panel_status():
        if routine is None:
            return 'Ready · contact monitor'
        if routine.finished:
            return 'Complete' if routine.report()['status'] == 'PASS' else 'Stopped · check report'
        if args.strike:
            return f'Strike {routine.cycle}/{routine.count} · {routine.state.title()}'
        return 'Slow contact check'

    print('14 joints; fixed base; green left tip, orange right tip.', flush=True)
    print(report['config']['inertia_status'], flush=True)
    if args.snapshot:
        camera = mujoco.MjvCamera()
        frame_camera(model, data, camera, args.robot_only)
        options = mujoco.MjvOption()
        options.sitegroup[3] = 1
        options.geomgroup[4] = int(args.show_collisions)
        options.sitegroup[5] = 1
        if args.collision_only:
            options.geomgroup[1:3] = 0
            options.sitegroup[3] = 0
        options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = args.show_collisions
        if args.labels:
            options.label = mujoco.mjtLabel.mjLABEL_SITE
        model.vis.global_.offwidth, model.vis.global_.offheight = 960, 720
        with mujoco.Renderer(model, 720, 960) as renderer:
            renderer.update_scene(data, camera=camera, scene_option=options)
            pixels = renderer.render()
            if panel_enabled:
                pixels = np.concatenate([pixels, telemetry.draw_panel(330, 720, panel_status())], axis=1)
            save_png(args.snapshot, pixels)
        print(f'Saved {args.snapshot}')
        return
    if args.headless:
        if routine is not None:
            while not routine.finished:
                step()
            if telemetry is not None:
                telemetry.save()
            if routine.report()['status'] != 'PASS':
                raise RuntimeError('Routine failed; see strike-report.json or contact-test.json')
            return
        else:
            while data.time < args.seconds:
                step()
        if telemetry is not None:
            telemetry.save()
        print(f'Completed {data.time:.3f}s; max target error = {np.max(np.abs(data.qpos[controller.q]-controller.target)):.6f} rad')
        return
    with mujoco.viewer.launch_passive(model, data, show_left_ui=not panel_enabled, show_right_ui=not panel_enabled) as viewer:
        with viewer.lock():
            frame_camera(model, data, viewer.cam, args.robot_only)
            viewer.opt.sitegroup[3] = 1
            viewer.opt.sitegroup[5] = 1
            viewer.opt.geomgroup[4] = int(args.show_collisions)
            if args.collision_only:
                viewer.opt.geomgroup[1:3] = 0
                viewer.opt.sitegroup[3] = 0
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = args.show_collisions
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = args.show_collisions
            if args.labels:
                viewer.opt.label = mujoco.mjtLabel.mjLABEL_SITE
        viewer.sync()
        # The viewer's initial UI synchronization can replace the initial pose.
        # Restore the requested starting state before the first physics step.
        with viewer.lock():
            (routine or controller).reset(data)
            frame_camera(model, data, viewer.cam, args.robot_only)
        if panel_enabled:
            with viewer.lock():
                telemetry.offset_camera(viewer)
        while viewer.is_running():
            start = time.monotonic()
            with viewer.lock():
                for _ in range(max(1, round(1/60/model.opt.timestep))):
                    step()
                if telemetry is not None:
                    telemetry.update_colors()
            viewer.sync()
            if panel_enabled:
                telemetry.present(viewer, panel_status())
            time.sleep(max(0, 1/60-(time.monotonic()-start)))
    if telemetry is not None:
        telemetry.save()


if __name__ == '__main__':
    main()
