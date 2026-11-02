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
    parser.add_argument('--snapshot', type=Path, help='Save a PNG of the initial scene and exit')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--seconds', type=float, default=10, help='Duration for headless runs')
    args = parser.parse_args()
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
    scene_report = None
    if args.robot_only:
        model, report = build()
    else:
        model, report, scene_report = build_scene()
        print('Surface targets: '+', '.join(s['name'] for s in scene_report['surfaces']))
        print('Primitive contact geometry enabled. '+scene_report['config']['scale_note'])
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
    contact_test = None
    if args.contact_test:
        from contact_check import ContactCheck
        contact_test = ContactCheck(model, report, scene_report)
        contact_test.reset(data)
        print(f'Slow contact check: {contact_test.side} stick / {contact_test.target}; {contact_test.duration:.1f}s')
    rest = controller.target.copy()
    last_joint = None

    def step():
        nonlocal last_joint
        if contact_test is not None:
            if not contact_test.finished:
                contact_test.step(data)
                if contact_test.finished:
                    contact_test.save()
                    print(json.dumps(contact_test.report(), indent=2), flush=True)
                    print('Contact check finished; scene frozen. Close the viewer to exit.', flush=True)
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

    print('14 joints; fixed base; green left tip, orange right tip.', flush=True)
    print(report['config']['inertia_status'], flush=True)
    if args.snapshot:
        camera = mujoco.MjvCamera()
        frame_camera(model, data, camera, args.robot_only)
        options = mujoco.MjvOption()
        options.sitegroup[3] = 1
        options.geomgroup[4] = int(args.show_collisions)
        if args.collision_only:
            options.geomgroup[1:3] = 0
            options.sitegroup[3] = 0
        options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = args.show_collisions
        if args.labels:
            options.label = mujoco.mjtLabel.mjLABEL_SITE
        model.vis.global_.offwidth, model.vis.global_.offheight = 960, 720
        with mujoco.Renderer(model, 720, 960) as renderer:
            renderer.update_scene(data, camera=camera, scene_option=options)
            save_png(args.snapshot, renderer.render())
        print(f'Saved {args.snapshot}')
        return
    if args.headless:
        if contact_test is not None:
            while not contact_test.finished:
                step()
            if contact_test.report()['status'] != 'PASS':
                raise RuntimeError('Contact check failed; see contact-test.json')
            return
        else:
            while data.time < args.seconds:
                step()
        print(f'Completed {data.time:.3f}s; max target error = {np.max(np.abs(data.qpos[controller.q]-controller.target)):.6f} rad')
        return
    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            frame_camera(model, data, viewer.cam, args.robot_only)
            viewer.opt.sitegroup[3] = 1
            viewer.opt.geomgroup[4] = int(args.show_collisions)
            if args.collision_only:
                viewer.opt.geomgroup[1:3] = 0
                viewer.opt.sitegroup[3] = 0
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = args.show_collisions
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = args.show_collisions
            if args.labels:
                viewer.opt.label = mujoco.mjtLabel.mjLABEL_SITE
        while viewer.is_running():
            start = time.monotonic()
            with viewer.lock():
                for _ in range(max(1, round(1/60/model.opt.timestep))):
                    step()
            viewer.sync()
            time.sleep(max(0, 1/60-(time.monotonic()-start)))


if __name__ == '__main__':
    main()
