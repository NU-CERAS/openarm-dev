"""Compose shared assets, fit visual target surfaces, and check tip reachability.

Target fits and IK are scene alignment aids, not collision-free motion plans.
"""
import hashlib
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from openarm_drummer.model import ROOT, OUTPUT, CONFIG, HoldController, build

SCENE_CONFIG = CONFIG.with_name('scene.json')
DRUMS = ROOT / 'robots/drums'


def read_obj():
    vertices, faces, material = [], {}, None
    for line in (DRUMS / 'drums.obj').open():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == 'v':
            vertices.append([float(x) for x in fields[1:4]])
        elif fields[0] == 'usemtl':
            material = fields[1]
        elif fields[0] == 'f':
            indices = [int(x.split('/')[0])-1 for x in fields[1:]]
            for i in range(1, len(indices)-1):
                faces.setdefault(material, []).append([indices[0], indices[i], indices[i+1]])
    colors = {}
    for line in (DRUMS / 'drums.mtl').open():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == 'newmtl':
            material = fields[1]
        elif fields[0] == 'Kd':
            colors[material] = [float(x) for x in fields[1:4]]
    return np.array(vertices), {name: np.array(f, dtype=int) for name, f in faces.items()}, colors


def components(vertices, faces):
    """Weld duplicate OBJ positions and group connected triangles per material."""
    used = np.unique(faces)
    points, inverse = np.unique(np.round(vertices[used], 6), axis=0, return_inverse=True)
    mapping = np.zeros(len(vertices), dtype=int)
    mapping[used] = inverse
    triangles = mapping[faces]
    parent = np.arange(len(points))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for triangle in triangles:
        a, b, c = map(find, triangle)
        parent[b] = a
        parent[c] = a
    labels = np.array([find(i) for i in range(len(points))])
    result = []
    for label in np.unique(labels):
        result.append((points[labels == label], faces[labels[triangles[:, 0]] == label]))
    return result


def fitted_surface(points, spec):
    center = points.mean(axis=0)
    _, _, axes = np.linalg.svd(points-center, full_matrices=False)
    normal = axes[-1]
    up = np.array([0, 0, -1]) if spec['kind'] == 'kick' else np.array([0, 1, 0])
    if normal @ up < 0:
        normal *= -1
    projected = points-center
    depth = projected @ normal
    radial = projected - depth[:, None]*normal
    radius = float(np.max(np.linalg.norm(radial, axis=1)))
    return center, normal, radius, float(np.max(np.abs(depth)))


def add_axes(parent, origin, length=0.12, prefix='world'):
    for index, color in enumerate(('1 .1 .1 1', '.1 1 .1 1', '.1 .4 1 1')):
        end = np.array(origin, dtype=float)
        end[index] += length
        ET.SubElement(parent, 'site', name=f'{prefix}_axis_{index}', type='capsule',
                      fromto=' '.join(map(str, [*origin, *end])), size='.002', rgba=color, group='3')


def build_scene(collisions=True):
    _, robot_report = build()
    cfg = json.loads(SCENE_CONFIG.read_text())
    elevation = float(cfg['robot_elevation_m'])
    support_xy = np.asarray(cfg['support_size_xy_m'], dtype=float)
    if not np.isfinite(elevation) or elevation <= 0:
        raise ValueError('Robot elevation must be positive and finite')
    if support_xy.shape != (2,) or not np.isfinite(support_xy).all() or np.any(support_xy <= 0):
        raise ValueError('Support width and depth must be positive and finite')
    scale = float(cfg['kit_scale'])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('kit_scale must be positive and finite')
    xy = np.asarray(cfg['kit_position_xy'], dtype=float)
    if xy.shape != (2,) or not np.isfinite(xy).all() or not np.isfinite(cfg['kit_yaw_degrees']):
        raise ValueError('Kit placement must contain finite XY and yaw values')
    if not np.isfinite(cfg['approach_height_m']) or cfg['approach_height_m'] <= 0:
        raise ValueError('Approach height must be positive and finite')
    if len({s['name'] for s in cfg['surfaces']}) != len(cfg['surfaces']):
        raise ValueError('Surface names must be unique')
    vertices, materials, colors = read_obj()
    grouped = {name: components(vertices, materials[name]) for name in ('Cylinder.015__0', 'Black_transp', 'Metal')}
    # Remove only the connected display slab; preserve all actual kit components.
    platforms = [faces for points, faces in grouped['Cylinder.015__0'] if np.ptp(points[:, 0]) > 3]
    if len(platforms) != 1:
        raise ValueError('Expected one display platform in source OBJ')
    platform_faces = {tuple(face) for face in platforms[0]}
    materials['Cylinder.015__0'] = np.array([f for f in materials['Cylinder.015__0'] if tuple(f) not in platform_faces])
    used = np.unique(np.concatenate(list(materials.values())))
    # Source +Y is up; source +Z is robot-forward; source +X is robot-left.
    base_rotation = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    angle = np.deg2rad(cfg['kit_yaw_degrees'])
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ base_rotation
    translation = np.array([*cfg['kit_position_xy'], -scale*vertices[used, 1].min()])
    transformed = scale*vertices @ rotation.T + translation
    tree = ET.parse(OUTPUT / 'opendrummer.xml')
    root = tree.getroot()
    root.set('model', 'opendrummer_playground')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='1280', offheight='960')
    ET.SubElement(visual, 'headlight', ambient='.35 .35 .35', diffuse='.7 .7 .7')
    asset = root.find('asset')
    world = root.find('worldbody')
    # Lift the entire robot, including fused base visuals and tip sites, together.
    robot_elements = [child for child in world if child.tag in ('body', 'geom', 'site')]
    mount = ET.SubElement(world, 'body', name='robot_mount', pos=f'0 0 {elevation}')
    for child in robot_elements:
        world.remove(child)
        mount.append(child)
    ET.SubElement(world, 'geom', name='robot_support_visual', type='box',
                  pos=f'0 0 {elevation/2}',
                  size=f'{support_xy[0]/2} {support_xy[1]/2} {elevation/2}',
                  rgba='.28 .32 .38 1', contype='0', conaffinity='0', group='2')
    ET.SubElement(asset, 'texture', name='floor_texture', type='2d', builtin='checker',
                  rgb1='.14 .16 .20', rgb2='.19 .21 .25', width='256', height='256')
    ET.SubElement(asset, 'material', name='floor_material', texture='floor_texture', texrepeat='8 8', reflectance='0')
    ET.SubElement(world, 'geom', name='floor_visual', type='plane', size='2 2 .01', material='floor_material',
                  contype='0', conaffinity='0', group='2')
    generated = OUTPUT / 'drum_visuals'
    generated.mkdir(exist_ok=True)
    for number, (material, faces) in enumerate(materials.items()):
        indices, inverse = np.unique(faces, return_inverse=True)
        path = generated / f'material_{number}.obj'
        with path.open('w') as f:
            for point in transformed[indices]:
                f.write('v '+' '.join(format(x, '.9g') for x in point)+'\n')
            for face in inverse.reshape(-1, 3)+1:
                f.write('f '+' '.join(map(str, face))+'\n')
        name = f'kit_material_{number}'
        ET.SubElement(asset, 'mesh', name=name, file=str(path))
        ET.SubElement(world, 'geom', name=name+'_visual', type='mesh', mesh=name,
                      rgba=' '.join(map(str, [*colors[material], 1])), contype='0', conaffinity='0', group='2')
    surfaces = []
    for spec in cfg['surfaces']:
        candidates = []
        for points, _ in grouped[spec['material']]:
            if len(points) < 20:
                continue
            center, normal, radius, deviation = fitted_surface(points, spec)
            distance = np.linalg.norm(center-np.array(spec['source_seed']))
            if radius >= spec['minimum_source_radius'] and distance < .06:
                candidates.append((distance, center, normal, radius, deviation))
        if not candidates:
            raise ValueError(f"No surface match for {spec['name']}")
        _, center, normal, radius, deviation = min(candidates, key=lambda row: row[0])
        position = scale*rotation @ center + translation
        normal = rotation @ normal
        radius *= scale
        quat = np.empty(4)
        mujoco.mju_quatZ2Vec(quat, normal)
        name = spec['name']
        ET.SubElement(world, 'site', name=f'{name}_surface', type='cylinder',
                      pos=' '.join(map(str, position+normal*.002)), quat=' '.join(map(str, quat)),
                      size=f'{radius*.9} .001', rgba='.2 .7 1 .12', group='3')
        ET.SubElement(world, 'site', name=f'{name}_target', pos=' '.join(map(str, position)),
                      size='.006', rgba='1 .8 .1 1', group='3')
        # Separate visual-only highlight: independent of target/debug visibility.
        ET.SubElement(world, 'site', name=f'{name}_hit_flash', type='cylinder',
                      pos=' '.join(map(str, position+normal*.004)), quat=' '.join(map(str, quat)),
                      size=f'{radius*.98} .0005', rgba='0 0 0 0', group='5')
        ET.SubElement(world, 'site', name=f'{name}_normal', type='capsule',
                      fromto=' '.join(map(str, [*position, *(position+normal*.07)])), size='.0015', rgba='1 .8 .1 1', group='3')
        surfaces.append({'name': name, 'kind': spec['kind'], 'center': position.tolist(),
                         'normal': normal.tolist(), 'radius': radius, 'source_center': center.tolist(),
                         'plane_deviation_m': deviation*scale,
                         'fit': 'Planar head fit' if spec['kind'] != 'cymbal' else 'Approximate cymbal plane; curvature/bell not modeled'})
    add_axes(world, [0, 0, .005])
    path = OUTPUT / 'scene.xml'
    ET.indent(tree)
    tree.write(path)
    model = mujoco.MjModel.from_xml_path(str(path))
    report = {'config': cfg, 'surfaces': surfaces, 'kit_transform': {'rotation': rotation.tolist(), 'translation': translation.tolist()},
              'robot_translation_m': [0, 0, elevation],
              'support_size_m': [*support_xy.tolist(), elevation],
              'kit_bounds_m': [transformed[used].min(0).tolist(), transformed[used].max(0).tolist()],
              'source_obj_sha256': hashlib.sha256((DRUMS/'drums.obj').read_bytes()).hexdigest(),
              'scene_config_sha256': hashlib.sha256(SCENE_CONFIG.read_bytes()).hexdigest(),
              'platform_removed_triangles': len(platform_faces), 'collision_enabled': False}
    if collisions:
        from openarm_drummer.collisions import add_collisions
        model, contact_report = add_collisions(model, report, vertices, materials, components)
        report['collision_enabled'] = True
        report['collision_geom_count'] = contact_report['collision_geom_count']
    (OUTPUT/'scene-report.json').write_text(json.dumps(report, indent=2)+'\n')
    return model, robot_report, report


def solve_approach(model, robot_report, scene_report, side, target_name, height=None, seed=None, wrist_angle=None, joint_margin=.001,
                   tip_direction=None, attempts=16, iterations=250, pose_validator=None):
    """Joint-limited static IK only; does not plan or certify a clear path."""
    target = next(s for s in scene_report['surfaces'] if s['name'] == target_name)
    if target['kind'] == 'kick':
        raise ValueError('Kick target is for scene annotation, not an arm approach')
    if height is None:
        height = scene_report['config']['approach_height_m']
    position = np.array(target['center'])+np.array(target['normal'])*height
    # A forward/downward stick allows a natural wrist pose above the head.
    direction = np.array([.65, 0, 0])-np.array(target['normal']) if tip_direction is None else np.array(tip_direction, dtype=float)
    direction /= np.linalg.norm(direction)
    controller = HoldController(model, robot_report)
    selected = np.arange(7) if side == 'left' else np.arange(7, 14)
    wrist = int(selected[-1])
    if wrist_angle is not None:
        if not controller.bounds[wrist, 0] <= wrist_angle <= controller.bounds[wrist, 1]:
            raise ValueError('Requested wrist angle exceeds joint limits')
        selected = selected[:-1]
    qids, vids = controller.q[selected], controller.v[selected]
    sid = model.site(f'{side}_stick_tip').id
    rng = np.random.default_rng(9)
    best = None
    accepted = False
    for attempt in range(attempts):
        data = mujoco.MjData(model)
        controller.reset(data)
        if attempt == 0 and seed is not None:
            data.qpos[controller.q] = seed
        elif attempt:
            data.qpos[qids] = rng.uniform(controller.bounds[selected, 0], controller.bounds[selected, 1])
        if wrist_angle is not None:
            data.qpos[controller.q[wrist]] = wrist_angle
        for _ in range(iterations):
            mujoco.mj_forward(model, data)
            z = data.site_xmat[sid].reshape(3, 3)[:, 2]
            error = np.r_[position-data.site_xpos[sid], .18*(direction-z)]
            score = float(np.linalg.norm(error))
            if best is None or score < best[0]:
                best = (score, data.qpos[controller.q].copy(), float(np.linalg.norm(error[:3])),
                        float(np.degrees(np.arccos(np.clip(z@direction, -1, 1)))))
            converged = best[2] < .0001 and best[3] < .1
            if pose_validator is not None:
                converged = np.linalg.norm(error[:3]) < .0001 and np.linalg.norm(direction-z) < .002
            if converged:
                break
            jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jp, jr, sid)
            skew = np.array([[0, -z[2], z[1]], [z[2], 0, -z[0]], [-z[1], z[0], 0]])
            jacobian = np.vstack([jp[:, vids], -.18*skew@jr[:, vids]])
            delta = jacobian.T @ np.linalg.solve(jacobian@jacobian.T+.0002*np.eye(6), error)
            data.qpos[qids] = np.clip(data.qpos[qids]+np.clip(delta, -.12, .12),
                                      controller.bounds[selected, 0]+joint_margin, controller.bounds[selected, 1]-joint_margin)
        if pose_validator is not None:
            mujoco.mj_forward(model, data)
            position_error = float(np.linalg.norm(position-data.site_xpos[sid]))
            direction_error = float(np.degrees(np.arccos(np.clip(data.site_xmat[sid].reshape(3, 3)[:, 2]@direction, -1, 1))))
            if position_error < .005 and direction_error < 10 and pose_validator(data.qpos[controller.q].copy()):
                best = (position_error, data.qpos[controller.q].copy(), position_error, direction_error)
                accepted = True
                break
        elif best[2] < .0001 and best[3] < .1:
            break
    return {'arm': side, 'target': target_name, 'approach_position': position.tolist(),
            'tip_direction': direction.tolist(), 'joint_positions': best[1].tolist(),
            'position_error_m': best[2], 'direction_error_degrees': best[3],
            'reachable': best[2] < .005 and best[3] < 10 and (pose_validator is None or accepted),
            'clearance_status': 'Not collision-certified; inspect geometry before planning motion'}


if __name__ == '__main__':
    model, robot_report, report = build_scene()
    print(json.dumps(report, indent=2))
    for side, target in report['config']['initial_targets'].items():
        print(json.dumps(solve_approach(model, robot_report, report, side, target), indent=2))
