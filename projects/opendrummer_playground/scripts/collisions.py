"""Primitive collision proxies; independent of render mesh/material groups."""
import json
import hashlib
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from model import CONFIG, OUTPUT

CONTACT_CONFIG = CONFIG.with_name('contact.json')


def fmt(values):
    return ' '.join(format(float(x), '.12g') for x in values)


def quat_z(normal):
    quat = np.empty(4)
    mujoco.mju_quatZ2Vec(quat, normal)
    return fmt(quat)


def add_collisions(model, scene_report, source_vertices, source_materials, split_components):
    cfg = json.loads(CONTACT_CONFIG.read_text())
    for key in ('head_thickness_m', 'rim_radius_m', 'shell_wall_thickness_m',
                'stick_shaft_radius_m', 'stick_tip_radius_m', 'stick_length_m'):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f'{key} must be positive and finite')
    if type(cfg['shell_segments']) is not int or cfg['shell_segments'] < 8:
        raise ValueError('shell_segments must be an integer >= 8')
    path = OUTPUT/'scene.xml'
    mujoco.mj_saveLastXML(str(path), model)
    tree = ET.parse(path)
    root = tree.getroot()
    root.find('compiler').set('fusestatic', 'false')
    root.find('compiler').set('inertiafromgeom', 'false')
    root.find('option').set('iterations', '100')
    root.find('option').set('tolerance', '1e-10')
    world = root.find('worldbody')
    bodies = {b.get('name'): b for b in root.findall('.//body')}
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    names = []

    def geom(parent, name, kind, dynamic=False, soft=False, **attributes):
        # Bit 1 = environment; bit 2 = moving robot. Environment never collides
        # with itself, but all robot/environment and robot/robot pairs are eligible.
        values = dict(name=name, type=kind, contype='2' if dynamic else '1',
                      conaffinity='3' if dynamic else '2', group='4', mass='0',
                      rgba='.95 .45 .12 .45' if dynamic else '.15 .75 .9 .35',
                      friction=fmt(cfg['friction']), solimp=fmt(cfg['solimp']),
                      solref=fmt(cfg['head_solref'] if soft else cfg['rigid_solref']),
                      priority='1' if soft else '0', condim='3')
        values.update(attributes)
        names.append(name)
        return ET.SubElement(parent, 'geom', **values)

    elevation = scene_report['config']['robot_elevation_m']
    geom(world, 'floor_collision', 'plane', size='2 2 .01')
    sx, sy, sz = scene_report['support_size_m']
    geom(world, 'support_collision', 'box', pos=fmt([0, 0, sz/2]), size=fmt([sx/2, sy/2, sz/2]))
    base = ET.SubElement(world, 'body', name='robot_base_collision', pos=fmt([0, 0, elevation]))
    geom(base, 'pedestal_collision', 'box', pos='0 0 .375', size='.032 .032 .375')
    geom(base, 'shoulder_bar_collision', 'capsule', fromto='0 -.0925 .697 0 .0925 .697', size='.055')
    geom(base, 'base_plate_collision', 'box', pos='0 0 .004', size='.11 .11 .004')

    contact = ET.SubElement(root, 'contact')
    for side in ('left', 'right'):
        body_names = []
        for joint in range(1, 8):
            bid = int(model.jnt_bodyid[model.joint(f'{side[0]}j{joint}').id])
            body_name = model.body(bid).name
            body_names.append(body_name)
            parent = bodies[body_name]
            if joint == 7:
                sid = model.site(f'{side}_stick_tip').id
                tip = (data.site_xpos[sid]-data.xpos[bid]) @ data.xmat[bid].reshape(3, 3)
                direction = data.xmat[bid].reshape(3, 3).T @ data.site_xmat[sid].reshape(3, 3)[:, 2]
                radius = cfg['stick_shaft_radius_m']
                start = tip-direction*(cfg['stick_length_m']-radius)
                geom(parent, f'{side}_stick_shaft_collision', 'capsule', True,
                     fromto=fmt([*start, *(tip-direction*.035)]), size=str(radius))
                radius = cfg['stick_tip_radius_m']
                geom(parent, f'{side}_stick_taper_collision', 'capsule', True,
                     fromto=fmt([*(tip-direction*.035), *(tip-direction*(2*radius))]), size=str(radius))
                geom(parent, f'{side}_stick_tip_collision', 'sphere', True,
                     pos=fmt(tip-direction*radius), size=str(radius))
                mount_points = []
                for gid in np.where(model.geom_bodyid == bid)[0]:
                    mid = model.geom_dataid[gid]
                    if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH or model.mesh(mid).name == 'Drumstick':
                        continue
                    vertices = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid]+model.mesh_vertnum[mid]]
                    vertices = vertices @ data.geom_xmat[gid].reshape(3, 3).T+data.geom_xpos[gid]
                    mount_points.append((vertices-data.xpos[bid]) @ data.xmat[bid].reshape(3, 3))
                mount_points = np.concatenate(mount_points)
                lo, hi = mount_points.min(0), mount_points.max(0)
                geom(parent, f'{side}_tool_mount_collision', 'box', True, pos=fmt((lo+hi)/2), size=fmt((hi-lo)/2))
            else:
                points = []
                for gid in np.where(model.geom_bodyid == bid)[0]:
                    mid = model.geom_dataid[gid]
                    if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
                        continue
                    vertices = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid]+model.mesh_vertnum[mid]]
                    vertices = vertices @ data.geom_xmat[gid].reshape(3, 3).T+data.geom_xpos[gid]
                    points.append((vertices-data.xpos[bid]) @ data.xmat[bid].reshape(3, 3))
                points = np.concatenate(points)
                center = (points.min(0)+points.max(0))/2
                # A local axis-aligned rounded segment avoids noisy PCA axes on
                # approximately cylindrical joints while covering their bulk.
                extent = np.ptp(points, axis=0)
                axis = np.eye(3)[np.argmax(extent)]
                radius = float(np.max(np.delete(extent, np.argmax(extent)))/2)
                half = max(.001, float(extent.max()/2-radius))
                geom(parent, f'{side}_link{joint}_collision', 'capsule', True,
                     fromto=fmt([*(center-axis*half), *(center+axis*half)]), size=str(radius))
        for a, b in zip(body_names, body_names[1:]):
            ET.SubElement(contact, 'exclude', body1=a, body2=b)
        # J5/J6/J7 form the compact wrist. Bulk proxies fill the real mounting
        # recesses, so omit J5/tool-body contacts as well as direct neighbors.
        ET.SubElement(contact, 'exclude', body1=body_names[4], body2=body_names[6])
        ET.SubElement(contact, 'exclude', body1='robot_base_collision', body2=body_names[0])
        # The first two shoulder axes share the fixed shoulder housing.
        ET.SubElement(contact, 'exclude', body1='robot_base_collision', body2=body_names[1])

    rotation = np.array(scene_report['kit_transform']['rotation'])
    translation = np.array(scene_report['kit_transform']['translation'])
    scale = scene_report['config']['kit_scale']
    shells = split_components(source_vertices, source_materials['Main_Color'])
    shell_reports = []
    for surface in scene_report['surfaces']:
        name = surface['name']
        center, normal = np.array(surface['center']), np.array(surface['normal'])
        radius = surface['radius']
        thick = cfg['head_thickness_m']
        geom(world, f'{name}_head_collision', 'cylinder', soft=True,
             pos=fmt(center-normal*thick/2), quat=quat_z(normal), size=fmt([radius, thick/2]))
        if surface['kind'] == 'cymbal':
            continue
        tangent = np.cross(normal, [1, 0, 0])
        if np.linalg.norm(tangent) < .1:
            tangent = np.cross(normal, [0, 1, 0])
        tangent /= np.linalg.norm(tangent)
        other = np.cross(normal, tangent)
        count = cfg['shell_segments']
        candidates = []
        for points, _ in shells:
            points = scale*points @ rotation.T+translation
            offset = points-center
            depth = offset@normal
            radii = np.linalg.norm(offset-depth[:, None]*normal, axis=1)
            if .8*radius < radii.max() < 1.2*radius and np.ptp(depth) > .06:
                candidates.append((abs(float(depth.max())), float(depth.min()), float(depth.max())))
        if not candidates:
            raise ValueError(f'No shell component for {name}')
        _, bottom, top = min(candidates)
        shell_reports.append({'name': name, 'bottom_offset_m': bottom, 'top_offset_m': top})
        for index in range(count):
            angle = index*2*np.pi/count
            radial = np.cos(angle)*tangent+np.sin(angle)*other
            next_radial = np.cos(angle+2*np.pi/count)*tangent+np.sin(angle+2*np.pi/count)*other
            geom(world, f'{name}_rim_{index}_collision', 'capsule',
                 fromto=fmt([*(center+radius*radial), *(center+radius*next_radial)]), size=str(cfg['rim_radius_m']))
            orientation = np.column_stack([radial, np.cross(normal, radial), normal])
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, orientation.ravel())
            geom(world, f'{name}_shell_{index}_collision', 'box',
                 pos=fmt(center+radius*radial+normal*(bottom+top)/2), quat=fmt(quat),
                 size=fmt([cfg['shell_wall_thickness_m']/2, radius*np.tan(np.pi/count), (top-bottom)/2]))

    stand_count = 0
    for points, _ in split_components(source_vertices, source_materials['Metal']):
        if len(points) < 12:
            continue
        center = points.mean(0)
        _, _, axes = np.linalg.svd(points-center, full_matrices=False)
        local = (points-center) @ axes.T
        length = np.ptp(local[:, 0])
        radius = np.max(np.linalg.norm(local[:, 1:], axis=1))
        if length < .15 or radius > .06 or length < 4*radius:
            continue
        start = center+axes[0]*local[:, 0].min()
        end = center+axes[0]*local[:, 0].max()
        start, end = scale*rotation@start+translation, scale*rotation@end+translation
        geom(world, f'stand_{stand_count}_collision', 'capsule', fromto=fmt([*start, *end]), size=str(max(.002, scale*radius)))
        stand_count += 1
    ET.indent(tree)
    tree.write(path)
    compiled = mujoco.MjModel.from_xml_path(str(path))
    report = {'config': cfg, 'collision_geom_count': len(names), 'stand_capsules': stand_count,
              'config_sha256': hashlib.sha256(CONTACT_CONFIG.read_bytes()).hexdigest(),
              'body_pair_exclusions': [dict(e.attrib) for e in contact],
              'shell_fits': shell_reports, 'geom_names': names,
              'scope': 'Simplified rigid shells, planar drumheads/cymbals, stand capsules, arm capsules and rounded sticks; no membrane deformation or acoustic model.'}
    (OUTPUT/'collision-report.json').write_text(json.dumps(report, indent=2)+'\n')
    return compiled, report
