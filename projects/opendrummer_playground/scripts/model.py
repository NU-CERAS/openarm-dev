"""Reproducible OpenDrummer import; source assets are never modified."""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ASSET = ROOT / 'robots/opendrummer'
SOURCE = ASSET / 'urdf/opendrummer.urdf'
CONFIG = Path(__file__).resolve().parents[1] / 'configs/import.json'
OUTPUT = ROOT / 'runs/opendrummer_playground'
JOINTS = [f'{side}j{i}' for side in ('l', 'r') for i in range(1, 8)]


def vector(text, default='0 0 0'):
    return np.fromstring(text or default, sep=' ')


def origin(element):
    """URDF fixed-axis roll/pitch/yaw transform."""
    result = np.eye(4)
    if element is None:
        return result
    r, p, y = vector(element.get('rpy'))
    cr, cp, cy = np.cos([r, p, y])
    sr, sp, sy = np.sin([r, p, y])
    result[:3, :3] = [[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                      [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                      [-sp, cp*sr, cp*cr]]
    result[:3, 3] = vector(element.get('xyz'))
    return result


def forward_urdf(robot, positions):
    """Independent URDF FK for checking imported anchors, axes, and tips."""
    children = {j.find('child').get('link') for j in robot.findall('joint')}
    roots = {l.get('name') for l in robot.findall('link')} - children
    if len(roots) != 1:
        raise ValueError('Expected one fixed robot root')
    transforms = {roots.pop(): np.eye(4)}
    pending = list(robot.findall('joint'))
    while pending:
        progressed = False
        for joint in pending[:]:
            parent = joint.find('parent').get('link')
            if parent not in transforms:
                continue
            transform = transforms[parent] @ origin(joint.find('origin'))
            if joint.get('type') == 'revolute':
                axis = vector(joint.find('axis').get('xyz'))
                axis /= np.linalg.norm(axis)
                x, y, z = axis
                skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
                angle = positions.get(joint.get('name'), 0.0)
                rotation = np.eye(4)
                rotation[:3, :3] += np.sin(angle)*skew + (1-np.cos(angle))*(skew @ skew)
                transform = transform @ rotation
            transforms[joint.find('child').get('link')] = transform
            pending.remove(joint)
            progressed = True
        if not progressed:
            raise ValueError('Disconnected or cyclic URDF')
    return transforms


def build():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(CONFIG.read_text())
    robot = ET.parse(SOURCE).getroot()
    source_mass = sum(float(m.get('value')) for m in robot.findall('link/inertial/mass'))
    for mesh in robot.findall('.//mesh'):
        filename = mesh.get('filename')
        if not filename.startswith('package://opendrummer/'):
            raise ValueError(f'Unexpected mesh URI: {filename}')
        path = ASSET / filename.removeprefix('package://opendrummer/')
        if not path.is_file():
            raise FileNotFoundError(path)
        mesh.set('filename', path.name)
    custom = ET.SubElement(robot, 'mujoco')
    ET.SubElement(custom, 'compiler', meshdir=str(ASSET / 'meshes'),
                  discardvisual='false', fusestatic='false', inertiafromgeom='false')
    imported = OUTPUT / 'import.urdf'
    ET.ElementTree(robot).write(imported)
    raw = mujoco.MjModel.from_xml_path(str(imported))
    intermediate = OUTPUT / 'import.xml'
    mujoco.mj_saveLastXML(str(intermediate), raw)
    mjcf = ET.parse(intermediate)
    root = mjcf.getroot()
    root.find('compiler').set('fusestatic', 'true')
    root.find('compiler').set('inertiafromgeom', 'false')
    ET.SubElement(root, 'option', timestep=str(cfg['timestep']), integrator='implicitfast', gravity='0 0 -9.81')
    # No collision geometry in step one; do not infer collisions from CAD skins.
    for geom in root.findall('.//geom'):
        geom.set('contype', '0')
        geom.set('conaffinity', '0')
        geom.set('group', '1')
    for joint in root.findall('.//joint'):
        joint.set('armature', str(cfg['armature']))
    world = root.find('worldbody')
    ET.SubElement(world, 'light', pos='1 -2 3', dir='-0.3 0.3 -1', directional='true')
    # Derive the stick tip from its mesh in source coordinates, undoing MuJoCo's
    # principal-axis mesh transform before applying the URDF visual origin.
    mid = raw.mesh('Drumstick').id
    vertices = raw.mesh_vert[raw.mesh_vertadr[mid]:raw.mesh_vertadr[mid]+raw.mesh_vertnum[mid]]
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, raw.mesh_quat[mid])
    vertices = vertices @ rotation.reshape(3, 3).T + raw.mesh_pos[mid]
    low, high = vertices.min(0), vertices.max(0)
    if np.argmax(high-low) != 2:
        raise ValueError('Expected stick length along local Z; review tip derivation')
    mesh_tip = np.array([*(0.5*(low+high))[:2], high[2], 1.0])
    bodies = {b.get('name'): b for b in root.findall('.//body')}
    parents = {j.find('child').get('link'): j for j in robot.findall('joint')}
    tips = {}
    for link_name in ('drumstick', 'drumstick_1'):
        chain, cursor = [], link_name
        while cursor in parents:
            joint = parents[cursor]
            if joint.get('type') == 'revolute':
                chain.append(joint.get('name'))
            cursor = joint.find('parent').get('link')
        side = 'left' if set(chain) == {f'lj{i}' for i in range(1, 8)} else 'right'
        if set(chain) != {f'{side[0]}j{i}' for i in range(1, 8)}:
            raise ValueError(f'Unexpected arm chain for {link_name}: {chain}')
        visual = robot.find(f"link[@name='{link_name}']/visual")
        local = (origin(visual.find('origin')) @ mesh_tip)[:3]
        name = f'{side}_stick_tip'
        ET.SubElement(bodies[link_name], 'site', name=name, pos=' '.join(map(str, local)),
                      size='0.008', rgba='0.1 0.9 0.3 1' if side == 'left' else '1 0.3 0.1 1', group='0')
        tips[name] = {'link': link_name, 'local_position': local.tolist()}
    actuator = ET.SubElement(root, 'actuator')
    limits = {}
    for name in JOINTS:
        limit = robot.find(f"joint[@name='{name}']/limit")
        limits[name] = {key: float(value) for key, value in limit.attrib.items()}
        root.find(f".//joint[@name='{name}']").set('range', f"{limit.get('lower')} {limit.get('upper')}")
        effort = limits[name]['effort']
        ET.SubElement(actuator, 'motor', name=f'{name}_motor', joint=name,
                      ctrllimited='true', ctrlrange=f'{-effort} {effort}',
                      forcelimited='true', forcerange=f'{-effort} {effort}')
    path = OUTPUT / 'opendrummer.xml'
    ET.indent(mjcf)
    mjcf.write(path)
    model = mujoco.MjModel.from_xml_path(str(path))
    report = {
        'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'config_sha256': hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        'mujoco_version': mujoco.__version__,
        'config': cfg, 'source_total_mass_kg': source_mass,
        'imported_total_mass_kg': float(raw.body_mass.sum()),
        'compiled_moving_mass_kg': float(model.body_mass.sum()),
        'body_count_before_fusion': raw.nbody, 'body_count_after_fusion': model.nbody,
        'joint_count': model.njnt, 'visual_geoms': model.ngeom,
        'tips': tips, 'limits': limits,
        'arm_mass_kg': {side: float(model.body_subtreemass[model.jnt_bodyid[model.joint(f'{side[0]}j1').id]])
                        for side in ('left', 'right')},
        'moving_body_masses_kg': {name: float(model.body_mass[model.jnt_bodyid[model.joint(name).id]]) for name in JOINTS},
    }
    (OUTPUT / 'build-report.json').write_text(json.dumps(report, indent=2)+'\n')
    return model, report


class HoldController:
    """Effort-limited PD + bias compensation; only for import/holding checks."""
    def __init__(self, model, report):
        self.model = model
        self.q = np.array([model.jnt_qposadr[model.joint(n).id] for n in JOINTS])
        self.v = np.array([model.jnt_dofadr[model.joint(n).id] for n in JOINTS])
        self.a = np.array([model.actuator(n+'_motor').id for n in JOINTS])
        self.bounds = np.array([[report['limits'][n]['lower'], report['limits'][n]['upper']] for n in JOINTS])
        self.velocity = np.array([report['limits'][n]['velocity'] for n in JOINTS])
        self.effort = np.array([report['limits'][n]['effort'] for n in JOINTS])
        self.kp = np.tile(report['config']['kp'], 2)
        self.kd = np.tile(report['config']['kd'], 2)
        self.target = np.array(report['config']['hold_left'] + report['config']['hold_right'], dtype=float)
        if self.target.shape != (14,) or not np.isfinite(self.target).all() or np.any(self.target < self.bounds[:, 0]) or np.any(self.target > self.bounds[:, 1]):
            raise ValueError('Configured holding pose exceeds joint limits')

    def reset(self, data):
        mujoco.mj_resetData(self.model, data)
        data.qpos[self.q] = self.target
        mujoco.mj_forward(self.model, data)

    def step(self, data, target=None):
        if target is not None:
            target = np.asarray(target, dtype=float)
            if target.shape != (14,) or not np.isfinite(target).all():
                raise ValueError('Expected 14 finite target angles')
            if np.any(target < self.bounds[:, 0]) or np.any(target > self.bounds[:, 1]):
                raise ValueError('Target exceeds joint limits')
            delta = self.velocity * self.model.opt.timestep
            self.target += np.clip(target-self.target, -delta, delta)
        torque = self.kp*(self.target-data.qpos[self.q]) - self.kd*data.qvel[self.v] + data.qfrc_bias[self.v]
        data.ctrl[self.a] = np.clip(torque, -self.effort, self.effort)
        mujoco.mj_step(self.model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError('Nonfinite simulation state')


if __name__ == '__main__':
    model, report = build()
    print(json.dumps(report, indent=2))
    print(f'Built {OUTPUT / "opendrummer.xml"}')
