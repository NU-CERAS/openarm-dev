"""Map pinned official rigid-body inertials into OpenDrummer's CAD frames.

Offline and reproducible: reads the preserved export and official URDF snapshots.
Only inertial elements change. Custom joint-7 tools remain separate estimates.
"""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET

import numpy as np

ASSET = Path(__file__).resolve().parents[1]
PROVENANCE = ASSET / 'provenance'
COMMIT = '10ca9bf2f49ef93a2f0f83f97ed2c0fb42dd9a90'


def transform(element):
    T = np.eye(4)
    if element is None:
        return T
    r, p, y = np.fromstring(element.get('rpy', '0 0 0'), sep=' ')
    cr, cp, cy = np.cos([r, p, y])
    sr, sp, sy = np.sin([r, p, y])
    T[:3, :3] = [[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                 [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                 [-sp, cp*sr, cp*cr]]
    T[:3, 3] = np.fromstring(element.get('xyz', '0 0 0'), sep=' ')
    return T


def zero_frames(robot):
    parents = {j.find('child').get('link'): j for j in robot.findall('joint')}
    frames = {}

    def visit(name):
        if name not in frames:
            if name not in parents:
                frames[name] = np.eye(4)
            else:
                j = parents[name]
                frames[name] = visit(j.find('parent').get('link')) @ transform(j.find('origin'))
        return frames[name]

    for link in robot.findall('link'):
        visit(link.get('name'))
    return frames, parents


def properties(inertial):
    mass = float(inertial.find('mass').get('value'))
    a = {k: float(v) for k, v in inertial.find('inertia').attrib.items()}
    tensor = np.array([[a['ixx'], a['ixy'], a['ixz']],
                       [a['ixy'], a['iyy'], a['iyz']],
                       [a['ixz'], a['iyz'], a['izz']]])
    return mass, tensor


def write_inertial(link, mass, com, tensor):
    inertial = ET.SubElement(link, 'inertial')
    ET.SubElement(inertial, 'origin', xyz=' '.join(format(x, '.17g') for x in com), rpy='0 0 0')
    ET.SubElement(inertial, 'mass', value=format(mass, '.17g'))
    fields = {'ixx': tensor[0, 0], 'ixy': tensor[0, 1], 'ixz': tensor[0, 2],
              'iyy': tensor[1, 1], 'iyz': tensor[1, 2], 'izz': tensor[2, 2]}
    ET.SubElement(inertial, 'inertia', **{k: format(v, '.17g') for k, v in fields.items()})


def update():
    source_path = PROVENANCE / 'original-opendrummer.urdf'
    official_path = PROVENANCE / 'official-v1.urdf'
    robot = ET.parse(source_path).getroot()
    official = ET.parse(official_path).getroot()
    frames, parents = zero_frames(robot)
    upstream_frames, _ = zero_frames(official)
    links = {l.get('name'): l for l in robot.findall('link')}
    upstream_links = {l.get('name'): l for l in official.findall('link')}
    joints = {j.get('name'): j for j in robot.findall('joint')}
    groups = {}
    for name in links:
        cursor = name
        while cursor in parents and parents[cursor].get('type') == 'fixed':
            cursor = parents[cursor].find('parent').get('link')
        groups[name] = parents[cursor].get('name') if cursor in parents else 'fixed_base'

    # First remove distributed CAD inertias from the assemblies being replaced.
    # Joint-7 descendants are exclusively Base/ScrewOn/screws/stick in this export.
    custom = {name for name, group in groups.items() if group in ('lj7', 'rj7')}
    expected_custom = {'base', 'base_1', 'screwon', 'screwon_1', 'drumstick', 'drumstick_1'}
    expected_custom.update('hex_socket_head_cap_screw_m3x0_50_x_8'+(f'_{i}' if i else '') for i in range(8))
    if custom != expected_custom:
        raise ValueError('Joint-7 components changed; review custom tool policy')
    for name, link in links.items():
        if name not in custom:
            inertial = link.find('inertial')
            if inertial is not None:
                link.remove(inertial)

    mappings = [('openarm_body_link0', 'root', 'fixed_base')]
    for side in ('left', 'right'):
        prefix = side[0]
        mappings.append((f'openarm_{side}_link0', joints[f'{prefix}j1'].find('parent').get('link'), 'fixed_base'))
        for i in range(1, 7):
            target = joints[f'{prefix}j{i}'].find('child').get('link')
            mappings.append((f'openarm_{side}_link{i}', target, f'{prefix}j{i}'))

    rows = []
    for upstream, target, group in mappings:
        inertial = upstream_links[upstream].find('inertial')
        mass, tensor = properties(inertial)
        # Both descriptions share the pedestal/world coordinate system at zero.
        # T maps official COM coordinates into the target CAD carrier frame.
        T = np.linalg.inv(frames[target]) @ upstream_frames[upstream] @ transform(inertial.find('origin'))
        rotated = T[:3, :3] @ tensor @ T[:3, :3].T
        write_inertial(links[target], mass, T[:3, 3], rotated)
        rows.append({'official_link': upstream, 'target_link': target, 'assembly': group,
                     'mass_kg': mass, 'com_in_target_m': T[:3, 3].tolist(),
                     'inertia_in_target_kg_m2': rotated.tolist()})

    custom_rows = []
    for name in sorted(custom):
        link = links[name]
        inertial = link.find('inertial')
        scale = 1000.0 if name in ('base', 'base_1', 'screwon', 'screwon_1') else 1.0
        mass, tensor = properties(inertial)
        inertial.find('mass').set('value', format(mass*scale, '.17g'))
        for key, value in list(inertial.find('inertia').attrib.items()):
            inertial.find('inertia').set(key, format(float(value)*scale, '.17g'))
        custom_rows.append({'link': name, 'mass_kg': mass*scale, 'source_scale': scale,
                            'status': 'Custom CAD estimate; not official or measured'})

    # Keep the kinematic/visual tree byte-equivalent at the XML element level.
    original = ET.parse(source_path).getroot()
    for tree in (original, ET.fromstring(ET.tostring(robot))):
        for link in tree.findall('link'):
            inertial = link.find('inertial')
            if inertial is not None:
                link.remove(inertial)
        for element in tree.iter():
            element.text = None
            element.tail = None
        if tree is original:
            expected = ET.tostring(tree)
        else:
            assert ET.tostring(tree) == expected, 'Non-inertial content changed'
    destination = ASSET / 'urdf/opendrummer.urdf'
    ET.indent(robot)
    ET.ElementTree(robot).write(destination, encoding='utf-8', xml_declaration=True)
    report = {
        'official_commit': COMMIT,
        'official_url': f'https://github.com/enactic/openarm_description/blob/{COMMIT}/assets/robot/openarm_v1.0/urdf/example/v1.urdf',
        'original_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
        'official_sha256': hashlib.sha256(official_path.read_bytes()).hexdigest(),
        'output_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
        'frame_policy': 'Shared world at zero; rotate COM inertia into each CAD carrier frame. Geometry/joint frames unchanged.',
        'official_assemblies': rows, 'custom_tool_components': custom_rows,
        'limitations': [
            'Official link7 end-effector assembly is absent and is not added to custom tools.',
            'Base/ScrewOn mass and inertia retain a provisional 1000x CAD unit correction; weigh these parts to calibrate.',
            'Official model properties are reference values, not measurements of this robot.',
            'CAD and official axes have small offsets; reference inertias are mapped at zero without changing CAD kinematics.',
        ],
    }
    (ASSET / 'inertial-provenance.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'Updated {destination}: {len(rows)} official assemblies, {len(custom_rows)} custom tool components')


if __name__ == '__main__':
    update()
