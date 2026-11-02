# OpenDrummer inertial provenance

The current URDF uses published OpenArm v1.0 model inertials for the pedestal
body, two fixed arm bases, and moving links 1–6 of both arms. These are official
reference model parameters, not measured properties of this particular robot.

Upstream repository: [enactic/openarm_description](https://github.com/enactic/openarm_description).
Pinned revision: `10ca9bf2f49ef93a2f0f83f97ed2c0fb42dd9a90`.

Included source snapshots:

- [Official generated v1 URDF](https://github.com/enactic/openarm_description/blob/10ca9bf2f49ef93a2f0f83f97ed2c0fb42dd9a90/assets/robot/openarm_v1.0/urdf/example/v1.urdf): `official-v1.urdf`.
- [Arm inertials](https://github.com/enactic/openarm_description/blob/10ca9bf2f49ef93a2f0f83f97ed2c0fb42dd9a90/assets/robot/openarm_v1.0/config/arm/inertials.yaml): `official-arm-inertials.yaml`.
- [Body inertials](https://github.com/enactic/openarm_description/blob/10ca9bf2f49ef93a2f0f83f97ed2c0fb42dd9a90/assets/robot/openarm_v1.0/config/body/inertials.yaml): `official-body-inertials.yaml`.
- Original Desktop export, before inertial edits: `original-opendrummer.urdf`.
  This is a provenance snapshot, not a second runnable model entry point.

Upstream license notices are retained. See `../OPENARM-LICENSE`.

## Mapping method

The two robot descriptions share the pedestal/world frame and the physical
zero pose. For each corresponding rigid assembly, the updater computes the
transform from the official inertial frame to the CAD carrier link at zero.
It transfers the center of mass and rotates the full COM inertia tensor using
`I_target = R I_official Rᵀ`. Translation moves the COM origin; no parallel-axis
term is added to a tensor that remains expressed about that COM.

The mass and inertia of each complete official assembly are placed on one
carrier link. Other CAD parts rigidly attached to that assembly retain their
visuals and fixed joints but lose their old individual inertials. This avoids
double-counting. The root carries the official body inertia; each joint-1
parent carries its side's official link-0 inertia.

CAD and official joint axes are not numerically identical (including a roughly
1.7 mm transverse difference at joint 7). Kinematics are preserved rather than
silently changed. Validation compares official link 1–6 COMs and tensors across
multiple poses with the documented axis-sign mapping, using small tolerances.
The custom tool follows its original CAD wrist axis.

| Assembly | Mass (kg) |
|---|---:|
| Body/pedestal | 3.740000 |
| Each fixed arm base, link 0 | 1.143228494 |
| Link 1 | 1.141668465 |
| Link 2 | 0.277509275 |
| Link 3 | 1.073863338 |
| Link 4 | 0.634853457 |
| Link 5 | 0.615658803 |
| Link 6 | 0.475202773 |
| Custom joint-7 tool, each side | 0.101493484 |

Each moving arm totals 4.320249594 kg. Total assembly mass is 14.666956176 kg.
Generated MJCF can differ by a few micrograms due to MuJoCo XML serialization.

## Custom tool exception

The official link-7 assembly is **not** installed in this export. Its 0.405497127
kg official inertia is therefore not added. The entire CAD subtree after each
seventh joint consists of these custom parts:

| Part | Count per arm | Mass per part (kg) | Treatment |
|---|---:|---:|---|
| Base | 1 | 0.054680600 | Original mass and inertia ×1000; provisional |
| ScrewOn | 1 | 0.023759000 | Original mass and inertia ×1000; provisional |
| M3 screw | 4 | 0.000943321 | Original export values retained |
| Drumstick | 1 | 0.019280600 | Original export values retained |

All custom tool properties remain CAD estimates, not measured or official
OpenArm values. Verify material units or weigh the mounts before interpreting
impact forces as physically calibrated. The old broad 1000x correction has
been removed from the playground importer.

## Reproduce and verify

From the repository root, with the existing Pixi environment active:

```sh
python robots/opendrummer/scripts/update_inertials.py
python projects/opendrummer_playground/scripts/validate.py
```

The updater reads the local pinned snapshots and preserved original URDF, so
it needs no network and never compounds a previous correction. It verifies
that non-inertial XML content is unchanged and writes `../inertial-provenance.json`
with source/output hashes, carrier mappings, transformed tensors, and exceptions.
The validator checks the official assignments, compiled moving inertias,
kinematics, mass aggregation, holding stability, and joint excursions.
