OpenDrummer repaired URDF with OpenArm v1.0 joint limits

Load urdf/opendrummer.urdf together with meshes.
Original Desktop files are unchanged.

All 20 arm shell bodies remain attached to the moving arm chains.
All 14 arm joints are revolute, with bounded position, velocity, and effort.
Limits are from the official OpenArm v1.0 bimanual example, matched to
this export's zero pose and reversed where the rotation axis is reversed.
See joint-limits.json for provenance and exact values.
Source: https://github.com/enactic/openarm_description/blob/1f8a3c76d8b043dfb4256061ddd61ca0441884e0/assets/robot/openarm_v1.0/urdf/example/v1.urdf

Position limits checked at all 28 endpoints using forward kinematics.
This checks the joint tree, not collision clearance with the drumsticks.
The custom drumstick geometry may collide before an arm reaches a limit.

Repository update: inertials now use pinned official OpenArm base/link 0-6
properties, plus documented custom-tool estimates. See provenance/README.md.
The preserved original URDF is provenance/original-opendrummer.urdf.
