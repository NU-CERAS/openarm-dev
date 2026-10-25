# Robot assets

- `v0.3/models/`: existing standalone OpenArm single-arm/bimanual assets.
- `v1.0/models/`: existing bimanual MuJoCo scene previously under the Windows setup guide.
- `v1.0/urdf/`: ROS-control overlay using the upstream OpenArm description submodule.

Each version has left/right seven-joint JSON configurations and a place for reviewed poses. Config limits are extracted from the matching MJCF. Hardware types/IDs/gains are templates requiring physical review; model limits alone do not validate a physical arm. Preserve upstream copyright notices and the included Apache-2.0 licenses.
