"""Inspect and display the imported drum-kit OBJ in MuJoCo.

From the repository root on macOS, run with the project's Pixi environment:
    ~/.pixi/bin/pixi run --manifest-path dependencies/mac/pixi.toml mjpython robots/drums/view_drums.py

Uses compiled mesh vertices (in the mesh geom's local frame) and the geom's
world transform to determine the visible bounds. Does not add mesh_pos or
mesh_quat a second time.
"""
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

XML_PATH = Path(__file__).resolve().parent / "drums.xml"

model = mujoco.MjModel.from_xml_path(str(XML_PATH))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

mesh_id = model.mesh("drums_mesh").id
geom_id = model.geom("drums_visual").id

start = int(model.mesh_vertadr[mesh_id])
count = int(model.mesh_vertnum[mesh_id])
vertices = np.asarray(model.mesh_vert[start : start + count], dtype=float)

# The compiled vertices are expressed in the mesh geom's local coordinates.
# The geom world transform accounts for MuJoCo's mesh recentering/rotation.
R = data.geom_xmat[geom_id].reshape(3, 3)
p = data.geom_xpos[geom_id]
world_vertices = vertices @ R.T + p

minimum = world_vertices.min(axis=0)
maximum = world_vertices.max(axis=0)
center = 0.5 * (minimum + maximum)
size = maximum - minimum

print("Drum mesh loaded!")
print("Vertex count:", count)
print("Mesh face count:", int(model.mesh_facenum[mesh_id]))
print("Geom world position:", np.round(p, 4))
print("Actual mesh bounds min:", np.round(minimum, 4))
print("Actual mesh bounds max:", np.round(maximum, 4))
print("Actual mesh center:", np.round(center, 4))
print("Actual mesh size (m):", np.round(size, 4))

with mujoco.viewer.launch_passive(model, data) as viewer:
    with viewer.lock():
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = center
        viewer.cam.distance = max(1.0, float(np.linalg.norm(size)) * 1.2)
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25

    while viewer.is_running():
        viewer.sync()
        time.sleep(0.02)
