import mujoco
import numpy as np


class CameraControls:
    def __init__(self, simulation):
        self.simulation = simulation
        self.camera = mujoco.MjvCamera()
        self.scene = mujoco.MjvScene()
        model = simulation.model
        camera_id = model.camera("overhead").id
        offset = model.cam_pos[camera_id] - simulation.camera_look_at
        self.camera.lookat[:] = simulation.camera_look_at
        self.camera.distance = float(np.linalg.norm(offset))
        self.camera.azimuth = float(np.degrees(np.arctan2(-offset[1], -offset[0])))
        self.camera.elevation = float(np.degrees(np.arctan2(-offset[2], np.linalg.norm(offset[:2]))))
        self.fovy = float(model.cam_fovy[camera_id])

    def move(self, action: mujoco.mjtMouse, dx: float, dy: float) -> None:
        if not np.isfinite([dx, dy]).all():
            raise ValueError("Camera movement must be finite")
        sim = self.simulation
        mujoco.mjv_moveCamera(sim.model, action, dx, dy, self.camera)
        mujoco.mjv_updateCamera(sim.model, sim.data, self.camera, self.scene)
        camera = mujoco.mjv_averageCamera(*self.scene.camera)
        position = np.asarray(camera.pos, dtype=float)
        look_at = position + np.asarray(camera.forward, dtype=float)*self.camera.distance
        sim.set_camera(position, look_at, self.fovy)
