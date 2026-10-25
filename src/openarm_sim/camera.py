from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CameraFrame:
    rgb: NDArray[np.uint8]
    position: NDArray[np.float64]
    rotation: NDArray[np.float64]
    timestamp: float
    focal: float


class OverheadCamera:
    def __init__(self, model: mujoco.MjModel, width: int = 1280, height: int = 720, plane_height: float = 0.30):
        self.width = width
        self.height = height
        self.plane_height = plane_height
        self.camera_id = model.camera("overhead").id
        self.model = model
        if model.cam_projection[self.camera_id] != mujoco.mjtProjection.mjPROJ_PERSPECTIVE:
            raise ValueError("Camera must use perspective projection")
        self.focal = height / (2 * np.tan(np.deg2rad(model.cam_fovy[self.camera_id]) / 2))
        self.cx = (width - 1) / 2
        self.cy = (height - 1) / 2
        self.renderer = mujoco.Renderer(model, height=height, width=width)

    def capture(self, data: mujoco.MjData) -> CameraFrame:
        self.focal = self.height / (2 * np.tan(np.deg2rad(self.model.cam_fovy[self.camera_id]) / 2))
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(data, camera=self.camera_id)
        rgb = self.renderer.render().copy()
        return CameraFrame(
            rgb=rgb,
            position=data.cam_xpos[self.camera_id].copy(),
            rotation=data.cam_xmat[self.camera_id].reshape(3, 3).copy(),
            timestamp=float(data.time),
            focal=self.focal,
        )

    def unproject(self, u: int, v: int, frame: CameraFrame) -> NDArray[np.float64]:
        if not 0 <= u < self.width or not 0 <= v < self.height:
            raise ValueError("Pixel coordinates are outside the image")
        # MuJoCo 相机沿局部坐标的 -Z 方向观察，图像行坐标向下增加。
        local = np.array([(u - self.cx) / frame.focal, -(v - self.cy) / frame.focal, -1.0])
        ray = frame.rotation @ local
        if abs(ray[2]) < 1e-9:
            raise ValueError("Camera ray is parallel to the target plane")
        distance = (self.plane_height - frame.position[2]) / ray[2]
        if distance <= 0:
            raise ValueError("Target plane is behind the camera")
        return frame.position + distance * ray

    def project(self, point: NDArray[np.float64], frame: CameraFrame) -> tuple[int, int]:
        local = frame.rotation.T @ (point - frame.position)
        if local[2] >= 0:
            raise ValueError("Point is behind the camera")
        return (
            round(self.cx + frame.focal * local[0] / -local[2]),
            round(self.cy - frame.focal * local[1] / -local[2]),
        )

    def close(self) -> None:
        self.renderer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
