from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.typing import NDArray

# 求交只看真实表面：跳过 group 4/5 的调试 marker（reach_target 等）。
SURFACE_GEOMGROUP = np.array([1, 1, 1, 1, 0, 0], dtype=np.uint8)
# 投射时把机器人 geom 临时挪到 group 5，正好被 SURFACE_GEOMGROUP 排除。
ROBOT_HIDDEN_GROUP = 5
MAX_SURFACE_DISTANCE = 10.0


@dataclass(frozen=True)
class CameraFrame:
    rgb: NDArray[np.uint8]
    position: NDArray[np.float64]
    rotation: NDArray[np.float64]
    timestamp: float
    focal: float
    # 沿相机光轴的深度（米）。没有 MjData 可求交时（真机深度相机）unproject 会用它。
    depth: NDArray[np.float32] | None = None


class OverheadCamera:
    def __init__(self, model: mujoco.MjModel, width: int = 1280, height: int = 720, plane_height: float = 0.30, include_depth: bool = False):
        """include_depth 会多渲染一遍深度图（1280x720 约 +11 ms/帧）。

        仿真里取点用射线求交，深度图只在没有 MjData 时（真机深度相机）才需要。
        """
        self.width = width
        self.height = height
        self.plane_height = plane_height
        self.include_depth = include_depth
        self.camera_id = model.camera("overhead").id
        self.model = model
        if model.cam_projection[self.camera_id] != mujoco.mjtProjection.mjPROJ_PERSPECTIVE:
            raise ValueError("Camera must use perspective projection")
        self.focal = height / (2 * np.tan(np.deg2rad(model.cam_fovy[self.camera_id]) / 2))
        self.cx = (width - 1) / 2
        self.cy = (height - 1) / 2
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        # 求交时忽略机械臂自身：点到自己的手臂上不能当成桌面目标。
        self._robot_bodies = {
            index for index in range(model.nbody)
            if model.body(index).name.startswith("openarm_")
        }
        self._robot_geoms = np.array(
            [index for index in range(model.ngeom) if int(model.geom_bodyid[index]) in self._robot_bodies],
            dtype=np.intp,
        )
        self._data: mujoco.MjData | None = None

    def capture(self, data: mujoco.MjData) -> CameraFrame:
        self.focal = self.height / (2 * np.tan(np.deg2rad(self.model.cam_fovy[self.camera_id]) / 2))
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(data, camera=self.camera_id)
        rgb = self.renderer.render().copy()
        depth = None
        if self.include_depth:
            self.renderer.enable_depth_rendering()
            self.renderer.update_scene(data, camera=self.camera_id)
            depth = self.renderer.render().copy()
        self._data = data
        return CameraFrame(
            rgb=rgb,
            position=data.cam_xpos[self.camera_id].copy(),
            rotation=data.cam_xmat[self.camera_id].reshape(3, 3).copy(),
            timestamp=float(data.time),
            focal=self.focal,
            depth=depth,
        )

    def unproject(self, u: int, v: int, frame: CameraFrame) -> NDArray[np.float64]:
        if not 0 <= u < self.width or not 0 <= v < self.height:
            raise ValueError("Pixel coordinates are outside the image")
        # MuJoCo 相机沿局部坐标的 -Z 方向观察，图像行坐标向下增加。
        local = np.array([(u - self.cx) / frame.focal, -(v - self.cy) / frame.focal, -1.0])
        ray = frame.rotation @ local
        # 1) 物理模型可用时按真实几何求交（会越过机械臂，找到它后面的表面）。
        surface = self._surface_along(frame, ray)
        if surface is not None:
            return surface
        # 2) 没有物理模型（例如真机深度相机）时退回 depth buffer。
        point = self._depth_along(frame, ray, u, v)
        if point is not None:
            return point
        # 3) 最后退回桌面平面（旧的平面假设）。
        if abs(ray[2]) < 1e-9:
            raise ValueError("Camera ray is parallel to the target plane")
        distance = (self.plane_height - frame.position[2]) / ray[2]
        if distance <= 0:
            raise ValueError("Target plane is behind the camera")
        return frame.position + distance * ray

    def _surface_along(self, frame: CameraFrame, ray: NDArray[np.float64]) -> NDArray[np.float64] | None:
        """Ray-cast from the captured camera pose, ignoring the arm's own geometry.

        机械臂的 visual mesh 是空心壳，逐层「穿透」会在同一层里反复命中，
        所以投射时把机器人的 geom 临时移出可见分组，只留场景和物体。
        """
        if self._data is None or self._robot_geoms.size == 0:
            return None
        direction = ray / np.linalg.norm(ray)
        saved_groups = self.model.geom_group[self._robot_geoms].copy()
        self.model.geom_group[self._robot_geoms] = ROBOT_HIDDEN_GROUP
        geom_id = np.full(1, -1, dtype=np.int32)
        try:
            distance = float(mujoco.mj_ray(self.model, self._data, frame.position, direction, SURFACE_GEOMGROUP, 1, -1, geom_id))
        finally:
            self.model.geom_group[self._robot_geoms] = saved_groups
        if geom_id[0] < 0 or not np.isfinite(distance) or distance <= 0.0 or distance > MAX_SURFACE_DISTANCE:
            return None
        return frame.position + distance * direction

    def _depth_along(self, frame: CameraFrame, ray: NDArray[np.float64], u: int, v: int) -> NDArray[np.float64] | None:
        """Return the depth-buffer point. MuJoCo depth is metres along the optical axis.

        ray 是未归一化的相机射线（相机坐标系 z=-1 转到世界系），乘以光轴深度即为世界坐标。
        """
        if frame.depth is None:
            return None
        depth = float(frame.depth[v, u])
        if not np.isfinite(depth) or depth <= 0.0 or depth >= MAX_SURFACE_DISTANCE:
            return None
        return frame.position + ray * depth

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
