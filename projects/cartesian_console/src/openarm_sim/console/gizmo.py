from dataclasses import dataclass

import numpy as np

from ..camera import CameraFrame, OverheadCamera


def hit_axis(pixel, handles, radius=12.0):
    point = np.asarray(pixel, dtype=float)
    tips = [(float(np.linalg.norm(point - end)), axis) for axis, (_, end) in handles.items()]
    if tips and min(tips)[0] <= radius:
        return min(tips)[1]
    candidates = []
    for axis, (start, end) in handles.items():
        direction = np.asarray(end) - start
        length_squared = float(direction @ direction)
        if length_squared < 18**2:
            continue
        fraction = float((point - start) @ direction / length_squared)
        if 0.25 <= fraction <= 1.0:
            distance = float(np.linalg.norm(point - (start + fraction * direction)))
            if distance <= radius:
                candidates.append((distance, axis))
    return min(candidates)[1] if candidates else None


@dataclass(frozen=True)
class AxisDrag:
    camera: OverheadCamera
    frame: CameraFrame
    origin: np.ndarray
    axis: np.ndarray
    index: int
    start_parameter: float

    @staticmethod
    def parameter(camera, frame, origin, axis, pixel):
        ray = camera.pixel_ray(*pixel, frame)
        ray /= np.linalg.norm(ray)
        solution, _, rank, singular = np.linalg.lstsq(np.column_stack((axis, -ray)), frame.position - origin, rcond=None)
        if rank != 2 or singular[-1] < 0.03 or solution[1] <= 0:
            return None
        return float(solution[0])

    @classmethod
    def begin(cls, camera, frame, origin, axis, index, pixel):
        origin = np.asarray(origin, dtype=float).copy()
        axis = np.asarray(axis, dtype=float).copy()
        axis /= np.linalg.norm(axis)
        parameter = cls.parameter(camera, frame, origin, axis, pixel)
        if parameter is None:
            return None
        return cls(camera, frame, origin, axis, index, parameter)

    def target(self, pixel):
        parameter = self.parameter(self.camera, self.frame, self.origin, self.axis, pixel)
        if parameter is None:
            return None
        return self.origin + self.axis * (parameter - self.start_parameter)
