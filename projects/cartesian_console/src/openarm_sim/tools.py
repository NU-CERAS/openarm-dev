import base64
import io

import jsonschema
from PIL import Image, ImageDraw

from .camera import CameraFrame, OverheadCamera
from .simulation import ReachSimulation


TOOL_DEFINITIONS = {
    "observe": {
        "description": "获取机器人头部相机图像、frame_id、相机标定参数和当前状态。",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    },
    "get_state": {
        "description": "读取实际 TCP 坐标与最近一次 reach 的执行状态。",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    },
    "reach_pixel": {
        "description": "根据最近一张相机图像上的像素选择桌面位置，执行指定机械臂的 reach，TCP 高度为桌面上方 0.14 米。",
        "parameters": {
            "type": "object",
            "properties": {
                "arm": {"type": "string", "enum": ["right", "left"]},
                "u": {"type": "integer", "minimum": 0},
                "v": {"type": "integer", "minimum": 0},
                "frame_id": {"type": "integer", "minimum": 1},
            },
            "required": ["arm", "u", "v", "frame_id"],
            "additionalProperties": False,
        },
    },
}


class SimulationTools:
    def __init__(self, width: int = 1280, height: int = 720, planner: str = "mink"):
        self.sim = ReachSimulation(planner=planner)
        self.camera = OverheadCamera(self.sim.model, width, height, self.sim.table_height)
        self.frame: CameraFrame | None = None
        self.frame_id = 0

    @staticmethod
    def schemas() -> list[dict]:
        return [
            {"type": "function", "function": {"name": name, "strict": True, **definition}}
            for name, definition in TOOL_DEFINITIONS.items()
        ]

    def state(self) -> dict:
        status = self.sim.status
        return {
            "state": status.state,
            "reason": status.reason,
            "error_m": status.error_m,
            "target": None if status.target is None else status.target.tolist(),
            "tcp": {arm: self.sim.tcp_position(arm).tolist() for arm in ["right", "left"]},
            "time_s": float(self.sim.data.time),
            "frame": "world",
            "robot": "OpenArm v1.0",
            "planner": self.sim.planner_backend,
        }

    def call(self, name: str, arguments: dict) -> dict:
        if name not in TOOL_DEFINITIONS:
            raise ValueError(f"未知工具：{name}")
        jsonschema.validate(arguments, TOOL_DEFINITIONS[name]["parameters"])
        if name == "get_state":
            return self.state()
        if name == "observe":
            self.frame = self.camera.capture(self.sim.data)
            self.frame_id += 1
            return {
                **self.state(), "frame_id": self.frame_id,
                "image_size": [self.camera.width, self.camera.height],
                "camera_position": self.frame.position.tolist(),
                "camera_rotation": self.frame.rotation.tolist(),
                "intrinsics": {"fx": self.camera.focal, "fy": self.camera.focal, "cx": self.camera.cx, "cy": self.camera.cy},
                "table_height_m": self.sim.table_height,
                "target_clearance_m": 0.14,
            }
        if self.frame is None or arguments["frame_id"] != self.frame_id:
            raise ValueError("frame_id 与当前图像不一致，需要重新 observe")
        target = self.camera.unproject(arguments["u"], arguments["v"], self.frame) + [0, 0, 0.14]
        acceptance = self.sim.start_reach(arguments["arm"], target)
        if not acceptance.accepted:
            return {**self.state(), "accepted": False, "reason": acceptance.reason}
        for _ in range(30000):
            self.sim.step()
            if self.sim.status.state != "running":
                return {**self.state(), "accepted": True}
        raise TimeoutError("reach 执行超过工具时间限制")

    def image_content(self) -> dict:
        if self.frame is None:
            raise RuntimeError("需要调用 observe 获取图像")
        image = Image.fromarray(self.frame.rgb)
        draw = ImageDraw.Draw(image)
        for u in range(0, self.camera.width, 100):
            draw.line((u, 0, u, self.camera.height), fill=(160, 160, 160), width=1)
            draw.text((u + 2, 3), str(u), fill=(255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0))
        for v in range(0, self.camera.height, 100):
            draw.line((0, v, self.camera.width, v), fill=(160, 160, 160), width=1)
            draw.text((3, v + 2), str(v), fill=(255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}

    def close(self) -> None:
        self.camera.close()
        self.sim.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
