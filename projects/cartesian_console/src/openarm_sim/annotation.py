from copy import deepcopy
import hashlib
import json
from pathlib import Path

import jsonschema
import mujoco
import numpy as np

from openarm_tools.recording.storage import write_json


def torso_origin(model):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    points = []
    for index in range(model.ngeom):
        if model.geom_bodyid[index] != 0 or model.geom_type[index] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if model.geom(index).name.startswith("kit_"):
            continue
        mesh = model.geom_dataid[index]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        points.extend(model.mesh_vert[start:start + count] @ data.geom_xmat[index].reshape(3, 3).T + data.geom_xpos[index])
    if not points:
        raise ValueError("Fixed torso mesh is required")
    points = np.asarray(points)
    result = (points.min(0) + points.max(0)) / 2
    result[2] = points[:, 2].max()
    return result


def identity(robot, scene):
    from openarm_drummer.collisions import CONTACT_CONFIG

    result = {"robot": robot["source_sha256"], "import": robot["config_sha256"],
            "scene": scene["scene_config_sha256"], "kit": scene["source_obj_sha256"],
            "contact": hashlib.sha256(CONTACT_CONFIG.read_bytes()).hexdigest()}
    if "layout_sha256" in scene:
        result["layout"] = scene["layout_sha256"]
    return result


class DrumLayout:
    def __init__(self, document):
        jsonschema.validate(document, {"type": "object", "required": ["schema", "robot_identity", "origin_m", "targets"],
            "properties": {"schema": {"const": "openarm.drum_layout.v1"}, "robot_identity": {"type": "object"},
                "origin_m": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "number"}},
                "targets": {"type": "object", "minProperties": 7, "maxProperties": 7}}})
        self.document = deepcopy(document)
        for name, item in self.document["targets"].items():
            if item["arm"] not in ("left", "right") or item["arm"] != ("left" if item["base_center_m"][1] > 0 else "right"):
                raise ValueError("Drum arm must match its source side")
            self.set_position(name, item["position_cm"])
            if "ready" in item:
                self.capture_ready(name, item["ready"]["joints_rad"], item["ready"]["source"])

    @classmethod
    def from_simulation(cls, sim):
        return cls.from_scene(sim.model, sim.robot_report, sim.scene_report)

    @classmethod
    def from_scene(cls, model, robot, scene):
        origin = torso_origin(model)
        return cls({"schema": "openarm.drum_layout.v1", "robot_identity": identity(robot, scene),
                    "origin_m": origin.tolist(), "targets": {s["name"]: {
                        "arm": "left" if s["center"][1] > 0 else "right", "base_center_m": s["center"],
                        "position_cm": ((np.array(s["center"]) - origin) * 100).tolist(),
                        "normal": s["normal"], "radius_m": s["radius"]}
                        for s in scene["surfaces"] if s["kind"] != "kick"}})

    @classmethod
    def load(cls, path, sim):
        layout = cls(json.loads(Path(path).read_text()))
        layout.validate(sim.model, sim.robot_report, sim.scene_report)
        return layout

    def validate(self, model, robot, scene):
        if self.document["robot_identity"] != identity(robot, scene):
            raise ValueError("Layout robot/scene identity differs")
        if not np.allclose(self.document["origin_m"], torso_origin(model), atol=1e-7):
            raise ValueError("Layout torso reference differs")
        surfaces = {s["name"]: s for s in scene["surfaces"] if s["kind"] != "kick"}
        if set(self.document["targets"]) != set(surfaces):
            raise ValueError("Layout target names differ")
        for name, item in self.document["targets"].items():
            if not np.allclose(item["base_center_m"], surfaces[name].get("annotation_base_center_m", surfaces[name]["center"]), atol=1e-7):
                raise ValueError("Layout source center differs")
            if not np.allclose(item["normal"], surfaces[name]["normal"], atol=1e-7) or not np.isclose(item["radius_m"], surfaces[name]["radius"]):
                raise ValueError("Layout surface geometry differs")

    def set_position(self, name, position_cm):
        values = np.asarray(position_cm, dtype=float)
        if values.shape != (3,) or not np.isfinite(values).all() or np.any(np.abs(values) > 200):
            raise ValueError("Position requires three finite centimeters within +/-200 cm")
        item = self.document["targets"][name]
        if "position_cm" in item and not np.array_equal(values, item["position_cm"]):
            item.pop("ready", None)
        item["position_cm"] = values.tolist()

    def center(self, name):
        return np.array(self.document["origin_m"]) + np.array(self.document["targets"][name]["position_cm"]) / 100

    def capture_ready(self, name, joints, source):
        values = np.asarray(joints, dtype=float)
        if values.shape != (14,) or not np.isfinite(values).all():
            raise ValueError("Ready pose requires fourteen finite joint positions")
        if source not in ("simulation_feedback", "physical_feedback", "simulation_preview"):
            raise ValueError("Ready pose source must be explicit")
        self.document["targets"][name]["ready"] = {"joints_rad": values.tolist(), "source": source}

    def apply(self, model, scene):
        for surface in scene["surfaces"]:
            name = surface["name"]
            if name not in self.document["targets"]:
                continue
            center = self.center(name)
            surface.setdefault("annotation_base_center_m", list(surface["center"]))
            delta = center - np.array(surface["center"])
            for index in range(model.ngeom):
                if model.geom_bodyid[index] == 0 and model.geom(index).name.startswith(name + "_"):
                    model.geom_pos[index] += delta
            for index in range(model.nsite):
                if model.site_bodyid[index] == 0 and model.site(index).name.startswith(name + "_"):
                    model.site_pos[index] += delta
            surface["center"] = center.tolist()
        for index in range(model.ngeom):
            name = model.geom(index).name
            if name.startswith("kit_material_"):
                model.geom_rgba[index, 3] = 0
            elif name.endswith("_head_collision"):
                model.geom_rgba[index] = [.3, .55, .75, .8]

    def save(self, path):
        write_json(Path(path), self.document)
