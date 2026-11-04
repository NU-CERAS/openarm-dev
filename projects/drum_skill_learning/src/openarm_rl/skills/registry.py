from dataclasses import dataclass
import json

import jsonschema

from ..paths import PROJECT_ROOT


@dataclass(frozen=True)
class StrikeSkill:
    name: str
    target: str
    arm: str

    @property
    def wrist(self) -> int:
        return 6 if self.arm == "left" else 13

    def schema(self) -> dict:
        return {"type": "function", "function": {"name": self.name,
                "description": f"Strike the center of {self.target} with the {self.arm} arm, then recover.",
                "parameters": {"type": "object", "properties": {
                    "target_speed_m_s": {"type": "number", "minimum": 0.35, "maximum": 0.85},
                    "timing_offset_s": {"const": 0.0, "type": "number"}},
                    "required": ["target_speed_m_s", "timing_offset_s"], "additionalProperties": False}, "strict": True}}


def load_skills() -> dict[str, StrikeSkill]:
    document = json.loads((PROJECT_ROOT / "configs/skills/strikes.json").read_text())
    jsonschema.validate(document, {"type": "object", "properties": {
        "schema": {"const": "openarm.strike_skills.v1"}, "skills": {"type": "array", "minItems": 7, "maxItems": 7,
        "items": {"type": "object", "properties": {"name": {"type": "string"}, "target": {"type": "string"},
                 "arm": {"enum": ["left", "right"]}}, "required": ["name", "target", "arm"], "additionalProperties": False}}},
        "required": ["schema", "skills"], "additionalProperties": False})
    records = [StrikeSkill(**row) for row in document["skills"]]
    if len({r.name for r in records}) != 7 or len({r.target for r in records}) != 7:
        raise ValueError("Strike tool names and targets must be unique")
    for record in records:
        if record.name != f"strike_{record.target}":
            raise ValueError("Strike names must identify their fixed drum target")
    return {r.target: r for r in records}


SKILLS = load_skills()
