import hashlib
import json
import shutil
from pathlib import Path

import mujoco

from openarm_drummer.model import CONFIG, SOURCE
from openarm_drummer.scene import build_scene


def prepare_scene(directory: Path, scene_config: Path | None = None, layout_file: Path | None = None) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    configuration = None if scene_config is None else json.loads(scene_config.read_text())
    model, robot, scene = build_scene(configuration=configuration)
    if layout_file is not None:
        from openarm_sim.annotation import DrumLayout

        layout = DrumLayout(json.loads(layout_file.read_text()))
        layout.validate(model, robot, scene)
        layout.apply(model, scene)
        scene["layout_sha256"] = hashlib.sha256(layout_file.read_bytes()).hexdigest()
        scene["annotations"] = layout.document
        shutil.copyfile(layout_file, directory / "layout.json")
    mujoco.mj_saveModel(model, str(directory / "model.mjb"), None)
    sources = [SOURCE, CONFIG, CONFIG.with_name("scene.json"), CONFIG.with_name("contact.json"), CONFIG.with_name("strike.json")]
    metadata = {"robot": robot, "scene": scene,
                "source_hashes": {str(p.relative_to(SOURCE.parents[3])): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    if scene_config is not None:
        metadata["source_hashes"]["scene_override"] = hashlib.sha256(scene_config.read_bytes()).hexdigest()
    (directory / "scene.json").write_text(json.dumps(metadata, indent=2) + "\n")
