import json

import mujoco
import pytest

from openarm_rl.planning.scene import prepare_scene
from openarm_rl.planning.strikes import plan_center
from openarm_rl.skills.registry import SKILLS


@pytest.fixture(scope="session")
def native_strike_fixture(tmp_path_factory):
    directory = tmp_path_factory.mktemp("native-camera-free")
    prepare_scene(directory / "scene")
    model = mujoco.MjModel.from_binary_path(str(directory / "scene/model.mjb"))
    metadata = json.loads((directory / "scene/scene.json").read_text())
    profile = plan_center(model, metadata["robot"], metadata["scene"], SKILLS["snare"])
    assert profile["status"] == "ready", profile
    (directory / "snare.json").write_text(json.dumps(profile))
    return directory
