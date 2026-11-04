import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from openarm_drummer.model import HoldController
from openarm_rl.music.reference import MUSIC_TRANSFER_CLEARANCE_M, TransferPlanner, load_strokes

SCENE = Path(__file__).parents[4]/"runs/rl/original-midi-preparation-20261103/scene"
PROFILES = Path(__file__).parents[4]/"runs/rl/tuned-references-20261104/profiles"
pytestmark = pytest.mark.skipif(not PROFILES.exists(), reason="prepared scene and profiles are not available")


def test_fast_transfer_keeps_room_for_tracking_lag_around_the_rack_tom_rim():
    model = mujoco.MjModel.from_binary_path(str(SCENE/"model.mjb"))
    metadata = json.loads((SCENE/"scene.json").read_text())
    controller = HoldController(model, metadata["robot"])
    _, references, _ = load_strokes(metadata, PROFILES, ["snare", "rack_tom_left"])
    start, goal = (references[target, "left"].positions[0] for target in ("snare", "rack_tom_left"))
    planner = TransferPlanner(model, controller, [{"transfer_limits": {"velocity_rad_s": 4., "acceleration_rad_s2": 30.}}])
    transfer = planner.plan(start, goal, "left")
    data, scratch = mujoco.MjData(model), np.zeros(6)
    tip = model.geom("left_stick_tip_collision").id
    rims = [index for index in range(model.ngeom) if (model.geom(index).name or "").startswith("rack_tom_left_rim")]
    gaps = []
    for time_s in np.linspace(0, transfer.ends[-1], 300):
        data.qpos[controller.q] = transfer.sample(time_s)[0]
        mujoco.mj_forward(model, data)
        gaps.append(min(mujoco.mj_geomDistance(model, data, tip, rim, .5, scratch) for rim in rims))
    assert min(gaps) >= MUSIC_TRANSFER_CLEARANCE_M-.002
