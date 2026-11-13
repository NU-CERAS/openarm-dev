import json
from pathlib import Path

import numpy as np
import pytest

from openarm_rl.control.reference import JointReference, SpeedConditionedReference


def bank(speeds):
    positions = np.zeros((3, 14))
    positions[1, 6] = .1
    reference = JointReference([0., .2, .4], positions, 6, .2).document()
    return {"schema": "openarm.speed_conditioned_strike.v1", "speeds_m_s": speeds,
            "references": [reference for _ in speeds]}


def test_single_operating_point_does_not_claim_interpolation():
    reference = SpeedConditionedReference(bank([.6]))
    assert reference.at(.6).impact_time == .2
    for speed in (.35, .85, np.nan):
        with pytest.raises(ValueError):
            reference.at(speed)


@pytest.mark.parametrize("speeds", [[], [np.nan], [0.], [-.1], [.6, .6], [.8, .4]])
def test_invalid_reference_speed_axes_rejected(speeds):
    with pytest.raises(ValueError):
        SpeedConditionedReference(bank(speeds))


def test_single_speed_environment_resets_at_its_operating_point(rhythm_options, tmp_path):
    from openarm_rl.envs.camera_free_strike import CameraFreeStrikeEnv

    profile = json.loads((Path(rhythm_options["profiles"])/"snare.json").read_text())
    profile["reference_bank"]["speeds_m_s"] = [.8]
    profile["reference_bank"]["references"] = profile["reference_bank"]["references"][:1]
    path = tmp_path/"single-speed.json"
    path.write_text(json.dumps(profile))
    with CameraFreeStrikeEnv(rhythm_options["scene_directory"], path, position_range=0, domain_range=0) as env:
        _, info = env.reset(seed=7)
        assert info["target_speed_m_s"] == .8
