import json

import mujoco
import numpy as np
import pytest

from openarm_sim.annotation import DrumLayout
from openarm_sim.simulation import reach_simulation


def test_layout_moves_geometry_and_roundtrips_reference(tmp_path):
    with reach_simulation("opendrummer") as sim:
        layout = DrumLayout.from_simulation(sim)
        original = np.array(layout.document["targets"]["snare"]["position_cm"])
        geom = sim.model.geom("snare_head_collision").id
        before = sim.model.geom_pos[geom].copy()
        layout.set_position("snare", original + [1, 2, 3])
        layout.apply(sim.model, sim.scene_report)
        np.testing.assert_allclose(sim.model.geom_pos[geom] - before, [.01, .02, .03])
        layout.apply(sim.model, sim.scene_report)
        np.testing.assert_allclose(sim.model.geom_pos[geom] - before, [.01, .02, .03])
        joints = np.zeros(14)
        layout.capture_ready("snare", joints, "simulation_feedback")
        path = tmp_path / "layout.json"
        layout.save(path)
        loaded = DrumLayout.load(path, sim)
        assert loaded.document == layout.document
        assert loaded.document["targets"]["snare"]["ready"]["source"] == "simulation_feedback"
        document = json.loads(path.read_text())
        document["robot_identity"]["robot"] = "invalid"
        path.write_text(json.dumps(document))
        with pytest.raises(ValueError, match="identity"):
            DrumLayout.load(path, sim)


def test_layout_rejects_motion_and_invalid_values():
    with reach_simulation("opendrummer") as sim:
        layout = DrumLayout.from_simulation(sim)
        with pytest.raises(ValueError):
            layout.set_position("snare", [np.nan, 0, 0])
        assert len(layout.document["targets"]) == 7
