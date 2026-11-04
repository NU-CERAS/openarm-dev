import numpy as np
import pytest

from openarm_rl.planning.kit_strikes import RIM_MARGIN, offset_scene, strike_point, tuned_stroke
from openarm_rl.planning.rhythmic_references import select_stroke, shared_ready


SCENE = {"surfaces": [{"name": "hi_hat", "center": [.44, .71, .86], "normal": [0., 0., 1.], "radius": .168},
                      {"name": "tilted", "center": [.6, .46, 1.03], "normal": [-.348, -.35, .869], "radius": .235}]}


def test_strike_point_moves_toward_the_playing_shoulder():
    surface = SCENE["surfaces"][0]
    point = strike_point(surface, "left", .08)
    shoulder = np.array([0., .0925, 1.0018])
    assert np.linalg.norm(point-np.array(surface["center"])) == pytest.approx(.08)
    assert np.linalg.norm(point-shoulder) < np.linalg.norm(np.array(surface["center"])-shoulder)


def test_strike_point_stays_on_a_tilted_surface():
    surface = SCENE["surfaces"][1]
    point = strike_point(surface, "left", .1)
    assert (point-np.array(surface["center"]))@np.array(surface["normal"]) == pytest.approx(0, abs=1e-9)


def test_offset_scene_keeps_the_stroke_inside_the_original_rim():
    shifted = offset_scene(SCENE, "hi_hat", "left", .08)
    surface = next(row for row in shifted["surfaces"] if row["name"] == "hi_hat")
    assert surface["radius"] == pytest.approx(.088)
    assert SCENE["surfaces"][0]["radius"] == .168 and surface["radius"] > RIM_MARGIN


def test_stroke_selection_prefers_the_shortest_period_near_the_requested_speed():
    rows = [{"speed_m_s": .43, "duration_s": .25}, {"speed_m_s": .55, "duration_s": .21},
            {"speed_m_s": .57, "duration_s": .19}, {"speed_m_s": .33, "duration_s": .15}]
    assert select_stroke(rows, .6)["duration_s"] == .19
    assert select_stroke(rows, .35)["duration_s"] == .15


def test_speed_bank_uses_one_ready_pose():
    first, second = [[0.]*14], [[.5]*14]
    rows = [{"speed_m_s": .6, "duration_s": .25, "reference": {"positions_rad": first}},
            {"speed_m_s": .4, "duration_s": .20, "reference": {"positions_rad": first}},
            {"speed_m_s": .35, "duration_s": .15, "reference": {"positions_rad": second}}]
    assert shared_ready(rows, .6) == rows[:2]


def test_tuned_stroke_keeps_arm_joints_and_lifts_only_the_wrist():
    ready = [.1, .2, .3, .4, .5, .6, -.3]
    document = tuned_stroke(ready, "left", impact=-1., lift=.4, depth=.02, downstroke=.07)
    positions = np.asarray(document["positions_rad"])
    assert np.allclose(positions[:, :6], ready[:6])
    assert positions[0, 6] == pytest.approx(-.6) and positions[3, 6] == pytest.approx(-1.02)
    assert np.allclose(positions[:, 7:], 0)
    assert document["active_joints"] == [6]
