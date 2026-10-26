import numpy as np
import pytest

from openarm_sim.tools import SimulationTools


def test_tools_execute_camera_grounded_reach():
    with SimulationTools(width=320, height=240) as tools:
        observation = tools.call("observe", {})
        pixel = tools.camera.project(np.array([0.43, -0.24, tools.sim.table_height]), tools.frame)
        result = tools.call("reach_pixel", {
            "arm": "right", "u": pixel[0], "v": pixel[1], "frame_id": observation["frame_id"],
        })
        assert result["state"] == "succeeded", result
        assert result["error_m"] < 0.01
        assert tools.image_content()["image_url"]["url"].startswith("data:image/png;base64,")


def test_tools_reject_stale_image_selection():
    with SimulationTools(width=320, height=240) as tools:
        first = tools.call("observe", {})
        tools.call("observe", {})
        with pytest.raises(ValueError, match="frame_id"):
            tools.call("reach_pixel", {"arm": "right", "u": 150, "v": 120, "frame_id": first["frame_id"]})


def test_tools_validate_arguments():
    with SimulationTools(width=320, height=240) as tools:
        with pytest.raises(ValueError, match="未知"):
            tools.call("execute_code", {})
