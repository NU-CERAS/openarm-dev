import json
import importlib.util
from pathlib import Path
import signal
import subprocess
import sys
import uuid

import mujoco
import numpy as np
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from openarm_rl.deployment.drum_actor import DrumActor
from openarm_rl.envs.camera_free_strike import CameraFreeStrikeEnv
from openarm_rl.models.asymmetric import AsymmetricStrikePolicy
from openarm_rl.skills.runtime import SimDrumTools
from openarm_rl.training.camera_free import save_bundle
from openarm_rl.paths import PROJECT_ROOT


@pytest.fixture(scope="module")
def native_bundle(native_strike_fixture, tmp_path_factory):
    root = tmp_path_factory.mktemp("drum-bundle")
    options = {"scene_directory": str(native_strike_fixture / "scene"), "profile_file": str(native_strike_fixture / "snare.json"),
               "target": "snare", "position_range": 0, "domain_range": 0, "history_frames": 8}
    raw = make_vec_env(CameraFreeStrikeEnv, n_envs=1, env_kwargs=options)
    try:
        env = VecNormalize(raw)
        policy = PPO(AsymmetricStrikePolicy, env, n_steps=128, batch_size=128, n_epochs=1, device="cpu", seed=2)
        policy.learn(512)
        save_bundle(policy, root / "snare", options)
    finally:
        raw.close()
    manifest = {"schema": "openarm.camera_free_tools.v1", "scene_directory": options["scene_directory"],
                "targets": {"snare": {"status": "trained_unqualified", "checkpoint": "snare", "arm": "left", "tool": "strike_snare"}}}
    (root / "tools.json").write_text(json.dumps(manifest))
    return root


def test_exported_actor_uses_encoders_and_rejects_stale_feedback(native_bundle):
    actor = DrumActor(native_bundle / "snare")
    data = mujoco.MjData(actor.model)
    data.qpos[actor.servo.q] = actor.reference.positions[0]
    mujoco.mj_forward(actor.model, data)
    q, dq = data.qpos[actor.servo.q], data.qvel[actor.servo.v]
    with pytest.raises(RuntimeError, match="stale"):
        actor.begin(q, dq, 0, 1)
    actor.begin(q, dq, 1, 1)
    command = actor.step(q, dq, 1, 1)
    assert command.shape == (14,) and np.isfinite(command).all()
    with pytest.raises(RuntimeError, match="deadline"):
        actor.step(q, dq, 1.1, 1.1)


def test_unqualified_tools_require_explicit_experimental_mode(native_bundle):
    tools = SimDrumTools(native_bundle / "tools.json")
    with pytest.raises(ValueError, match="unavailable"):
        tools.checkpoint("snare")
    tools = SimDrumTools(native_bundle / "tools.json", allow_experimental=True)
    assert tools.checkpoint("snare") == native_bundle / "snare"


def test_consecutive_native_tool_calls_preserve_robot_state_and_clock(native_bundle):
    tools = SimDrumTools(native_bundle / "tools.json", allow_experimental=True)
    first = tools.call("strike_snare")
    final = np.array(first["final_joints_rad"])
    before = tools.data.qpos[tools.controller.q].copy()
    np.testing.assert_array_equal(final, before)
    second = tools.call("strike_snare")
    assert second["start_time_s"] == first["end_time_s"]
    assert second["end_time_s"] > first["end_time_s"]
    assert first["encoder_verified"] and second["encoder_verified"]
    assert first["recovered"] and second["recovered"]


@pytest.mark.skipif(importlib.util.find_spec("rclpy") is None, reason="Requires ROS Jazzy in the learning environment")
def test_actual_ros_actor_stream_and_feedback_cycle(native_bundle, tmp_path):
    from openarm_rl.deployment.ros_drum import JointMapping, RosDrumTool

    namespace = f"/drum_actor_test_{uuid.uuid4().hex[:10]}"
    mapping = PROJECT_ROOT / "configs/deployment/drum_joint_mapping.example.json"
    actor = DrumActor(native_bundle / "snare")
    with pytest.raises(ValueError, match="reviewed"):
        JointMapping(json.loads(mapping.read_text()), actor.limits, physical=True)
    tool = None
    with (tmp_path / "backend.log").open("w") as log:
        process = subprocess.Popen([sys.executable, "-u", "-m", "openarm_sim.console.ros_backend", "--model", "opendrummer", "--namespace", namespace],
                                   stdout=log, stderr=subprocess.STDOUT)
        try:
            tool = RosDrumTool(namespace, mapping)
            before = tool.client.read().positions.copy()
            preview = tool.execute(native_bundle / "snare", execute=False)
            assert not preview["execute"]
            np.testing.assert_allclose(tool.client.read().positions, before, atol=.002)
            first = tool.execute(native_bundle / "snare", execute=True)
            assert first["encoder_verified"] and first["recovered"], first
            assert np.linalg.norm(first["final_joints_rad"]) > .2
            second = tool.execute(native_bundle / "snare", execute=True)
            assert second["encoder_verified"] and second["recovered"], second
            np.testing.assert_allclose(second["measured_joints_rad"], first["final_joints_rad"], atol=.03)
            for result in (first, second):
                assert result["command_count"] > 200
                assert result["max_control_period_s"] <= .05
                assert result["max_feedback_age_s"] <= actor.limits["feedback_timeout_s"]
        finally:
            if tool is not None:
                tool.close()
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
            assert process.wait(timeout=10) == 0, (tmp_path / "backend.log").read_text()
