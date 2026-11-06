import json
import sys
import time
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from openarm_rl.deployment.rhythm_actor import ReferenceRuntime
from openarm_rl.envs.rhythmic import RhythmicDrummingEnv
from openarm_rl.music.rehearse import judge, rehearse


def options(rhythm_options):
    return rhythm_options["scene_directory"], rhythm_options["profiles"], rhythm_options["midi_file"]


def test_reference_runtime_commands_the_validated_teacher(rhythm_options):
    runtime = ReferenceRuntime(*options(rhythm_options))
    with RhythmicDrummingEnv(**rhythm_options, score_only=True) as env:
        env.reset(seed=1)
        sampled_at, q, dq = env.sensor_cache
        runtime.begin(q, dq, sampled_at, env.elapsed)
        while True:
            expected = env.reference_action()
            sampled_at, q, dq = env.sensor_cache
            command = runtime.step(q, dq, sampled_at, env.elapsed)
            if command is None:
                break
            np.testing.assert_allclose(runtime.last_action, expected, atol=1e-5)
            previous_q, previous_dq = env.data.qpos[env.controller.q].copy(), env.data.qvel[env.controller.v].copy()
            _, _, done, _, info = env.step_command(*command, target_position=runtime.previous_target)
            np.testing.assert_array_equal(env.trace[-1]["motor_target_rad"], command[0])
            np.testing.assert_array_equal(env.trace[-1]["motor_velocity_rad_s"], command[1])
            np.testing.assert_array_equal(env.trace[-1]["previous_measured_joints_rad"], previous_q)
            np.testing.assert_array_equal(env.trace[-1]["previous_measured_velocity_rad_s"], previous_dq)
            np.testing.assert_array_equal(env.trace[-1]["measured_velocity_rad_s"], env.data.qvel[env.controller.v])
            if done:
                break
    assert info["correct_hits"] == 2 and not info["unsafe_contact"], info


def test_one_failed_required_run_blocks_the_performance():
    good = {"condition": "deployment", "required": True, "is_success": True, "completed": True, "failure": None}
    bad = {**good, "is_success": False}
    nominal = {**good, "condition": "nominal"}
    assert judge([nominal, good, good])
    assert not judge([nominal, good, bad])
    assert judge([nominal, good, {**bad, "required": False}])
    assert not judge([])
    assert not judge([nominal])
    assert not judge([good])
    assert not judge([nominal, {**good, "required": False}])
    assert not judge([nominal, {**good, "unsafe_contact": True}])


@pytest.mark.parametrize("argument,value", [("seeds", 0), ("seeds", -1), ("seeds", 1.5),
                                             ("workers", 0), ("workers", -1), ("workers", False)])
def test_rehearsal_rejects_empty_or_invalid_work_before_loading_models(argument, value):
    with pytest.raises(ValueError, match="positive integer"):
        rehearse("missing", "missing", "missing", **{argument: value})


def test_rehearsal_certifies_only_the_exact_score(rhythm_options, tmp_path):
    scene, profiles, midi = options(rhythm_options)
    report = rehearse(scene, profiles, midi, output=tmp_path/"rehearsal.json", seeds=1, sweep=False, workers=1)
    assert report["passed"], report["runs"]
    runtime = ReferenceRuntime.from_rehearsal(tmp_path/"rehearsal.json")
    assert report["runtime_limits"]["position_gains"] == runtime.limits["position_gains"]
    changed = json.loads((tmp_path/"rehearsal.json").read_text())
    changed["midi_sha256"] = "0"*64
    (tmp_path/"changed.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="MIDI differs"):
        ReferenceRuntime.from_rehearsal(tmp_path/"changed.json")
    (tmp_path/"failed.json").write_text(json.dumps({**report, "passed": False}))
    with pytest.raises(ValueError, match="did not pass"):
        ReferenceRuntime.from_rehearsal(tmp_path/"failed.json")
    nominal_only = {**report, "runs": [run for run in report["runs"] if run["condition"] == "nominal"]}
    (tmp_path/"nominal-only.json").write_text(json.dumps(nominal_only))
    with pytest.raises(ValueError, match="did not pass"):
        ReferenceRuntime.from_rehearsal(tmp_path/"nominal-only.json")


class FakeClient:
    """Encoders that follow the compiled path, so the real streaming loop runs without ROS."""
    def __init__(self, runtime):
        self.runtime, self.started = runtime, None

    def read(self, wait=True):
        elapsed = 0. if self.started is None else time.monotonic()-self.started
        q, dq = self.runtime.reference.sample(min(elapsed, self.runtime.reference.duration))
        return SimpleNamespace(q=q, dq=dq, received_at=time.monotonic())


class FakeMapping:
    def __init__(self, document, limits, physical):
        pass

    def measured(self, snapshot):
        return snapshot.q, snapshot.dq


@pytest.fixture
def ros_music(monkeypatch):
    for name, attributes in {"trajectory_msgs": {}, "trajectory_msgs.msg": {"JointTrajectory": object, "JointTrajectoryPoint": object},
                             "rclpy": {}, "rclpy.parameter_client": {"AsyncParameterClient": object},
                             "openarm_tools.ros_execution": {"RosExecutionClient": object, "arm_joint_names": None,
                                                             "duration": None, "trajectory_message": None}}.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    for name in ("openarm_rl.deployment.ros_music", "openarm_rl.deployment.ros_drum"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    import openarm_rl.deployment.ros_music as module
    yield module
    for name in ("openarm_rl.deployment.ros_music", "openarm_rl.deployment.ros_drum"):
        sys.modules.pop(name, None)


def test_ros_tool_streams_the_reference_runtime_to_completion(ros_music, rhythm_options, monkeypatch):
    import threading
    runtime = ReferenceRuntime(*options(rhythm_options))
    tool = ros_music.RosMusicTool.__new__(ros_music.RosMusicTool)
    tool.client = FakeClient(runtime)
    tool.reservation, tool.cancelled = threading.Lock(), threading.Event()
    tool.namespace, tool.physical = "/fake", True
    tool.mapping_document = {"robot_source_sha256": runtime.identity["robot"], "import_config_sha256": runtime.identity["import"],
                             "layout_sha256": runtime.identity.get("layout", "")}
    published = []
    tool.publish = lambda mapping, q, dq: published.append(np.array(q))
    monkeypatch.setattr(ros_music, "JointMapping", FakeMapping)
    monkeypatch.setattr(ros_music, "prepare_transfers", lambda *_: [])
    begin = runtime.begin
    def timed(*arguments):
        tool.client.started = time.monotonic()
        begin(*arguments)
    runtime.begin = timed
    result = tool.perform(runtime, execute=True)
    assert result["command_count"] >= int(runtime.reference.duration*90)
    assert len(published) >= result["command_count"] and np.isfinite(published).all()
    assert not runtime.running


def test_physical_execution_refuses_a_learned_actor(ros_music, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["ros_music", "--actor", str(tmp_path/"actor.pt"), "--scene-directory", str(tmp_path),
                                      "--profiles", str(tmp_path), "--midi", str(tmp_path/"a.mid"), "--mapping", str(tmp_path/"m.json"),
                                      "--physical", "--execute"])
    with pytest.raises(ValueError, match="rehearsal"):
        ros_music.main()


def test_stroke_overshoot_at_a_joint_limit_is_tolerated_but_bounded(rhythm_options):
    runtime = ReferenceRuntime(*options(rhythm_options))
    q = runtime.controller.bounds[:, 1].copy()
    runtime.validate(q+.005, np.zeros(14), 1., 1.)
    with pytest.raises(RuntimeError, match="exceed model limits"):
        runtime.validate(q+.02, np.zeros(14), 1., 1.)
