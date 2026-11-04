import numpy as np
import pytest
import torch

from openarm_rl.envs.rhythmic import RhythmicDrummingEnv


pytest.importorskip("mujoco_warp")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="MuJoCo Warp environment requires CUDA")


@pytest.fixture
def batched(rhythm_options):  # noqa: F811
    from openarm_rl.envs.warp_rhythmic import WarpRhythmicVecEnv

    def build(worlds=2, **options):
        return WarpRhythmicVecEnv(worlds, rhythm_options["scene_directory"], rhythm_options["profiles"],
                                  midi_files=[rhythm_options["midi_file"]], **options)
    return build


def test_batched_episode_matches_native_contact_physics(rhythm_options, batched):  # noqa: F811
    env = batched()
    cpu = RhythmicDrummingEnv(**rhythm_options, score_only=True)
    expected, _ = cpu.reset(seed=0)
    observed = env.reset()
    for name in expected:
        assert np.abs(expected[name]-observed[name][0]).max() < 1e-5
    total, batched_total = 0., 0.
    while True:
        action = cpu.reference_action()
        expected, reward, done, _, info = cpu.step(action)
        observed, rewards, dones, infos = env.step(np.tile(action, (2, 1)))
        total += reward
        batched_total += rewards[0]
        assert bool(dones[0]) == done
        if done:
            break
        assert np.abs(expected["actor"]-observed["actor"][0]).max() < 1e-3
        assert rewards[0] == pytest.approx(rewards[1])
    assert batched_total == pytest.approx(total, abs=1e-3)
    assert infos[0]["f1"] == pytest.approx(info["f1"])
    assert infos[0]["is_success"] == info["is_success"]
    assert infos[0]["episode"]["l"] == cpu.steps


def test_kit_shift_moves_static_contact_geometry(batched):
    from openarm_rl.envs.warp_pickup import simulation_stream
    env = batched(worlds=2, position_range=.002)
    nominal = env.kit_xpos.clone()
    shift = torch.tensor([[.002, -.001, 0.], [0., 0., 0.]], device="cuda")
    simulation_stream(lambda self: self._reset_worlds(torch.arange(2, device="cuda"), shift=shift))(env)
    moved = env.geom_xpos[:, env.kit_geoms]-nominal
    assert torch.allclose(moved[0], shift[0].expand_as(moved[0]), atol=1e-6)
    assert torch.allclose(moved[1], torch.zeros_like(moved[1]), atol=1e-6)


def test_domain_randomization_gives_each_world_its_own_parameters(batched):
    env = batched(worlds=4, domain_range=.05, position_range=.002, seed=3)
    env.reset()
    assert len({round(value, 6) for value in env.parameters[:, 0].tolist()}) == 4
    assert torch.all(torch.abs(env.parameters-1) <= .05)
    assert torch.all(torch.abs(env.shift[:, :2]) <= .002)


def test_finished_world_reports_terminal_data_and_restarts(batched):
    env = batched(worlds=1)
    env.reset()
    hold = np.asarray(env.last_action.cpu())
    for _ in range(100000):
        observation, _, dones, infos = env.step(hold)
        if dones[0]:
            break
    assert {"f1", "is_success", "episode", "terminal_observation"} <= set(infos[0])
    assert env.state["steps"][0].item() == 0
    assert observation["actor"].shape == (1, 269)
