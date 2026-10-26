from gymnasium.utils.env_checker import check_env
import numpy as np

from openarm_sim.pickup import StickPickupEnv


def test_pickup_uses_a_free_stick_and_gripper_action():
    with StickPickupEnv() as env:
        observation, info = env.reset(seed=0)
        assert env.action_space.shape == (8,)
        assert env.observation_space.contains(observation)
        assert env.sim.model.joint("stick_free").type == 0
        assert env.sim.model.body("stick").parentid == 0
        assert not info["is_success"]
        assert not info["grasped"]


def test_physical_contact_can_lift_and_hold_the_stick():
    with StickPickupEnv() as env:
        env.reset(seed=0)
        total = 0.0
        milestones = 0.0
        successes = 0.0
        for action in env.reference_pickup_actions():
            _, _, terminated, truncated, info = env.step(action)
            total += sum(info["reward_terms"].values())
            milestones += info["reward_terms"]["milestone"]
            successes += info["reward_terms"]["success"]
            if terminated or truncated:
                break
        assert info["is_success"], info
        assert info["grasped"]
        assert info["lift_height_m"] > 0.10
        assert info["gripper_contacts"] == 2
        assert milestones == 3.0
        assert successes == 50.0
        assert total > 0
        assert not env.stick_supported()


def test_pickup_follows_gymnasium_api():
    with StickPickupEnv() as env:
        check_env(env, skip_render_check=True)


def test_repeated_physical_lifting_does_not_repeat_milestone_rewards():
    with StickPickupEnv() as env:
        env.reset(seed=0)
        for index, action in enumerate(env.reference_pickup_actions()):
            _, _, terminated, _, _ = env.step(action)
            assert not terminated
            if index == 124:
                break
        assert env.grasp_contacts() == 2
        bonuses = []
        heights = []
        for height in (0.065, 0.005, 0.065):
            target = env.stick_position()
            target[2] = env.sim.table_height + 0.012 + height
            solution = env.sim._solve("right", target, env.tool_orientation)
            assert solution is not None
            start = env.sim._desired[env.sim.actuator_ids["right"]].copy()
            goal = solution[env.sim.qpos_ids["right"]]
            bonus = 0.0
            for step in range(40):
                fraction = min((step + 1) / 35, 1)
                blend = 10 * fraction**3 - 15 * fraction**4 + 6 * fraction**5
                desired = start + blend * (goal - start)
                action = np.zeros(8, dtype=np.float32)
                action[:7] = np.clip((desired - env.sim._desired[env.sim.actuator_ids["right"]]) / 0.025, -1, 1)
                action[7] = -1
                _, _, terminated, _, info = env.step(action)
                assert not terminated, info
                bonus += info["reward_terms"]["milestone"]
            bonuses.append(bonus)
            heights.append(info["lift_height_m"])
        assert heights[0] > 0.05 and heights[1] < 0.02 and heights[2] > 0.05
        assert bonuses == [3.0, 0.0, 0.0]
