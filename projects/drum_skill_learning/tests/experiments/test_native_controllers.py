import mujoco
import numpy as np

from openarm_rl.envs.rhythmic import RhythmicDrummingEnv
from openarm_rl.experiments.compare import measured_state
from openarm_rl.experiments.controllers import IKController, MPCController, NominalPhysics, ReferenceController


def test_native_predictor_matches_one_period_and_does_not_copy_plant_parameters(rhythm_options):
    with RhythmicDrummingEnv(**rhythm_options, score_only=True) as env:
        env.reset(seed=0)
        physics = NominalPhysics(rhythm_options["scene_directory"], env.scene)
        teacher = ReferenceController(env.reference, physics)
        for _ in range(4):
            q, dq = measured_state(env)
            command = teacher.command(q, dq, env.elapsed)
            predicted_q, predicted_dq, _, unsafe = physics.predict(q, dq, command)
            assert not unsafe
            env.step_command(command.motor, command.velocity, target_position=command.target)
            actual_q, actual_dq = measured_state(env)
            np.testing.assert_allclose(predicted_q, actual_q, atol=1e-10)
            np.testing.assert_allclose(predicted_dq, actual_dq, atol=1e-9)
        nominal = physics.model.body_mass.copy()
        env.model.body_mass[:] *= 1.2
        np.testing.assert_array_equal(physics.model.body_mass, nominal)


def test_mpc_runs_real_rollouts_and_ik_reduces_cartesian_error(rhythm_options):
    with RhythmicDrummingEnv(**rhythm_options, score_only=True) as env:
        env.reset(seed=0)
        physics = NominalPhysics(rhythm_options["scene_directory"], env.scene)
        q, dq = measured_state(env)
        mpc = MPCController(env.reference, physics, horizon=2, candidates=3)
        command = mpc.command(q, dq, env.elapsed)
        assert physics.forward_evaluations == 6
        assert np.isfinite(command.motor).all() and np.isfinite(mpc.last_cost)
        perturbed = q.copy()
        perturbed[2] += .03
        data = mujoco.MjData(physics.model)
        data.qpos[physics.controller.q] = q
        mujoco.mj_forward(physics.model, data)
        desired = data.site_xpos[physics.tip_sites].copy()
        data.qpos[physics.controller.q] = perturbed
        mujoco.mj_forward(physics.model, data)
        before = np.linalg.norm(desired-data.site_xpos[physics.tip_sites])
        ik = IKController(env.reference, physics)
        solved, _ = ik.solve(perturbed, 0.)
        data.qpos[physics.controller.q] = solved
        mujoco.mj_forward(physics.model, data)
        after = np.linalg.norm(desired-data.site_xpos[physics.tip_sites])
        assert after < before*.1
