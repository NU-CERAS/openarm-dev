import numpy as np
import pytest

from openarm_rl.envs.rhythmic import RhythmicDrummingEnv


@pytest.mark.parametrize("options", [
    {"height_offset_m": .051}, {"height_offset_m": float("nan")},
    {"contact_stiffness_scale": 0.}, {"contact_damping_scale": float("inf")},
    {"gain_scale": 1.6},
])
def test_invalid_physical_perturbations_fail_before_loading_models(options):
    with pytest.raises(ValueError):
        RhythmicDrummingEnv("missing", "missing", "missing", **options)


def test_direct_command_preserves_boundaries_and_rejects_invalid_commands(rhythm_options):
    with RhythmicDrummingEnv(**rhythm_options, score_only=True) as env:
        env.reset(seed=1)
        target = env.previous_target.copy()
        # Runtime motor commands already include gravity compensation and joint-stop clipping.
        motor = np.clip(env.servo.position_target(target, env.encoders()[0]),
                        env.controller.bounds[:, 0], env.controller.bounds[:, 1])
        velocity = np.linspace(-.01, .01, 14)
        env.step_command(motor, velocity, target_position=target)
        np.testing.assert_array_equal(env.trace[-1]["motor_target_rad"], motor)
        np.testing.assert_array_equal(env.trace[-1]["motor_velocity_rad_s"], velocity)
        assert env.trace[-1]["performance_time_s"] - env.trace[-1]["state_time_s"] == pytest.approx(env.period)
        before = env.data.qpos.copy()
        for q, dq in ((motor[:13], velocity), (motor, np.full(14, np.nan)),
                      (env.controller.bounds[:, 1]+.001, velocity),
                      (motor, env.controller.velocity+.001)):
            with pytest.raises(ValueError):
                env.step_command(q, dq)
            np.testing.assert_array_equal(env.data.qpos, before)


def test_normalized_actions_cannot_exploit_gravity_commands_outside_motor_limits(rhythm_options, monkeypatch):
    with RhythmicDrummingEnv(**rhythm_options, score_only=True) as env:
        env.reset(seed=1)
        upper = env.controller.bounds[:, 1].copy()
        monkeypatch.setattr(env.servo, 'position_target', lambda desired, measured: upper+.1)
        env.step(env.last_action)
        np.testing.assert_array_equal(env.trace[-1]['motor_target_rad'], upper)


def test_physical_perturbations_are_reproducible_and_restore_nominal(rhythm_options):
    with RhythmicDrummingEnv(**rhythm_options, height_offset_m=.001,
                             contact_stiffness_scale=1.5, contact_damping_scale=.8,
                             gain_scale=1.1, domain_range=.05) as env:
        first, info = env.reset(seed=19)
        parameters, solref = env.parameters.copy(), env.model.geom_solref.copy()
        q = env.data.qpos.copy()
        np.testing.assert_allclose(env.model.geom_pos[env.kit_geoms, 2],
                                   env.nominal_geom[env.kit_geoms, 2]+.001)
        for head in env.heads:
            np.testing.assert_allclose(solref[head], env.nominal_solref[head]*[1/.8, .8/np.sqrt(1.5)])
        assert info["kit_shift_m"][2] == .001
        assert info["contact_stiffness_scale"] == 1.5 and info["gain_scale"] == 1.1
        env.reset(seed=19)
        np.testing.assert_array_equal(env.parameters, parameters)
        np.testing.assert_array_equal(env.data.qpos, q)
        np.testing.assert_array_equal(env.model.geom_solref, solref)
        env.restore_nominal()
        np.testing.assert_array_equal(env.model.geom_pos, env.nominal_geom)
        np.testing.assert_array_equal(env.model.geom_solref, env.nominal_solref)


def test_contact_stiffness_changes_measured_contacts(rhythm_options):
    def run(env):
        env.reset(seed=1)
        while env.active:
            env.step(env.reference_action())
        return env.info()

    with RhythmicDrummingEnv(**rhythm_options, score_only=True) as nominal:
        before = run(nominal)
        nominal.restore_nominal()
        with RhythmicDrummingEnv(**rhythm_options, score_only=True,
                                 contact_stiffness_scale=.5, source=nominal) as softer:
            after = run(softer)
    assert before["actual_hits"] > 0 and after["actual_hits"] > 0
    assert abs(before["peak_force_N"]-after["peak_force_N"]) > .01
