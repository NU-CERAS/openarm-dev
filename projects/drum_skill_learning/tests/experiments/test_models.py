import argparse
from types import SimpleNamespace

import numpy as np
import pytest

from openarm_rl.experiments.compare import aggregate, seeds, validate_split
from openarm_rl.experiments.controllers import BCController, MPCController, NominalPhysics, Command
from openarm_rl.experiments.models import DynamicsResidual, RidgeModel, dynamics_features


def test_dynamics_residual_learns_state_prediction_error_and_round_trips(tmp_path):
    rng = np.random.default_rng(10)
    states = rng.normal(size=(300, 56))
    coefficients = rng.normal(scale=.0001, size=(56, 28))
    errors = states @ coefficients + np.linspace(-.001, .001, 28)
    model = RidgeModel(.001).fit(states[:200], errors[:200])
    assert np.mean((model.predict(states[200:])-errors[200:])**2) < 1e-12
    model.save(tmp_path/"residual.npz")
    loaded = RidgeModel.load(tmp_path/"residual.npz")
    np.testing.assert_array_equal(loaded.predict(states[200:]), model.predict(states[200:]))
    # A state residual is an additive next-state error, not a motor action.
    zero = np.zeros(14)
    assert DynamicsResidual(loaded).correction(zero, zero, zero, zero).shape == (28,)
    np.testing.assert_array_equal(dynamics_features(zero, zero, np.ones(14), zero)[28:42], 1.)


def test_unseen_dynamics_corrections_are_bounded():
    model = SimpleNamespace(predict=lambda _: np.full(28, 100.))
    correction = DynamicsResidual(model).correction(*(np.zeros(14) for _ in range(4)))
    np.testing.assert_array_equal(correction[:14], .01)
    np.testing.assert_array_equal(correction[14:], .5)


def test_learning_split_rejects_seed_leakage_and_missing_validation():
    validate_split([10, 11, 12], [0, 1, 2])
    with pytest.raises(ValueError, match="disjoint"):
        validate_split([10, 11, 12], [12])
    with pytest.raises(ValueError, match="three"):
        validate_split([10, 11], [0])
    for value in ("", "1,1", "-1", "not-a-seed"):
        with pytest.raises(argparse.ArgumentTypeError):
            seeds(value)


def test_behavior_clone_never_samples_reference_at_inference():
    class Reference:
        initial = np.zeros(14)
        preroll = .2

        def sample(self, *_):
            raise AssertionError("Behavior clone must not read the reference trajectory")

    c = SimpleNamespace(velocity=np.ones(14), bounds=np.tile([-1., 1.], (14, 1)))
    physics = SimpleNamespace(controller=c, period=.01,
                              servo=SimpleNamespace(position_target=lambda target, _: target))
    score = SimpleNamespace(future=lambda _: np.zeros(80))
    policy = SimpleNamespace(predict=lambda features: np.full(14, .2))
    controller = BCController(Reference(), physics, score, policy)
    command = controller.command(np.zeros(14), np.zeros(14), .1)
    np.testing.assert_allclose(command.target, .002)
    assert controller.features(np.zeros(14), np.zeros(14), .1).shape == (136,)
    np.testing.assert_array_equal(controller.features(np.zeros(14), np.zeros(14), .1)[42:56], command.target)


@pytest.mark.parametrize("contacts,forces,expected_peak,expected_unsafe", [
    ([(0, 2, 0.), (1, 2, 0.)], [18., 18.], 36., True),
    ([(0, 2, 0.), (1, 2, 0.)], [18., 10.], 28., False),
    ([(3, 4, 0.)], [.11], 0., True),
    ([(3, 4, 0.)], [.1], 0., False),
    ([(0, 2, -.004)], [1.], 1., True),
])
def test_prediction_contact_stops_match_summed_force_and_unexpected_contact(
        monkeypatch, contacts, forces, expected_peak, expected_unsafe):
    physics = object.__new__(NominalPhysics)
    physics.model = object()
    physics.data = SimpleNamespace(contact=[SimpleNamespace(geom1=a, geom2=b, dist=d) for a, b, d in contacts])
    physics.tip_arms, physics.head_targets = np.array([0, 0, -1, -1, -1]), np.array([-1, -1, 0, -1, -1])
    physics.safety = {"force_stop_N": 30., "penetration_stop_m": .003}

    def contact_force(model, data, index, wrench):
        wrench[:] = 0.
        wrench[0] = forces[index]

    monkeypatch.setattr("openarm_rl.experiments.controllers.mujoco.mj_contactForce", contact_force)
    assert physics.contact_safety() == (expected_peak, expected_unsafe)


def test_mpc_rejects_candidates_that_predict_unsafe_contact():
    zero = np.zeros(14)
    c = SimpleNamespace(velocity=np.ones(14), bounds=np.tile([-1., 1.], (14, 1)))
    physics = SimpleNamespace(controller=c, period=.01, set_state=lambda *_: None,
                              advance=lambda *_: (zero, zero, 36., True),
                              servo=SimpleNamespace(position_target=lambda target, _: target))
    reference = SimpleNamespace(initial=zero, strokes=[])
    controller = MPCController(reference, physics, horizon=2, candidates=3)
    controller.reference_command = lambda *_: Command(zero, zero, zero)
    with pytest.raises(RuntimeError, match="Every MPC candidate was unsafe"):
        controller.command(zero, zero, 0.)
    assert controller.last_unsafe_candidates == 3


def test_residual_corrected_terminal_state_is_checked_for_unsafe_contact(monkeypatch):
    physics = object.__new__(NominalPhysics)
    indices, zero, one = np.arange(14), np.zeros(14), np.ones(14)
    physics.controller = SimpleNamespace(q=indices, v=indices, a=indices, kp=one, kd=one, effort=one)
    physics.model = object()
    physics.data = SimpleNamespace(qpos=zero.copy(), qvel=zero.copy(), ctrl=zero.copy())
    physics.substeps, physics.forward_evaluations = 1, 0
    inspected = []

    def contact_safety():
        inspected.append(physics.data.qpos.copy())
        corrected = bool(np.any(physics.data.qpos))
        return (1. if corrected else 0.), corrected

    physics.contact_safety = contact_safety
    monkeypatch.setattr("openarm_rl.experiments.controllers.mujoco.mj_step", lambda *_: None)
    monkeypatch.setattr("openarm_rl.experiments.controllers.mujoco.mj_forward", lambda *_: None)
    residual = SimpleNamespace(correction=lambda *_: np.r_[np.full(14, .001), zero])
    _, _, peak, unsafe = physics.advance(Command(zero, zero, zero), residual)
    assert unsafe and peak == 1. and len(inspected) == 2
    np.testing.assert_array_equal(inspected[0], zero)
    np.testing.assert_array_equal(inspected[1], .001)


def test_aggregation_keeps_missing_intensity_and_timing_as_null():
    row = {"variant": "reference", "scenario": "nominal", "completed": False,
           "controller_failure": None, "controller_p50_ms": 1., "controller_p95_ms": 2.,
           "deadline_miss_fraction": 0., "metrics": {"f1": 0., "unsafe_contact": False,
           "timing_p95_s": None, "velocity_mae": None, "peak_force_N": 0., "missed_hits": 2, "extra_hits": 0}}
    summary = aggregate([row])[0]
    assert summary["velocity_mae_mean"] is None and summary["timing_p95_s_mean"] is None
    assert summary["completed_fraction"] == 0. and summary["missed_hits_mean"] == 2.
