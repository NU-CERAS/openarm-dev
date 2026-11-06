"""Small, inspectable supervised models for the controller comparison.

The dynamics model learns *state prediction error*, not a policy correction.
Neither model accepts randomized plant parameters or contact ground truth.
"""
from pathlib import Path

import numpy as np


class RidgeModel:
    """Standardized ridge regression, optionally with fixed nonlinear features."""

    def __init__(self, regularization=1., hidden=0, seed=1729):
        if regularization <= 0 or hidden < 0:
            raise ValueError("Positive regularization and nonnegative hidden size required")
        self.regularization, self.hidden, self.seed = regularization, hidden, seed

    def features(self, x):
        z = np.clip((np.asarray(x)-self.mean)/self.scale, -10., 10.)
        if self.hidden:
            z = np.concatenate([z, np.tanh(z @ self.projection + self.bias)], axis=-1)
        return np.concatenate([z, np.ones((*z.shape[:-1], 1))], axis=-1)

    def fit(self, x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 2:
            raise ValueError("At least two paired feature/target rows required")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Training rows must be finite")
        self.mean, self.scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-3)
        rng = np.random.default_rng(self.seed)
        self.projection = rng.normal(size=(x.shape[1], self.hidden))/np.sqrt(x.shape[1])
        self.bias = rng.uniform(-1., 1., self.hidden)
        design = self.features(x)
        penalty = np.eye(design.shape[1])*self.regularization
        penalty[-1, -1] = 0.
        self.weights = np.linalg.solve(design.T @ design + penalty, design.T @ y)
        return self

    def predict(self, x):
        return self.features(x) @ self.weights

    def save(self, path):
        np.savez_compressed(Path(path), mean=self.mean, scale=self.scale, weights=self.weights,
                            projection=self.projection, bias=self.bias, hidden=self.hidden,
                            regularization=self.regularization, seed=self.seed)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            model = cls(float(data["regularization"]), int(data["hidden"]), int(data["seed"]))
            for name in ("mean", "scale", "weights", "projection", "bias"):
                setattr(model, name, data[name].copy())
        return model


def dynamics_features(q, dq, motor, velocity):
    """56 measured/control features; no mass, friction, surface shift or gains."""
    return np.concatenate([q, dq, np.asarray(motor)-q, velocity])


def policy_features(q, dq, previous_velocity, previous_target, anchor, future):
    """Joint feedback, last commanded velocity and the next eight MIDI events."""
    return np.concatenate([np.asarray(q)-anchor, dq, previous_velocity, np.asarray(previous_target)-anchor, future])


class DynamicsResidual:
    """One-period q/dq error with explicit correction limits for unseen states."""

    def __init__(self, model, position_limit=.01, velocity_limit=.5):
        self.model = model
        self.limits = np.r_[np.full(14, position_limit), np.full(14, velocity_limit)]

    def correction(self, q, dq, motor, velocity):
        return np.clip(self.model.predict(dynamics_features(q, dq, motor, velocity)),
                       -self.limits, self.limits)
