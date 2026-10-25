import numpy as np


REWARD_VERSION = "pickup_milestones_v1"


class PickupReward:
    def __init__(self, count: int, device: str | None = None):
        if device is None:
            self.xp = np
            self.options = {}
        else:
            import torch

            self.xp = torch
            self.options = {"device": device}
        self.count = count
        self.elapsed = self.zeros(count)
        self.previous_potential = self.zeros(count)
        self.awarded = self.zeros((count, 2))
        self.clear_time = self.zeros((count, 2))
        self.held = self.zeros(count)
        self.best_height = self.zeros(count)
        self.stagnant_time = self.zeros(count)
        self.lost_time = self.zeros(count)
        self.has_lifted = self.zeros(count)
        self.grasp_steps = self.zeros(count)
        self.term_totals = {name: self.zeros(count) for name in (
            "progress", "time", "milestone", "success", "failure", "action_change", "stagnation")}

    def zeros(self, shape):
        return self.xp.zeros(shape, dtype=self.xp.float32, **self.options)

    def potential(self, distance, grasped, lift):
        return (0.5 * self.xp.exp(-(distance / 0.05)**2) + 0.5 * grasped
                + 2.0 * grasped * self.xp.clip(lift / 0.10, 0, 1))

    def reset(self, indices, distance):
        for value in (self.elapsed, self.previous_potential, self.awarded, self.clear_time,
                      self.held, self.best_height, self.stagnant_time, self.lost_time,
                      self.has_lifted, self.grasp_steps, *self.term_totals.values()):
            value[indices] = 0
        self.previous_potential[indices] = 0.5 * self.xp.exp(-(distance / 0.05)**2)

    def features(self):
        return self.xp.concatenate((self.awarded, self.clear_time / 0.06, self.held[:, None] / 0.3,
                                    self.best_height[:, None] / 0.10,
                                    self.xp.clip(self.stagnant_time[:, None] / 0.5, 0, 1),
                                    self.lost_time[:, None] / 0.10, self.has_lifted[:, None],
                                    self.xp.clip(1 - self.elapsed[:, None] / 6, 0, 1)), axis=1)

    def step(self, distance, grasped, lift, supported, speed, unsafe, below_table, action, previous_action):
        xp = self.xp
        self.elapsed += 0.02
        airborne_grasp = grasped & ~supported
        self.has_lifted = xp.maximum(self.has_lifted, (airborne_grasp & (lift >= 0.02)) * 1.0)
        self.lost_time = xp.where((self.has_lifted > 0) & ~grasped, self.lost_time + 0.02, 0.0)
        failure = unsafe | below_table | (self.lost_time >= 0.10 - 1e-6)
        stable = airborne_grasp & (lift > 0.10) & (speed < 0.12) & ~failure
        self.held = xp.where(stable, self.held + 0.02, 0.0)
        success = self.held >= 0.3 - 1e-6
        timeout = (self.elapsed >= 6.0 - 1e-5) & ~success & ~failure
        done = success | failure | timeout
        next_potential = xp.where(done, 0.0, self.potential(distance, grasped, lift))
        terms = {"progress": 0.99 * next_potential - self.previous_potential,
                 "time": self.zeros(self.count) - 0.01, "milestone": self.zeros(self.count),
                 "success": 50.0 * success, "failure": -10.0 * failure - 5.0 * timeout,
                 "action_change": -0.002 * ((action - previous_action)**2).mean(axis=1),
                 "stagnation": self.zeros(self.count)}
        for index, (height, bonus) in enumerate(((0.02, 1.0), (0.05, 2.0))):
            qualified = airborne_grasp & (lift >= height) & ~failure
            self.clear_time[:, index] = xp.where(qualified, self.clear_time[:, index] + 0.02, 0.0)
            newly_reached = (self.clear_time[:, index] >= 0.06 - 1e-6) & (self.awarded[:, index] == 0)
            terms["milestone"] += bonus * newly_reached
            self.awarded[:, index] = xp.maximum(self.awarded[:, index], newly_reached * 1.0)
        progressing = lift > self.best_height + 1e-5
        self.best_height = xp.maximum(self.best_height, lift)
        self.stagnant_time = xp.where(progressing | ~grasped, 0.0, self.stagnant_time + 0.02)
        terms["stagnation"] = -0.02 * ((self.stagnant_time >= 0.5) & ~success)
        self.grasp_steps += grasped
        self.previous_potential = next_potential
        for name, value in terms.items():
            self.term_totals[name] += value
        return sum(terms.values()), success, failure, timeout, terms
