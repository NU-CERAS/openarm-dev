from bisect import bisect_right
import json
import hashlib
from pathlib import Path

import mujoco
import numpy as np

from openarm_drummer.model import HoldController
from openarm_drummer.transfers import TransferTrajectory, clearance_checker, plan_transfer

from ..control.reference import JointReference, SpeedConditionedReference
from ..skills.registry import SKILLS
from ..planning.strikes import scene_identity


# Compact strokes end with joint motion still decaying; recovery is judged after this hold.
SETTLE_TAIL_S = .40
# Cross-hand strokes can park an arm in the other arm's path, so they are used only when needed.
CROSS_HAND_PENALTY_S = .25


def profile_hashes(directory: Path) -> dict:
    directory = Path(directory)
    names = [name for target in SKILLS for name in (f"{target}.json", f"{target}.left.json", f"{target}.right.json")]
    return {name: hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in names if (directory/name).is_file()}


class ScoreFeasibilityError(ValueError):
    def __init__(self, message, *, time_s=None, target=None, arm=None):
        super().__init__(message)
        self.time_s, self.target, self.arm = time_s, target, arm


def load_strokes(metadata, profiles, targets, speed=.6):
    """Calibrated strokes for each available hand; targets without a profile are simply absent."""
    identity = scene_identity(metadata["robot"], metadata["scene"])
    loaded, available = {}, {}
    for target in targets:
        available[target] = []
        for filename in (f"{target}.json", f"{target}.left.json", f"{target}.right.json"):
            path = Path(profiles) / filename
            if not path.is_file():
                continue
            profile = json.loads(path.read_text())
            if (profile["status"] != "ready" or profile["target"] != target or profile["arm"] not in ("left", "right")
                    or profile["scene_identity"] != identity):
                raise ValueError(f"Reference profile does not match the performance scene: {filename}")
            if "reference_bank" not in profile:
                raise ValueError(f"Calibrated timed references are required: {filename}")
            key = (target, profile["arm"])
            if key in loaded:
                raise ValueError(f"Duplicate hand-to-drum profile: {key}")
            loaded[key] = profile
            available[target].append(profile["arm"])
    references = {key: SpeedConditionedReference(p["reference_bank"]).at(speed) for key, p in loaded.items()}
    return loaded, references, available


class TransferPlanner:
    """Collision-checked arm transfers; repeated pose pairs reuse their planned trajectory."""
    def __init__(self, model, controller, profiles):
        limits = [profile.get("transfer_limits", {"velocity_rad_s": .5, "acceleration_rad_s2": 1.2}) for profile in profiles]
        if any(limit != limits[0] for limit in limits):
            raise ValueError("Performance profiles must share explicit transfer limits")
        self.model, self.controller = model, controller
        self.velocity, self.acceleration = limits[0]["velocity_rad_s"], limits[0]["acceleration_rad_s2"]
        if (not np.isfinite([self.velocity, self.acceleration]).all()
                or not 0 < self.velocity <= float(np.min(controller.velocity))*.8
                or not 0 < self.acceleration <= 120):
            raise ValueError("Performance transfer limits exceed configured command bounds")
        self.cache = {}

    @classmethod
    def for_profiles(cls, model, metadata, profiles):
        loaded, _, _ = load_strokes(metadata, profiles, list(SKILLS))
        return cls(model, HoldController(model, metadata["robot"]), list(loaded.values()))

    def plan(self, initial, ready, arm):
        key = arm, np.round(initial, 5).tobytes(), np.round(ready, 5).tobytes()
        if key not in self.cache:
            try:
                path = plan_transfer(self.model, self.controller, initial, ready, arm=arm)
                self.cache[key] = TransferTrajectory(path, np.full(14, self.velocity), np.full(14, self.acceleration),
                                                     clearance_checker(self.model, self.controller, initial, ready, arm))
            except ValueError as error:
                self.cache[key] = error
        if isinstance(self.cache[key], ValueError):
            raise self.cache[key]
        return self.cache[key]


class PerformanceReference:
    def __init__(self, model, metadata, score, profiles, speed=.6, planner=None):
        self.model, self.score = model, score
        self.surfaces = {surface["name"]: surface for surface in metadata["scene"]["surfaces"]}
        self.controller = HoldController(model, metadata["robot"])
        self.strokes, self.lanes = [], {"left": [], "right": []}
        self.profiles, references, available = load_strokes(metadata, profiles, {goal.target for goal in score.goals}, speed)
        for target, arms in available.items():
            if not arms:
                raise ValueError(f"No calibrated reachable hand for {target}")
        self.planner = planner or TransferPlanner(model, self.controller, list(self.profiles.values()))
        self.preroll = max(reference.impact_time for reference in references.values()) + .20
        self.assignment = self._assign(available, references)
        self.initial = np.zeros(14)
        for arm in self.lanes:
            goals = [g for g in score.goals if self.assignment[g.event_id] == arm]
            if goals:
                selected = slice(0, 7) if arm == "left" else slice(7, 14)
                self.initial[selected] = references[(goals[0].target, arm)].positions[0, selected]
        state = self.initial.copy()
        for arm in self.lanes:
            selected = slice(0, 7) if arm == "left" else slice(7, 14)
            previous_end = 0.
            for goal in [g for g in score.goals if self.assignment[g.event_id] == arm]:
                reference = references[(goal.target, arm)]
                start = goal.time_s + self.preroll - reference.impact_time
                if start < previous_end - 1e-8:
                    raise ScoreFeasibilityError(f"Infeasible {arm} score at {goal.time_s:.3f}s: prior stroke ends at score time {previous_end-self.preroll:.3f}s",
                                                time_s=goal.time_s, target=goal.target, arm=arm)
                ready = state.copy()
                ready[selected] = reference.positions[0, selected]
                if np.max(np.abs(ready-state)) > .001:
                    transfer = self._transfer(state, ready, arm, goal)
                    transfer_start = start - float(transfer.ends[-1])
                    if transfer_start < previous_end - 1e-8:
                        raise ScoreFeasibilityError(f"Insufficient transfer time for {arm}/{goal.target} at {goal.time_s:.3f}s",
                                                    time_s=goal.time_s, target=goal.target, arm=arm)
                    self.lanes[arm].append((transfer_start, start, transfer, False))
                end = start + reference.duration
                self.lanes[arm].append((start, end, reference, True))
                self.strokes.append({"goal": goal, "arm": arm, "start": start, "end": end, "reference": reference})
                state[selected] = reference.positions[-1, selected]
                previous_end = end
        self.index_lanes()
        self.duration = max(max(stroke["end"] for stroke in self.strokes), self.score.duration_s+self.preroll) + SETTLE_TAIL_S
        self._validate()

    def _assign(self, available, references):
        beam = [(0., {}, {"left": (None, 0.), "right": (None, 0.)})]
        costs = {}
        goals = sorted(self.score.goals, key=lambda g: (g.time_s, len(available[g.target])))
        for goal in goals:
            next_beam = []
            for cost, assignments, hands in beam:
                for arm in available[goal.target]:
                    reference = references[(goal.target, arm)]
                    previous, end = hands[arm]
                    start = goal.time_s+self.preroll-reference.impact_time
                    if start < end-1e-8:
                        continue
                    duration = 0.
                    if previous is not None and previous != goal.target:
                        key = (previous, goal.target, arm)
                        if key not in costs:
                            selected = slice(0, 7) if arm == "left" else slice(7, 14)
                            initial, ready = np.zeros(14), np.zeros(14)
                            initial[selected] = references[(previous, arm)].positions[-1, selected]
                            ready[selected] = reference.positions[0, selected]
                            motion = self._transfer(initial, ready, arm, goal)
                            costs[key] = float(motion.ends[-1])
                        duration = costs[key]
                        if start-duration < end-1e-8:
                            continue
                    updated = dict(hands)
                    updated[arm] = (goal.target, start+reference.duration)
                    next_beam.append((cost+duration+CROSS_HAND_PENALTY_S*(arm != SKILLS[goal.target].arm),
                                      {**assignments, goal.event_id: arm}, updated))
            if not next_beam:
                raise ScoreFeasibilityError(f"No feasible hand assignment at {goal.time_s:.3f}s for {goal.target}; inspect stroke/transfer durations and simultaneous events",
                                            time_s=goal.time_s, target=goal.target)
            beam = sorted(next_beam, key=lambda item: item[0])[:32]
        return beam[0][1]

    def _transfer(self, initial, ready, arm, goal):
        try:
            return self.planner.plan(initial, ready, arm)
        except ValueError as error:
            raise ScoreFeasibilityError(f"Transfer to {arm}/{goal.target} at {goal.time_s:.3f}s failed: {error}",
                                        time_s=goal.time_s, target=goal.target, arm=arm) from error

    def index_lanes(self):
        self.lane_starts = {arm: [row[0] for row in lane] for arm, lane in self.lanes.items()}

    def current(self, arm, time_s):
        index = bisect_right(self.lane_starts[arm], time_s)-1
        return self.lanes[arm][index] if index >= 0 else None

    def sample(self, time_s):
        q, dq = self.initial.copy(), np.zeros(14)
        for arm in self.lanes:
            selected = slice(0, 7) if arm == "left" else slice(7, 14)
            motion = self.current(arm, time_s)
            if motion is None:
                # Before an arm's first motion it holds the ready pose of that first motion.
                continue
            start, end, trajectory, _ = motion
            desired, velocity = trajectory.sample(min(time_s, end)-start)
            q[selected], dq[selected] = desired[selected], velocity[selected] if time_s < end else 0
        return q, dq

    def motor_target(self, time_s, measured, servo):
        q, _ = self.sample(time_s)
        motor = servo.position_target(q, measured)
        for arm in self.lanes:
            motion = self.current(arm, time_s)
            if motion is not None and motion[3] and time_s <= motion[1]:
                reference = motion[2]
                motor[reference.wrist] = reference.motor_target(time_s-motion[0], measured, servo)[reference.wrist]
        return motor

    def _validate(self):
        data = mujoco.MjData(self.model)
        c = self.controller
        for t in np.arange(0, self.duration, .01):
            q, dq = self.sample(t)
            if np.any(q < c.bounds[:, 0]) or np.any(q > c.bounds[:, 1]) or np.any(np.abs(dq) > c.velocity):
                raise ScoreFeasibilityError(f"Performance reference exceeds motor limits at {t:.3f}s", time_s=float(t-self.preroll))
            data.qpos[c.q] = q
            mujoco.mj_forward(self.model, data)
            for contact in data.contact:
                if contact.dist >= -1e-5:
                    continue
                names = {self.model.geom(g).name for g in (contact.geom1, contact.geom2)}
                intended = False
                for stroke in self.strokes:
                    arm, target = stroke["arm"], stroke["goal"].target
                    if not stroke["start"] <= t <= stroke["end"] or f"{target}_head_collision" not in names:
                        continue
                    tip = data.site_xpos[self.model.site(f"{arm}_stick_tip").id]
                    surface = self.surfaces[target]
                    height = float((tip-np.array(surface["center"]))@np.array(surface["normal"]))
                    intended |= height < .005 and any(name.startswith(arm+"_stick_") for name in names)
                if not intended:
                    raise ScoreFeasibilityError(f"Coupled performance path intersects {sorted(names)} at {t:.3f}s", time_s=float(t-self.preroll))

    def document(self):
        return {"schema": "openarm.performance_reference.v1", "score": self.score.document(), "preroll_s": self.preroll,
                "duration_s": self.duration, "event_assignments": self.assignment,
                "profiles": {f"{target}.{arm}": profile for (target, arm), profile in self.profiles.items()}}
