import mujoco
import numpy as np

from openarm_drummer.model import HoldController
from openarm_drummer.scene import solve_approach


def plan_wrist_strike(model, robot, scene, cfg, seed=None, tip_direction=None, captured_ready=None):
    arm, target = cfg["arm"], cfg["target"]
    if arm not in ("left", "right") or target == "kick":
        raise ValueError("Choose a stick arm and a playable drum surface")
    c = HoldController(model, robot)
    wrist = 6 if arm == "left" else 13
    surface = next(s for s in scene["surfaces"] if s["name"] == target)
    center, normal = np.array(surface["center"]), np.array(surface["normal"])
    impact_height = float(cfg.get("impact_height_m", 0.0))
    impact_position = center + impact_height * normal
    sid = model.site(f"{arm}_stick_tip").id
    tip_geom = model.geom(f"{arm}_stick_tip_collision").id
    head = model.geom(f"{target}_head_collision").id
    probe = mujoco.MjData(model)
    selected = None

    def geometry(pose):
        q7 = pose[wrist]

        def height(angle):
            probe.qpos[c.q] = pose
            probe.qpos[c.q[wrist]] = angle
            mujoco.mj_forward(model, probe)
            return float((probe.site_xpos[sid] - center) @ normal)

        height(q7)
        if np.linalg.norm(probe.site_xpos[sid] - impact_position) > 0.005:
            return None
        gradient = (height(q7 + .0001) - height(q7 - .0001)) / .0002
        if abs(gradient) < .02:
            return None
        result = {"impact_pose": pose.copy()}
        for name, wanted in (("ready", cfg["ready_height_m"]), ("approach", cfg["approach_height_m"]),
                             ("bottom", -cfg["command_depth_m"])):
            direction = np.sign(wanted / gradient)
            previous, previous_error = q7, height(q7) - wanted
            found = None
            for angle in np.linspace(q7, np.clip(q7 + direction * 1.5, *c.bounds[wrist]), 301)[1:]:
                error = height(angle) - wanted
                if previous_error * error <= 0:
                    low, high = previous, angle
                    for _ in range(40):
                        middle = (low + high) / 2
                        if (height(low) - wanted) * (height(middle) - wanted) <= 0:
                            high = middle
                        else:
                            low = middle
                    found = pose.copy()
                    found[wrist] = (low + high) / 2
                    break
                previous, previous_error = angle, error
            if found is None:
                return None
            result[name] = found
        arc = []
        for angle in np.linspace(result["ready"][wrist], result["bottom"][wrist], 101):
            h = height(angle)
            radial = float(np.linalg.norm(probe.site_xpos[sid] - center - h * normal))
            if radial > surface["radius"] - .01:
                return None
            for contact in probe.contact:
                virtual = h < 0 and head in (contact.geom1, contact.geom2) and any(
                    model.geom(g).name.startswith(arm + "_stick_") for g in (contact.geom1, contact.geom2))
                if contact.dist < -1e-5 and not virtual and {contact.geom1, contact.geom2} != {tip_geom, head}:
                    return None
            arc.append({"joint7_rad": float(angle), "height_m": h, "radial_offset_m": radial})
        result["arc"] = arc
        return result

    def accept(pose):
        nonlocal selected
        candidate = geometry(pose)
        if candidate is None:
            return False
        selected = candidate
        return True

    if captured_ready is not None:
        ready = np.asarray(captured_ready, dtype=float)
        if ready.shape != (14,) or not np.isfinite(ready).all() or np.any(ready < c.bounds[:, 0]) or np.any(ready > c.bounds[:, 1]):
            raise ValueError("Captured ready pose exceeds model joint limits")
        probe.qpos[c.q] = ready
        mujoco.mj_forward(model, probe)
        clearance = float((probe.site_xpos[sid] - center) @ normal)
        if clearance <= .02 or any(contact.dist < -1e-5 for contact in probe.contact):
            return None
        cfg["ready_height_m"] = clearance
        cfg["approach_height_m"] = min(cfg["approach_height_m"], clearance * .6)
        angles = np.linspace(c.bounds[wrist, 0] + .03, c.bounds[wrist, 1] - .03, 601)
        heights = []
        for angle in angles:
            probe.qpos[c.q] = ready
            probe.qpos[c.q[wrist]] = angle
            mujoco.mj_forward(model, probe)
            heights.append(float((probe.site_xpos[sid] - center) @ normal))
        brackets = [(a, b) for a, b, ha, hb in zip(angles[:-1], angles[1:], heights[:-1], heights[1:]) if ha * hb <= 0]
        for lower, upper in sorted(brackets, key=lambda pair: abs(sum(pair) / 2 - ready[wrist])):
            for _ in range(40):
                middle = (lower + upper) / 2
                values = []
                for angle in (lower, middle):
                    probe.qpos[c.q] = ready
                    probe.qpos[c.q[wrist]] = angle
                    mujoco.mj_forward(model, probe)
                    values.append(float((probe.site_xpos[sid] - center) @ normal))
                if values[0] * values[1] <= 0:
                    upper = middle
                else:
                    lower = middle
            pose = ready.copy()
            pose[wrist] = (lower + upper) / 2
            candidate = geometry(pose)
            if candidate is not None and np.max(np.abs(candidate["ready"] - ready)) < .005:
                candidate["ready"] = candidate["staging"] = ready.copy()
                probe.qpos[c.q] = pose
                mujoco.mj_forward(model, probe)
                candidate["tip_direction"] = probe.site_xmat[sid].reshape(3, 3)[:, 2].tolist()
                return candidate
        return None

    impact = solve_approach(model, robot, scene, arm, target, height=impact_height,
                            wrist_angle=cfg["impact_joint7_rad"], joint_margin=.03,
                            seed=seed, tip_direction=tip_direction, pose_validator=accept)
    if not impact["reachable"]:
        return None
    mode = cfg.get("staging_mode", "vertical")
    if mode == "wrist_ready":
        selected["staging"] = selected["ready"].copy()
    elif mode == "vertical":
        staging = solve_approach(model, robot, scene, arm, target, height=cfg["ready_height_m"],
                                 seed=selected["ready"], tip_direction=tip_direction)
        if not staging["reachable"]:
            return None
        selected["staging"] = np.array(staging["joint_positions"])
    else:
        raise ValueError("Unknown staging mode")
    probe.qpos[c.q] = selected["staging"]
    mujoco.mj_forward(model, probe)
    if any(contact.dist < -1e-5 for contact in probe.contact):
        return None
    selected["tip_direction"] = impact["tip_direction"]
    return selected
