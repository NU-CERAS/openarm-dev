"""Worst-case link loads for the OpenArm via inverse dynamics.

For each sampled (qpos, qvel, qacc) it calls mj_inverse, which returns the joint
torques needed to produce that motion, and reads the force/torque sensor at each
joint, which is the wrench transmitted across the joint into the child link.
No time stepping, so the sweep is fast and needs no GPU.

Usage:
    python link_load_sweep.py [--n 200000] [--accel 10] [--rigid]
"""
import argparse
import itertools
from pathlib import Path

import mujoco
import numpy as np

MODEL = Path(__file__).resolve().parent.parent / "models" / "single_arm.xml"
ARM_JOINTS = [f"rev{i}" for i in range(1, 8)]

# Limits from openarm_description/.../v2.0/urdf/example/v2.urdf (joints 1-7).
# The MJCF ctrlrange values are placeholders, so they are not used here.
VEL_LIMIT = np.array([16.755, 16.755, 5.4454, 5.4454, 20.944, 20.944, 20.944])  # rad/s
EFFORT_LIMIT = np.array([40.0, 40.0, 27.0, 27.0, 7.0, 7.0, 7.0])  # N*m


def load_model(rigid):
    """Load the MJCF and add a force/torque sensor pair at each joint frame."""
    spec = mujoco.MjSpec.from_file(str(MODEL))
    for name in ARM_JOINTS:
        body = spec.joint(name).parent  # joint frame == child body frame
        site = body.add_site(name=f"{name}_site")
        spec.add_sensor(name=f"{name}_force", type=mujoco.mjtSensor.mjSENS_FORCE,
                        objtype=mujoco.mjtObj.mjOBJ_SITE, objname=site.name)
        spec.add_sensor(name=f"{name}_torque", type=mujoco.mjtSensor.mjSENS_TORQUE,
                        objtype=mujoco.mjtObj.mjOBJ_SITE, objname=site.name)
    model = spec.compile()
    # We prescribe the motion, so limit/contact constraints would only add spurious
    # forces (e.g. at range corners). Keep the gripper equality constraint.
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_LIMIT | mujoco.mjtDisableBit.mjDSBL_CONTACT
    if rigid:  # pure rigid-body loads: drop rotor inertia and Coulomb friction
        model.dof_armature[:] = 0
        model.dof_frictionloss[:] = 0
    return model


def sample_states(model, n, accel, rng):
    ids = [model.joint(j).id for j in ARM_JOINTS]
    lo, hi = model.jnt_range[ids].T
    # Corners of the joint-range box (2^7) plus random interior poses.
    corners = np.array(list(itertools.product(*zip(lo, hi))))
    q = np.vstack([corners, rng.uniform(lo, hi, (n, len(ids)))])
    # Joints at +/- max velocity and +/- max acceleration, randomly signed.
    qd = rng.choice([-1.0, 0.0, 1.0], q.shape) * VEL_LIMIT
    qdd = rng.choice([-1.0, 0.0, 1.0], q.shape) * accel
    return q, qd, qdd


def sweep(model, q, qd, qdd):
    data = mujoco.MjData(model)
    jq = [model.jnt_qposadr[model.joint(j).id] for j in ARM_JOINTS]
    jv = [model.jnt_dofadr[model.joint(j).id] for j in ARM_JOINTS]
    sens = {j: (model.sensor(f"{j}_force").adr[0], model.sensor(f"{j}_torque").adr[0])
            for j in ARM_JOINTS}
    tau = np.empty_like(q)
    force = np.empty((len(q), len(ARM_JOINTS)))
    torque = np.empty_like(force)
    for k in range(len(q)):
        data.qpos[jq] = q[k]  # gripper stays at its default (closed) position
        data.qvel[jv] = qd[k]
        data.qacc[jv] = qdd[k]
        mujoco.mj_inverse(model, data)
        tau[k] = data.qfrc_inverse[jv]
        for i, j in enumerate(ARM_JOINTS):
            f, t = sens[j]
            force[k, i] = np.linalg.norm(data.sensordata[f:f + 3])
            torque[k, i] = np.linalg.norm(data.sensordata[t:t + 3])
    return tau, force, torque

def show_worst(model, q_worst):
    """Open the MuJoCo viewer holding the worst-case pose."""
    import time
    import mujoco.viewer

    data = mujoco.MjData(model)
    jq = [model.jnt_qposadr[model.joint(j).id] for j in ARM_JOINTS]
    data.qpos[jq] = q_worst
    mujoco.mj_forward(model, data)  # pose only, no stepping
    with mujoco.viewer.launch_passive(model, data) as v:
        while v.is_running():
            v.sync()
            time.sleep(0.02)


def plot_loads(tau, out=None):
    import matplotlib.pyplot as plt

    x = np.arange(len(ARM_JOINTS))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, np.abs(tau).max(axis=0), label="max |tau| from sweep")
    ax.step(np.r_[x - 0.5, x[-1] + 0.5], np.r_[EFFORT_LIMIT,
EFFORT_LIMIT[-1]],
            where="post", color="r", label="effort limit")
    ax.set_xticks(x, ARM_JOINTS)
    ax.set_ylabel("N*m")
    ax.legend()
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150)
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--accel", type=float, default=10.0, help="peak |qacc| per joint, rad/s^2")
    ap.add_argument("--rigid", action="store_true", help="zero armature and frictionloss")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", nargs="?", const="", metavar="PNG", help="bar chart; optional PNG path")
    ap.add_argument("--view", action="store_true", help="show worst-case pose in the viewer")
    args = ap.parse_args()

    model = load_model(args.rigid)
    q, qd, qdd = sample_states(model, args.n, args.accel, np.random.default_rng(args.seed))
    tau, force, torque = sweep(model, q, qd, qdd)

    print(f"{len(q)} states, accel={args.accel} rad/s^2, rigid={args.rigid}")
    print(f"{'joint':6} {'max|tau| Nm':>12} {'limit Nm':>9} {'max|F| N':>10} {'max|M| Nm':>10}")
    for i, j in enumerate(ARM_JOINTS):
        print(f"{j:6} {np.abs(tau[:, i]).max():12.2f} {EFFORT_LIMIT[i]:9.1f} "
              f"{force[:, i].max():10.2f} {torque[:, i].max():10.2f}")
    worst = np.abs(tau / EFFORT_LIMIT).max(axis=1).argmax()
    print("\nworst state vs. effort limits:")
    print(" q  =", np.round(q[worst], 3))
    print(" qd =", np.round(qd[worst], 3))
    print(" tau=", np.round(tau[worst], 2))


    if args.plot is not None:
        plot_loads(tau, args.plot or None)
    if args.view:
        show_worst(model, q[worst])



if __name__ == "__main__":
    main()
