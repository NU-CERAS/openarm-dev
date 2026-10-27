import sys
import os

import mujoco
import numpy as np
import pytest

from openarm_sim.simulation import ReachSimulation


@pytest.mark.parametrize("with_obstacle", [False, True])
def test_ompl_plans_and_executes_actual_collision_free_motion(with_obstacle):
    obstacles = ({"name": "reach_obstacle", "position": [0.4, -0.24, 0.4], "half_size": [0.012, 0.012, 0.08]},) if with_obstacle else ()
    with ReachSimulation(planner="ompl", obstacles=obstacles) as sim:
        target = np.array([0.52, -0.35, 0.44] if with_obstacle else [0.43, -0.24, 0.44])
        if with_obstacle:
            goal = sim._solve("right", target)
            assert goal is not None
            data = mujoco.MjData(sim.model)
            collisions = []
            for fraction in np.linspace(0, 1, 80):
                data.qpos[:] = sim.data.qpos + fraction * (goal - sim.data.qpos)
                mujoco.mj_forward(sim.model, data)
                collisions.append(sim._collision(data))
            assert not collisions[0] and not collisions[-1]
            assert any(collisions[1:-1])
        result = sim.start_reach("right", target)
        assert result.accepted, result.reason
        assert sim.trajectory.source == "ompl"
        probe = mujoco.MjData(sim.model)
        probe.qpos[:] = sim.data.qpos
        for elapsed in np.arange(0, sim.trajectory.duration + 0.01, 0.01):
            probe.qpos[sim.qpos_ids["right"]] = sim.trajectory.sample(elapsed)
            mujoco.mj_forward(sim.model, probe)
            assert not sim._collision(probe)
        for _ in range(30000):
            sim.step()
            if sim.status.state != "running":
                break
        assert sim.status.state == "succeeded", sim.status
        assert np.linalg.norm(sim.tcp_position("right") - target) < 0.01


def test_tabletop_geometry_and_click_target_use_robot_mount_frame():
    with ReachSimulation(planner="ompl", tabletop_mount=True) as sim:
        assert sim.table_height == 0
        table = sim.model.geom("worktable").id
        assert sim.model.geom_pos[table, 2] + sim.model.geom_size[table, 2] == 0
        assert sim.model.geom_pos[sim.model.geom("floor").id, 2] == -0.75
        result = sim.start_reach("right", [0.43, -0.24, 0.14])
        assert "outside the demo workspace" not in result.reason


def test_cpu_planner_does_not_import_cuda_packages():
    before = set(sys.modules)
    with ReachSimulation(planner="ompl") as sim:
        assert sim.start_reach("right", [0.43, -0.24, 0.44]).accepted
    imported = set(sys.modules) - before
    assert not any(name.split(".")[0] in ("torch", "curobo", "warp", "mujoco_warp") for name in imported)


@pytest.mark.skipif(os.environ.get("LIBGL_ALWAYS_SOFTWARE") != "1", reason="Requires explicit Mesa software rendering")
def test_software_rendering_uses_cpu_mesa():
    from OpenGL.GL import GL_RENDERER, glGetString

    context = mujoco.GLContext(64, 64)
    try:
        context.make_current()
        renderer = glGetString(GL_RENDERER).decode().lower()
        assert "llvmpipe" in renderer or "softpipe" in renderer or "swrast" in renderer
    finally:
        context.free()
