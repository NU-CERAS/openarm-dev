from copy import copy

import mujoco
import numpy as np
from ompl import base as ob
from ompl import geometric as og

from ..execution import JointTrajectory
from . import PlanningResult


class OmplPlanner:
    def __init__(self, simulation, timeout=5.0):
        if timeout <= 0 or not np.isfinite(timeout):
            raise ValueError("Planning timeout must be positive and finite")
        self.sim = simulation
        self.timeout = timeout

    def plan(self, arm, target, orientation=None):
        if self.sim.model.nq != 18:
            raise ValueError("Free-object transport requires an attached-object collision model")
        goal_q = self.sim._solve(arm, target, orientation)
        if goal_q is None:
            return PlanningResult(None, "No feasible IK solution found for the requested target")
        model = copy(self.sim.model)
        model.geom_margin[:] = np.maximum(model.geom_margin, 0.004)
        indices = self.sim.qpos_ids[arm]
        limits = model.jnt_range[self.sim.joint_ids[arm]]
        data = mujoco.MjData(model)
        data.qpos[:] = self.sim.data.qpos
        space = ob.RealVectorStateSpace(7)
        bounds = ob.RealVectorBounds(7)
        for index, (lower, upper) in enumerate(limits):
            bounds.setLow(index, float(lower))
            bounds.setHigh(index, float(upper))
        space.setBounds(bounds)
        setup = og.SimpleSetup(space)

        def valid(state):
            values = np.array([state[index] for index in range(7)])
            if np.any(values < limits[:, 0]) or np.any(values > limits[:, 1]):
                return False
            data.qpos[indices] = values
            mujoco.mj_forward(model, data)
            for contact in data.contact:
                if contact.dist < 0.003 and any(int(body) in self.sim.robot_bodies
                                               for body in model.geom_bodyid[[contact.geom1, contact.geom2]]):
                    return False
            return True

        setup.setStateValidityChecker(valid)
        information = setup.getSpaceInformation()
        information.setStateValidityCheckingResolution(0.002 / space.getMaximumExtent())
        start, goal = space.allocState(), space.allocState()
        for index in range(7):
            start[index] = float(self.sim.data.qpos[indices[index]])
            goal[index] = float(goal_q[indices[index]])
        if not valid(start) or not valid(goal):
            return PlanningResult(None, "Start or goal configuration is in collision or outside joint limits")
        setup.setStartAndGoalStates(start, goal)
        planner = og.RRTConnect(information)
        planner.setRange(0.25)
        setup.setPlanner(planner)
        setup.solve(self.timeout)
        if not setup.haveExactSolutionPath():
            return PlanningResult(None, "OMPL did not find an exact collision-free path within the time limit")
        setup.simplifySolution(0.1)
        path = setup.getSolutionPath()
        if not path.check():
            raise RuntimeError("OMPL returned a path that failed collision validation")
        positions = np.array([[state[index] for index in range(7)] for state in path.getStates()])
        delta = np.max(np.abs(np.diff(positions, axis=0)), axis=1)
        durations = np.maximum(0.15, np.maximum(1.5 * delta / 0.6, np.sqrt(6 * delta / 1.5)))
        trajectory = JointTrajectory(arm, np.r_[0.0, np.cumsum(durations)], positions,
                                     np.zeros_like(positions), "ompl").retime()
        return PlanningResult(trajectory)
