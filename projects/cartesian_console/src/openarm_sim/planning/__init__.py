from dataclasses import dataclass

from ..execution import JointTrajectory


@dataclass(frozen=True)
class PlanningResult:
    trajectory: JointTrajectory | None
    reason: str = ""
