from openarm_tools.robot.config import load, root


ROOT = root()
PROJECT_ROOT = ROOT / "projects" / "cartesian_console"
RUNS = ROOT / "runs" / "cartesian_console"
ROBOT_CONFIGS = {arm: load(robot="v1.0", arm=arm) for arm in ("right", "left")}
MODEL = ROOT / ROBOT_CONFIGS["right"]["model"]
