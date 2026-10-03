#!/usr/bin/env bash
# Checks that the environment is healthy. Every student runs this on day one.
source /opt/ros/jazzy/setup.bash
[ -f install/setup.bash ] && source install/setup.bash

pass() { echo "  [ok]   $1"; }
fail() { echo "  [FAIL] $1"; FAILED=1; }
FAILED=0

echo "ROS 2"
[ "$ROS_DISTRO" = "jazzy" ] && pass "ROS_DISTRO=jazzy" || fail "ROS_DISTRO is '$ROS_DISTRO'"
ros2 pkg list | grep -q '^rviz2$' && pass "rviz2 installed" || fail "rviz2 missing"
command -v check_urdf >/dev/null && pass "check_urdf found" || fail "check_urdf missing"

echo "Pinocchio"
python3 - <<'PY' && pass "pinocchio imports and computes gravity torques" || fail "pinocchio test failed"
import pinocchio as pin, numpy as np
model = pin.buildSampleModelManipulator()
data = model.createData()
q = pin.neutral(model)
g = pin.computeGeneralizedGravity(model, data, q)
assert np.all(np.isfinite(g))
PY

echo "OpenArm packages"
for p in openarm_description openarm_bringup; do
  ros2 pkg prefix "$p" >/dev/null 2>&1 && pass "$p built" || fail "$p not found (run: cb && cs)"
done
if ros2 pkg prefix openarm_description >/dev/null 2>&1; then
  echo "       description files:"
  ls "$(ros2 pkg prefix --share openarm_description)" | sed 's/^/         /'
fi

echo "Virtual CAN (needed for the CAN phase)"
if ip link show vcan0 >/dev/null 2>&1; then
  pass "vcan0 exists"
else
  echo "  [info] vcan0 not set up yet. Run: bash scripts/setup_vcan.sh"
fi

echo
[ "$FAILED" = "0" ] && echo "All good." || { echo "Some checks failed."; exit 1; }
