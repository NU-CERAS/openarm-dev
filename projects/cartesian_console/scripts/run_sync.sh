#!/usr/bin/env bash
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
exec "$PYTHON_BIN" -u -m openarm_sim.console.app "$@"
