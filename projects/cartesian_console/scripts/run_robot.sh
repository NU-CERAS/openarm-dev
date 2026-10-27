#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$PROJECT_DIR/scripts/run_sync.sh" \
  --hardware-config "$PROJECT_DIR/../../runs/cartesian_console/sync/hardware.json" "$@" --backend can
