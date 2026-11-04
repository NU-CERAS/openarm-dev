#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
mkdir -p runs/rl/cache
export TMPDIR="$ROOT/runs/rl/cache"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "${OPENARM_PYTHON:-python}" -m openarm_rl.skills.cli "$@"
