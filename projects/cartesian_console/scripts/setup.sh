#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd -- "$PROJECT_DIR/../.." && pwd)"
case "$(uname -s)" in
  Linux) exec bash "$REPO_DIR/setup/linux.sh" --console "$@" ;;
  Darwin) exec bash "$REPO_DIR/setup/mac.sh" --console "$@" ;;
  *) echo 'Use setup/windows.ps1 -Console from Windows, then run the console inside WSL.' >&2; exit 1 ;;
esac
