#!/usr/bin/env bash
set -eo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if ! command -v docker >/dev/null 2>&1; then
  echo 'Docker is required. Install Docker and start it, then rerun this script.' >&2
  exit 1
fi
docker compose version >/dev/null
docker info >/dev/null

docker compose -f .devcontainer/compose.yaml up -d --build
docker compose -f .devcontainer/compose.yaml exec -T dev bash scripts/post-create.sh

echo 'Setup complete. Open http://localhost:6080/vnc.html (password: openarm).'
echo 'Container terminal: docker compose -f .devcontainer/compose.yaml exec dev bash'
