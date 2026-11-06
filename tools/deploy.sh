#!/usr/bin/env bash
# Copy this repo into the running ComfyUI container, run the tests there, restart ComfyUI.
# Usage: tools/deploy.sh [container]    (from the repo root)
set -euo pipefail
export MSYS_NO_PATHCONV=1

container="${1:-comfyui}"
target=/opt/ComfyUI/custom_nodes/comfyui-h3-take
governor="${GOVERNOR_URL:-http://127.0.0.1:8199}"

docker exec "$container" sh -c "mkdir -p $target && find $target -mindepth 1 -maxdepth 1 ! -name __pycache__ -exec rm -rf {} +"
git ls-files --cached --others --exclude-standard -z | tar --null -T - -cf - | docker exec -i "$container" tar -xf - -C "$target"

docker exec -w "$target" -e PYTHONPATH=/opt/ComfyUI -e PYTHONDONTWRITEBYTECODE=1 "$container" \
    /opt/venv/bin/python -m unittest discover -s tests

curl -fsS -m 60 -X POST -H "Content-Type: application/json" -d '{"strategy":"graceful"}' "$governor/api/comfyui/restart"
echo
