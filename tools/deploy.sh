#!/usr/bin/env bash
# Copy this repo into the running ComfyUI container and run the tests there.
# Then reload the pack in place, which keeps models loaded. If the running pack
# has no reload endpoint yet, or with --restart, restart ComfyUI instead.
# Usage: tools/deploy.sh [--restart] [container]    (from the repo root)
set -euo pipefail
export MSYS_NO_PATHCONV=1

restart=0
if [ "${1:-}" = "--restart" ]; then
    restart=1
    shift
fi
container="${1:-comfyui}"
target=/opt/ComfyUI/custom_nodes/comfyui-h3-take
governor="${GOVERNOR_URL:-http://127.0.0.1:8199}"
comfy="${COMFY_URL:-http://127.0.0.1:8188}"

docker exec "$container" sh -c "mkdir -p $target && find $target -mindepth 1 -maxdepth 1 ! -name __pycache__ -exec rm -rf {} +"
git ls-files --cached --others --exclude-standard -z | tar --null -T - -cf - | docker exec -i "$container" tar -xf - -C "$target"
docker exec "$container" touch "$target/dev.enable"

docker exec -w "$target" -e PYTHONPATH=/opt/ComfyUI -e PYTHONDONTWRITEBYTECODE=1 "$container" \
    /opt/venv/bin/python -m unittest discover -s tests

if [ "$restart" = 0 ]; then
    code=$(curl -s -m 60 -o /tmp/h3take-reload.json -w '%{http_code}' -X POST "$comfy/h3take/reload" || true)
    cat /tmp/h3take-reload.json 2>/dev/null || true
    echo
    if [ "$code" = 200 ]; then
        echo "reloaded in place. Refresh the browser."
        exit 0
    fi
    if [ "$code" != 404 ]; then
        echo "reload failed with $code" >&2
        exit 1
    fi
fi

curl -fsS -m 60 -X POST -H "Content-Type: application/json" -d '{"strategy":"graceful"}' "$governor/api/comfyui/restart"
echo
