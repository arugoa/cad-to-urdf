#!/usr/bin/env bash
# Create the repo's single virtualenv (.venv, Python 3.12).
#
#   scripts/setup_venv.sh            # conversion + MuJoCo, PyBullet, SAPIEN/ManiSkill, viewer (~7 GB)
#   scripts/setup_venv.sh --isaac    # the same plus Isaac Sim 6.1 in the same venv (~27 GB)
#
# With --isaac, Isaac Sim goes in first and everything else is held to the versions it pins
# (only coacd and pillow are allowed to move), so both sides keep working in one environment.
set -euo pipefail
cd "$(dirname "$0")/.."
uv venv -p 3.12 .venv
if [[ "${1:-}" == "--isaac" ]]; then
  uv pip install -p .venv --prerelease=allow "isaacsim[all,extscache]==6.1.*" --extra-index-url https://pypi.nvidia.com
  uv pip freeze -p .venv | grep -E '^[A-Za-z0-9_.-]+==' | grep -viE '^(coacd|pillow)==' > .venv/isaac-pins.txt
  uv pip install -p .venv -r requirements.txt -c .venv/isaac-pins.txt \
    --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match
else
  uv pip install -p .venv -r requirements.txt
fi
env -u PYTHONPATH .venv/bin/python -m pytest -q tests
