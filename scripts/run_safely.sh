#!/usr/bin/env bash
# Run a heavy command at full speed on idle CPU without starving the rest of the machine.
#
#   scripts/run_safely.sh python -m cad2urdf.route ... --run ...
#   RESERVE_GB=6 scripts/run_safely.sh <command>      # keep more RAM for other apps
#
# - CPU: `nice -n 19`, so every idle core is used, but browsers/editors win whenever they need CPU.
# - Memory: a systemd user scope with a hard cap of (total RAM - RESERVE_GB, default 5 GB) and no swap.
#   Past the cap only THIS command is killed (exit 137); other apps are untouched. It starts
#   reclaiming/throttling at 90 % of the cap first.
# Falls back to nice-only (with a warning) where systemd user scopes aren't available.
set -euo pipefail
reserve_gb=${RESERVE_GB:-5}
total_kb=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
cap_kb=$(( total_kb - reserve_gb * 1024 * 1024 ))
(( cap_kb < 1024 * 1024 )) && cap_kb=$(( 1024 * 1024 ))
high_kb=$(( cap_kb * 9 / 10 ))
if command -v systemd-run >/dev/null && systemd-run --user --scope --quiet true 2>/dev/null; then
  echo "run_safely: memory cap $(( cap_kb / 1024 / 1024 )) GB, CPU priority lowest (nice 19)" >&2
  exec systemd-run --user --scope --quiet -p MemoryMax=${cap_kb}K -p MemoryHigh=${high_kb}K -p MemorySwapMax=0 \
       nice -n 19 env -u PYTHONPATH "$@"
fi
echo "run_safely: systemd user scopes unavailable; running with nice 19 only (no memory cap)" >&2
exec nice -n 19 env -u PYTHONPATH "$@"
