#!/usr/bin/env bash
# Run any heavy command inside the same shared memory cap as cad2urdf's own entry points.
#
#   scripts/run_safely.sh <command ...>
#   CAD2URDF_RESERVE_GB=6 scripts/run_safely.sh <command>   # keep more RAM for other apps
#
# - Memory: joins the systemd user slice cad2urdf.slice. ONE cap (half the RAM by default, no swap) covers all
#   cad2urdf jobs together; past it only these jobs are killed (exit 137).
# - OOM: oom_score_adj 1000, so the kernel kills these jobs before your editor/browser.
# - CPU: nice 19, so idle cores are used but other apps win.
set -euo pipefail
total_kb=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
# same default as cad2urdf/safety.py: leave half the RAM (at least 5 GB) to the desktop
reserve_kb=$(( total_kb / 2 )); (( reserve_kb < 5 * 1024 * 1024 )) && reserve_kb=$(( 5 * 1024 * 1024 ))
[ -n "${CAD2URDF_RESERVE_GB:-${RESERVE_GB:-}}" ] && reserve_kb=$(awk -v g="${CAD2URDF_RESERVE_GB:-$RESERVE_GB}" 'BEGIN{printf "%d", g*1024*1024}')
cap_kb=$(( total_kb - reserve_kb ))
(( cap_kb < 1024 * 1024 )) && cap_kb=$(( 1024 * 1024 ))
inner=(sh -c 'echo 1000 > /proc/self/oom_score_adj 2>/dev/null; exec "$@"' sh)
if command -v systemd-run >/dev/null && systemd-run --user --scope --quiet --slice=cad2urdf.slice true 2>/dev/null \
   && systemctl --user set-property --runtime cad2urdf.slice MemoryMax=${cap_kb}K MemoryHigh=$(( cap_kb * 9 / 10 ))K MemorySwapMax=0 2>/dev/null; then
  echo "run_safely: shared cad2urdf memory cap $(( cap_kb / 1024 / 1024 )) GB (no swap), killed first under pressure, nice 19" >&2
  exec systemd-run --user --scope --quiet --slice=cad2urdf.slice nice -n 19 env -u PYTHONPATH CAD2URDF_SANDBOXED=1 "${inner[@]}" "$@"
fi
echo "run_safely: no systemd user scopes; nice 19 + killed first under pressure, no hard cap" >&2
exec nice -n 19 env -u PYTHONPATH CAD2URDF_SANDBOXED=1 "${inner[@]}" "$@"
