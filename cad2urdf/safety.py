"""Keep big conversions from running the machine out of memory.

Every heavy entry point (route, the compiler, scene, validate, the viewer) calls ``sandbox()`` first.
It re-launches the command inside the systemd user slice ``cad2urdf.slice``, which has:

* ONE shared memory cap for all cad2urdf jobs together: total RAM minus ``CAD2URDF_RESERVE_GB``
  (default 5 GB), no swap. Running a batch, a validation and a viewer at the same time can't add up
  past it. Over the cap, only cad2urdf processes are killed (exit 137); the rest of the desktop is untouched.
* ``oom_score_adj = 1000`` for every cad2urdf process, so that if the machine still runs out of memory
  for another reason, the kernel kills our jobs before your editor or browser.
* ``nice 19``: every idle core is used, but other apps win whenever they need CPU.

Set ``CAD2URDF_NO_SANDBOX=1`` to disable (CI, containers). Without systemd user scopes it still renices
and raises the OOM score, but can't enforce the cap.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

_MARK = "CAD2URDF_SANDBOXED"
SLICE = "cad2urdf.slice"


def _mem_total_kb() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                return int(line.split()[1])
    return 0


def prefer_oom_kill() -> None:
    """Ask the kernel to kill this process (and its children) first under memory pressure."""
    try:
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("1000")
    except OSError:
        pass


def cap_kb() -> int:
    reserve_gb = float(os.environ.get("CAD2URDF_RESERVE_GB", "5"))
    return max(int(_mem_total_kb() - reserve_gb * 1024 * 1024), 1024 * 1024)


def _have_scopes() -> bool:
    return bool(shutil.which("systemd-run")) and subprocess.run(
        ["systemd-run", "--user", "--scope", "--quiet", f"--slice={SLICE}", "true"],
        capture_output=True).returncode == 0


def _limit_slice(cap: int) -> bool:
    """(Re)apply the shared cap to the slice every cad2urdf job runs in."""
    r = subprocess.run(["systemctl", "--user", "set-property", "--runtime", SLICE, f"MemoryMax={cap}K",
                        f"MemoryHigh={cap * 9 // 10}K", "MemorySwapMax=0"], capture_output=True)
    return r.returncode == 0


def sandbox(module: str | None = None) -> None:
    """Re-exec the current command (``python -m module`` or this script) inside the capped slice, once."""
    if os.environ.get(_MARK):
        prefer_oom_kill()
        return
    if os.environ.get("CAD2URDF_NO_SANDBOX") or not sys.platform.startswith("linux"):
        return
    cap = cap_kb()
    env = {**os.environ, _MARK: "1"}
    env.pop("PYTHONPATH", None)  # ROS's PYTHONPATH breaks the venv
    target = ["-m", module] if module else [os.path.abspath(sys.argv[0])]
    cmd = [sys.executable, *target, *(sys.argv[1:])]
    if _have_scopes() and _limit_slice(cap):
        print(f"[cad2urdf] all cad2urdf jobs share a {cap / 1024 / 1024:.1f} GB memory cap (no swap), lowest CPU "
              f"priority, killed first under memory pressure (CAD2URDF_NO_SANDBOX=1 to disable)", file=sys.stderr)
        full = ["systemd-run", "--user", "--scope", "--quiet", f"--slice={SLICE}", "nice", "-n", "19", *cmd]
    else:
        print("[cad2urdf] no systemd user scope: lowest CPU priority and killed first under memory pressure, "
              "but no hard memory cap", file=sys.stderr)
        full = ["nice", "-n", "19", *cmd]
    r = subprocess.run(full, env=env)
    if r.returncode == 137:
        print(f"[cad2urdf] stopped: cad2urdf jobs together needed more than {cap / 1024 / 1024:.1f} GB. The rest of "
              "the desktop was protected. Run fewer jobs at once, set CAD2URDF_RESERVE_GB lower, or coarsen "
              "tessellation (visual.tessellation.linear_mm in the spec).", file=sys.stderr)
    sys.exit(r.returncode)
