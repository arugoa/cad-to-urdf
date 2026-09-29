"""Out-of-memory protection for heavy entry points.

``sandbox()`` re-launches the command in the systemd user slice ``cad2urdf.slice``: one memory cap shared
by all cad2urdf jobs (RAM minus ``CAD2URDF_RESERVE_GB``, default half the RAM; no swap), oom_score_adj
1000 so the kernel kills these jobs first, and nice 19. ``CAD2URDF_NO_SANDBOX=1`` disables it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

from .util import meminfo_gb

_MARK = "CAD2URDF_SANDBOXED"
SLICE = "cad2urdf.slice"



def prefer_oom_kill() -> None:
    """Ask the kernel to kill this process (and its children) first under memory pressure."""
    try:
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("1000")
    except OSError:
        pass


def cap_kb() -> int:
    total = int(meminfo_gb("MemTotal") * 1024 * 1024)
    default = max(5.0, total / 2 / 1024 / 1024)  # leave at least half the RAM to the desktop (VS Code, browser)
    reserve_gb = float(os.environ.get("CAD2URDF_RESERVE_GB", default))
    return max(int(total - reserve_gb * 1024 * 1024), 1024 * 1024)


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
