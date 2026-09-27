"""Keep big conversions from freezing the machine.

Every heavy entry point (route, the compiler, scene) calls ``sandbox()`` first. It re-launches the
current command inside a systemd user scope with:

* a hard memory cap of total RAM minus ``CAD2URDF_RESERVE_GB`` (default 5 GB) and no swap: past the cap
  only this command is killed (exit 137), your browser and editor are untouched;
* ``nice 19``: every idle core is used, but other apps win whenever they need CPU.

Set ``CAD2URDF_NO_SANDBOX=1`` to disable (e.g. in CI or containers). Where systemd user scopes aren't
available it only renices and prints a warning.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

_MARK = "CAD2URDF_SANDBOXED"


def _mem_total_kb() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                return int(line.split()[1])
    return 0


def sandbox(module: str) -> None:
    """Re-exec ``python -m <module> <args>`` inside the memory-capped scope, once."""
    if os.environ.get(_MARK) or os.environ.get("CAD2URDF_NO_SANDBOX") or not sys.platform.startswith("linux"):
        return
    reserve_gb = float(os.environ.get("CAD2URDF_RESERVE_GB", "5"))
    total = _mem_total_kb()
    cap_kb = max(int(total - reserve_gb * 1024 * 1024), 1024 * 1024)
    env = {**os.environ, _MARK: "1"}
    env.pop("PYTHONPATH", None)  # ROS's PYTHONPATH breaks the venv
    cmd = [sys.executable, "-m", module, *sys.argv[1:]]
    if shutil.which("systemd-run") and subprocess.run(
            ["systemd-run", "--user", "--scope", "--quiet", "true"], capture_output=True).returncode == 0:
        print(f"[cad2urdf] memory cap {cap_kb / 1024 / 1024:.1f} GB, lowest CPU priority "
              f"(CAD2URDF_NO_SANDBOX=1 to disable)", file=sys.stderr)
        full = ["systemd-run", "--user", "--scope", "--quiet", "-p", f"MemoryMax={cap_kb}K",
                "-p", f"MemoryHigh={cap_kb * 9 // 10}K", "-p", "MemorySwapMax=0", "nice", "-n", "19", *cmd]
    else:
        print("[cad2urdf] no systemd user scope: running at lowest CPU priority without a memory cap",
              file=sys.stderr)
        full = ["nice", "-n", "19", *cmd]
    r = subprocess.run(full, env=env)
    if r.returncode == 137:
        print(f"[cad2urdf] stopped: the conversion needed more than {cap_kb / 1024 / 1024:.1f} GB. Other apps "
              "were protected. Close some apps or set CAD2URDF_RESERVE_GB lower, or coarsen tessellation "
              "(visual.tessellation.linear_mm in the spec).", file=sys.stderr)
    sys.exit(r.returncode)
