"""Small helpers shared by the writers and front ends."""

from __future__ import annotations

import re
import warnings

import numpy as np
from scipy.spatial.transform import Rotation


def fmt(v, digits=6) -> str:
    """Space-separated numbers for XML attributes."""
    return " ".join(f"{x:.{digits}g}" if abs(x) > 1e-12 else "0" for x in np.ravel(v))


def rpy(T: np.ndarray) -> np.ndarray:
    """URDF fixed-axis roll/pitch/yaw of a 4x4 or 3x3 transform."""
    with warnings.catch_warnings():  # gimbal lock still gives a valid decomposition
        warnings.simplefilter("ignore", UserWarning)
        return Rotation.from_matrix(np.asarray(T)[:3, :3]).as_euler("xyz")


def quat(T: np.ndarray) -> str:
    """MJCF quaternion (w x y z) of a 4x4 or 3x3 transform."""
    x, y, z, w = Rotation.from_matrix(np.asarray(T)[:3, :3]).as_quat()
    return fmt((w, x, y, z))


def slug(s: str, lower: bool = False) -> str:
    """A name usable as an XML/file identifier."""
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s).strip("_") or "x"
    return s.lower() if lower else s


def meminfo_gb(field: str) -> float:
    """A /proc/meminfo field (e.g. MemTotal, MemAvailable) in GB."""
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1]) / 1024 / 1024
    return 0.0


class UnionFind:
    def __init__(self, items=()):
        self.p = {i: i for i in items}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


FASTENER = re.compile(r"screw|bolt|nut(?![a-z])|washer|rivet|standoff|dowel|(threaded|heat.?set)[_ ]?insert|"
                      r"(?<![a-z])(shcs|bhcs|fhcs|msb\d+|m\d+x\d)", re.I)
NOT_FASTENER = re.compile(r"lead.?screw|ball.?screw|acme|trapezoidal|threaded.?rod|worm", re.I)  # drive screws


def is_fastener(name: str) -> bool:
    n = name.split("/")[-1]
    return bool(FASTENER.search(n)) and not NOT_FASTENER.search(n)
