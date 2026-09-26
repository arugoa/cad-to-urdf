"""Parametric sample CAD: a 4-DOF arm with a parallel-jaw gripper.

Builds 24 named solids with build123d (OpenCascade) and writes them as a single
STEP assembly (``cad/arm4.step``). The STEP file stands in for "an assembly
exported from Onshape / SolidWorks / Fusion / Creo": it keeps part names,
placement and exact B-rep geometry, but — like almost every real STEP export —
it carries *no mates/joints*. Joints are recovered by ``cad2urdf.joints``
(geometric inference) and fixed by ``robot_spec.yaml`` (the granularity spec).

The design is posed in its zero configuration (arm pointing straight up) and
modelled in millimetres, as CAD tools do; the converter scales to metres.

Run:  python examples/arm4/build_cad.py
"""

from pathlib import Path

from build123d import (
    Box,
    Color,
    Compound,
    Cylinder,
    Pos,
    Rot,
    export_step,
)

OUT = Path(__file__).parent / "cad" / "arm4.step"

# Key dimensions (mm). Changing these regenerates a consistent assembly.
Z_BASE_TOP = 82.0  # shoulder_yaw axis height (top of base housing)
Z_SHOULDER = 150.0  # shoulder_pitch axis height
L_UPPER = 250.0  # shoulder -> elbow
L_FORE = 230.0  # elbow -> wrist
Z_ELBOW = Z_SHOULDER + L_UPPER
Z_WRIST = Z_ELBOW + L_FORE
CLEARANCE = 0.2  # diametral clearance of every bore around its shaft


def cyl_y(radius, length, at):
    """Cylinder whose axis is the world Y axis, centred at ``at``."""
    return Pos(*at) * Rot(90, 0, 0) * Cylinder(radius, length)


def cyl_z(radius, z0, z1, xy=(0, 0)):
    return Pos(xy[0], xy[1], (z0 + z1) / 2) * Cylinder(radius, z1 - z0)


def box(sx, sy, sz, center):
    return Pos(*center) * Box(sx, sy, sz)


def build_parts():
    p = {}
    shaft_r = 14.8
    bore_r = shaft_r + CLEARANCE / 2

    # ---- base (4 bolts + plate + housing: 6 parts that never move relative to each other)
    plate = box(160, 160, 12, (0, 0, 6))
    for sx in (-60, 60):
        for sy in (-60, 60):
            plate -= cyl_z(4.5, 0, 12, (sx, sy))
    p["base_plate"] = (plate, "aluminum")
    i = 1
    for sx in (-60, 60):
        for sy in (-60, 60):
            bolt = cyl_z(4.0, 0, 12, (sx, sy)) + cyl_z(7.0, 12, 18, (sx, sy))
            p[f"base_bolt_{i}"] = (bolt, "steel")
            i += 1
    housing = cyl_z(50, 12, Z_BASE_TOP) - cyl_z(bore_r, Z_BASE_TOP - 20, Z_BASE_TOP)
    p["base_housing"] = (housing, "aluminum")

    # ---- turret (rotates about Z): disc+shaft, two clevis plates, motor can
    turret = cyl_z(55, Z_BASE_TOP, Z_BASE_TOP + 15) + cyl_z(shaft_r, Z_BASE_TOP - 20, Z_BASE_TOP)
    p["turret_disc"] = (turret, "aluminum")
    pin_r = 6.0
    for side, y in (("left", 27.0), ("right", -27.0)):
        plate = box(60, 10, 80, (0, y, Z_BASE_TOP + 15 + 40))
        plate -= cyl_y(pin_r + CLEARANCE / 2, 10, (0, y, Z_SHOULDER))
        p[f"turret_clevis_{side}"] = (plate, "aluminum")
    p["shoulder_motor"] = (cyl_y(25, 40, (0, 52, Z_SHOULDER)), "steel")

    # ---- upper arm (rotates about Y at the shoulder)
    beam = box(36, 40, L_UPPER, (0, 0, Z_SHOULDER + L_UPPER / 2))
    beam += cyl_y(20, 40, (0, 0, Z_SHOULDER)) + cyl_y(20, 40, (0, 0, Z_ELBOW))
    beam -= cyl_y(pin_r, 40, (0, 0, Z_SHOULDER))  # pin is pressed in (same link)
    beam -= cyl_y(5.0 + CLEARANCE / 2, 40, (0, 0, Z_ELBOW))  # elbow bore
    beam -= box(20, 50, L_UPPER - 80, (0, 0, Z_SHOULDER + L_UPPER / 2))  # lightening slot
    p["upper_arm_beam"] = (beam, "aluminum")
    p["shoulder_pin"] = (cyl_y(pin_r, 60, (0, 0, Z_SHOULDER)), "steel")  # ends inside the clevis plates

    # ---- forearm (rotates about Y at the elbow): two side plates, spacer, pin
    for side, y in (("left", 24.0), ("right", -24.0)):
        fp = box(30, 6, L_FORE, (0, y, Z_ELBOW + L_FORE / 2 - 10))
        fp += cyl_y(15, 6, (0, y, Z_ELBOW))
        fp -= cyl_y(5.0, 6, (0, y, Z_ELBOW))
        p[f"forearm_plate_{side}"] = (fp, "aluminum")
    spacer = box(30, 42, 30, (0, 0, Z_WRIST - 15))
    spacer -= cyl_z(8.0 + CLEARANCE / 2, Z_WRIST - 15, Z_WRIST)  # wrist bore
    p["forearm_spacer"] = (spacer, "aluminum")
    p["elbow_pin"] = (cyl_y(5.0, 54, (0, 0, Z_ELBOW)), "steel")
    p["elbow_motor"] = (cyl_y(20, 30, (0, -42, Z_ELBOW)), "steel")  # bolted to the right forearm plate

    # ---- gripper base (rolls about Z at the wrist): flange+shaft, palm, rail, rail posts
    flange = cyl_z(20, Z_WRIST, Z_WRIST + 15) + cyl_z(8.0, Z_WRIST - 15, Z_WRIST)
    p["wrist_flange"] = (flange, "aluminum")
    z_palm = Z_WRIST + 15
    p["gripper_palm"] = (box(30, 90, 20, (0, 0, z_palm + 10)), "pla")
    z_rail = z_palm + 20 + 12
    p["gripper_rail"] = (cyl_y(4.0, 90, (0, 0, z_rail)), "steel")
    for side, y in (("left", 41.0), ("right", -41.0)):
        post = box(16, 8, 24, (0, y, z_palm + 20 + 12))
        post -= cyl_y(4.0, 8, (0, y, z_rail))
        p[f"gripper_rail_post_{side}"] = (post, "pla")

    # ---- fingers (slide along Y on the rail)
    for side, s in (("left", 1), ("right", -1)):
        carriage = box(24, 20, 18, (0, s * 18, z_rail))
        carriage -= cyl_y(4.0 + CLEARANCE / 2, 20, (0, s * 18, z_rail))
        blade = box(20, 6, 60, (0, s * 11, z_rail + 9 + 30))
        p[f"finger_{side}"] = (carriage + blade, "pla")
    return p


MATERIAL_COLOR = {
    "aluminum": Color(0.72, 0.74, 0.78),
    "steel": Color(0.30, 0.31, 0.34),
    "pla": Color(0.95, 0.45, 0.15),
}


def main():
    parts = build_parts()
    children = []
    for name, (shape, material) in parts.items():
        shape.label = name
        shape.color = MATERIAL_COLOR[material]
        children.append(shape)
    assembly = Compound(label="arm4", children=children)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    export_step(assembly, OUT)
    print(f"wrote {OUT} with {len(children)} parts")


if __name__ == "__main__":
    main()
