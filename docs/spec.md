# Spec and outputs

The spec is a small YAML file. For STEP input it is drafted from the geometry, with each guess marked `# REVIEW`. Pass corrections with `--spec`; they are merged over the draft. Anything the rules can't settle goes here as an explicit value, so the next run gives the same result.

```yaml
materials: {aluminum: 2700, steel: 7850, pla: 1240}          # kg/m^3
part_materials: {"*bolt*": steel, "*": aluminum}              # first match wins; "*" is required
part_classes: {fastener: ['hold-down clamp'], servo: ['^st3215']}   # name regexes, merged over the library
joints:
  base_to_turret: {limits: [-2.97, 2.97], effort: 40, velocity: 3}
  gripper_to_finger_2: {mimic: {joint: gripper_to_finger}, axis_sign: -1}
  some_bearing_joint: {type: fixed}                           # weld a joint that is not a mechanism
dynamics:  {default: {damping: 0.5, friction: 0.05, armature: 0.01}}
actuators: {default: {kind: position, kp: 200, kv: 10}, gripper_to_finger_2: {kind: none}}
collision: {default: {mode: auto, max_geoms: 12}, finger: {mode: decompose}}
simplify:  {drop_fasteners: true, visual_faces_per_link: 20000}   # or `simplify: false`
root_rpy:  [1.5708, 0, 0]                                     # Y-up export -> Z-up
closures:                                                     # loops a URDF tree can't hold
  pump_hinge: {link1: upper_arm, link2: pump_rod, point: [14.91, 88.89, 51.34]}   # CAD units
srdf: {group_states: {home: {group: arm, joints: {base_to_turret: 0}}}}
```

`type: fixed` works on every route and every output. Deciding which joints are real mechanisms is done with the [`cad2sim` skill](../.agents/skills/cad2sim/SKILL.md).

## Part classes

The code knows nothing about part names. A part counts as a fastener, bearing, gear, servo, non-physical solid or placeholder only if `part_classes:` has a pattern for it. [`part_classes.yaml`](../.agents/skills/cad2sim/part_classes.yaml) is the default pattern set: pass it with `--part-classes` and add or override classes in the spec. Without it nothing is dropped as a fastener and no bearing, servo or gear is recognised. Fasteners are removed from visuals and collision (their mass stays) and their mates never become joints.

## Collision

Modes per link: `none`, `box`, `spheres`, `primitives`, `hull`, `auto` (default), `decompose`, `keep` (an input URDF's own shapes). `auto` uses a primitive where it fills at least 80% of a part, a hull for small or nearly convex parts, and CoACD otherwise. Pieces are convex with at most 64 vertices. A link gets at most `max_geoms` pieces (12 by default); a small link gets one piece if it fits tightly, otherwise up to four.

## Output folder

```
build/robot/
├── <robot>.urdf, <robot>.srdf
├── mjcf/<robot>.xml
├── usd/<robot>.usda, usd/configuration/   see usd.md
├── maniskill/<robot>_agent.py
├── isaaclab/<robot>_cfg.py
├── gazebo/<robot>.gazebo.urdf, gazebo/config/controllers.yaml
├── meshes/visual/*.stl           per link and material, up to 100k triangles each
├── meshes/collision/*.stl        one convex piece each
├── robot_spec.yaml               the spec used
├── robot_spec.draft.yaml         STEP only: the draft with REVIEW notes
├── report.json                   joint candidates, masses, collision fit, joint-limit sweep
├── validation.json               what each simulator loaded and did
└── asset_test.json               from `python -m cad2urdf.asset_test`
```
