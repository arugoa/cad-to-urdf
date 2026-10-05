# CAD sources

Choose the CAD package, the export format and the target simulator. Where the CAD gives real mates, use them: they say what moves. STEP has shapes but no mates, so joints are inferred from geometry. All routes end in the same compiler. `python -m cad2urdf.route --list` prints the matrix; [RESEARCH.md §9](RESEARCH.md#9-routing-cad--format--simulator) explains the choices.

```bash
python -m cad2urdf.route --cad onshape --format native --sim maniskill        # print the plan only

python -m cad2urdf.route --cad solidworks --format step --sim maniskill --run \
    --input robot.step --out build/robot [--spec overrides.yaml] \
    --part-classes .agents/skills/cad2sim/part_classes.yaml

python -m cad2urdf.route --cad fusion --format native --sim mujoco --run \
    --input exported/robot.urdf --out build/robot
```

Every route writes every output. `--sim` only sets collision defaults and which checks run. Output goes to `build/` (git-ignored).

## Onshape

- **URDF export:** right-click the assembly tab, Export, URDF, then run the router on the unzipped URDF with `--cad onshape --format urdf-export`.
- **API (preferred):** reads mates, limits, gear relations, masses and meshes. Needs keys in `.env` ([ONSHAPE_API_KEYS.md](ONSHAPE_API_KEYS.md)). Pass the assembly tab's URL:

```bash
python -m cad2urdf.route --cad onshape --format native --sim maniskill --run \
    --input "https://cad.onshape.com/documents/<doc>/w/<workspace>/e/<assembly>" --out build/myrobot \
    --part-classes .agents/skills/cad2sim/part_classes.yaml
```

Conventions from onshape-to-robot: if any mate is named `dof_*`, only those mates are joints (`_inv` flips the axis); `closing_*` mates close loops; joints named `*passive*` get no actuator, `*_speed` a velocity actuator; `frame_*` parts are markers.

How other mates are handled:

- Planar mates between the same two bodies are combined. One is two slides and a spin, two leave one slide, three fix the part.
- A mate to the assembly origin (including a mate that lists one entity) moves the body against the grounded part. A REVIEW line names the part used.
- A part in the `fastener` class stays fixed to whatever it is mated to. Parts named like FOV cones or keep-out volumes, or in the `ignore` class, are skipped.
- Default mate names ("Revolute 4") get the moved link's name as a prefix.

Onshape does not report a document's up axis. If the robot loads sideways, set `root_rpy: [1.5708, 0, 0]`.

## SolidWorks, Fusion, Creo

Run an exporter inside the CAD tool (sw2robot, ACDC4Robot, creo2urdf) and pass the URDF it writes as `--input`. With no access to the CAD tool, export one STEP file of the whole assembly.

## STEP

The draft uses fixed rules: touching parts form a link, a running fit (shaft in a bore, 0.005 to 0.15 mm radial clearance) is a joint, a press fit is fixed. Bearings, servo horns, gears and fasteners are recognised through `part_classes` ([spec.md](spec.md)). Guesses are marked `# REVIEW` in `robot_spec.draft.yaml`; pass corrections with `--spec`.

If the draft misses joints (motors flat against a link, zero-clearance pivots), write the links and joints by hand, taking axes from the inspector:

```bash
python -m cad2urdf.step robot.step --part-classes .agents/skills/cad2sim/part_classes.yaml
```

[`examples/step/Haro380.spec.yaml`](../examples/step/Haro380.spec.yaml) is a worked example. It needs your own copy of the STEP file.

## URDF input

`--input robot.urdf` goes through the same compiler. Use `package_dirs` in the spec to resolve `package://` paths and `collision: {default: {mode: keep}}` to keep the file's own collision shapes.

## Static scenes

Surface-only STEP files (competition fields) have no solids to make links from. They become a static scene:

```bash
python -m cad2urdf.scene field.step -o build/field
python tests/view_urdf.py build/robot --sim maniskill --scene build/field --at 0 0 0.02
```
