# Development

## Layout

```
cad2urdf/
  route.py         router CLI and routing table
  step.py          STEP front end: loading, joint inference, spec draft, inspector
  frontends.py     Onshape API and exporter-URDF front ends
  model.py         intermediate representation
  geometry.py      visual decimation, collision modes, joint-limit sweep
  writers.py       URDF, MJCF, SRDF, Isaac Lab / ManiSkill / Gazebo files
  usd_asset.py     the layered USDA asset
  asset_test.py    asset audit and stress tests
  sim2sim.py       one scenario run in several simulators and compared
  scene.py         static scenes and drop test
  validate.py, isaac_probe.py   simulator checks (the probe runs in Isaac Sim's Python)
  util.py          memory cap and shared helpers
.agents/skills/cad2sim/   agent skill: SKILL.md, part_classes.yaml
.agents/skills/sim2sim/   agent skill: cross-simulator comparison
examples/                 arm4 (parametric sample), sigmaban, step (Kaya base, a Haro380 spec)
tests/                    pytest suite, view_urdf.py
docs/                     these docs, RESEARCH.md, ONSHAPE_API_KEYS.md, demo GIFs
scripts/                  setup_venv.sh, run_safely.sh
```

## Memory cap

`route`, the compiler, `scene`, `validate` and the viewer re-launch themselves in the systemd user slice `cad2urdf.slice`. All cad2urdf jobs share one cap (half the RAM, no swap), run at `nice 19`, and are killed first if memory runs out. `CAD2URDF_RESERVE_GB` changes the cap, `CAD2URDF_NO_SANDBOX=1` turns it off, `scripts/run_safely.sh <command>` runs any command in the slice. Run heavy jobs one at a time. If ROS is sourced, its `PYTHONPATH` breaks the venv: prefix commands with `env -u PYTHONPATH`.

## Determinism

The same inputs give the same output bytes: sampling is seeded and ordering is sorted. To check, build twice with different `PYTHONHASHSEED` values and compare `find . -type f | sort | xargs md5sum`. Judgment calls (what a part is, which joints matter, limits) are not made in the code. They are spec data or agent-skill decisions.

## Import pitfalls

Import `pxr` and Isaac Sim packages lazily. Don't name a module like one of Isaac's top-level packages (`usd`, `omni`, `carb`, `isaacsim`): the Isaac probe runs with `python -P` for this reason, and a module called `usd.py` once broke Isaac's extension loading.
