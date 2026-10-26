# Zhang et al. 2021: independent OneString integration experiment (2026-10-27)

**Isolation:** branch `agent/zhang2021-independent-20261027` starts from frozen
commit `3e91e23bbbcc54031efe2c4839c7283ef0b7ea6e`.
No changes are made to `main`, `feature/planar-quad-audit-20261026`,
Claude's branch, launchers or the production app. Do not merge automatically.

## Research status: NOT a reference implementation yet

Zhang, Chai, Liu, Fu, *Quad Meshing with Coarse Layouts for Planar
Domains*, CAD 140, 103084 (2021):
https://doi.org/10.1016/j.cad.2021.103084

The authors' project page lists Code, Exe, Data and Suppl:
https://kfckfckf.github.io/
The Code URL is https://rec.ustc.edu.cn/share/8081ab10-dd29-11eb-b5e1-5fc36251b03b
The USTC REC site announced on 2026-08-06 that REC stopped service
and migrated to pan.ustc.edu.cn: https://rec.ustc.edu.cn/download

**No actual author source code, its license, or its CLI format could be
retrieved and verified during this experiment.** Do not describe the
current module as implementing paper stages (field-guided segmentation,
template layout, or global optimization). In particular, do not silently
run Gmsh, a cropped regular grid, or invented heuristics as Zhang 2021.

## Second stage: ACTUALLY GENERATING experimental Quad Meshes

Added `src/onestring_physics/zhang2021_field_layout_experiment.py`.

This is a **separate independent experimental algorithm** (not the original
Zhang et al. 2021 C++ implementation). It does **not** call Gmsh. It performs:

1. Source Omega simplification **only within** the original area bound.
2. 4-RoSy directional estimation from weighted boundary-edge directions.
3. Geometrically constrained planar triangulation.
4. Field-weighted adjacent-triangle pairing into coarse, convex quad patches.
5. Conforming midpoint/centroid template quadrangulation (three/four quads
   per triangle/quad patch).
6. Constrained center-only improvement of 2D angle and edge-length quality.
7. Original OptCuts UV/3D barycentric inverse and **mandatory** seam audit.
8. A separate Streamlit dropdown labeled
   `OptCuts → independent coarse-patch Quad experiment (NOT Zhang reference)`.
   The original Gmsh mode is unchanged.

The paper's *specific* low-distortion straightening, field-guided
segmentation solver, template library, patch-count global optimizer
and original C++ reference are NOT implemented and are NOT claimed.

The engine can produce actual all-quad M2D. It has been tested on square,
concave L/stepped polygons and a built-in open dome throughout the
OneString M2D→M3D→K3D→T3D→K2D→T2D stages. The dome input uses
`pca_debug`, NOT an actual OptCuts Bunny run. The actual 30–35 quad
target remains approximate, not hard-constrained.

### Run on a saved OptCuts source NPZ

```bash
cd onestring-physics-simulator
python3 -m pip install -e . pytest 'shapely>=2.1'
python3 scripts/run_field_layout_experiment.py \
  --source /path/to/optcuts_source.npz \
  --output /tmp/field_layout_quads.npz \
  --report /tmp/field_layout_report.json \
  --target-quads 35
```

The exporter `zhang2021_reference_bridge.export_source_chart()` writes the
required `optcuts_source.npz` format.

Streamlit: run the existing OptCuts launcher; select the new
`2026-10-27 | independent coarse-patch Quad experiment` option, with
`Paper-oriented computation mode` disabled. Official OptCuts binary
is still required for actual Bunny OptCuts input. The updated mode
remains an **experimental** mesh generator and may fail on a seam or
CSF Split constraint rather than silently return an invalid design.

## What IS implemented and tested

`src/onestring_physics/zhang2021_reference_bridge.py`

- Immutable NPZ export of the already-generated OptCuts chart, its paired
  3D triangles and its original Omega boundary.
- External **adapter** invocation with explicit, separately verified
  `{input}` and `{output}` paths, process timeout, nonzero-exit handling.
- Explicit NPZ output contract; the published C++ executable is **not**
  claimed to accept this contract directly.
- Strict candidate audit: all-quad disk topology, orientation/convexity,
  overlapping area, outside-Omega fraction and uncovered Omega fraction,
  agreement of UV triangulation with the declared Omega polygon,
  barycentric inverse at every new vertex, every quad edge midpoint
  and each quad centroid.
- Original OptCuts seam provenance, checked at real boundary *edges*
  through the existing `discover_exact_seam_topology()` auditor.
- A verified mesh can be adapted to the existing `QuadMesh` and
  `UnstructuredQuadGrid` structures **without activating the app**.
- Metrics intentionally include `zhang2021_algorithm_verified=False`.
  Passing an NPZ geometry validation DOES NOT prove the upstream
  algorithm was Zhang's. Reference identity and reproduction must be
  established independently, from inspected original source.

## Our adapter contract (NOT the author's native input format)

Inputs in `optcuts_source.npz`:

`source_uv_vertices[N,2]`, `source_uv_faces[T,3]`,
`source_xyz_vertices[M,3]`, `source_xyz_faces[T,3]`,
`omega_boundary[B,2]`.

Outputs in `candidate_quads.npz`:

`vertices_uv[V,2]`, `faces[Q,4]`.

Example of the *API shape*, not a runnable paper command:

```python
audited = run_reference_adapter(
    optcuts_parameterization,
    ["python3", "my_verified_zhang_cpp_adapter.py",
     "--input", "{input}", "--output", "{output}"],
)
m2d = to_onestring_m2d(audited, tile_size=1., gap_size=.05)
```

The supplied command will run only when an actual adapter exists.
Without it, `REFERENCE_ADAPTER_NOT_CONFIGURED` is an intended failure.

## Standalone candidate inspection

If a paper-derived backend eventually produces a candidate NPZ, inspect it
without importing or modifying any Streamlit launcher:

```bash
cd onestring-physics-simulator
python3 scripts/audit_zhang2021_candidate.py \
    --source /path/to/optcuts_source.npz \
    --candidate /path/to/candidate_quads.npz \
    --report /path/to/audit.json
```

This CLI **audits** supplied quads, it does not generate them. The report
explicitly says `zhang2021_algorithm_verified: false` even after success.

## Tests and CI

```bash
cd onestring-physics-simulator
python3 -m pip install -e . pytest shapely
python3 -m pytest -q tests/test_zhang2021_reference_bridge.py
```

Workflow: `.github/workflows/zhang2021-independent-20261027.yml`.
It triggers only on this independent branch. It uses an explicitly
identified synthetic rectangle to test protocol, never mislabels that
geometry as a Zhang algorithm reproduction.

## Safe worktree checkout (does not switch Claude's local working directory)

```bash
git fetch origin
git worktree add --detach ../onestring-zhang2021-independent \
  origin/agent/zhang2021-independent-20261027
```

Never run `git reset --hard` or switch away from Claude's active branch.

## Next genuine research milestones

1. Obtain the C++ Code/Supplement from relocated USTC hosting or author;
   inspect repository, license, dependencies, build, exact domain I/O.
2. Run *unchanged* original reference on its own samples; record meshes
   and timings. If unrecoverable, say so; do not silently substitute.
3. Build a real, versioned native-format-to-NPZ adapter.
4. Record the original backend executable or source fingerprint.
5. Use the existing bridge to verify real OptCuts Omega and M2D->M3D.
6. Add synchronized cut-side subdivision as constraints **during**
   layout generation. Audit is necessary but insufficient for this.
7. Independently redesign row/column-based CSF Split for irregular
   quad adjacency. The current regular-grid Split is NOT certified
   for this mode.
8. Only after verified standalone and seam tests, add a separately
   named selection to the UI and test K3D->T3D->K2D->T2D.

Do not report full pipeline success from the standalone bridge tests.
