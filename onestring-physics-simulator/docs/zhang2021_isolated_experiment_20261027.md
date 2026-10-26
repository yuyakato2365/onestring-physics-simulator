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
