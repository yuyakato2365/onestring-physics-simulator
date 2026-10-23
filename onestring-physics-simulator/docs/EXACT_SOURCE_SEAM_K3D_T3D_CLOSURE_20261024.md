# OptCuts source-Seam: exact K3D closure / T3D extrusion (2026-10-24)

## Requirements and what the implementation actually supports

- Work only on the experimental `optcuts_test` carrier with an **accepted**
  `optcuts_area_grid_boundary_enabled` parameterization. Do not change
  ordinary OptCuts, native Grid-OptCuts, original source surface, M2D selected
  mask, or the original cut topology.
- Identify *true* original OptCuts cuts from the pair of UV realizations of
  each common physical source triangle edge; do NOT group spatially close
  but unrelated 3D vertices. Detect genuine source-open edges separately.
- Inspect the **actual resulting M2D boundary edges**, not the source seam
  picture only. Project their UV endpoints onto original UV cut segments
  and recover physical source seam points by interpolation. Each physical
  seam segment must be represented on both sides. The initial method
  currently requires identical physical edge endpoint samples on both sides.
- If **nonmatching Grid subdivisions** make that impossible, raise a
  `SEAM_K3D_NONCONFORMING_GRID_EDGE_SAMPLING` error with counts; do not
  silently make panels disappear, bend source data, or declare "closed" from
  just nearest-neighbor pairs. Likewise, unclassified M2D boundary edges
  trigger `SEAM_K3D_UNCLASSIFIED_GRID_BOUNDARY`.
- For paired, conforming seam samples, solve exactly shared 3D K3D position
  unknowns through equality-variable elimination. Reproject quad faces
  through a planarity projection while maintaining equality. Record
  before/after max physical seam gap.
- The Paper Sec.4.2 T3D solver uses the **same paired vertex groups** for both
  its top and bottom unknowns. Shared vertex normals are averaged before
  normal offset; both top and bottom vertex identities are eliminated in
  the local/global Eq.(2) least-squares solve. The resulting top is
  promoted to final K3D exactly as before. Therefore T3D cannot reopen a
  *matched, conforming* seam merely by planarity optimization.

## Full-area limitations

Exact coincidence of all vertices in *some* pairs is **not** proof that
all seam edges coincide. Most Bunny/Grid configurations can create
different sampling patterns on the two sides, and not all such cases can
be made geometrically identical with only a handful of vertex welds.

In general exact seam closure with noncorresponding edge segmentations
requires either:
- a *certified* geometric correspondence (e.g., both splines are proven
  straight/collinear and cover the same physical segment even with
  different vertices), or
- modifying the **M2D panel topology**/subdivision on the seam before
  K3D, while preserving the manufacturable quad-panel constraint,
  and then redoing M3D inverse mapping.

The current guard is conservative: it does not automatically use the
collinear nonconforming special case. This is intentional so a run never
reports full seam closure without a geometric equivalence proof.

Hard K3D seam fitting might compete with surface proximity, squareness,
quad planarity, and T3D finite-thickness collision avoidance. Bunny
native end-to-end results are NOT yet verified. Passing synthetic
tests only proves the paired-edge constraint is correctly wired.

## Expected runtime logs

```
[SEAM-K3D] exact_groups=... matched_edges=... gap_before=... gap_after=0 quad_planarity_after=...
[PAPER-T3D] ... seam_top=0 seam_bottom=0 ...
```

Errors indicating the mesh needs a *conforming seam boundary* stage:

```
SEAM_K3D_NONCONFORMING_GRID_EDGE_SAMPLING:
  unmatched_grid_boundary_edges=... of ...
SEAM_K3D_UNCLASSIFIED_GRID_BOUNDARY:
  edges=... sample=...
```

These are not errors to bypass by changing the matching tolerance.
The corresponding Grid/UV boundaries must be reconciled upstream.

## Code

- `src/onestring_physics/optcuts_seam_exact_closure.py`
- `src/onestring_physics/paper_extrusion.py`
- `app_optcuts_core_20260916.py` (last/exterior hook)
- `tests/test_optcuts_seam_exact_closure.py`
