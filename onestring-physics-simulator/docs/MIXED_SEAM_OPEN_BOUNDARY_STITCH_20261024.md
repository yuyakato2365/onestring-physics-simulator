# Mixed source-Seam / original-open boundaries on one Grid edge
Date: 2026-10-24

## Verified root cause from Bunny log

```
edge=[425,426], cut_coverage=0.64445591, open_coverage=0.35554409,
                 all_source_boundary_coverage=1.0
edge=[772,815], cut_coverage=0.94897182, open_coverage=0.05102818,
                 all_source_boundary_coverage=1.0
```

This **does not** show a Grid/Ω mismatch: every M2D Grid edge
lies on the original UV source boundary. A true physical OptCuts
two-sided cut *terminates inside a whole quadrilateral boundary edge*;
the remaining fraction is a genuine one-sided **original source-open
surface edge**. Treating an entire M2D edge as either 100% paired
source cut or 100% physical opening was mathematically wrong and
caused `SEAM_K3D_UNCLASSIFIED_GRID_BOUNDARY`.

## Implemented fix (no artificial M2D cutting)

1. Original paired source UV chart cut edges are identified via two
   topological boundary incidents of the **same physical source edge**.
   One-incidence source edges are *real physical open boundaries*.
2. Each M2D Grid boundary edge intersects each source cut edge in
   the source UV chart. Its cut fraction and original open fraction are
   recorded separately. If their union accounts for exactly 100% of
   the Grid edge, it is a valid **mixed** boundary edge, not an error.
   Missing total coverage still fails.
3. For each paired source cut 3D edge, build parametric intervals on
   each UV copy, independently of original Grid corner subdivision.
   They MUST cover both sides of the full physical source cut. Construct
   linear equations at all source/target Grid partition breakpoints:

       (1-sA)*V_A0 + sA*V_A1 =
       (1-sB)*V_B0 + sB*V_B1

   The variables V_A* and V_B* are unchanged four-corner K3D quad
   vertices. Correspondence is by the **same original physical 3D
   source edge**, not nearest-distance matching. These equalities force
   the two entire piecewise-linear seam boundaries to coincide, even
   when one original source endpoint lies inside a Grid edge.
4. `AffineSeamProjector` projects K3D coordinates and analytic
   SmoothMax gradients onto the exact nullspace of these affine linear
   equations. The K3D Augmented Lagrangian plane-restoration steps
   apply the SAME projection. No priority weighting can soften the
   source-cut equality.
5. On the authoritative final K3D, verify maximum physical seam
   constraint residual. If the planarity and affine constraints cannot
   jointly converge under current constraints/AL budgets, fail with
   `SEAM_K3D_AFFINE_SMOOTHMAX_PLANARITY_NOT_CONVERGED`, including
   numerical residuals rather than pretending success.
6. T3D projects both normal-offset top and bottom vertex positions
   onto the corresponding doubled affine equality set. Apply this
   after every Paper Eq.(2) local/global face-planarity solve and
   audit the final constraint residual, before promoting the solved
   top layer back into K3D.

The **real original open boundary fraction** remains open. There is
no fabricated opposite sheet and no attempt to turn a genuine
physical source hole into a closed surface. To close that physical
opening, the source mesh topology/geometry must explicitly be repaired,
which is a *different* operation from OptCuts seam closure.

## Tests and remaining limitations

Synthetic tests explicitly cover:

- A genuine source cut that terminates at 40% along an unsplit Grid
  quad edge (60% seam, 40% real opening)
- Both user-reported cases: 64.445591% / 35.554409% and
  94.897182% / 5.102818%
- Straight source physical 3D seam classification, exact vertex-on-
  OTHER-edge coupling and different Grid edge partitions
- SmoothMax enabled, affine constraints retained through hard
  planarity-restoration sweeps
- T3D top AND bottom exact physical seam equations and final K3D promotion

Bunny with the actual OptCuts binary is **not available in CI**,
so success on the user's 16k-triangle model is not yet established.
The constraints guarantee geometric coincidence of *source cut*
piecewise-linear polylines if the K3D/T3D solver converges. They do
not invent a shared topological mesh edge or guarantee manifold
connectivity at a T-junction with different tessellations; this would
require a topology-aware panel splitting/remeshing step. Nor does
this fix the independent 4-corner curved-surface approximation error.

## Runtime proof

```
[SEAM-K3D-PREFLIGHT] ... affine_rows=... mixed_seam_open_edges=2
[K3D-SEAM-AFFINE] enabled=True ...
[SEAM-K3D] unified_affine_SmoothMax_AL_feasible ...
[PAPER-T3D] ... 
```

Check metrics:

- `seam_closure_mixed_seam_open_grid_edges == 2`
- `seam_k3d_affine_constraint_residual < 1e-9 * scale`
- `seam_exact_source_open_edge_count` (real open Grid edges)
- `seam_exact_full_shape_closed` (False when a genuine open portion remains)
- `t3d_seam_affine_max_residual < 1e-9 * scale`
- `t3d_final_k3d_smoothmax_energy_change`: T3D can still
  alter SmoothMax; this field is an honest audit, not a guarantee
  that the 3D square energy never increases.

## Modified modules

- `optcuts_seam_affine_constraints.py`: provenance-based source-
  edge/Quad-edge intersection, linear constraint row builder, robust
  sparse affine orthogonal projection
- `optcuts_seam_exact_closure.py`: accepts completely accounted-for
  source cut/open mixed Grid edges; requires exact physical source-cut
  coverage on both sides; validates final K3D
- `optcuts_test_k3d_augmented_lagrangian_patch.py`: SmoothMax,
  hard-planarity and source-cut constraints optimized together
- `paper_extrusion.py`: applies identical affine constraints to
  top and bottom normal-offset faces and all planarity iterations
- `tests/test_optcuts_seam_mixed_boundary_affine.py`: exact
  mixed-edge synthetic geometry, AL and T3D regression.
