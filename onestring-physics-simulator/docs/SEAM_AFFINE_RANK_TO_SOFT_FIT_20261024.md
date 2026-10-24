# Seam rank collapse: implementation correction (2026-10-24)

## Root cause verified

The external 10-page assessment titled "OneString K3D Seam制約の理論的成否と修正方針" identifies a structural flaw in the 2026-10-24 hard affine cut-stitch implementation. The original `source_cut_affine_rows()` imposes

    (1-s_A(t)) V_A0 + s_A(t) V_A1
      = (1-s_B(t)) V_B0 + s_B(t) V_B1

at the union of both Grid breakpoints along each physical source cut. When the two sides use incompatible, non-affinely related parameterizations, exact equality of their piecewise-linear polylines can force the entire seam to a straight line or even a single point. The affine nullspace can have only one free scalar per 3D coordinate (translation): the hard projector then collapses the seam **before** the SmoothMax optimizer executes.

This explains why `source_cut_gap=2.2e-16` and `plane_distance=inf` can occur simultaneously. The former is perfect satisfaction of the *wrong* constraint set; the latter indicates a degenerate quad reference triangle. The earlier signed-area barrier and K3D restart cannot enlarge the feasible set of those hard constraints.

The mathematically correct statement is conditional: not every nonmatching partition necessarily collapses to a point, but exact pointwise equality of two independently parameterized polygonal chains severely restricts curvature; an eight-source-edge synthetic witness demonstrates the actual low-rank failure.

## Implemented (Plan B: safe immediate restoration)

1. Keep proven topological source cut/open correspondences and preserve `source_cut_affine_rows()` for matching. M2D tile selection, Ω mapping, original source triangle mesh and Grid remain unchanged.
2. Compute rank/nullity of the constraint matrix restricted to participating seam Grid vertices when <=200 vertices. Log `seam_stitch_mode` and `affine_rank_diagnostics`.
3. **Only fully matched** Grid seam edges without mixed cut/open cells or mismatched source partition can use the legacy exact hard equality. Other valid physical cut pairings are explicitly `mode=soft`; an evaluated low-rank constraint set also disqualifies the hard route.
4. K3D AL now optimizes the objective

       E = E_surface/anchor + E_planarity_AL + E_SmoothMax
           + E_reference_shape + E_oriented_area
           + 0.5*w_seam/(m*h²) ||C X||²

   with analytic gradient `w_seam/(m*h²) C^T C X`.
   Crucially, **NO hard affine projector**, `project(strict=True)`, or exact `C X=0` acceptance check occurs on the soft path.
   `w_seam` defaults to 30 (override
   `ONESTRING_K3D_SEAM_SOFT_WEIGHT` or optional params field).
5. After K3D, soft mode checks actual seam residual stats and stores
   `seam_soft_gap_rms/max` and `seam_exact_closure_enabled=False`.
   It never claims exact closure. Existing source-open regions stay open.
6. The T3D Eq.(2) top/bottom system includes the same soft rows as
   additional least-squares equations. It **does not project** its
   vertices and does not demand a 1e-9 gap. Logs
   `t3d_seam_stitch_mode=soft`,
   `t3d_seam_soft_top_rms`, `_bottom_rms` and max values.
   Its solved top is still promoted to final K3D, so this downstream
   residual matters and is recorded independently.
7. A source-Seam quotient run with reduced vertex IDs reindexes
   linear rows and exact groups, instead of incorrectly applying
   source M2D vertex IDs to the reduced K3D mesh.

## Regression coverage

The added `tests/test_optcuts_seam_nonaffine_rank_soft.py` constructs:
- 8 physical source-cut edges (not the prior 1-edge-only test);
- A-side and B-side with 2 vs 3 Grid segments and a deliberately
  nonaffine source-arc-length UV parameter relationship;
- original hard `C` rank/nullity and the resulting projective
  edge collapse as a *negative witness*;
- analytic soft least-squares gradient vs finite differences;
- soft K3D AL + SmoothMax with noncollapsed source-triangle area,
  a measurably smaller seam residual, and no hard equality;
- an explicitly soft T3D extrusion with no hard projection and
  noncollapsed promoted K3D quad geometry;
- remapping of soft rows under reduced geometry vertex IDs.

All native Bunny 947-quad results still require validation with the
actual OptCuts binary on the user's Mac. Passing synthetic tests does
not establish a global energy optimum or manufacturing clearance.

## Not implemented in this change (Plan A)

The final desired *exact and curved* geometric source-Seam closure
requires making both sides of each physical cut use common breakpoints
before K3D, by modifying Ω boundary Grid fitting in
`optcuts_area_grid_boundary.py` and its UV seam chain metadata. If
the required boundary corner counts cannot match without changing
the quad lattice, the Grid must be refined/split or its topology
changed. Do **not** restore hard affine projection on unmatched
partitions: it will reproduce the collapse regardless of optimizing
weights, area guard, or AL iteration count.

The paper's suggested `O(h² κ)` residual threshold is an asymptotic
approximation estimate under regular smoothness and bounded curvature,
**not** an automatically validated numerical threshold without
measuring mesh scale and curvature. Here we record the measured
residuals first and make no unsupported closure guarantee.

## Verify locally

    git pull --ff-only origin fix/optcuts-real-m2d-footprint-20261023
    python3 -m streamlit run app_optcuts_20260923_extrusion_aware.py

Expected messages for nonconforming original source seams:

    [SEAM-K3D-PREFLIGHT] ... mode=soft rank=... nullity=... AL_hard_equality_ready=False
    [K3D-SEAM-SOFT] ... projector=DISABLED exact_seam=False
    [SEAM-K3D-SOFT] ... rms=... max=... no hard projection
    [PAPER-T3D-SEAM-SOFT] ... projector=DISABLED

If an AL hard projection or `SEAM_K3D_AFFINE_SMOOTHMAX_PLANARITY_NOT_CONVERGED`
still occurs in soft mode, inspect the runtime branch and older
installation flags; the active execution route did not reach the new
soft-mode path.
