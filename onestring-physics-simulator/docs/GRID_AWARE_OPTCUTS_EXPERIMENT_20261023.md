# Grid-aware OptCuts: first controlled UV-stage experiment (2026-10-23)

## Purpose and scope

Question: Given the **working 2026-09-18 OptCuts Omega + paper-aligned K3D/K2D** setup,
can we reduce the damage from seams crossing the regular M2D grid?

This branch is an opt-in experiment, **not a replacement for the successful baseline**:
`feature/grid-aware-optcuts-20261023`.

UI (revised: Grid-aware is its own Omega parameterization mode):
- **Paper mode: OFF**
- **Omega parameterization mode**:
  - existing `optcuts_test2`: successful 2026-09-18 paper-aligned baseline;
  - **new `optcuts_grid_aware`** shown as
    **2026-10-23 | Grid-aware OptCuts Ω + paper-aligned K3D/K2D (UV refinement)**.
- **No separate enable checkbox.** Selecting the new Omega entry activates
  the refinement; returning to `optcuts_test2` restores the original route.
- Under the new Omega mode, expand **Grid-aware OptCuts のUV設定（実験）**
  to adjust strength (default 3.0).
- The visible mode is mapped to the same internal `optcuts_test` carrier as
  `optcuts_test2` with `ONESTRING_OPTCUTS_TEST_VARIANT=2`, retaining the
  successful K3D/K2D and strict full-cell M2D crop behavior.
  A per-session selection flag becomes
  `PipelineParameters.optcuts_grid_aware_enabled`, so the OptCuts
  backend can activate its UV-only extra stage without changing later stages.
  The metrics `requested_omega_parameterization_mode` and
  `optcuts_grid_aware_selected_as_omega_mode` retain the actual UI choice.
- The original `optcuts`, `optcuts_test`, `optcuts_test2`, native
  `optcuts_grid`, and all Paper/BFF modes remain available unmodified.

The `optcuts_test` entry wraps its official OptCuts backend using
`replace(params, omega_parameterization_mode="optcuts")` in
`optcuts_test_boundary_reparameterization_patch.py`, so the same
post-OptCuts UV experiment runs there *before* downstream paper-aligned
M2D clipping. It does not affect `optcuts_grid`, Paper mode, BFF or LSCM.

## Actual algorithm (do not call it full joint Grid-OptCuts)

1. Run the **unmodified official OptCuts executable**, which jointly selects
   a seam topology and UV with its original Symmetric Dirichlet/length tradeoff.
2. Perform the existing **CSF-tail UV refinement**, also with fixed seam topology.
3. When the new toggle is ON, weld exact 3D geometric duplicates only for
   identifying corresponding **UV copies of each internal OptCuts seam edge**;
   never weld or reindex their actual UV vertices.
4. Determine the downstream free-boundary grid density
   `h_u=span(U)/(2*nx)`, `h_v=span(V)/(2*ny)`
   (matching `_rebuild_domain_overlay_for_general_omega`).
5. For each seam edge/copy, choose its initial dominant grid direction and
   target the nearest **single constant-u or constant-v lattice line**.
   The two UV sides use **separate** target lines to preserve seam gaps.
   At junctions, conflicting soft targets are blended, never hard-overwritten.
6. Compute a smooth UV displacement by solving a sparse
   **uniform graph-Laplacian + anchored seam-coordinate** linear system.
   Test fractions of the proposed displacement:
   `0.125, 0.25, 0.5, 0.75, 1`.
7. Reject any candidate that changes triangle orientation, collapses triangles,
   causes triangle overlap, increases mean area-weighted SD by >0.5%,
   exceeds the configured mean SD bound if the baseline already met that bound,
   or reduces an approximate fully-contained quad-cell fraction by >1 percentage point.
8. Accept the best **actually improving** valid alignment candidate,
   otherwise **keep the exact baseline OptCuts UV**. The cut topology and
   face-wise S/UV vertex correspondence do not change.

Normalized per-seam alignment energy:

```text
E_align = weighted mean over existing seam UV edge copies of

  min {
    (delta_u / h_u)^2 + distance_to_nearest_vertical_lattice_line(mid_u)^2 / h_u^2,
    (delta_v / h_v)^2 + distance_to_nearest_horizontal_lattice_line(mid_v)^2 / h_v^2
  }.
```

Complete-grid-cell *yield proxy* counts bounding-rectangle cells with four
corners and a center inside the OptCuts boundary polygon. It is **not**
an exact M2D validity / quad retention count. It caps sampling at 48x48
for performance. Production M2D clipping and inverse-lift validity remain
authoritative. The grid currently can have `h_u != h_v`, so this is grid
**alignment**, not yet a mathematically guaranteed equal-square lattice.

## Comparison and tests

1. Keep the **identical input, grid size, OptCuts executable, initial cut,
   distortion bound, K3D/K2D settings**.
2. Select `optcuts_test2` as the baseline; save Omega/M2D/Split/K2D.
3. Change **only** Omega mode to `optcuts_grid_aware`; inspect
   **View stage > Omega > Grid-aware OptCuts 実験の結果**, compare:
   - `optcuts_grid_aware_status` (accepted or rejected),
   - `optcuts_grid_aware_alignment_before/after` (lower better),
   - `optcuts_grid_aware_cell_yield_proxy_before/after` (higher better),
   - mean Symmetric Dirichlet before/after (preserved within guardrails),
   - actual M2D retained panels, K3D/T2D final seam appearance.
4. The experiment is a **candidate study**, not yet a modification to the C++
   topology optimizer. The latter would additionally need cut/merge proposal
   scoring under Grid constraints and potentially mesh-edge subdivision.

Code:
- `src/onestring_physics/optcuts_grid_aware_refine.py` (metric and harmonic step)
- `src/onestring_physics/optcuts_backend.py` (after existing CSF-tail, on each start)
- `src/onestring_physics/optcuts_pipeline_patch.py` (actual grid resolution and gating)
- `app_runtime_base.py` (toggle and Omega report)
- `tests/test_optcuts_grid_aware_refine.py` (gating, a valid improving seam,
  topology, guards)

Safety: existing `optcuts_grid` C++ study, standard `optcuts` with toggle OFF,
Paper/BFF, and all K3D/K2D/T2D solvers are unchanged.
