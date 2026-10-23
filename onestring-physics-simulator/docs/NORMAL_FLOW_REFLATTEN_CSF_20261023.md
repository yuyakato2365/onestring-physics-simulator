# Reflatten tile-area CSF in the normal (Paper OFF) flow

This change reuses the **existing Paper-oriented `reflatten_tile_area` engine unchanged** for the normal OneString pipeline. It is an experimental reconstruction and is not verified author code.

## UI
1. Turn **Paper-oriented computation mode** OFF.
2. Choose the usual target mesh, Omega parameterization and grid (for example, the existing `rectangular_harmonic_legacy` mode). Note: harmonic/LSCM initial maps are **not** official BFF, even though reflatten assessment uses official free-boundary BFF.
3. Select **Split CSF判定（Paper / 通常フロー共通）** -> **3Dピース自由境界BFF再展開 → タイル面積比 ≤ 2**.
4. Supply an official free-boundary `bff-command-line` binary via the existing **official bff-command-line path (nonrectangle)** field (or the existing environment variable).
5. Run the existing OneString pipeline. `fixed_uv` remains the default in Paper OFF with non-reference Omega, so old experiments are not silently changed.

## Shared algorithm
For each current 3D source patch:
1. Compute a new official *free-boundary* BFF parameterization of that patch.
2. For each original M2D grid quad, lift its four sampled corners onto the original 3D source and transfer them into the new patch UV using barycentric coordinates.
3. Compute `J_q=A3D_q/A2D_q(reBFF)` and `sigma_P=max_q J_q / min_q J_q`.
4. A patch needs a Split if `sigma_P>2`. Candidate complete grid-row or grid-column cuts pass through a smoothed-Gaussian-curvature peak. Compute official free-boundary BFF again for each candidate child and choose the direction minimizing the worse child `sigma_P`, unless manually overridden.
5. Add all planned cuts to actual M2D topology, preserving existing downstream layout, 3D optimization, and independently accepted collision-driven cuts.

## Safety and interpretation
- The reflatten UVs are **for scoring only** and are **not** used to reshape the final manufacturing M2D tiles. A passing score is not a physical deployability or collision-free certificate.
- Any initial Omega mapping must be finite, nondegenerate and locally unfolded. The normal flow refuses locally flipped/degenerate charts; it never silently repairs them.
- The initial S->Omega method is recorded as `csf_split_initial_omega_method`. With harmonic/LSCM Omega, it is explicitly *not* a literal Paper reproduction.
- `sigma_P` is an **area-span ratio**, not a linear conformal factor `sqrt(sigma_P)`; the threshold 2 retains the previous Paper experimental convention.
- The reflatten calculation requires the native official BFF CLI. No fallback to harmonic mapping is permitted on failed child BFF.
- Imported meshes are checked as single connected disk-topology triangular surfaces when this option is active.

## Regression tests
`python -m pytest -q tests/test_paper_split_reflatten.py`

The tests check dispatch from both Paper ON and Paper OFF, UI selection/default, and collision-driven seam preservation with native BFF mocked. Full end-to-end native-BFF validation on a representative mesh remains necessary.

## Verifying the downstream pipeline with a forced cut

A **zero-Split result is valid** when the initial BFF area span already satisfies the threshold. The Split decision can be inspected in the Results view as `max_csf_before_split` and `csf_split_step_count`.

To exercise the full production pipeline when there is no naturally required cut:

1. Keep `Split CSF判定` set to `reflatten_tile_area` and choose a disk-topology input mesh with several interior grid lines (`grid size` around 8–14 for a quick test).
2. Enable **検証用：CSFが2以下でもSplitを1本入れる** in the BFF settings.
3. Run the normal pipeline. The planner uses its regular curvature-guided complete grid-line candidates and computes each child's area-span by official free-boundary BFF, even if the initial span was <=2.
4. Check the Split process view, M2D/M3D stage views, K3D, K2D, T3D, T2D and any downstream errors. The UI labels the cut as **verification-only**. It is NOT evidence of an actually necessary Split.
5. Turn the checkbox OFF before collecting research results. The natural CSF policy and threshold are not modified.

The stored report explicitly distinguishes:
- `csf_split_validation_force_one_cut_requested`: debug control was enabled
- `csf_split_validation_forced_count`: number of cuts deliberately inserted despite the parent already passing CSF
- `reflatten_split_report.steps[].validation_forced`: per-cut debug provenance
- `max_csf_before_split` and `max_csf_after_split`: real, unmodified BFF area-span measurements

If the geometry has no complete admissible internal grid line, or if the Split budget is zero, **no cut is generated**; the UI shows a warning instead of fabricating a successful result.

**Scope:** The debug mode verifies that the Split-to-downstream execution path is reachable and surfaces real failures. A unit test verifies a geometrically flat `sigma=1` case can yield one *explicitly marked* debug cut. It does not by itself prove mechanical assembly or that any natural Split is required.


## ReBFF ON/OFF comparison (2026-10-23)

The **Split CSF判定** menu has an area-span option and a separate checkbox
**SplitのCSF判定で各3D領域を再BFFする**:

- **ON** (`paper_split_criterion="reflatten_tile_area"`): each candidate 3D piece
  is independently reflattened with official free-boundary BFF, and the
  3D-to-new-2D quad area span determines its Split score. This is an
  *evaluation-only* UV; downstream M2D still retains the original grid.
- **OFF** (`paper_split_criterion="fixed_chart_tile_area"`): the exact same
  curvature peak, full grid-line candidates, threshold (`max(J)/min(J) <= 2`),
  direction chooser, collision-cut handling and final M2D topology operations
  are used. The only difference is that the piece inherits the original
  S-to-Omega UV coordinates through triangle clipping. It does **not** call
  BFF when evaluating candidate source patches, and does not independently
  rescale or reparameterize the child patch. An **initial Omega mode that uses
  BFF still needs BFF for its initial Omega creation**.
- **Legacy fixed_uv** (the other menu option): previously implemented
  area/linear stretch test, **not the same formula**. Do not use it as the
  one-variable reBFF ON/OFF comparison.

In both new modes, for each sampled M2D tile `q`, `J_q=A3D_q/A2D_q` and
`sigma_piece=max(J_q)/min(J_q)`. The 2D chart differs; the 3D geometry,
quad IDs, scan of curvature peaks, choice of complete row/column cuts, and
physical seam-duplication code are the same.

The evaluation chart must be finite, non-flipped and non-degenerate.
Both modes support the **検証用：CSFが2以下でもSplitを1本入れる** option.
The report stores `csf_split_evaluation_uv`,
`csf_split_bff_evaluation_enabled`, `csf_split_criterion`,
`tile_area_split_report`, and the existing compatibility report
`reflatten_split_report`; per-cut metadata includes
`fixed_chart_area_cut` or `reflatten_csf_cut`.

Neither a small reBFF score nor a small fixed-chart area span alone proves
global conformal distortion, physical deployability, or completed T2D
assembly. The area span on an anisotropic initial Omega map is **not**
a guarantee on the linear CSF.

To compare fairly, keep the input surface, Omega solver, grid size,
grid rotation, curvature rings, Split budget, and collision feedback settings
the same, then toggle only reBFF, rerun and compare
`max_csf_before_split`, `csf_split_step_count`,
`max_csf_after_split`, cut lines and downstream failure diagnostics.
