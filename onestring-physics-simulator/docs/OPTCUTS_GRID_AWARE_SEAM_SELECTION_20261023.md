# Experimental Grid-aware OptCuts seam candidate selection (2026-10-23)

Branch: `fix/optcuts-real-m2d-footprint-20261023`

## Scope and limitations

This is **approach A, first-stage implementation**: rank genuine OptCuts
initial-cut candidates by the shape of the resulting **actual M2D grid**
footprint, instead of ranking only original OptCuts UV distortion/CSF.
It does **not** yet modify the C++ OptCuts cut-placement objective. The current
official CLI offers initial_cut_option 0/1, so closed genus-0 inputs normally
provide **at most two distinct initial seam/topology candidates**. Open surfaces
and unsupported meshes may provide a single candidate. There is no guarantee
that two starts have distinct final cuts or that either admits the target
bijective stair-boundary mapping.

## Pipeline

1. Run the official OptCuts configurations already provided by
   `run_official_optcuts`, including the alternative initial cut when valid.
2. For **each real official UV/3D paired candidate**, call the original
   `_flatten_to_domain` and extract its real overlay:
   `origin, (dx,dy), overlay_nx, overlay_ny, margins`.
3. Compute exact per-cell overlap area `Area(cell ∩ Omega0)/Area(cell)`.
   **Do not modify the threshold** (default 0.50), rephase the grid, add
   panels, or use a separately inferred overlay.
4. Construct the selected-cell outline, refusing disconnected, holed,
   pinched, or otherwise invalid topology.
5. Measure `N_reflex` (270-degree interior corners), `N_corners`,
   `L_grid` (outline length expressed in x/y tile-step units), and `N_tiles`.
6. Rank source-valid candidates by

   ```
   E_grid = (2.5*N_reflex + 0.4*N_corners + 0.08*L_grid) / sqrt(max(1,N_tiles))
   E_quality = 1.5*max(0,log(SD_mean/SD_best))
               + 2*max(0,CSF_over_2_area_fraction)
               + 0.2*max(0,log(seam_length/seam_best))
   score = E_grid + E_quality
   ```

   Source candidates with invalid UV geometry or original mean SD greater
   than `max(1.5*SD_best, SD_best+0.25)` are rejected. Weights are experimental
   heuristics, **not** paper-specified or calibrated for Bunny.
7. Try full exact-grid-boundary reparameterization in score order, without
   changing each candidate's original lattice and selected-cell mask.
   If one fit fails, report its failure and try the next official
   **different source candidate**. The fit must pass orientation, overlap
   and SD checks. If every fit fails, raise
   `GRID_SEAM_ALL_RANKED_OFFICIAL_CUTS_FAILED`, never silently reverting to
   unmodified Omega.
8. Copy the **complete paired chart** from the selected candidate back into
   the original `SurfaceParameterization` object referenced by the
   OneString pipeline. Clear unfitted alternative charts to prevent a
   downstream multistart swap. Restore the exact pre-fit M2D lattice
   when recomputing derived CSF information.

## Diagnostics

The Omega stage records and displays:

- `optcuts_area_grid_seam_ranking`: original-start index, source SD, CSF
  tail, seam length, grid angles, reflex corners, perimeter in grid units,
  retained tiles, normalized composite score, fitting status/error.
- `optcuts_area_grid_seam_selected_start` and
  `optcuts_area_grid_seam_selected_score`.
- All previous fixed-grid / area-threshold / boundary-fit metrics.

The new mode only is affected. Normal 2026-09-18 OptCuts, regular OptCuts,
Paper mode, M3D/K3D/K2D/T2D solver equations, and existing fabrication
checks are not modified.

## Validation

CI: `pytest -q tests/test_optcuts_area_grid_boundary.py`, the larger
existing regressions, and Python compilation on GitHub Actions.
Tests include a nonuniform source boundary, a rotated versus axis-aligned
official-candidate fixture, quality penalty, and a forced failure of the
first candidate followed by a second accepted candidate.

**Not validated** in these synthetic tests: actual native OptCuts generation,
Bunny, Lilium, final M3D/K3D/T2D output, and whether either official start
avoids the reported 99.67% UV fitting failure. The ranking may favor an
outline with fewer corners yet have a worse true minimum Jacobian. Assess
those metrics on the real input before claiming research success.


## 2026-10-23 correction after valid_seam_candidates=1 failure

The initial candidate-ranking implementation did **not** make new seams, and
could leave only one valid input. A single remaining chart reproduced the
original `last_valid_fraction ≈ 0.98669` boundary fitting error.

The amended experimental mode therefore:

1. Records the real number of official OptCuts source seam candidates
   separately from grid candidate orientations, including the existing
   OptCuts multistart skip/failure status.
2. Searches rigid pre-grid UV rotations of **0, 22.5, 45, 67.5 degrees**
   by default. The same official seam topology is kept at each rotation.
   For each rotated chart, obtain the real production lattice for that
   chart **before** selecting the area-threshold panels; that lattice is then
   fixed exactly. There is no after-the-fact grid translation, rephasing,
   scaling or seam movement.
3. Retains a valid secondary seam source even when its original SD is
   >1.5x the best source: the existing lexicographic ranking penalizes
   larger SD, and the final inverse map's hard checks still reject excessive
   distortion, overlaps, inversion or degeneration.
4. On failure, reports the `official_cut_sources`,
   `official_multistart_status`, `evaluated_uv_orientations`, and the
   tested cut+rotation combinations, without claiming a mathematical proof
   of impossibility.
5. Exposes a Streamlit checkbox to opt out of the extra orientations.
   It is included in the pipeline cache key.

**This is a UV orientation experiment, not a C++ Grid-aware cut optimizer.**
With one official OptCuts source there is still only **one seam topology**.
The added orientations only change its alignment with the original production
M2D grid and therefore its 50%-cell staircase. If every combination fails,
the next research step is genuinely generating/modifying cut topology within
OptCuts (or a separate cut graph optimizer), then reparameterizing the newly
cut surface and confirming inverse-mapping topology and distortion.

Native Bunny failure resolution is **not verified by the current CI**.
