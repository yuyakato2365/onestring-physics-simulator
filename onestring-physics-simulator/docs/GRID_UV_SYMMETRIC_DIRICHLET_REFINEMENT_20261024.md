# Low-distortion exact-Grid Ω refinement (2026-10-24)

## Motivation

On the Bunny chart (15,982 source triangles, 8,222 UV vertices), the
ear-collapse recovery reached 100% of the prescribed Grid staircase but
then rejected the resulting UV for excessive 3D-to-UV symmetric Dirichlet
(SD) distortion: baseline area-weighted mean SD ~4.0999; required mean SD
<=5.3298 (unchanged 1.3x rule). For the first 67.5-degree candidate,
the fixed-boundary harmonic map had mean SD 8.6629.

The existing harmonic projection minimized a vertex-position energy.
Its rescue untangler minimized vertex displacement plus an oriented-area
penalty. **Neither minimized symmetric Dirichlet distortion itself.**
Trying numerous corner correspondences without directly optimizing SD
could thus reject a good orientation needlessly.

## Implemented numerical refinement

On every correspondence that reaches the *full* Grid outline, is locally
non-inverted and globally overlap-free, but fails the original SD limit:

1. Construct the exact per-triangle 3D source metric G from the paired
   original 3D faces and source UV faces (the exact same face pairing
   as OptCuts quality reporting).
2. Minimize the area-weighted symmetric Dirichlet energy

       E_SD(U) = Σ_t (A_t / Σ A) [ ||J_t||_F² + ||J_t^-1||_F² ]

   using the 2x2 Gram matrix identity, without numerical SVD gradients:

       ||J||_F² = tr(U G^-1 U^T)
       ||J^-1||_F² = ||J||_F² det(G) / det(U)²

   UV gradients are analytic and checked against central differences.
   The numerical energy is compared with the official
   `optcuts_quality_audit.quality` on the initial candidate and cannot
   silently optimize a different quantity.
3. **Stage 1:** L-BFGS-B adjusts *only interior UV vertices*. The entire
   boundary remains unchanged, and the existing footprint is untouched.
4. **Stage 2:** if SD is still over the original limit, L-BFGS-B adjusts
   interior vertices plus non-corner boundary vertices along the
   *original corresponding Grid edge*. Each scalar coordinate receives
   a conservative bound based on the midpoint to each neighbor, so no
   boundary vertex can change to another edge or cross a neighbor.
   Every original Grid corner remains exactly fixed.
5. Independently check the **existing** signed-area ratio >=0.005
   relative to the original OptCuts UV, no flips, no degenerate faces,
   no positive-area UV overlaps and no missing Grid corners.
   Maintain a copy of the best fully audited lower-SD candidate.
   Never adopt a bad numerical iterate or relax the 1.3x SD limit.
6. If this candidate passes the original SD limit, accept. Otherwise
   continue trying the remaining corrected boundary correspondences,
   and then the remaining UV orientations, exactly as before.

The production M2D uses the same frozen >=50%-area-overlap mask as the
target Grid before fitting. No regridding or source-Seam changes.

### New logs

```
[GRID-BOUNDARY-FIT] correspondence=1/12 ...
[GRID-UV-CONT] complete; steps=4 ...
[GRID-SD] improved phase=interior mean_sd=... limit=5.3298 ...
[GRID-SD] phase_done=interior ...
[GRID-SD] phase_done=grid_edge_slide ...
[GRID-SD] correspondence=1/12 mean_sd=8.6629->... limit=5.3298 status=...
[GRID-BOUNDARY-FIT] accepted=... mean_sd=... limit=...
```

For controlled A/B tests (only), disable the new step with:

```bash
ONESTRING_GRID_SD_REFINEMENT=0 python3 -m streamlit run app_optcuts_20260923_extrusion_aware.py
```

Default is **enabled**. New telemetry is stored in the accepted chart:

- `optcuts_area_grid_sd_refinement_attempts`
- `optcuts_area_grid_sd_refinement_improvements`
- `optcuts_area_grid_sd_refinement` (phase details and before/after SD)

## What this does **not** prove

The new optimization is local. It does not guarantee that the given Bunny
Seam, rectangular lattice, chosen 50%-cell mask and staircase contour admit
a mapping with SD <=5.3298. Even if the target meets the cutoff, it does
not guarantee the full M3D surface is watertight: that downstream issue
must be verified separately after M2D face-mask identity and inverse
mapping are reached.

CI tests check the exact analytic SD equality with the SVD audit,
finite-difference gradient, interior improvement, ordered edge sliders,
corner retention, non-flip/non-overlap acceptance, and the production
quality-retry route. Native OptCuts/Bunny convergence remains unverified
until the user's Mac run.
