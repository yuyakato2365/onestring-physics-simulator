# Grid-boundary UV failure: evidence-first debugging

**2026-10-23** — Experimental `optcuts_area_grid_boundary` Omega only.
These additions collect diagnostic evidence; they do **not** claim the
previous `last_valid_fraction≈0.99368` failure is resolved.

## Why this diagnosis is needed

The current official OptCuts backend returned **one cut topology** for the
reported Bunny experiment. Rigid rotations changed the grid's 50%-cell
outline, but none of the first boundary correspondences had yet reached a
valid full deformation. The 67.5° variant contained 53 reflex (270°) corners,
with an interior area ratio approaching the existing 0.005 guard.

The available logs cannot distinguish:

- a non-simple *intermediate* boundary under straight-line homotopy;
- local UV compression concentrated around reflex staircase corners;
- poor boundary correspondence (monotone / DP landmark assignment);
- distributed distortion, awkward source triangles or a numerical
  limitation of the fixed-boundary harmonic + L-BFGS-B untangler.

**Do not assume that the 270° corners are the cause without evidence.**

## Automatic failure snapshot

At the end of each rejected boundary correspondence, the app saves:

`logs/grid_boundary_diagnostics/<timestamp>_<cut>_<rotation>_<attempt>/`

- `diagnosis.json`: rejected/last-valid fraction; minimum/p1/p5
  signed-area ratio; counts of flipped, below 0.005 and below
  the 0.015 barrier target; 40 worst face IDs, centroid positions, distance
  to nearest 270° corner; whether triangles cluster near reflex corners;
  proper non-adjacent crossings of the *actual interpolated boundary*;
  an explicitly qualified working hypothesis and suggested next check.
- `overview.svg`: source UV outline (gray), required fixed-Grid
  polyomino (blue), last accepted boundary (green), rejected boundary
  (red), reflex corners (orange), and most compressed triangles.
- `worst_triangle_zoom.svg`: localized view of the minimum-area triangle.
- `reproduce_uv.npz`: full untouched source UV and face IDs, last
  accepted/rejected/harmonic UV, selected original Grid cells and exact
  coverage, target boundary mapping, and signed-area ratios.

A line in `index.jsonl` summarizes **every saved failure**. Each file
is written only after an attempted UV continuation fails; it never changes
the solver's acceptance, source topology, tile threshold or Grid selection.
Failed diagnostic I/O cannot mask the original mapping error.

## Compare across correspondences and rotations

From the `onestring-physics-simulator` working directory:

```bash
python3 scripts/summarize_grid_boundary_failures.py
```

This creates:

- `logs/grid_boundary_diagnostics/analysis_summary.md` — strongest
  candidate and across-orientation evidence, repeated failing triangle IDs
  and prioritized **tests**, not unsupported conclusions.
- `logs/grid_boundary_diagnostics/comparison.csv` — every attempt's
  cut, rotation, correspondence, phase, maximum fraction, min ratio,
  reflex-enrichment and boundary intersections.
- `logs/grid_boundary_diagnostics/grid_failure_evidence.zip` — a
  shareable bundle containing the report, CSV, and up to six representative
  fully reproducible failures. **Upload this ZIP for follow-up analysis.**

The default aggregates the most recent 48 attempts. Set `--last N`
to focus on a smaller experiment and `--max-bundles N` if necessary.
Different runs with different mesh/topology and coincident triangle IDs
should not be aggregated when judging recurring hot triangles.

Use the finder to open the most recent diagnosis:

```bash
open "$(find logs/grid_boundary_diagnostics -name worst_triangle_zoom.svg | sort | tail -1)"
```

## Decision table based on actual evidence

| Evidence | Next experiment, keeping existing production rules intact |
| --- | --- |
| Intermediate contour self-crossing | Change the boundary interpolation path or source-target monotone corner matching before reworking cuts |
| Strong compression enrichment near reflex corners | Examine nearest reflex corner and try *localized* seam placement or chart boundary decoupling |
| Same triangle repeatedly worst across candidate correspondences | Inspect its original mesh geometry, triangle quality, correspondence and local Jacobian first |
| Weak distributed compression, no boundary self-crossing | Investigate harmonic initial guess, area-barrier solver and conditioning rather than assuming the staircase is impossible |
| All tested source cuts/rotations fail | Need additional **real cut topology** from official OptCuts changes or a source-surface seam graph optimizer; UV rotations are not additional cuts |

**Important:** Tests can establish the diagnostic machinery works. They
cannot establish which of the above hypotheses is true for the real Bunny
without running that model. Exact source/target boundary and final overlap
checks remain unchanged.

Disable file output (to diagnose performance only) via:

```bash
ONESTRING_GRID_BOUNDARY_DIAGNOSTICS=0 python3 -m streamlit run app_optcuts_20260923_extrusion_aware.py
```

Change output folder via `ONESTRING_GRID_DIAGNOSTICS_DIR`.

The local `logs/` directory is ignored by Git but is **not deleted**.
