# Why the original Grid-boundary numerical fit cannot finish (sample 2026-10-23)

## Reproduced from the uploaded diagnostic ZIP

Source `grid_failure_evidence.zip` contained one original official OptCuts
cut, **rotation 0°**, original fixed 30×30 Grid mask with 422 selected cells,
and original correspondence `phase=72, method=corner_dp`.

- 8,222 UV vertices, 15,982 UV triangles, 460 boundary vertices
- 82 target grid-outline corners (39 reflex, i.e. 270-degree corners)
- Last accepted boundary fraction **0.9866875**
- First recorded rejected fraction **0.98749140625**
- Rejected minimum oriented area ratio **0.0048815063**
- Existing hard lower bound: **0.005**

Crucially, three triangles have *all three UV vertices* in the constrained
boundary loop: IDs **11456, 11703, 14877**. When all vertices are constrained,
moving interior UV cannot affect their mapped areas.

In the uploaded failed correspondence, both triangle **11456** and
**11703** have three target UV vertices on one straight grid-boundary edge,
so their full-target signed areas are mathematically zero.

For triangle 11456, its vertices (UV IDs 8060,8062,8061) would be at:

- (-0.75405686, 1.36835547)
- (-0.75405686, 1.39515151)
- (-0.75405686, 1.38189772)

All x-coordinates coincide: the signed area is zero regardless of the
interior optimizer's iteration count. The previous 0.9867 ceiling was a
consequence of this exact boundary mapping, **not a reason to relax the
0.005 guard**.

### Offline candidate study on that actual failed UV snapshot

Recreated all 98 original phase/arclength/DP correspondences from the
recorded source UV, ordered boundary and required Grid outline:

- **0/98** original correspondences satisfied the necessary positive-area
  constraint for every three-boundary-vertex triangle.
- **31/98** corresponding seed assignments could be repaired by greedily
  relocating existing target grid-corner anchors to the median vertex of
  offending boundary ears, preserving cyclic vertex order.
- The repaired best candidate used **phase 73 / corner_dp** and one
  source-anchor correction at ordered boundary position **87 → 85**.
  Its three fully-constrained triangle area ratios are positive; minimum
  approximately **0.9547**, far above 0.005.

These numbers come from this **one specific input snapshot**. They are not
new native OptCuts execution and do not prove a globally bijective solution
for the repaired target.

## Implemented code correction

- `optcuts_grid_ear_correspondence.py`: exact signed-area preflight on
  triangles fully pinned by the target boundary; fail before optimization
  if any is structurally guaranteed to collapse. Generate an alternative
  *ordered* corner-to-source-anchor assignment for each such triangle;
  accept only changes that strictly decrease the number of forced invalid
  boundary-ear triangles.
- `optcuts_area_grid_boundary.py`: invoke the ear-aware generator only
  in the experimental area-threshold mode. Rank surviving correspondences
  by their original movement score; run the expensive harmonic and L-BFGS-B
  solver only on matches with no proven forced-degenerate ear.
- `scripts/replay_grid_ear_correspondence.py`: replay the diagnostic ZIP
  locally and output the exact original forced-degenerate face IDs and the
  count/best scores of repaired options.

The source mesh, OptCuts Seam topology, production Grid origin/spacing,
50%-intersection cell mask, grid-outline polygon, exact target corners,
ordered boundary, existing 0.005 area-ratio guard and final global overlap
audit are left intact. Only **which original UV boundary vertex is assigned
to each existing Grid corner** is changed.

### Reproduce without another full optimization

From the repository's nested `onestring-physics-simulator` directory:

```bash
python3 scripts/replay_grid_ear_correspondence.py \\
  logs/grid_boundary_diagnostics/grid_failure_evidence.zip
```

This writes `logs/grid_boundary_diagnostics/grid_ear_replay_result.json`.
Successful preflight is a **necessary, not sufficient**, condition for final
bijective area-grid UV fitting. The newly generated candidates still must
pass all expensive checks, and the real Bunny native end-to-end result
cannot be claimed until the user reruns with its original project settings.

A failure after ear-aware repair would justify diagnosing *interior*
triangles, boundary self-intersections during continuation, and solver
conditioning, rather than continuing to try structurally impossible
correspondences.
