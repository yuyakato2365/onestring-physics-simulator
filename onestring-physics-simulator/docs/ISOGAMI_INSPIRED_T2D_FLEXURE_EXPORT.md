# Experimental IsoGami-inspired PP lamella hinges on final T2D

## What this adds

After the **final T2D Dual Hinge** layout, a separate, optional STL exporter
creates thin lamella-like flexure strips alongside the **existing tile prisms**.
This is a geometric adaptation of the polypropylene lamella concept in
Han et al., *IsoGami: Rigid-Deployable Kirigami Materials From Isohedral
Tilings* (SIGGRAPH 2026, Sec. 5.2), not a claim to reproduce the authors'
undisclosed print geometry or parameters.

**Do not confuse kinematics:** IsoGami hinges facilitate planar rotation;
OneString hinges fold panels in 3D. Mechanical viability of this new geometry
has not been demonstrated.

## Authoritative hinge surface and split handling

- Uses only the final state **state.hinge_graph.hinges**, produced by the T2D
  dual-hinge optimizer. No closest-vertex search, reclassified curvature
  sign, or connection across split seams.
- Each Hinge supplies tile_a, tile_b, local_vertex_a, local_vertex_b, and
  surface ("top" or "bottom").
- top requires indices 0..3 and puts the lamella on the upper (+Z) face.
- bottom requires indices 4..7 and puts it on the lower (-Z) face.
- Violations raise exceptions instead of quietly flipping side or welding
  disconnected components.

## Mesh algorithm

1. Reuse the **existing final dual T2D** panel solid coordinates and
   _t2d_stl_mesh_and_metrics. The separate print export scales the median
   panel edge to an explicit size in **millimetres**.
2. Normalize the vertical thickness to an adjustable nominal panel thickness
   through the existing export helper (does not mutate source T2D).
3. For every *listed* final joint, find points a,b via its local indices.
   Verify their 3D positions are within max_joint_gap_mm and their
   designated faces are approximately horizontal and at similar Z.
4. Find inward anchor points along each hinge-vertex-to-face-centroid direction.
   Form a bent three-point centerline: anchor A -> joint midpoint -> anchor B.
5. Extrude one closed mitered lamella of adjustable in-plane width and Z
   thickness along that centerline. Each end intentionally overlaps the
   respective panel on the **designated top or bottom face** by embed_mm.
6. Append lamella triangles to the panel triangles as **separate intersecting
   closed shells**. A later slicer/mesh Boolean union is required.

The bridge creates physical geometry across small optimizer residual gaps, but
**does not fix T2D hinge-coincidence constraints**. For example, a 0.732 mm
residual gap passes the default 1.5 mm fabrication ceiling, but would fail
if the user requires a 0.6 mm ceiling. The UI reports the measured maximum
and warns when bridging a nonzero gap. Large gaps still fail explicitly.

The algorithm intentionally rejects major gaps, incompatible hinge-side
indices, tilted face Z spreads and extreme miter angles. This does **not**
prove the resulting flat pattern has no new accidental intersections or
that its out-of-plane folding is mechanically feasible.

## UI / defaults

Launch the same usual Streamlit app (for example:

    streamlit run app_optcuts_20260923_extrusion_aware.py

after repository installation). Once the final "T2D Dual Hinge" tab can
export its normal STL, expand:

    実験：IsoGami型PP薄肉ヒンジ付き T2D STL（mm）

and tick the generation checkbox. This intentionally does **not** run if
hard T2D nonpenetration was not satisfied.

Suggested *unverified experimental starting values*:

| Parameter | Default | Meaning |
|---|---:|---|
| panel size | 40 mm | Median edge length after scaling |
| panel thickness | 2 mm | Z extent of the source tile, normalized |
| flexure width | 1.2 mm | XY strip width |
| flexure thickness | 0.6 mm | Total flexible Z cross-section |
| anchor inset | 3 mm | Connection depth from each stored joint |
| embed | 0.2 mm | Strip intersection with chosen rigid-panel face |
| max bridge gap | 1.5 mm | Hard ceiling for residual final-T2D attachment gaps; adjustable from 0.1 to 5 mm |

The STL format has no unit metadata; its vertex numbers mean mm in the new
export. Original T2D STL exports still use their unchanged settings.

## Handling imperfect T2D coplanarity

The source design optimizes coplanarity but can retain small numerical
out-of-plane residuals and different panel Z offsets. These deviations
must **not** be treated as a missing hinge. For this OPTIONAL fabrication
export only, XY corners and the authoritative top/bottom hinge IDs are
preserved, while every tile top is projected to Z = panel_thickness_mm and
bottom to Z = 0. Original optimized T2D vertices and normal STL are not
modified. No split-edge re-welding occurs.

The exporter records original per-tile Z spread, across-tile top/bottom
height spread and best-fit-plane residual in millimetres. It warns about
larger projection changes instead of aborting because of minor nonplanarity.
This is a **manufacturing approximation**, not a correction to the T2D
optimization or a proof that the resulting flattened prisms remain exactly
congruent to the 3D target. A deformable lamella can be generated even when
its two source joint vertices had slightly different heights.

Top lamella material lies in the upper skin [thickness-skin, thickness],
and bottom lamella material in the lower skin [0, skin]. Both remain in the
printable slab, rather than extending below the print bed. The independent
shells still need a successful mesh/slicer union to be reliably attached.

## Limitations before printing

- No Boolean union between overlapping shells. Inspect/merge in slicer and
  verify every intended connector is **one continuous printable body**.
- No thickness-dependent stress calculation, flexure fatigue, printer profile,
  supports, orientation, collision sweep, relief notch or tolerance tuning.
- Thin PP hinges should be tested on small samples before printing the full
  panel layout. In particular, the small free length near a point hinge may
  not sustain large-angle folding; a later relief geometry may be needed.
- The solver's rigid pin-joint kinematics do not model flexure bending
  torque. Do not interpret this exporter as a OneString fabrication proof.
- If the graph contains no hinges, **none are fabricated**. Disconnected
  components remain disconnected.

## Automated tests

    python -m pytest -q tests/test_t2d_flexure_fabrication.py \
        tests/test_paper_t2d.py tests/test_t2d_pipeline_ui_wiring.py

The targeted branch also contains a GitHub Actions job that runs these tests.
