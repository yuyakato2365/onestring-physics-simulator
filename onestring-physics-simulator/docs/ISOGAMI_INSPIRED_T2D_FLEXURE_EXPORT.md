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

## Mesh algorithm — exact T2D panel shape preservation

1. Read the **final T2D Dual Hinge 8-vertex panel geometry** without
   modifying its coordinates or changing any panel's own thickness.
2. Convert source units to millimetres using a **single global uniform**
   scale calculated from the representative panel edge. Apply only one
   **global Z translation** to set the minimum vertex height to zero.
   No per-panel flattening, leveling, thickness setting, local rotation
   or independent rescaling is permitted.
3. Re-triangulate each source 8-vertex prism from its own exact scaled
   vertex coordinates, retaining nonplanar face and height residuals.
4. For each listed hinge, preserve its top/bottom label and local
   vertex indices; never infer an unlisted hinge across a split seam.
5. Interpolate each attachment anchor from the actual panel face
   joint vertex toward the actual face centroid in 3D. A central
   point bridges the original joint pair. Generate a **ruled 3D lamella**
   whose Z heights vary along the strip, following the stored
   top/bottom face on both tiles.
6. Append flexure triangles as separate intersecting closed shells.
   Slicer/mesh Boolean union and physical checking remain required.

Source panel vertex pairwise distances in millimetres are
`distance_print = global_scale * distance_T2D`. No other deformation
is allowed. The printable hinge may span a finite optimizer residual
gap but does not fix the optimizer's connectivity constraints.

The configurable bridge limit is at most 50 mm (5 cm). Large bridge
gaps may create geometries that fail mechanical deployment; allowing
their export is **not** a fabrication guarantee. Original face warp
and joint height mismatches are reported, not silently flattened.

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
| panel thickness | unchanged | Original T2D vertex geometry, no thickness control |
| flexure width | 1.2 mm | XY strip width |
| flexure thickness | 0.6 mm | Total flexible Z cross-section |
| anchor inset | 3 mm | Connection depth from each stored joint |
| embed | 0.2 mm | Strip intersection with chosen rigid-panel face |
| max bridge gap | 1.5 mm | Hard ceiling for residual final-T2D attachment gaps; adjustable from 0.1 to 50 mm |

The STL format has no unit metadata; its vertex numbers mean mm in the new
export. Original T2D STL exports still use their unchanged settings.

## Handling imperfect T2D coplanarity

K3D/T2D can have residual face nonplanarity and between-panel height
offsets. The print exporter keeps **those original XYZ values**; it
never adjusts top or bottom vertices to a common plane and never
changes panel geometry to fix these errors.

A hinge strip's 3D centerline and cross-sections follow the original
top/bottom vertex heights. Its panels remain exactly the original
T2D prisms after a single global scale and translation.

The measured best-fit-plane residual and original hinge Z mismatch
are diagnostics. Significant deviations trigger warnings, not a
silent geometry rewrite. The mesh may require supports and additional
relief/collision design before it can be printed or deployed.

## Limitations before printing

- Original panel tilt and height offsets are preserved. Some panels may
  float above the print bed after global Z translation; **supports may be
  required**, and no automatic support design is included.
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
