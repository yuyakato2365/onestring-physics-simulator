# 2026-10-26 OptCuts → Planar Quad Omega experiment

**Status: first-stage prototype, not a reproduction of Zhang et al. 2021.**
The dated Omega selector is `optcuts_planar_quad_20261026`.

## Purpose

The former Oct 23 real-grid-area mode crops cells using a 50% overlap mask,
and then forces the original OptCuts Omega onto the staircase boundary.
For an irregular UV disk this may disconnect selected cells or produce narrow
3D quads. This isolated experiment tests a different approach:

```
S → Official OptCuts (3D cut and vertex/face pairing unchanged)
  → conservative boundary UV smoothing + harmonic interior continuation
  → optional coarse-boundary straightening with SD/flip/overlap guards
  → planar Gmsh 2D quadrangulation with target edge length
  → strict all-quad / original-Ω coverage / UV inverse tests
  → M2D + quantitative M3D shape diagnostics
  → conditional M3D-aware interior-vertex quad adjustment
  → existing audited CSF Split attempt + conservative unstructured T2D adapters
```

The generating mesher currently uses **Gmsh recombination**, NOT the reference
implementation or algorithms of *Quad Meshing with Coarse Layouts for Planar
Domains* by Zhang, Chai, Liu, Fu (CAD 2021, DOI 10.1016/j.cad.2021.103084).
That published code is separately linked from Xiao-Ming Fu's project page but
its executable interface has not been integrated or validated.

## Setup (macOS)

```bash
git fetch origin feature/planar-quad-omega-20261026
git switch feature/planar-quad-omega-20261026
cd onestring-physics-simulator
python3 -m pip install gmsh shapely
python3 -m streamlit run app_optcuts_20260923_extrusion_aware.py --server.port 8502
```

Depending on the macOS Python/Gmsh build, Gmsh may require additional system
libraries. The adapter reports dependency failure; it does not fall back to
the cropped-square routine.

In the sidebar choose **Omega parameterization mode**:
`2026-10-26 | OptCuts → boundary-conforming Planar Quad M2D`.
Set *Target Planar Quad panel count (approximate)*. Gmsh's final face
count is **not** guaranteed to equal the requested count.

## Strict acceptance

- Input: original paired OptCuts UV/3D triangles and one simple UV disk.
- The source paired UV triangles may be moved by bounded harmonic deformation; every accepted chart retains winding, one simple boundary and SD ≤ its guard. 3D vertices / triangle IDs / 3D cut remain fixed.
- UV triangle correspondences are kept intact; no silent resampling of S.
- Target edge length is based on Ω area and requested count.
- Mesh output **must** be all quads with consistent winding and positive
  2D areas. Triangle remnants are rejected, not converted to quads silently.
- Poly-domain coverage must be at least 98%; any measurable out-of-domain
  quad area or overlapping faces is rejected.
- Quad vertices, centroids, and edge midpoints are inversely located on
  original UV triangles with barycentric interpolation, with no nearest
  surface fallback.
- Metrics include 2D and 3D smallest angle, edge-length max/median,
  smallest area, area max/median, and Ω coverage loss.
- The existing OneString CSF Split stage is invoked with the actual unstructured faces. It may reject all complete cuts; diagnostics expose `csf_split_bound_certified=False`. No made-up grid seam or unverified bound is labeled success.

## Explicitly unfinished

1. The specific Zhang et al. 2021 coarse-layout implementation, field-guided
   segmentation and template optimizations have **not** been ported.
2. This version adds bounded-SD source UV boundary adjustment and harmonic interior continuation, but **does not yet implement** a global joint SD/quad objective or the published cross-field rectangular-patch layout algorithm.
3. M3D shape is measured, and a conservative interior-vertex M2D relocation is attempted only for poor 3D panels. This is local search, not a globally optimal 3D shape guarantee.
4. Full seam-sampling parity and fabrication verification for arbitrary
   all-quad topology are not completed.
5. Some downstream row/column assumptions are replaced by face connectivity. The original CSF Split may fail to certify the threshold for irregular meshes; K3D, T2D and exact seam pairing still need full Bunny end-to-end verification at 30/35 panels. **This is not yet a demonstrated complete Bunny assembly.**
6. Gmsh's recombination is heuristic; an unpaired boundary or difficult
   concave shape may still yield triangles. This mode fails explicitly.

## Tests

```bash
python3 -m pytest -q tests/test_optcuts_planar_quad_20261026.py
```

CI also installs Gmsh and Shapely, generates a small all-quad square disk, and
verifies source UV→3D correspondence.

## Next research steps

A. Integrate the actual published planar coarse-layout mesher after verifying
   its licensing and CLI/input/output correspondence.
B. Preserve source triangle barycentric location for every new quad vertex
   while optimizing UV positions with explicit global injectivity constraints.
C. Couple 3D quad quality and target size to parameterization and quad layout.
D. Replace row/column Grid assumptions in T2D, seams and CSF splitting with
   explicit graph adjacency; test Bunny neck cases at 30 and 35 panels.
