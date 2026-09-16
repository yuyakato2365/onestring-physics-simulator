# Variable-topology T3D recovery

## Geometry ownership

`ConvexTileSolid` is the authoritative T3D geometry. It stores variable vertices,
polygon faces, top-face ids, contact-face mapping, recovery status, reasons, and
per-tile manufacturing metrics. `TileAssembly.vertices` remains an eight-vertex
one-sided proxy only for existing T2D and deployment code.

The viewer and T3D STL exporter prefer authoritative solids. They do not treat
the compatibility proxy or the old render-only intersection trim as recovered
manufacturing geometry.

## Recovery sequence

1. Validate the K3D top quad and global normal orientation.
2. Reject positive-area intersections between mandatory nonadjacent top faces.
3. Construct a convex solid by clipping a large one-sided slab with the top,
   depth, boundary, shared-miter, and split-contact half-spaces.
4. Add lateral cap planes when miter intersections exceed the allowed range.
5. Accept a watertight positive-volume wedge or pyramid when the bottom quad
   collapses.
6. Record local thickness when the feasible solid terminates before the nominal
   bottom plane; otherwise search down to the configured minimum thickness.
7. Keep shared contact planes synchronized by construction.
8. Detect nonadjacent convex-solid collisions with AABB broad phase and SAT,
   attempt a symmetric top-preserving clip, and fail explicitly if unresolved.

If every geometric recovery tier fails and
`allow_legacy_normal_prism_emergency_fallback` is enabled, only the affected
tile is replaced by a one-sided normal prism. It receives status
`T3D_RECOVERED_LEGACY_EMERGENCY_PRISM`, is rendered gray, and is explicitly
marked `manufacturing_authoritative=False`. Invalid/self-intersecting K3D top
faces are never replaced this way. Any collision remaining after the emergency
replacement stays visible in the metrics.

The API default keeps this emergency fallback disabled. The selectable
`2026-07-12-one-sided-t3d` application version enables it explicitly.

## Diagnostics

Each tile report includes status, triggers, recovery steps, requested and actual
depth, volume, minimum feature size, topology counts, collision counts,
watertightness, and manifoldness. Assembly metrics aggregate nominal, recovered,
failed, cap, wedge, pyramid, local-thickness, synchronized-pair, junction, global
clip, remaining collision, volume, and feature-size counts.

The T3D viewer colors recovery status and can highlight generated cap faces. STL
triangulates the authoritative polygon faces directly.
