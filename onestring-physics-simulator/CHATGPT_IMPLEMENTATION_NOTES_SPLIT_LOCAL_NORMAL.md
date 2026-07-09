# Split local segment + normal orientation patch

This patch addresses three issues observed with the half-snowman / necked target:

1. **Do not re-weld CSF split boundaries**
   - `_canonicalize_faces_by_coincident_vertices()` and `_canonicalize_faces_by_coincident_tile_tops()` now preserve the face vertex ids as-is.
   - Coincident duplicate vertices created by `_split_m2d_along_existing_grid_line()` remain real cut boundaries.
   - Metrics still report candidate coincident groups, but `split_virtual_weld_applied=False`.

2. **Stabilize extrusion normal orientation for disconnected split components**
   - `_extrude_tiles()` now applies `_orient_tile_normals_to_outward_reference()` after shared-edge BFS normal orientation.
   - The added reference is `tile_center - global_tile_centroid`, so small disconnected side components are less likely to extrude in the opposite direction from the main component.
   - Metrics include `t3d_surface_reference_normal_*` fields.

3. **Localize CSF split cuts instead of always cutting the whole Omega width/height**
   - `PipelineParameters` now exposes `localize_csf_splits=True` and several tuning fields.
   - `_localized_csf_split_segments()` converts full row/column candidates into local segments around high-CSF clusters.
   - `_split_m2d_along_existing_grid_line()` supports segment tuples `(axis, value, other_min, other_max)`.
   - Metrics include `csf_split_localized_segments`, `csf_split_localized_segment_count`, and `raw_split_locations`.

Expected outcome:

- Split map should no longer create only full-width/full-height bands by default.
- T2D Top Hinge should be less likely to collapse side split parts into dense bundles.
- T3D/T2D extrusion direction inconsistency should be reduced for split components.

Limitations:

- This is still a local heuristic split implementation, not the full OneString paper optimizer.
- The outward normal reference is robust for convex/star-shaped snowman-like targets but may need a true surface-normal reference for highly concave arbitrary meshes.
