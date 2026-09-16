# Extrusion hard validity guard

This patch keeps the existing miter/contact-plane extrusion attempt, but adds a hard per-tile validity guard before T3D is accepted.

If the mitered bottom candidate is non-finite, inverted, too thin, has a large thickness error, or jumps too far from normal translation, the entire tile is replaced with a simple prism:

```text
bottom = top - thickness * oriented_tile_normal
```

The goal is to guarantee that T3D is a valid thick-tile input for T2D layout.  This intentionally prioritizes robust non-inverted panels over experimental miter/contact side-face exactness.

New metrics:

- `t3d_miter_candidate_min_signed_thickness_before_guard`
- `t3d_miter_candidate_max_thickness_error_before_guard`
- `t3d_normal_translation_guard_applied`
- `t3d_normal_translation_guard_tile_count`
- `t3d_normal_translation_guard_reasons`
- `t3d_extrusion_hard_validity_guard`

After applying the patch, the key validity metrics should satisfy:

- `t3d_reversed_extrusion_vertex_count = 0`
- `t3d_min_signed_thickness > 0`
