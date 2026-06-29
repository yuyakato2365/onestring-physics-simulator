# OneString paper-flow alignment notes

This patch keeps the existing strict paper-flow pipeline and adds a deployment-only target contact guard.

## Change in this patch

The deployment simulation can still show green animated tiles visually passing through or sitting inside the translucent blue T3D target.  This is not a K3D design problem; it is a deployment projection issue.  Snap/lift/hinge constraints can pull individual vertices into the target volume before rigid projection catches up.

This patch adds two projective-dynamics style constraints during actuation:

1. Per-tile rigid target pose fit
   - Each tile is rigidly fitted toward its corresponding T3D pose.
   - The tile shape is not warped; a Kabsch rigid fit is used.

2. One-sided T3D target contact guard
   - Uses the designed T3D tile top-face normal.
   - If the animated tile goes inside the corresponding T3D tile pose, the whole tile is pushed outward.
   - A rigid projection immediately follows, so panels remain rigid.

These are applied in both CPU and CUDA deployment paths.

## New controls

- target T3D fit guard
- target penetration guard
- target guard start progress
- target clearance
- target guard projection passes

## New metrics

- target_penetration_count
- target_penetration_max
- target_penetration_mean
- target_contact_model

## 2026-06-27 Top/bottom orientation guard

The deployment contact guard now derives the per-tile outward normal from the actual tile thickness direction, bottom-center to top-center, rather than from the top-face winding with a z-positive flip.  This avoids false top/bottom inversions when a tile has inconsistent local winding or when a curved target has strong lateral normals.  Metrics now report target and animated tile top/bottom orientation diagnostics.

## Smooth animation and orientation diagnostics patch

- Adds a browser-side Plotly animation player for Paper PD frames. Frames are preloaded into Plotly and played in the browser, avoiding Streamlit reruns for every frame.
- The Plotly scene uses `uirevision` and stable traces so users can rotate/zoom the camera during playback.
- Adds a visible toggle for the blue T3D target mesh overlay in the physical animation view.
- Adds an expander explaining the current extrusion convention:
  - T3D: `bottom = top - thickness * quad_normal(K3D)`, where `_quad_normal` is flipped to have non-negative z.
  - T2D: top face comes from K2D; bottom face uses the per-tile T3D top-to-bottom transform.
  - Animation target contact normal uses `bottom_center -> top_center` rather than triangle winding.
- Ensures Paper PD animation parameters include target guard settings in the simulation key and in DeploymentParameters.
