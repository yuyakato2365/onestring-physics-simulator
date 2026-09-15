# Side-face contact patch + free layout + UI/cache fixes

This ZIP keeps the old Copy-Item replacement workflow:

```powershell
Expand-Archive .\onestring_sideface_contact_patch.zip -DestinationPath .\sideface_contact_tmp -Force
Copy-Item .\sideface_contact_tmp\onestring_physics\* .\src\onestring_physics\ -Recurse -Force
Copy-Item .\sideface_contact_tmp\app.py .\app.py -Force
Copy-Item .\sideface_contact_tmp\PAPER_ALIGNMENT.md .\PAPER_ALIGNMENT.md -Force
```

## Geometry/layout changes kept

- T3D uses shared-edge side-face/contact-aware mitered extrusion.
- T2D avoids affine/shear top-to-bottom transforms and preserves rigid tile shape.
- T2D/dual-hinge layout is more permissive: hard hinge closure, open voids/collision clearance, weak initial-layout anchor.

## New UI changes

- All Plotly 3D charts preserve camera state with a stable `uirevision`.
- Plotly config always exposes 3D pan/orbit/turntable/zoom/reset controls.
- Middle-mouse drag is mapped to temporary Plotly 3D pan mode where the browser permits event forwarding.
  - Left drag remains orbit/rotation.
  - Mouse wheel remains zoom.
  - Middle drag pans the camera center.
- Plotly animation frame redraws now restore the current `scene.camera` after every frame, so Smooth play does not snap back to the initial view.
- If the server-frame player recreates the chart, the last camera in the page is applied to the new chart before the frame is shown.

## New animation cache

- `simulate_onestring_deployment()` is wrapped with a Streamlit `st.session_state` cache.
- Once an animation/simulation is generated for the same pipeline and deployment settings, returning to the same settings reuses the cached result instead of recomputing frames.
- The cache keeps up to 8 recent simulation results.

## Notes

The middle-mouse pan helper is a browser-side compatibility layer around Plotly. If a browser blocks synthetic mouse event forwarding, the modebar `pan3d` button still provides the same camera-center movement.
