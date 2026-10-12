# Native T2D OneString physical deployment (experimental)

## Scope and source

Reference: Zaman et al., *One String to Pull Them All*, SIGGRAPH Asia 2025,
https://onestringtopullthemall.github.io/ (paper DOI 10.1145/3763357).
The authors demonstrate rigid-body deployment with contacts and a routed
tensioned string; this implementation is an **independent approximate prototype**
rather than an implementation of their unpublished solver source.

The former "Legacy Projective Dynamics" applies snap/lift and T3D pose
constraints. Those terms can create target-looking movement even when
single-string actuation would fail. It is preserved for comparison only.

## New Native backend

Select **Native rigid-body + routed string** under **Actuation Simulation**.
Open **Assembly Animation** and select **Run / refresh native string simulation**.

- Initial frame is exactly `state.tiles_2d_dual_hinge.vertices`. No best-fit
  T3D recopying and no staging-scale tricks.
- Each 8-corner T2D tile moves as a single six-degree-of-freedom rigid body:
  the per-tile local shape stays fixed under SO(3) rotation + translation.
- Pin constraints use the actual top/bottom hinge material vertices.
- Guide points follow the computed `state.string_path.gap_ids` in route
  order. Each guide is attached to the closest top corner of the first
  surrounding tile; other, more accurate channel routing is future work.
- A fixed world anchor is placed above the first guide. String total length
  is commanded from initial length down to `native_pull_end_ratio` times
  its initial value. The unilateral constraint acts only when
  `L_actual > L_command`, not when the string is slack.
- Gravity, fixed tabletop support and approximate 3D OBB-SAT panel contacts
  are applied during PBD time integration.
- T3D is **not** an actuation target. It appears only as an optional
  translucent reference and end-state error metric.
- Every physics frame includes a corresponding 3D routed-string polyline.
- Diagnostics: initial/final 3D contact count, RMS hinge gap, actual/target
  string length and inequality residual, peak height rise, rigid shape
  deviation. Not every computed sequence is considered physically valid.

## Limitations (do not overclaim)

This is a position-based **approximation** without continuous collision
detection, IPC barrier constraints, exact frustum-frustum contact, Coulomb
rope-channel capstan friction, full mass/inertia calibration, or physical
experiment validation. It is not identical to the original authors' engine.

Starting from an intersecting T2D arrangement is *not* a physically
realizable rigid-body initial condition. The simulator currently reports
initial approximate contacts and continues diagnostically, but cannot
certify such a deployment path. Reduce initial T2D overlaps first.

There is no T3D goal attraction and no fabricated success guarantee.
The string pull may lift only part of the object, leave constraints violated,
or fail entirely; such failure is important research evidence.

For stronger 3D contact fidelity use the separate **Autodesk ABD + IPC/CCD**
backend, which requires a locally built and extended external solver.
The ABD route is kept separate and not silently substituted.

## Suggested settings / reproduction

Use a small grid initially; approximately 8–12 `actuation steps` and
8–16 `solver iterations` to check mechanics, then scale up to 40–100
steps with 16–30 iterations as CPU budget allows. Start with pull ratio
0.65–0.85 to avoid abrupt constraints; adjust anchor height. View
`Smooth browser player` to see the red string along with the rigid tiles.

Run targeted tests:

```bash
python3 -m pytest -q tests/test_native_string_deployment.py \
  tests/test_abd_backend.py tests/test_fast_assembly_animation.py \
  tests/test_t2d_pipeline_ui_wiring.py
```

The tests verify rigid shape invariance, a true rising two-panel
linkage under pull, initial geometry fidelity, tension-only behavior,
3D approximate contacts and absence of T3D target forces. They are
**not** a real-mesh or fabrication deployment validation.
