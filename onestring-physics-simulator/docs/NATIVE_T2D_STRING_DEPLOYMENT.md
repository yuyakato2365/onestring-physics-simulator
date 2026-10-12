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

**Native PHYSICS steps** and **native timestep** control how much time the
solver advances. **Animation frames** is only the number of snapshots
sampled *uniformly across this duration* for playback: increasing it
does not make the physical simulation longer. The old sampling filled
most of the snapshots early in the simulation, skipping later changes.

The Native sidebar now allows 40–600 physical steps (default 180) and
a **Pull phase fraction** (default 0.70): the prescribed string length
decreases over the first 70% of the physical time, then remains
constant over the final 30% so hinge constraints and rigid-body
contacts can relax. At timestep 0.02s, 180 steps yield 3.6s:
2.52s pulling followed by 1.08s holding.

Use a small grid initially; on larger cases CPU OBB contacts can make
180 steps expensive. Start at 60–100 steps, then try 180–300 with
8–20 solver iterations for diagnosis. Use 40–80 displayed frames and
a pull end ratio 0.65–0.85 to reduce abrupt constraints. If the
actual string length remains far above the commanded length, or
hinge gaps grow, more steps alone will not solve the actuation problem.
Inspect the contact count and string length residuals. View the
`Smooth browser player` to see the red string and rigid tiles.

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


## T2D rigid reassembly diagnostic (added 2026-10-13)

The T2D Dual Hinge stage now includes an additional 3D view directly below
the flat collision map. It takes EACH actual fabricated T2D eight-vertex
solid and uses a proper Kabsch SE(3) rigid fit (det R=+1) to its corresponding
T3D 8-vertex solid. It does not use T3D's local geometry as the result.

It reports per-tile fit error, vertex RMS error, hinge gaps and approximate
3D OBB SAT collision pairs in the reconstructed target placement. This is
a *kinematic feasibility diagnostic*, not a claim that one routed string
can reach those poses or avoid collisions along a deployment path.

The flat T2D collision overlay is 2D SAT in XY, while the new
post-reassembly and Native dynamic contact diagnostics check approximate
three-dimensional overlaps between thick panel prisms. Zero flat
collision pairs DOES NOT imply zero 3D collisions after rotation.

## Native contact and bounded extra settling (added 2026-10-13)

Native physical mode now projects approximate rigid-body contact positional
impulses at support-region contact points, affecting both translation and
rotation, instead of translating two intersecting panels only. The final
projection pass reapplies collision correction after hinge/string constraints.

A new `Extra settling steps after scheduled physics` control defaults to
40. It continues physical time integration after the original pull+hold
schedule with the final commanded string length fixed. The default
180 commanded + 40 extra steps at 0.02s corresponds to 4.4s of simulation.
This is bounded: it does NOT converge to the T3D goal by applying invisible
forces. A failed result is explicitly labeled
`ended_with_constraint_residuals`. Contact counts and hinge RMS by
display frame can be inspected alongside string length histories.

These changes are still an approximate PBD prototype. In particular,
3D OBB SAT is not exact geometry-on-geometry CCD/IPC, and the heuristic
contact point may not match the real contacting feature of a tile.
Validate on real geometries and use ABD if verified contact fidelity
is required. Physics failure may reflect an infeasible linkage/string
route rather than missing animation frames.
