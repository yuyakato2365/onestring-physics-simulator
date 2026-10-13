# OneString paper-aligned actuation reference (Sec. 5.2–5.4 / Supplement C–E)

Source: Zaman et al., *One String to Pull Them All*, ACM TOG 44(6), 2025.
- Paper: https://onestringtopullthemall.github.io/static/pdfs/onestringpull_authors_version_compressed.pdf
- Supplement: https://onestringtopullthemall.github.io/static/pdfs/OneStringPull-supplement.pdf

## What is implemented (Paper PD / default Assembly Animation)

1. **Gap graph from true explicit corner-linkage topology** via `paper_t2d.build_gap_graph`.
   Reverse boundary darts `(tile,corner)` are retained as
   `gap.boundary_perimeter_darts`. For each gap and surrounding tile, the
   physical material side-face edge `((c-1)%4,c)` is stored as
   `gap.material_side_edges`, preventing accidental nearest-corner routing.

2. **Lift selection**: the existing `_select_lift_points` uses discrete
   gap GPE peaks, gradient-ascent basins and tau threshold coupling. This is
   a *discrete graph approximation* to Sec. 5.2 / Supplement C, not a
   faithful Morse–Smale complex algorithm.

3. **String Path**: `paper_string_routing.build_string_path` solves a
   closed-cycle MILP with turn-angle cost, boundary-tile coverage, all
   selected lift gaps included, and restrictions on split-boundary
   interior entry. The paper prescribes the objective, not this MILP
   implementation. It uses a discretized gap graph and must report
   `string_path_model='Sec. 5.3 minimum-turn closed-cycle MILP'` and
   `string_path_valid=True`. Paper PD refuses unresolved, invalid, open,
   repeated, or unsupported routes. It never falls back to invented edges.

4. **Snap** (Supplement Eq. 9): paired T2D *material side-face
   midpoints* along the selected cycle are attracted to coincidence.
   Four-tile voids construct two diametrically opposite side-face
   pairs, determined using actual perimeter-dart side faces; the
   pairing across a four-tile gap remains a disclosed discretization
   because exact individual fabrication channel entry/exit features
   are not provided by the public source.

5. **Lift** (Supplement E): selected GPE lift gaps are attracted to their
   designated assembled 3D positions, by the top edge of actual side
   faces derived from the gap perimeter. Exit-point location within the
   channel and its physical clearance remain approximations.

6. **Projective Dynamics**: `paper_pd_global.PaperPDGlobal` assembles
   inertial, rigid, hinge, Snap and Lift terms into a *single sparse*
   quadratic global step with local 8-vertex rigid Kabsch projections.
   A sparse LU factorization is reused over steps; this is materially
   different from the legacy sequential PBD/snap/lift preview. Weight
   scales are independent defaults, not numbers cited by the authors.

7. **Time integration**: Supplement E position prediction
   `x_pred = x + dt*v + dt^2*a` and damped velocity update
   `v_next = (1-gamma)*v + dt*a`; initial frame is exact T2D dual.
   T3D tile poses are NOT used for direct per-panel attraction.
   Only chosen lift goals are actuated, as in the paper.

8. **Collision / nonpenetration**: 3D discrete 15-axis OBB-SAT contact
   projection with rigid translation/rotation correction, integrated
   into the nonlinear solver loop. This approximation can still miss
   contacts or leave residual penetration, unlike guaranteed
   noninterpenetration. It is NOT IPC or CCD.

## What is explicitly NOT claimed

- This is **not** the authors' released code or a validated identical
  implementation of all numerical algorithms.
- No mathematically guaranteed CCD/IPC noninterpenetration, no
  physical rope dynamics, no capstan/contact friction as a force in PD.
  The paper intentionally *approximates* string actuation with
  geometric Snap/Lift. Routing minimizes its capstan-derived turn cost.
- Physical 3D string threading inside fabricated channels is not
  reconstructed: the path graph is a *routing abstraction*;
  `gap.material_side_edges` identifies adjacent material faces.
  The String Path page draws red **Snap pair connectors**, not a
  literal embedded 3D string tunnel. Each 4-tile gap pairing and lift
  exit location remains an independent discretization.
- Initial T2D SAT nonintersection does not imply a collision-free
  assembled motion. `stable_state=False` or nonzero collision count
  means a deployment result must NOT be called physically successful.
- Large meshes still require runtime/performance validation; the
  toy unit tests are not proof of full Bunny deployment.

## Reproduction

Select `Paper Sec 5.4 / Appendix E — Snap + Lift PD (reference)`
(the new default). Re-run the **entire** OneString pipeline after
switching branches: earlier states lack the recorded material side faces.
The String Path view includes a dedicated expander displaying which
T2D side faces are paired. Use the Assembly Animation view to inspect
Snap RMS, Lift RMS, hinge RMS, and residual 3D contacts.

Legacy target-guided and Native total-path-length physical prototypes
remain opt-in and clearly separate; ABD/IPC also remains an independent
external solver option. They are NOT asserted to be the paper's PD model.

Targeted tests:

```bash
python3 -m pytest -q tests/test_paper_pd_deployment.py \
    tests/test_paper_t2d.py tests/test_fast_assembly_animation.py \
    tests/test_t2d_pipeline_ui_wiring.py
```
