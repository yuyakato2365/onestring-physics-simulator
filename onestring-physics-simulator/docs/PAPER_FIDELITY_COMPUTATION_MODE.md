# 2025 OneString paper-oriented computation mode

This experimental mode is meant to make the **actual calculations**, not
merely result captions, closer to Zaman et al., *One String to Pull Them All*,
ACM TOG 44(6), 2025.

Source:
- https://onestringtopullthemall.github.io/static/pdfs/onestringpull_authors_version_compressed.pdf
- https://onestringtopullthemall.github.io/static/pdfs/OneStringPull-supplement.pdf

## Selecting the mode

Enable **Paper-oriented computation mode (2025 Sec.4–5)** in the sidebar
(default on this branch). The UI records the effective forced parameter
values. This overrides legacy choices, including ones left in Streamlit
session state. The parameter override is applied BEFORE the pipeline run.

It requires a working official BFF CLI for `paper_reference_bff`: the
historical mode named `bff` actually performs rectangular cotangent
harmonic flattening, and is never accepted by this preset. Missing binaries
must produce an error; there is no silent substitute.

Stages actually selected:

- S→Ω: official BFF library with `target_rectangle` boundary.
  Four ordered boundary vertices approximately spaced by 3D perimeter
  quarters have prescribed external angle +pi/2; all others receive zero.
  This invokes the **independently compiled** `bff-rect-command-line`
  linked against the unmodified official BFF C++ library. The normal
  `bff-command-line` cannot prescribe these curvature constraints.
  The output's 4 straight sides and 4 corners are checked geometrically.
  No harmonic/LSCM/PCA fallback is allowed.
- M2D grid: regular fixed-size grid, strict vertex crop, no additional
  speculative expansion or heuristic collision split.
- Sec.4.5 Split: high Gaussian curvature, complete grid-direction
  cuts, with a maximum of 64 by default. **NEW** for this mode: compare
  the per-component area-Jacobian span
  `max(J) / min(J)` against the paper's quadrilateral area-expansion
  limit 2 (Supplement Fig.18). This is the smallest possible area
  span after independent uniform scaling of each piece.
- K3D/T3D: no variable-number-of-vertices patches, clipping/
  geometry-changing intersection repair, emergency prism fallback or
  independent extrusion-aware experiments. Keep eight rigid corners.
- T2D: rigid SE(2) local panel poses, multi-start search with at least
  700 iterations and 30s search budget per trial, plus the project's
  optional hard nonpenetration repair. This repair is NOT an exact
  replication of the authors' soft Eq.(6) numerical solver.
- String routing: existing Sec.5.3 minimum-turn closed-cycle MILP and
  topology-derived gap material-side edges; no fallback to a random route.
- Assembly Animation: select `Paper Sec 5.4 / Appendix E — Snap + Lift
  PD (reference)`, which uses sparse global inertial/rigid/hinge/Snap/
  Lift optimization. This is not the independent Native total-rope-length
  solver and does not use IPC.

## Important, unresolved fidelity gaps

A fixed UV map does **not** change pointwise Jacobian just because its
topology has been cut. Dividing the area-Jacobian range into separate
components lowers the minimum *theoretical per-piece normalized range*
if those pieces can later be rescaled. But no such rescaling/conformal
reparameterization is applied to the resulting actual M2D geometry by
the current code. Consequently `csf_split_bound_certified=False`
remains mandatory even if
`csf_split_candidate_area_bound_satisfied=True`.
This must not be represented as proof of manufacture-ready scale.

Original paper Sec.4.5 describes curvature-guided hierarchical full
cuts and the ratio condition; it does not prescribe a unique numerical
component-rescaling algorithm. We do NOT invent and claim one.

The T2D hard repair and 2D convex footprint SAT are project-specific
approximations; the full thick-solid collision audit must also be zero.
The authors' exact optimizer weights, mesh preprocessing, BFF boundary
conditions and tile counts for Lilium are not fully specified.
The Lilium mesh used in their published example is not bundled with
this branch. Grid=14 on an unrelated or modified OBJ cannot be assumed
to reproduce the published flat layout.

The paper's assembly uses geometric Snap and Lift rather than physical
rope dynamics; the optional ABD/IPC experiment is a different extension.
Side-face correspondence follows explicit T2D boundary topology, but
channel interior clearances, four-tile void pairing and discrete collision
resolution are still approximated.

Every completed run records `paper_fidelity_stage_results`,
`paper_fidelity_residual_reasons` and `paper_fidelity_verified=False`
until the actual shape, scale, collision and solver checks are proven.

## Comparing with legacy

Uncheck Paper mode to return to the existing experimental choices.
The preset copies dataclass parameters and leaves original UI values
untouched. It never commits an unverified intermediate Collision Split
or changes the user's local files.

### Verification

```bash
python3 -m pytest -q tests/test_paper_fidelity_computation_mode.py \
  tests/test_paper_pd_deployment.py tests/test_paper_t2d.py \
  tests/test_split_and_hinge_audit.py
```

These are regressions for parameter wiring and toy geometry. They are
not a whole-Lilium numerical reproduction test.


## Rectangular BFF initial installation (macOS)

The ordinary CLI previously used for `automatic_reference` is NOT able
to accept prescribed exterior-angle boundary data. It cannot be used
for `target_rectangle`. The extra adapter must be built once, using
the official BFF C++ source already installed in the previous OneString
checkout. From the new worktree's `onestring-physics-simulator` folder:

```bash
python3 scripts/setup_bff_rectangle.py --jobs 4
python3 scripts/verify_rectangular_bff.py
```

If its source is elsewhere, specify
`--source /absolute/path/to/boundary-first-flattening`.
Setup generates
`third_party/bff_rectangle_build/bff-rect-command-line`
and leaves the original official source checkout untouched.
The adapter is auto-discovered. Alternatively set
`ONESTRING_BFF_RECTANGLE_EXECUTABLE` to its absolute path.
If Xcode's C++ tools/CMake/SuiteSparse are missing, install the relevant
build dependencies first. A prebuilt ordinary `bff-command-line` is NOT a
substitute for the C++ library source.

The paper does not specify which four source boundary vertices are used
as rectangle corners for a particular Lilium input. The equal-arclength
choice is an explicitly labeled reproducibility hypothesis, not an
identical reproduction of the paper author's four corners.
