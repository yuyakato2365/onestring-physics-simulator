# Reflatten/area-span Split integrated with current OneString (2026-10-21)

## Scope and provenance

Ported the **algorithmic design** of the user-supplied
\`work_branch_reflatten_split.patch\` (2026-10-20 handoff) to the actively
developed \`feature/isogami-flexure-t2d-20261021\` branch. The source
patch targets the substantially older \`work/complete-k3d-optcuts-dualhinge\`
commit \`8126a92\`; applying its five large old-file replacements verbatim
would overwrite later K3D, collision Split, BFF, and UI improvements.

This is an **experimental reconstruction** of Zaman et al. (2025),
Sec. 4.5 and Supplemental Fig. 18. It is NOT a verified implementation
of the paper authors' actual algorithm and does not certify a physically
deployable linkage.

## Pipeline

S → **official rectangular BFF** Ω (optionally choose four high-boundary-
turning \`lobe_tips\` corners) → M2D (optional centered square grid fit)
→ **free-boundary BFF reflatten of each cut source 3D patch** →
tile area-ratio decision → **real complete grid cuts in the original M2D** →
existing M3D → existing K3D → existing T3D/K2D/T2D →
existing optional IsoGami-inspired fabrication flexures.

Only the selection of Ω corners, initial square-grid alignment and
**Split criterion** change. Reflattened charts are never fed into
K3D/T2D or used to redefine original tile geometry.

## Numerical criterion

For each connected source patch P, run official default/free-boundary
BFF on the cropped 3D triangles. Re-evaluate each of the ORIGINAL
M2D quads in that chart, computing

\[
J_q=\frac{A_{\mathrm{3D}}(q)}
          {A_{\mathrm{free\ BFF,2D}}(q)},\qquad
\sigma_P=\frac{\max_{q\in P}J_q}{\min_{q\in P}J_q}.
\]

Require \(\sigma_P\le2\) as the reconstructed rotating-square *area*
expansion proxy. Sample input signed Gaussian curvature, smooth over
three vertex rings by area-weighted averaging, select highest interior
eligible grid vertex, and evaluate both full grid directions. Choose
the direction minimizing the worst child \(\sigma\), including neutral
or temporarily non-improving cuts. Stop after all leaves pass or no
valid complete grid cut/budget remains. Fail visibly on missing/invalid
free BFF; never replace it with a fake identity chart.

The child charts are used only for this criterion; cut seams use the
existing \`_split_mesh_single\` source grid line and preserved
\`split_boundary_pairs\` records.

## Compatibility with recent features

* The K3D Augmented-Lagrangian launcher wiring added in this feature
  branch is left intact.
* Original \`fixed_uv\` Split code remains the fallback for other Omega
  modes and when the new criterion is not selected.
* Collision-driven Split retains its independent full-pipeline trial
  loop and forced-line list. If both mechanisms are enabled, free-BFF
  CSF cuts are applied first, and trial/accepted collision-cut forced
  seams after them. Diagnostics separate both provenance classes.
* Final T2D panel vertices, thickness, shape and relative positions
  are unaffected by reflattened UV charts; the fabrication-STL flexure
  export also retains its independent optional behavior.
* The pre-existing Split Map can replay the actual cuts and CSF values.
  A compact 3D M3D source geometry replay has also been added.
* The Figure 5 stage outline receives its CSS on every render (old
  UI patch required this during progress rerenders).

## UI

In the main sidebar, disable Paper-oriented mode only if desired.
Select Omega parameterization \`paper_reference_bff\`; choose
\`target_rectangle\` with \`lobe_tips\`; select
\`reflatten_tile_area\` under "Split CSF判定"; optionally enable
"矩形BFF...グリッドを合わせる". The paper preset may force some
other model parameters but not this opt-in criterion. The source patch
was based on the BFF backend, not the "Paper" checkbox itself.

The official \`bff-command-line\` executable must be available for
**every child patch** and the \`bff-rect-command-line\` for the initial
rectangle. \`scripts/repair_postcut_bff_mac.sh\` can repair/rebuild the
macOS BFF tools in an existing worktree.

## Validation

GitHub Actions runs a pure-geometry regression suite and existing
flexure, K3D launcher and T2D unit regressions. Tests cover:

- clip area conservation and connected child triangulations;
- exact area span = 1 for identity chart on a planar surface;
- no cuts required for planar regular quads;
- four curvature-maximal lobe-tip corners;
- fail-fast on missing official BFF;
- separate provenance when collision-driven cuts follow CSF cuts;
- non-mutating 3D component display;
- isolated opt-in selection and unchanged existing K3D/Collision paths.

**The user-supplied Lilium reference figures have NOT been reproduced
in this integration**. The ZIP reports grid=14, initial span 2.2025,
one cut at x=11, children 154 and 42 with spans 1.9326 and 1.2402,
but those remain *claims about the original patch*, not measured
results of this branch. Exact author equivalence, precise Mac build
availability, and numerical end-to-end manufacturability remain open
verification tasks.

To run the new test group locally:

\`\`\`bash
python3 -m pytest -q tests/test_paper_split_reflatten.py \
 tests/test_t2d_flexure_fabrication.py tests/test_k3d_al_launcher_wiring.py
\`\`\`

A full Lilium end-to-end check additionally requires the verified
\`lilium.obj\` and both external official BFF executables. No result
should be treated as a paper reproduction until the numerical output
has been measured.
