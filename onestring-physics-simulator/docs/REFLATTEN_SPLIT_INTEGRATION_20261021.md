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


## General inputs beyond Lilium (2026-10-21 follow-up)

The new reflatten Split has **never** needed a Lilium-specific source topology.
Its generality was limited by the UI's Lilium-specific rectangular BFF
corner preset, and by the Lilium-named verification script.

The Streamlit \`mesh upload\` widget already supports OBJ/STL/PLY. The
generalized mode now automatically chooses \`source_boundary_shape_fit\`
for non-Lilium input names, and preserves \`lobe_tips\` for named
\`lilium*.obj\` reference meshes. These options remain manually selectable.
The UI reinitializes the corner-selection widget when the selected input
file changes, so a previous Lilium choice is not silently carried over.

The standalone generic audit is:

\`\`\`bash
python scripts/run_reflatten_split.py --mesh /path/to/any-open-surface.obj --grid 14
python scripts/run_reflatten_split.py --mesh /path/to/open-surface.stl --grid 20
python scripts/run_reflatten_split.py --mesh /path/to/open-surface.ply --grid 16
python scripts/run_reflatten_split.py --builtin dome --grid 14
python scripts/run_reflatten_split.py --builtin saddle --grid 14
# Repeat the Lilium benchmark, preserving its original boundary preset:
python scripts/run_reflatten_split.py --mesh /path/to/lilium.obj --grid 14 --expect-splits 1
\`\`\`

External official BFF CLI executables must still be built/available.
\`--mesh\` and \`--builtin\` are exclusive and neither hard-codes a
split count. The \`--expect-splits\` flag is only an assertion after
all numerical work has finished. The original
\`scripts/verify_reflatten_lilium.py\` stays available for compatibility.

### Input restrictions

The official BFF and present source-piece slicing require a **single,
oriented, manifold triangulated disk with exactly one open boundary**.
An OBJ/STL/PLY describing a closed genus-0 animal, a multi-boundary
surface, multiple disconnected islands, or a nonmanifold mesh is
not automatically unfolded or cut for this algorithm. Its topology
is validated with \`validate_reference_mesh\`, and a clear error is
raised without automatic face deletion, unsupported hole filling, or
fabrication claims. Producing a valid opening cut for a closed 3D
surface is a separate inverse-design/topology problem.

\`source_boundary_shape_fit\` tests multiple rectangular-BFF corner
candidates and scores fit to the source boundary. It is less tied to
Lilium's four petal extrema but is **an expensive unverified heuristic**:
it may fail on unusually shaped boundaries, and does not guarantee
numerical convergence. It is NOT the authors' published criterion.

### Comparison with Zaman et al., Section 4.5

Paper: generate corresponding M2D/M3D from a conformal map and grid;
introduce hierarchical complete grid-direction bisections passing
through highest Gaussian curvature where the CSF is too high, until
every part meets the criterion \(\sigma\le2\). The section does
**not specify** the discrete 3-ring curvature averaging, how to
choose between row/column, per-candidate free-BFF reflattening, or
this implementation's tile-by-tile area range.

Current reconstruction: initial official rectangular BFF, grid
cropping, curvature per source vertex from discrete angle defect over
vertex area averaged over 3 adjacency rings; interpolate it onto
interior grid vertices, select the maximum; evaluate one complete
row and column cut, intersect the triangulated 3D surface with
each trial cut line, do **official free-boundary BFF separately for
both child pieces**, compute the per-tile 3D/free-BFF 2D area ratios
and their maximum/minimum spread, prefer the smaller worst-child
spread. Continue recursively until spread <=2, budget exhausted, or
no admissible complete cut. The new BFF charts are diagnostic:
they are **not** substituted for original M2D panel coordinates.

The paper calls the local conformal scale factor \(\lambda\) and an
upper bound \(\sigma=2\), while Supplement Fig. 18 discusses an
area-expansion limit for quadrilateral auxetics. For a locally
conformal map the area Jacobian is \(\lambda^2\).
This reconstruction compares an *area ratio of extremes* directly
to 2; this is a research interpretation and should not be relabeled
as a verified bound on local linear CSF or the physical mechanism.

Previous \`fixed_uv\` Split: assesses all components on the **same
initial conformal UV map** (with user-selected per-triangle or
M2D/M3D area proxies) and no per-child free-BFF solve. It can
therefore overestimate distortion that a child could release upon
reflattening. The new method evaluates that release but adds
BFF cost, more boundary dependence, and more failure cases.
Neither is a faithful published reconstruction of undocumented
authors' solver details.

Independently, \`collision_split_enabled\` is the downstream
SAT/panel-overlap response: candidate complete seams are evaluated
by full M2D->T2D rebuild and collision/hinge acceptance. It
addresses a **different feasibility failure** from conformal scale.

Validation: general-input tests in
\`tests/test_generic_reflatten_inputs.py\` cover open OBJ inputs,
built-in dome, expected result report, unchanged named Lilium preset,
and explicit rejection of closed tetrahedra. Existing
\`tests/test_paper_split_reflatten.py\` covers area conservation,
planar zero-cut, curvature-peak boundary corners, BFF failure,
split provenance and display. These tests **do not yet replace
real end-to-end BFF/M3D/K3D/T2D runs** on non-Lilium geometry.


## When ALL initial official rectangular BFF charts fold or degenerate

The general source-boundary-fit choice can fail on a valid, oriented
manifold disk: the 3D source may be perfectly legal, while every
official rectangular BFF result for the tested four corner configurations
has a flipped or tiny UV triangle. Merely increasing the number of corner
candidates DOES NOT guarantee a valid conformal rectangular chart.

To make the general experimental pipeline usable without falsifying a
Paper reproduction, we added an **explicitly labeled non-Paper** mode:

- UI: disable **Paper-oriented computation mode**; select
  \`paper_reference_bff\` as Omega route and
  \`source_boundary_shape_fit\` for corners.
- Check **「実験：矩形BFFが全滅したら非等角Tutte写像で救済」**.
- First, all the same official rectangular BFF candidates run and
  undergo the same stringent area/orientation validation.
- ONLY if all official choices are rejected, build a **positive-weight
  graph-harmonic (Tutte) embedding into a convex rectangle**. Every
  boundary vertex is placed monotonically along an arclength-parameterized
  rectangle; every interior vertex is the arithmetic mean of its mesh
  neighbors (solved by a sparse linear system).
- The alternative is accepted only when its UV is finite, no triangle
  has near-zero signed UV area, all triangle orientations agree, and
  the rectangular boundary validator passes. The 3D source mesh
  and original triangle connectivity are NOT modified.
- Metrics identify \`experimental_tutte_rectangle\`, report
  \`nonpaper_rectangle_rescue_used=true\`, and preserve
  \`fallbacks_used=["explicit_nonconformal_tutte_rectangle_rescue"]\`.
- **This is NOT BFF, NOT conformal, NOT the author algorithm.**
  The conformal scale-factor interpretation and any "same Split count
  as paper" conclusion are NOT transferable to this fallback.
  Reflatten Split continues using official free-boundary BFF on
  each cut 3D subpiece, but its INITIAL chart is not BFF.
- The rescue is blocked when strict Paper mode is on, even if the
  parameter is set manually. Lilium's \`lobe_tips\` setting is
  unchanged. Disabling the rescue preserves the previous fail-fast
  behavior exactly.

If even the alternative map fails, the application still stops:
it never calls an invalid folded/degenerate chart valid. This does
not make arbitrary closed surfaces or topologies supported.

To compare the paper method, keep the rescue disabled and repair/remesh
the input or use a valid official BFF chart. For exploratory T2D
processing, the clearly marked experimental rescue can allow progress.

The relevant tests are \`tests/test_tutte_rectangle_rescue.py\`
and \`tests/test_bff_rectangle_selection.py\`.


## Official BFF natural-boundary recovery for invalid rectangle charts (Oct 22)

The root symptom reported on an arbitrary disk-type mesh was:
25 attempted rectangular boundary corner configurations all invalid.
Some candidates produced exactly zero UV triangle area at face 275;
other corner choices inverted four triangles around face 286.
Tweaking area epsilon, repeating the same rectangle corners, using
nonconformal Tutte, or simply writing more decimals did NOT establish
a correct rectangular BFF embedding for this source surface.

For the **Paper ON + source_boundary_shape_fit** path ONLY:

1. Try all the existing official constrained rectangular BFF candidates,
   with their strict orientation / UV validity tests.
2. If and only if every rectangle chart is invalid, invoke the SAME official
   GeometryCollective BFF library with its native ZERO boundary scale
   factors / **free-boundary BFF** via the independent native adapter's
   \`--free-boundary\` option. Export the original BFF UVs to OBJ using
   17-digit round-trip precision (the old official OBJ writer quantized
   to six decimals).
3. Require valid and finite differential UV Jacobians, consistent triangle
   orientation, no UV area collapses and no interior triangle overlaps.
   On failure, keep throwing an error; do not inject Tutte or LSCM maps.
4. If valid, preserve the source 3D mesh and the EXACT existing
   \`reflatten_tile_area\` Split code. The source Ω is now a FREE-boundary
   conformal chart, **not** the prescribed rectangle. The grid is still
   square and cropped to Ω, but the rectangle-specific fit option is
   bypassed only for this recovery case.
5. Emit \`official_bff_free_boundary_recovery_used=true\`,
   \`paper_rectangle_requirement_met=false\` and a prominent Streamlit
   warning. The Paper fidelity report must explicitly state that the
   strict prescribed-rectangle condition was **not** reproduced.

For the Paper OFF path, this recovery remains disabled; no changes to
its original BFF/T2D/Split choices are intended. Likewise, valid
Lilium rectangular BFF outputs remain unchanged.

The recovery uses OFFICIAL BFF, not the previously added optional
nonconformal Tutte solution. Nevertheless the different **boundary
condition** can produce different source charts and Split numbers, and
must not be presented as a strict reproduction of the original
rectangle-preset numerical results. The authors' exact original
boundary choice remains unverified.

Native source changes require **rebuilding**:
\`\`\`bash
python3 scripts/setup_bff_rectangle.py --jobs 4
export ONESTRING_BFF_RECTANGLE_EXECUTABLE="$PWD/third_party/bff_rectangle_build/bff-rect-command-line"
python3 scripts/verify_rectangular_bff.py
\`\`\`

This verifies the standard rectangular CLI AND native 17-digit
official natural-boundary recovery. The programmatic failure-path
tests in \`tests/test_bff_free_boundary_recovery.py\` simulate
all candidate failures and do not substitute nonconformal charts.

This implementation has NOT yet passed the exact user's source mesh
through an actual full S->Omega->M2D->M3D->K3D->T2D run; the optional
previously added \`ONESTRING_BFF_DEBUG_DIR\` forensic output can reveal
the input-specific root cause if even free BFF is non-injective.
