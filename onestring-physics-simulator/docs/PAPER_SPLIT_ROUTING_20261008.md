# Mesh splitting and string routing (2026-10-08)

Base: `b7cd9a9`, branch `agent/fix-audit-findings-20261007`.
Scope: OneString Sec. 4.5, Fig. 6, Sec. 5.3, Supplement Fig. 18.
Sources: https://onestringtopullthemall.github.io/ and the locally supplied
`onestringpull_authors_version_compressed.pdf`, `OneStringPull-supplement.pdf`.

## Active path

The 2026-09-23 launcher reaches `app_runtime_base.py` through `app.py`.
`app_backup_before_mitered_t3d.py` is now a fallback UI, not the current UI.
Both UIs have matching split defaults. The seam bridge no longer rewrites 1.9
to 2.0 or suppresses the split policy for OptCuts.

`_flatten_to_domain` prepares diagnostics without pre-cutting. The final
cropped M2D is passed once to `apply_paper_split` / `split_mesh`. The active
Simple Split wrapper defers the upstream cut owner using a copied domain.
There is no coordinate re-weld, duplicate snap, cut-band relief, mirrored
cut generation, or localized slit in this path. Direct pipeline and reference
BFF initialization also reach this same cut implementation.

## Numerical choices

- Supplement Fig. 18 explicitly shows an **area** expansion of two; its linear
  expansion is sqrt(2). Split CSF therefore uses the area Jacobian of the
  retained piecewise-affine UV-to-surface correspondence, not sigma_max with
  a bound of two. Linear singular values and anisotropy remain separate
  diagnostics. An area bound alone does not certify a nonconformal input map.
  UV and surface triangle indices are matched explicitly, including duplicated
  or reordered UV vertices at OptCuts seams.
- Source triangles are intersected with the retained quads. Each component's
  measured ratio is max(area Jacobian)/min(area Jacobian), using only source
  triangles with positive-area overlap. Component normalization corresponds
  to independently selectable similarity scales; the original chart and
  inverse map are retained. No post-cut reparameterization is claimed.
- Curvature is the **signed** angle defect divided by barycentric dual area.
  Chart-boundary vertices are excluded. Within each violating component,
  select the highest-curvature sample admitting a complete grid bisection.
  Snap to the nearest existing grid coordinate. If both axes are admissible,
  use the more balanced face partition (deterministic tie break).
- A cut is confined to its current connected component, but crosses that
  component completely. Preserve tile order and record paired tile/local-side
  identities. Re-evaluate child components after every accepted cut.
- The default threshold is 2 and the safety budget is 64 successful cuts.
  Explicit custom thresholds, including 1.9, are honored. The budget is exposed
  in the active UI and participates in the cached-design key.
- Unresolved cases are reported as `budget_exhausted`, `no_admissible_cut`,
  or `disabled`; they are never reported as having reached the threshold.
  A grid that cannot resolve the shape may remain unresolved. This is visible
  in Split Map through per-component sigma and remaining high-CSF faces.

These normalization, discrete curvature, axis, and snapping choices fill in
details the paper does not specify. They should not be described as an exact
reproduction of the authors' code. No Bunny quality, convergence, performance,
or fabrication validation was performed for this change.

## Boundary and route flow

`split_boundary_pairs` travels from M2D into `LinkageTopology` and through T2D.
Gap construction marks those sides `-2` and creates channel bridge edges only
between explicitly paired split sides. It does not weld hinges or infer new
connections from coincident points. Sec. 5.2 shared-tile adjacency is retained
for lift selection; Sec. 5.3 uses separate shared-corner routing adjacency.

`paper_string_routing.py` uses SciPy/HiGHS MILP to select one closed cycle:

- cover every boundary tile and selected lift gap;
- visit each chosen gap once (apart from writing the closing node twice);
- forbid edges between split boundary gaps and interior gaps;
- require global connectivity with a single-commodity flow;
- minimize the sum of geometric predecessor/current/successor turn angles,
  including the closing turn; the boundary anchor is optimized as well.

MILP is an implementation choice, not a solver prescribed by the paper.
The 30-second solver limit distinguishes an optimal solution, a feasible
incumbent, and no feasible route. An invalid route has an explicit status and
cannot be passed to the deployment wrapper. This change does not replace the
existing lift-selection or physical actuation solvers, or certify their
agreement with the full paper.

## Minimal flow checks performed

- Syntax/import checks and `git diff --check`.
- Actual launcher/install chain through `app_runtime_base.py` entry; numerical
  function bindings checked, UI body intentionally skipped. The local test
  environment lacks `streamlit_plotly_events`, so this was not a full UI run.
- A 6x6 synthetic chart passed flatten -> final M2D cuts -> M3D lift. Explicit
  threshold 1.9 survived the bridge; a budget of two produced two cuts, three
  components, stable face order and an honestly reported exhausted budget.
- A small rotating-square fixture passed split-pair propagation, `-2` labels,
  seam bridging and one closed connected route covering all boundary tiles.
- An intact fixture passed a route through an interior lift gap.
- Split/residual display data passed through the installed OptCuts display
  wrapper; reordered/duplicated UV indices preserved the same face CSF.
- Existing `test_t2d_string_graph_is_connected` passed.

These checks establish control/data flow only. Shape-quality verification is
left to the user as requested.
