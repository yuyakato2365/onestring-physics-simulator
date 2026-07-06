# Current OneString Algorithm Overview

This document summarizes the current implementation, not an idealized paper-only pipeline.
The implementation follows the OneString flow, but several stages are discrete graph or
interactive-performance approximations rather than exact reference implementations.

## Pipeline Summary

The current pipeline is:

```text
S target surface
 -> Omega planar domain
 -> M2D regular quad overlay
 -> M3D inverse map to S
 -> K3D assembled 3D quad optimization
 -> T3D thick panels
 -> K2D flattened edge-length layout
 -> T2D top/dual hinge fabrication layout
 -> Gap graph / LiftPoint / StringPath
 -> animation / deployment simulation
```

## 1. S to Omega: Surface Flattening

The mathematical goal is to build a planar parameterization

```math
c: S \rightarrow \Omega
```

from the target surface `S` to a planar domain `Omega`.
For a conformal map, the local Jacobian should satisfy

```math
J^\top J \approx \lambda^2 I,
```

where `J` is the local differential of the map and `lambda` is a local scale
factor. This means angles are approximately preserved, while lengths and areas
may scale.

The requested main mode is `bff`, but the implementation does not silently call
the rectangular-boundary harmonic substitute BFF. Instead it uses this dispatch:

1. Try an optional external conformal backend. A wired reference BFF backend
   would report `bff_implemented=True` and `flattening_backend` such as
   `libigl_bff` or `geometry_central_bff`.
2. If no reference BFF backend is available, use an explicit free-boundary LSCM
   fallback. This reports `bff_implemented=False`,
   `bff_reference_backend_available=False`, and
   `flattening_backend="local_free_boundary_lscm_fallback"`.
3. Keep rectangular-boundary cotangent harmonic parameterization separate as
   `omega_parameterization_mode="rect_harmonic"`.

The local fallback solves a least-squares conformal map. In each triangle it
minimizes the discrete Cauchy-Riemann residual

```math
\frac{\partial u}{\partial x} - \frac{\partial v}{\partial y} \approx 0,
\qquad
\frac{\partial u}{\partial y} + \frac{\partial v}{\partial x} \approx 0.
```

Two boundary vertices are pinned only to remove translation, rotation, and
scale nullspaces. The remaining boundary is free; it is not forced onto a
rectangle, circle, or projected outline.

The old fixed-boundary harmonic alternative solves

```math
L_{II} u_I = -L_{IB} u_B,
```

where `u_B` is a prescribed rectangular boundary. This mode is now labeled
`rect_harmonic` and should not be called BFF.

## 2. Conformal Stretch Factor and Split

The conformal stretch factor (CSF) estimates where `S -> Omega` stretches too
much. The implementation compares corresponding 3D and UV edge lengths:

```math
s_e = \frac{\lVert x_i - x_j \rVert_S}{\lVert u_i - u_j \rVert_\Omega}.
```

Per-vertex CSF is estimated from local edge stretch statistics. Regions with
CSF above the threshold, typically near 2, are treated as over-stretched regions.

The split implementation is discrete. It does not cut along an arbitrary
continuous curve. Instead, it detects high-CSF bands in Omega and chooses grid
row or column split lines. After splitting, residual CSF diagnostics are stored
so the UI can show whether high-stretch regions remain.

## 3. Omega to M2D

M2D is built by overlaying a regular quad grid on Omega and cropping outside
cells. A quad is kept when its center lies inside Omega:

```math
\operatorname{keep}(Q) \iff \operatorname{center}(Q) \in \Omega.
```

This center policy is intentionally less strict than requiring all four vertices
to lie inside Omega. Strict vertex inclusion removes too many boundary tiles for
curved or non-rectangular domains.

## 4. M2D to M3D: Inverse Parameterization

Each M2D UV vertex is mapped back to the surface through

```math
c^{-1}: \Omega \rightarrow S.
```

The implementation finds a containing or nearby UV triangle and computes
barycentric coordinates:

```math
u = \alpha u_1 + \beta u_2 + \gamma u_3,
\quad
\alpha + \beta + \gamma = 1.
```

The same weights are applied to the corresponding 3D triangle:

```math
x = \alpha x_1 + \beta x_2 + \gamma x_3.
```

Triangle lookup is accelerated with `scipy.spatial.cKDTree` candidate searches
instead of scanning every triangle for every point.

## 5. M3D to K3D

K3D optimizes the lifted mesh into a more fabrication-friendly assembled 3D
quad mesh. The objective is approximately

```math
E_{\text{assembled}}
=
w_1 E_{\text{planar}}
+
w_2 E_{\text{square}}
+
w_3 E_{\text{surface}}.
```

The terms mean:

- `E_planar`: each quad should be close to planar.
- `E_square`: each quad should avoid extreme shape distortion.
- `E_surface`: vertices should remain close to the target surface.

Depending on size and backend availability, this stage uses PyTorch/CUDA,
SciPy `least_squares(method="trf")`, or a faster projection approximation.
A quality guard rejects results that collapse or drift too far, falling back to
M3D when necessary.

## 6. K3D to T3D: Thick Panel Extrusion

T3D converts each K3D quad into an eight-vertex thick tile. The simple extrusion
model would be

```math
x_{\text{bottom}} = x_{\text{top}} - h n,
```

where `h` is panel thickness and `n` is the tile normal.

The current implementation is more careful. It uses miter/contact planes along
shared edges and split-contact edges. Each bottom vertex is solved as the
intersection of three planes:

1. The bottom plane offset from the top face by thickness.
2. One adjacent side plane.
3. The other adjacent side plane.

This is a linear system

```math
A x = b.
```

If the system is ill-conditioned or produces an unreasonable jump, the code
falls back to the normal-translation extrusion for that vertex.

## 7. M2D to K2D

K2D is a flattened layout whose edge lengths should match K3D. The objective is

```math
E_{\text{flat}}
=
w_1 E_{\text{edge}}
+
w_2 E_{\text{collision}}
+
w_3 E_{\text{fab}}.
```

The edge term is

```math
E_{\text{edge}}
=
\sum_{(i,j)}
\left(
\lVert u_i-u_j\rVert - \lVert x_i-x_j\rVert
\right)^2.
```

This says: the 2D K2D edge length should match the corresponding 3D K3D edge
length.

The implementation may use:

- PyTorch/CUDA optimization when available.
- SciPy nonlinear least squares for smaller systems.
- A projective edge-length solver for larger systems.
- A strict edge-length projection fallback in strict mode.

The projective edge step computes, for each edge, a correction based on current
length `ell` and target length `ell*`:

```math
\Delta =
\frac{\ell - \ell^\*}{\ell}(u_j-u_i).
```

The two endpoint vertices receive opposite half corrections. This is not a
Runge-Kutta time integration method. It is iterative constraint projection.

## 8. K2D to T2D: Independent Tile Layout

K2D is an abstract shared-vertex mesh. T2D is a fabrication layout with
independent thick panels. Each tile is treated as a rigid body in the plane:

```math
p' = R p + t,
\quad
R \in SO(2).
```

The local/global layout objective is

```math
E_{\text{hinge}}
=
w_{\text{conn}} E_{\text{conn}}
+
w_{\text{coll}} E_{\text{collision}}
+
w_{\text{anchor}} E_{\text{anchor}}.
```

The terms mean:

- `E_conn`: hinge vertices should coincide.
- `E_collision`: panel footprints should not overlap.
- `E_anchor`: the result should not drift too far from the initial layout.

The solver builds desired vertex targets, then fits each tile rigidly to those
targets by a weighted Procrustes/Kabsch-style solve. Collision is detected with
the Separating Axis Theorem (SAT). Candidate collision pairs are reduced with a
spatial hash / AABB broad phase so not every pair has to be tested precisely.

This is a local/global `SE(2)` rigid projection solver, not a global exact
nonlinear optimizer.

## 9. Dual Hinge

The dual hinge stage extends the T2D layout to account for both top and bottom
hinges. It uses the same rigid tile placement principles:

- rigid per-tile shape preservation,
- hinge point coincidence,
- thick footprint collision handling,
- hard hinge repair where possible.

For split-coincident vertices, the implementation uses virtual welding for graph
connectivity. This preserves topology for hinge/gap reasoning without literally
undoing the split geometry.

## 10. Gap Graph

The gap graph is a discrete graph

```math
G = (V, E),
```

where each node is a gap between tiles and each edge represents adjacency between
gaps. A gap stores:

- surrounding tile ids,
- 2D centroid,
- 3D centroid,
- boundary/internal type,
- GPE value.

This graph is the substrate for LiftPoint selection and StringPath routing.

## 11. LiftPoint Selection

The paper motivates LiftPoints through GPE peaks and Morse-Smale style
segmentation. A Morse-Smale complex divides a scalar field into regions that
flow toward critical points such as maxima, minima, and saddles.

The current implementation is a discrete graph approximation:

1. Use internal gaps as candidates, falling back to all gaps if needed.
2. Find local GPE maxima.
3. Assign each gap to a peak by steepest ascent on the gap graph.
4. Keep peaks above `tau * max(GPE)`.
5. Suppress coupled peaks that share the same physical tile neighborhood.

The discrete ascent step is:

```math
p(i) = \arg\max_{j \in N(i)} \operatorname{GPE}(j),
```

repeated while GPE increases. This gives graph basins analogous to continuous
gradient-flow basins.

The UI records the chosen LiftPoints with gap id, GPE, basin size, 2D/3D
positions, and selection reason.

## 12. StringPath

StringPath is routed on the gap graph. The current route:

1. Orders all boundary gaps into a boundary loop.
2. Inserts that boundary loop into the actual string path.
3. Visits LiftPoint gaps in descending GPE order.
4. Uses weighted shortest paths between route targets.
5. Returns to the boundary anchor.

The shortest path algorithm is Dijkstra's algorithm. The edge cost is based on
centroid distance, with small modifiers for boundary traversal and GPE attraction:

```math
\operatorname{cost}(i,j)
\approx
\frac{
\lVert c_i-c_j\rVert(1+\text{boundary penalty})
}{
1+\text{GPE attraction}
}.
```

The path also estimates turn angle

```math
\theta = \sum_k \angle(g_{k-1}, g_k, g_{k+1})
```

and a capstan-style friction estimate

```math
T_{\text{out}} = T_{\text{in}} e^{\mu \theta}.
```

This equation means rope tension grows exponentially with wrap angle and
friction coefficient.

## 13. Deployment Simulation

The deployment simulation is Projective Dynamics / Position Based Dynamics
style. It is not Runge-Kutta.

The CPU path keeps current and previous positions. It first forms a damped
Verlet-style velocity:

```math
v = (x - x_{\text{prev}})(1-d),
```

then updates

```math
x \leftarrow x + v.
```

After that, it repeatedly projects constraints:

- Lift constraints,
- Snap / gap contraction constraints,
- Hinge constraints,
- Rigid tile projection,
- AABB collision projection,
- Target pose fitting,
- Target contact guard,
- Optional bending targets.

The energy reported by the simulation is conceptually

```math
E =
w_{\text{rigid}}E_{\text{rigid}}
+
w_{\text{collision}}E_{\text{collision}}
+
w_{\text{actuation}}(E_{\text{snap}}+E_{\text{lift}}).
```

The key numerical idea is constraint projection: move the positions toward a
constraint-satisfying set repeatedly. This differs from force-based simulation,
where one would compute forces and integrate an ODE with Euler, Verlet, or
Runge-Kutta.

Rigid tile preservation uses a Kabsch/Procrustes projection:

```math
R = \arg\min_{R \in SO(3)} \lVert R P - Q \rVert^2.
```

This finds the best rotation aligning a tile's rest shape to its current shape,
usually through SVD.

## 14. UI Animation Preview

The Streamlit assembly animation is a lightweight preview, separate from the
full deployment solve. The default motion mode is `simultaneous_hinge_contraction`.

It uses:

- Smooth interpolation of tile centers from T2D to T3D.
- Smooth interpolation of height.
- Rigid local tile shape interpolation so panels do not shear or collapse.

The smooth interpolation function is

```math
s(t) = t^2(3 - 2t).
```

This function has zero slope at `t=0` and `t=1`, so motion starts and stops
smoothly.

For local tile orientation, the code estimates a best-fit rotation by SVD, then
interpolates the rotation through SciPy `Rotation` when available. Conceptually,
this is close to

```math
R(t) = \exp(t \log R),
```

which interpolates rotation on the rotation group instead of linearly blending
coordinates.

## Practical Interpretation

The current codebase is best understood as a hybrid of:

- discrete differential geometry for `S -> Omega`,
- edge-length and stretch diagnostics for split decisions,
- barycentric inverse mapping for `Omega -> S`,
- nonlinear least squares and projective edge constraints for K2D/K3D,
- rigid-body Procrustes projection and SAT collision for T2D,
- graph algorithms for LiftPoint and StringPath,
- PBD/PD-style constraint projection for deployment,
- smoothstep plus rigid rotation interpolation for preview animation.

It does not currently use Runge-Kutta integration. Where it looks physical, the
dominant numerical method is iterative geometric projection rather than
force-based ODE integration.
