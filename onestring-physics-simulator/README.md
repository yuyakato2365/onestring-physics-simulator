# onestring-physics-simulator

A Python research prototype inspired by **One String to Pull Them All: Fast Assembly of Curved Structures from Flat Auxetic Linkages**.

This simulator implements a paper-faithful approximation of the OneString pipeline. It is not a direct morphing animation to the target surface. The design pipeline explicitly stores `S`, `Omega`, `M2D`, `M3D`, `K2D`, `K3D`, `T2D`, and `T3D`. The actuation simulation follows a Projective-Dynamics-style constraint formulation using rigid tiles, hinge constraints, collision constraints, snap constraints, and lift constraints.

The default Streamlit workflow is:

```text
S -> Omega -> M2D / M3D -> K2D / K3D -> T2D / T3D
-> hinge optimization
-> lift point selection
-> boundary-first string path generation
-> snap + lift + rigid + hinge + collision actuation
```

The deployed physical error is evaluated against the designed assembled tile configuration `T3D`, not against the raw target surface `S`.

## Quick Start

```powershell
cd onestring-physics-simulator
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
streamlit run app.py
```

If you do not want editable install:

```powershell
pip install -r requirements.txt
streamlit run app.py
```

## Default Demo

The default demo uses:

- target: dome
- grid: 3x3
- paper-like surface parameterization to `Omega`
- inverse parameterization lift `c^-1` from `M2D` in `Omega` back to `M3D` on `S`
- 3D optimization for `K3D` with planarity, square, and surface objectives
- 2D edge matching for `K2D`
- frustum extrusion for `T3D`, and `T2D` whose top vertices are generated from the optimized `K2D`
- dual-hinge-ready hinge data
- GPE-based lift point selection
- boundary-first gap-graph string routing
- quasi-static snap/lift deployment

## What Is Implemented

- Built-in height-field targets: flat, dome, saddle, wave, gaussian bump
- OBJ/STL/PLY loading through `trimesh`
- `OneStringDesignState` with formal intermediate representations
- Fig. 5-style pipeline viewer
- `SurfaceParameterization` representation for `c : S -> Omega` and `c^-1 : Omega -> S`
- M3D generation by UV triangle lookup and barycentric interpolation on the parameterized surface mesh
- K3D full-3D vertex optimization with a flattening guard
- `S`, `Omega`, `M2D`, `M3D`, `K3D`, `K2D`, `T3D`, `T2D top hinge`, and `T2D dual hinge` views
- Assembled 3D optimization metrics:
  - M3D construction method
  - parameterization method
  - M3D surface distance mean / max
  - UV triangle lookup failures
  - height-field shortcut flag
  - planarity error before / after
  - surface fit error before / after
  - square error before / after
  - edge length variance before / after
- Flat 2D edge matching metrics:
  - edge matching error before / after
  - max absolute `K2D.z` to prove the layout is planar
  - RMS and max displacement from `M2D` to `K2D`
  - independent-tile overlap count and minimum clearance
  - gap angle range
  - fabrication clearance proxy
- `T2D` correctness metrics:
  - top vertices matching `K2D`
  - top-vertex RMS displacement from `M2D`
  - 8 vertices per tile
  - side-face count
- Gap graph visualization with paper-style node labels:
  - `0` vertical gap
  - `1` horizontal gap
  - `-1` virtual boundary entrance
  - `-2` split boundary gap placeholder
- GPE-based lift point selection
- Boundary-first string path with turn angle and simplified channel friction
- Snap/lift actuation with metrics:
  - target surface fit error to `S`
  - final deployment error to `T3D`
  - snap error
  - lift error
  - rigid error
  - hinge error
  - collision count
  - turn angle total
  - estimated channel friction
  - kinetic energy
  - stable state
- High-Fidelity Physical Mode controls:
  - hinge rotational stiffness
  - hinge damping
  - tile mass and inertia proxy
  - gravity
  - contact friction
  - string channel friction
  - quasi-static pulling speed
  - solver substeps
  - damping ratio
- Complexity panel for grid-size scaling estimates
- Batched Plotly rendering for quad meshes and tile assemblies
- Optional PyTorch CUDA backend for tensorized K3D optimization when CUDA is available

## Removed From Default Actuation

The paper-faithful mode does not use these as the default actuator:

- central tendon
- debug goal attraction
- direct target-surface attraction
- rope-particle Verlet simulation as the primary deployment mechanism

Legacy rope/tendon code remains in the package for compatibility with earlier tests and examples, but the app defaults to the OneString-style constraint path.

## Current Approximations

- The default M3D construction is a paper-like inverse map: M2D vertices live in `Omega`, then each UV point is mapped back to `S` using UV triangle lookup and barycentric interpolation on the corresponding surface triangle.
- Direct height-field lifting `[u, v, z=f(u,v)]` is available only through the explicit `analytic_scaled_heightfield_debug` M3D construction mode and is not the default paper-like path.
- The current default parameterization solves a simple uniform-Laplacian harmonic map with the mesh boundary fixed to a rectangle. It is not BFF/LSCM, and the UI reports it as `harmonic` only because the Laplacian solve is actually performed.
- CSF estimation and split placement are simplified.
- `K3D` optimization uses a compact least-squares height-field approximation rather than the full projection stack from the paper.
- `K3D` optimization rejects invalid flattened results and falls back to `M3D` rather than passing a collapsed assembled state downstream.
- `K2D` edge matching uses a simplified optimizer/relaxation model. The stored mesh remains planar with `z = 0`, and the app renders an independent per-tile top-face layout with visible gaps instead of a continuous terrain-like quad surface.
- `T2D` is a fabrication-layout approximation generated from the independent `K2D` tile top vertices and projected frustum offsets, with side faces and hinge markers exposed for inspection.
- Frustum extrusion uses per-tile normal offsets plus face-planarity reporting.
- Dual hinge placement uses a local dihedral proxy rather than the full global fabrication optimization.
- Morse-Smale lift point selection is approximated with GPE peaks and threshold clustering.
- Collision handling is AABB-based with projection penalties.
- String channel friction is a simplified Capstan-style estimate from cumulative turn angle.
- High-fidelity mode is an exposed extension mode, not a validated physical contact simulator.

## Performance And GPU Notes

The app avoids generating every stage figure on each Streamlit rerun. Use the `View stage` selector to render only one stage at a time. Animation is generated on demand from the final deployment view.

GPU acceleration is used for tensorized optimization when PyTorch CUDA is available and the compute backend is set to `auto` or `cuda`. UI rendering, Plotly visualization, file I/O, and graph routing remain CPU-side. If `cuda` is explicitly requested but CUDA is unavailable in the Streamlit Python environment, the app raises a visible error instead of silently falling back to CPU.

Current GPU coverage:

- K3D optimization: optional PyTorch CUDA path for analytic height fields
- K2D optimization: optional PyTorch CUDA path, otherwise SciPy/projective NumPy path
- Deployment simulation: optional PyTorch CUDA constraint path, otherwise CPU constraint projection path

The `Complexity / Backend` view separates topology growth from backend status so slowdowns are easier to attribute. It reports `sys.executable`, PyTorch version, PyTorch CUDA version, torch import path, CUDA device count/current device, GPU name, capability, and memory counters. It also includes:

- `Run GPU self-test`, which allocates a CUDA tensor of shape `(4096, 4096)`, runs `x @ x.T`, synchronizes, and reports elapsed time plus peak memory.
- `nvidia-smi` probing, which checks whether the NVIDIA driver can see the GPU independently of PyTorch.

If `nvidia-smi` sees a GPU but `torch_available` or `cuda_available` is false, install a CUDA-enabled PyTorch build into the same `.venv` used by Streamlit, following the current official PyTorch selector for your driver/CUDA combination.

The string routing metrics use `log_channel_cost = mu_c * theta_total` as the stable routing cost. The display-only Capstan friction estimate uses a guarded `expm1` calculation and returns `inf` instead of raising `OverflowError` when the exponent is too large.

## Command Line Smoke Test

```powershell
python -m pytest
```

You can also run the OneString pipeline in Python:

```python
from onestring_physics.input_shape import create_builtin_shape
from onestring_physics.onestring_pipeline import (
    DeploymentParameters,
    PipelineParameters,
    build_onestring_design,
    simulate_onestring_deployment,
)

target = create_builtin_shape("dome", {"amplitude": 0.75, "radius": 2.2})
state = build_onestring_design(target, PipelineParameters(nx=3))
state.simulation_result = simulate_onestring_deployment(
    state,
    DeploymentParameters(steps=32, solver_iterations=12),
)
print(state.simulation_result.metrics)
```

## Repository Layout

```text
onestring-physics-simulator/
  app.py
  src/onestring_physics/
    onestring_pipeline.py
    visualization.py
    animation.py
    ...
  examples/
  tests/
  docs/
```

See `docs/physics_model.md`, `docs/limitations.md`, and `docs/roadmap.md` for older v0.1 notes. The Streamlit app and `onestring_pipeline.py` are now the canonical entry points for the paper-faithful approximation.
