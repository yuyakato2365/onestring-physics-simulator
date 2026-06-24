# onestring-physics-simulator

A small Python research prototype inspired by **One String to Pull Them All: Fast Assembly of Curved Structures from Flat Auxetic Linkages**.

This simulator is a simplified research prototype inspired by OneString. It includes a physical deployment simulation using rigid tiles, hinge constraints, rope particles, gravity, and time stepping. It does not reproduce the full OneString inverse design, friction-aware string path optimization, rigid-body contact simulation, or fabrication pipeline.

The goal of v0.1 is to make the distinction visible:

- **Mode A: Design Optimization** fits a small quad-tile linkage to a height-field target.
- **Mode B: Physical Deployment Simulation** starts from a flat initial state and evolves with gravity, damping, rigid tile constraints, hinge constraints, rope particles, and a pull handle.

The deployment mode is not a direct shape interpolation. A debug goal-attraction option exists in the UI, but it is off by default.

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
- tile size: 1.0
- gap size: 0.08
- design iterations: 30
- physics `dt`: 0.01
- substeps: 4
- solver iterations: 20
- frames: 200
- high rope/hinge stiffness
- moderate damping

## What v0.1 Includes

- Built-in height-field targets: flat, dome, saddle, wave, gaussian bump
- Simple OBJ/STL/PLY loading through `trimesh`
- Small quad grid generation
- Least-squares target fitting with rigidity, hinge, smoothness, and flat-compatibility residuals
- Flat initial configuration generation
- Lightweight custom XPBD/PBD-style physical deployment
- Rigid tile behavior through edge and diagonal constraints
- Point hinge constraints between neighboring tiles
- Rope particles, distance constraints, pull handle, and rope-tile coupling
- Plotly 3D visualization and animation frames
- Metrics comparing optimized assembled state and physically simulated final state

## What v0.1 Does Not Fully Handle

- The complete OneString inverse design pipeline
- Friction-aware string path optimization
- Exact 3D printability
- Robust arbitrary high-resolution quad remeshing
- Contact-rich rigid body simulation
- 8-vertex frustum tiles
- Top/bottom hinge switching
- Split optimization

## Command Line Smoke Test

```powershell
python -m pytest
```

You can also run a quick demo in Python:

```python
from onestring_physics.examples import run_default_demo

result = run_default_demo(num_frames=60)
print(result.metrics)
```

## Repository Layout

```text
onestring-physics-simulator/
  app.py
  src/onestring_physics/
  examples/
  tests/
  docs/
```

See [docs/physics_model.md](docs/physics_model.md), [docs/limitations.md](docs/limitations.md), and [docs/roadmap.md](docs/roadmap.md) for details.
