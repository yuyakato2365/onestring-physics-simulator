# Autodesk ABD backend

## Backend boundary

`legacy` runs the existing rigid projection and SAT collision pass. `abd` runs
an external executable built from Autodesk/affine-body-dynamics. The bridge does
not contain an approximate ABD solver and never uses SAT after an ABD failure.

The upstream source was inspected at commit
`fb65c94a5ea895405bc0f537e9f53e08381c32ba`. Its stock headless interface is:

```text
abd_sim --ngui --scene-path scene.json --output-path result --output-name sim.json
```

The executable writes `sim.json` and `sim.glb`. The JSON contains body position
and affine transform per frame, Newton iteration counts, active contact counts,
minimum distances, and solver statistics.

## OneString bridge

`src/onestring_physics/abd_backend.py` exports each authoritative T3D solid as a
closed OBJ mesh and rigidly places it in the initial T2D configuration. Hinge
anchors become Autodesk `pin_joint` constraints. Adjacent bodies keep distinct
collision groups, so hinge adjacency does not disable volume contact globally.

The separate `onestring_manifest.json` contains:

- tile thickness, density, initial position/orientation;
- hinge anchors and axes;
- tile-local string guide points;
- `L(q) <= L_command(t)` pull schedule;
- zero compression force while slack;
- prescribed shake amplitude, frequency, direction, start, and end time;
- required frame-log field names.

The stock upstream executable has no unilateral total-guide-length constraint.
For that reason, a full OneString run requires an extension that advertises the
CLI option `--onestring-manifest`. Capability probing happens before simulation;
missing capability is a hard error.

## Build and verification status on this machine

The official repository was downloaded successfully, but Release compilation
could not be performed because CMake, a Visual Studio C++ toolchain, and CUDA
`nvcc` were not installed/discoverable. No claim is made that ABD or IPC ran on
this machine. The bridge serialization smoke test does pass.

After installing the toolchain, run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_autodesk_abd.ps1
powershell -ExecutionPolicy Bypass -File scripts\verify_autodesk_abd.ps1
```

Only after the official sample passes should the OneString extension be built
and selected in Streamlit.

## CPU/GPU reporting

The bridge does not infer GPU coverage from CUDA availability. It records the
extension's explicit `device_report` for CCD, Hessian assembly, and Newton solve.
If stock ABD does not report these fields, the result says `not reported` rather
than claiming GPU execution.
