# Original Zhang et al. CAD 2021 Native Integration (2026-10-27)

## 2026-10-27: verified blocker in genuine Segment (not a mesher fallback)

Native Windows run now gets through the ORIGINAL `Straighten.exe` using a
Qt 5.10-matched QPA plugin. Original `Segment.exe` still terminates with
`0xC0000409` on both the authors' original sample and generated flat
source, with targets 35 and 4000 and with `OMP_NUM_THREADS=1`.

A Windows first-chance C++ exception trace from the *actual* author
`Segment/SplitPoly.exe` identifies its uncaught exception message:

```text
Wrong username or hostname .
```

This is thrown during Segment's solver stage (authored `PardisoSolver`,
which calls the distributed `libpardiso600-WIN-X86-64.dll`).
The original authors' README specifies a PARDISO license requirement.
**This is a solver license/host-validation failure**, not proof the source
Omega chart, mesh topology, target count or four-direction field is invalid.
The native published package's `pardiso.lic` does not authorize this CI
runner. A legitimate license tied to the execution host is needed for
this binary; do NOT patch host checks or use forged license keys.

Non-circumvention options:
- Supply a valid PARDISO license for this host and repeat the original three
  stage test with that authorized license.
- Build the unmodified mathematical solver calls against an independently
  licensed compatible sparse linear solver after checking type, indexing,
  factorization mode and numerical equivalence, with this solver replacement
  explicitly identified and separately benchmarked. This is not yet done.

**No claim is made that Bunny + OptCuts + seam locks + CSF Split are
complete or validated.** Python bridge protocol unit tests are separate
from the authors' native geometric algorithm.


**Purpose:** OneString M2D generation via the **genuine authors' original**
*Quad Meshing with Coarse Layouts for Planar Domains* software, not Gmsh,
the dated independent triangle-pair heuristic, or regular Grid cropping.

**Isolation:** Branch `agent/zhang2021-independent-20261027` only. This
branch is independent from Claude's work and has not been merged.

## Original files: REC guest-share API still functional

Authors' homepage: https://kfckfckf.github.io/
Paper DOI: https://doi.org/10.1016/j.cad.2021.103084
Authors' Code: https://rec.ustc.edu.cn/share/8081ab10-dd29-11eb-b5e1-5fc36251b03b
Authors' Exe: https://rec.ustc.edu.cn/share/911f9e50-dd29-11eb-aa39-130e36684f1f

The official REC website gives a 2026 closure notice, but public legacy
guest-share API endpoints still returned original archives on 2026-10-27:

- `Code_new.zip`, 23,394,797 bytes, three authors' **C++ source** solutions
  `Straighten`, `Segment`, `QuadMeshSimplify`. They are VS2017 projects
  using OpenMesh, Qt5, CGAL 4.12, Eigen, MKL, PARDISO and other dependencies.
- `ExeProgram.zip`, 43,181,657 bytes, SHA256
  `be7971f41821800ce32e0e2c0c9a53e081b002ec33e60521ccb4e2c9be848d37`,
  genuine `Straighten.exe`, `Segment.exe`, `QMSim.exe`, DLLs,
  `WholePipeline.bat`, samples and original `readme.txt`.

The authors' README specifies Windows 10 and PARDISO
(https://www.pardiso-project.org/) as runtime requirements. Respect their
license terms. No author code, third-party DLL or `pardiso.lic` file is
redistributed or checked into this repository.

## Download to a separate worktree (does not affect Claude)

```bash
git fetch origin
git worktree add --detach ../onestring-zhang2021-independent \
    origin/agent/zhang2021-independent-20261027
cd ../onestring-zhang2021-independent/onestring-physics-simulator
python3 scripts/download_zhang2021_authors.py --directory ./output/zhang2021_official
python3 scripts/download_zhang2021_executables.py --directory ./output/zhang2021_official
```

If repository root is already the Python project folder, omit the
`/onestring-physics-simulator` path component above.

Official executable directory after download:
`output/zhang2021_official/authors_executables/ExeProgram`.

Note: download scripts check the official public share identity, file size,
and (for executable ZIP) original release SHA256. They *do not* compile or
modify the authors' algorithm.

## Real native pipeline and source encoding

The authors' `WholePipeline.bat` defines this exact chain:

```text
Straighten  Source/Input/model.obj   Source/before/model.obj  Source/after/model_axis.obj
Segment     Source/after/model_axis.obj  Result/ Source/before/model.obj  35  0.01
QMSim       Result/model.obj  Result/finalOut/model_out.obj  Source/before/model.obj  0  35  0.01  0.05
```

Here 35 is target quad count and 0.01 is the authors' Hausdorff bound.
The `QMSim` fourth positional argument 0 preserves features.
The original authors' executable requires a **triangle mesh OBJ with
texture coordinates (vt)**; UV texture coordinates form the planar domain.

## How OneString calls the EXACT original pipeline

`scripts/zhang2021_authors_native_adapter.py` is a wrapper around those
three binaries. It:

1. Reads the immutable OptCuts source NPZ with matched UV and XYZ triangles.
2. Writes OBJ with separate original 3D `v` and planar UV `vt` indices;
   never substitutes Gmsh or overwrites the original OptCuts seam.
3. Invokes the three original authors' EXEs in a temporary, isolated
   runtime copy (or under explicitly configured Wine). Enforces stage exit
   codes and output existence.
4. Parses the authors' final OBJ's quad faces using explicit texture UV;
   accepts planar XY-only OBJ only if geometrically planar. Never guesses
   an XYZ→UV projection.
5. Outputs candidate `vertices_uv` and `faces` NPZ for the existing
   `zhang2021_reference_bridge.py` strict source-area/topology/seam audits.

Standalone Windows example:

```powershell
python scripts/zhang2021_authors_native_adapter.py `
  --input C:\path\to\optcuts_source.npz `
  --output C:\path\to\candidate_quads.npz `
  --exe-root C:\path\to\ExeProgram `
  --target-quads 35
```

The source NPZ must be exported from the real official OptCuts source chart
using `export_source_chart(parameterization, path)`.

In the dedicated Streamlit `optcuts_zhang2021_reference_only` mode,
set "Official authors' ExeProgram folder". Alternatively set
`ONESTRING_ZHANG2021_EXE_ROOT`. A custom adapter command may be provided
with literal `{input}` / `{output}` / `{target_quads}` placeholders.
Only reference mode uses this wrapper; the old Gmsh mode stays independent.
Without the native executables, execution fails **without fallback**.

macOS cannot execute Windows `.exe` natively. A configured Wine runtime is
possible in principle (`--wine-command wine64`) but is not yet certified for
the old Qt/PARDISO runtime. Never claim Mac compatibility from Windows CI.

## Verification gates: indispensable for thesis reliability

- **Paper provenance:** original ZIP SHA256 and original EXE files
- **Algorithm:** must execute Straighten -> Segment -> QMSim, and never
  redirect to the old experimental mesh generator
- **Omega:** require UV triangulation and original polygon agreement;
  no inverted quads, outside-Omega area, overlap, excessive uncovered area
- **Lift:** each output UV vertex and selected interior/edge samples must
  admit the original paired OptCuts triangle barycentric inverse
- **Seams:** cut boundary edges on the two sides must agree in original
  3D cut location and subdivision; otherwise fail explicitly
- **Downstream:** M3D→K3D→T3D→K2D→T2D and unscripted irregular
  CSF Split must be tested separately. Regular-grid Split is not certified
  by the current synthetic-only CI.

### Critical honest limitations (not to claim completed)

- The original author ZIP and source code are successfully retrieved, but
  reproducing all output with OptCuts Bunny requires a native Windows/Wine
  execution that passes the strict original 3D seam tests.
- The authors' Quad Meshing method was not designed to synchronize
  two independently cut UV boundary banks of a 3D OptCuts seam. Post-hoc
  rejection is implemented, **generation-time seam locking is not**.
- The published executables are Windows-only and depend on a licensed
  PARDISO runtime. The VS2017 source doesn't compile directly on a
  modern fresh runner (v141, SDK 16299, missing Eigen and Qt/SuiteSparse
  configuration). Do not disguise this with a replacement algorithm.
- The reference-only native flow does not make arbitrary 30/35 quad targets
  unconditionally feasible; returned actual quad count is reported.
- This document is an implementation audit, not a claim the user's full
  Bunny/OptCuts/CSF Split research requirements have been completed.

## Exact separate GitHub Actions

- `.github/workflows/zhang2021-original-pipeline-smoke-20261027.yml`:
  runs official three exe stages on authors sample (real native code)
- `.github/workflows/zhang2021-official-executables-20261027.yml`:
  downloads/checks original official binary release
- `.github/workflows/zhang2021-original-windows-build-20261027.yml`:
  diagnoses compilation of original authors C++ sources under MSVC
- `.github/workflows/zhang2021-independent-20261027.yml`:
  protocol/geometry/CLI/dome regression; success does NOT imply original
  authors' native end-to-end or OptCuts Bunny success.

All changes are isolated from the user's active Claude branch.
