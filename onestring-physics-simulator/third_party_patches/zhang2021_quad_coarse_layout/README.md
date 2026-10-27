# Portability patch for the authors' code of Zhang et al. (CAD 2021)

Paper: C. Zhang, S. Chai, L. Liu, X.-M. Fu, *Quad meshing with coarse layouts
for planar domains*, Computer-Aided Design 140 (2021) 103084,
doi:10.1016/j.cad.2021.103084.
Code: `Code_new.zip` from the authors' share linked on Xiao-Ming Fu's
publication page (<https://rec.ustc.edu.cn/share/8081ab10-dd29-11eb-b5e1-5fc36251b03b>),
SHA-256 `49b70af5e6272c96aec2c3cb3aec84709477c9e51d033e24924218709917f4b3`.

**The authors' archive has no license file.  This repository therefore does
not contain their code**; `scripts/setup_zhang2021_quad_official.py` downloads
it, verifies the hash, applies `onestring_port.patch` and builds it under
`third_party/Zhang2021_QuadCoarseLayout_official/` (git-ignored).  The patch
contains, as any unified diff does, the few original lines it replaces plus one
line of context.  Ask the authors before redistributing anything further.

The authors' package consists of three Windows/Visual Studio programs, run by
their `WholePipeline.bat`:

| program | paper section | OneString build |
|---|---|---|
| `Straighten.exe` (Qt GUI, batch via command line) | 3.2.1 boundary edge clustering, bijective low-distortion straightening | `z21_straighten` |
| `Segment.exe` | 3.2.2 field-guided segmentation; 3.3.1 template quad meshing; 3.3.2 T-junction elimination and poly-chord subdivision | `z21_segment` |
| `QMSim.exe` | 3.3.3 flip removal, modified poly-chord collapse, SD geometric optimisation, Hausdorff control | `z21_qmsim` |

## What the patch changes (and what it does not)

No change to any algorithm, parameter, threshold, iteration count or output
format.  Every modified line is marked `portability:` / `OneString` in the
source.  Categories:

1. **Text normalisation (setup script, not in the diff).**  UTF-16-LE and
   UTF-8-BOM files to UTF-8; CRLF to LF; applied only to `.c .cc .cpp .cxx .h
   .hh .hpp .txt`.  Byte-for-byte reproducible (checked against the recorded
   normalised tree).
2. **MSVC-only syntax, no semantic change.**  `#include` paths with `\` →
   `/`; include case on case-sensitive file systems (`Eigen/Eigenvalues`,
   `Viewer2D_Segments.h`); `long double(x)` → `(long double)(x)`;
   `void main` → `int main` (and `return;` → `return 1;` on the usage-error
   path); MSVC's binding of a temporary to a non-const reference in default
   arguments (`std::vector<double>& g = std::vector<double>()`) expressed with
   `z21_lvalue()`, which yields the same temporary with the same lifetime;
   `__int64`; POSIX `LINE_MAX` macro clash with a local variable name.
3. **Library API.**  `SparseLU::m_factorizationIsOk` is protected in Eigen
   3.3/3.4; replaced by `info() != Eigen::Success`, which `factorize()` sets
   together with that flag (equivalent for `SparseLU`).
4. **Undefined behaviour.**  `SplicePipeline` deleted three never-initialised
   pointers in its destructor (crash with GCC); initialised to `NULL`.
5. **Preserving MSVC numerical semantics.**
   * With MSVC `<cmath>`, unqualified `abs(double)` is the floating-point
     overload; with libstdc++/libc++ it resolves to C `int abs(int)` and
     silently truncates (`abs(-0.3) == 0`).  The force-included prelude
     `onestring_port/include/z21_compat.h` re-exports `std::abs` (and
     `std::min/max`, which `<windows.h>` provided).
   * `-ffp-contract=off` (MSVC `/fp:precise` without `/arch:AVX2` never
     fuses multiply-add).
   * OpenMP exactly where the authors' Release|x64 projects enable it
     (Segment only).
   * `fmath.hpp` (herumi, BSD-3): x86 intrinsics are compiled only on x86;
     on other CPUs (Apple Silicon) the authors' own scalar `#else` branch of
     `expd()` is used.  Both branches perform the same IEEE operations; the
     results of 2,000,000 random arguments hash identically (GCC and Clang).
6. **Platform services.**  `windows.h` shim (`CreateDirectory` → `mkdir`),
   empty `io.h`; Windows text-mode CRLF translation in the OBJ reader
   (`Mesh_doubleIO.cpp` drops a trailing `\r`).
7. **Proprietary dependencies.**
   * PARDISO 6.0 (licence bound to a user/machine) and MKL PARDISO →
     `onestring_port/src/PardisoSolverEigen.cpp`, the authors'
     `PardisoSolver` class interface implemented with Eigen `SparseLU`
     (symmetric upper CSR expanded to the full matrix, one iterative
     refinement step as PARDISO's default `iparm[7]`).
   * MOSEK (`ConvexQuadOptimization`) is not reachable from the batch
     straightening path; it is linked to a stub that aborts loudly if ever
     called.
8. **GUI removal (Straighten only).**  The Qt/OpenGL classes `Viewer2D*` are
   replaced by headless classes that keep the non-rendering state and logic
   verbatim; `onestring_port/straighten/headless_main.cpp` replicates the
   command-line path of `main.cpp`, `ProcessContainer` and
   `InteractiveViewerWidget` used by `WholePipeline.bat` (open mesh, load
   parameterisation, mirror negative charts, cut bridges, pre-process,
   `polysquare` / `polysquareWithHole`, write `before`/`after`).  Exit code 4
   if straightening reports failure (the GUI program would continue; the
   OneString runner refuses).
9. **Build system.**  CMake (Release, C++14; Straighten C++17 because of the
   bundled rectpack2D); OpenMesh 7.1, ANN and Triangle from the authors'
   bundled sources (Triangle with the authors' `TRILIBRARY`/`REAL=double`
   defines); Straighten keeps its bundled Eigen 3.3.5 include order.

Known harmless quirk of the original code: it writes debug text files to the
hard-coded path `D:\AfterSummerHoloday_8_11\OutInfo\...`; on POSIX these become
oddly named files in the working directory (the OneString runner uses a
temporary directory).

## Validation against the authors' executables

The authors' `ExeProgram.zip` (`Straighten.exe`, `Segment.exe`,
`QMSim.exe`) was run under Wine 9 (Xvfb).  Their PARDISO 6.0 DLL requires a
personal licence, so it was replaced by a MinGW-built DLL exporting the same
`pardisoinit`/`pardiso` API on top of Eigen `SparseLU` (identical to the
port's solver).  Five of the authors' sample inputs, paper defaults
(N_tar = 4000, ε = 0.01, l_collapse = 0.05):

| input | port N_q / N_p / J_min / d_H% | original N_q / N_p / J_min / d_H% | original, input × (1 ± 1e-12) |
|---|---|---|---|
| model_15 | 3666 / 84 / 0.414 / 0.875 | 3380 / 78 / 0.397 / 0.939 | 6874 / 338 / 0.323 / 0.904 |
| sharp_3 | 4004 / 27 / 0.736 / 0.701 | 4043 / 27 / 0.768 / 0.700 | 4043 / 27 / 0.758 / 0.700 |
| boundary254_mesh | 3399 / 14 / 0.601 / 0.690 | 3399 / 14 / 0.601 / 0.690 | 3257 / 12 / 0.584 / 0.781 |
| AMesh_mesh_input_milu | 3732 / 62 / 0.322 / 0.946 | 3909 / 67 / 0.322 / 0.937 | 3867 / 68 / 0.322 / 0.929 |
| AMesh_mesh_input_squirrel | 3327 / 38 / 0.534 / 0.486 | 3315 / 38 / 0.535 / 0.489 | 3315 / 38 / 0.535 / 0.488 |

* Straighten's pre-processed domain (`before.obj`) is bitwise identical to the
  original program's; Segment's output is identical up to element ordering
  when given identical input.
* The discrete decisions in Segment/QMSim (poly-chord collapse order, field
  singularity placement) are chaotic in the last bits: perturbing the input
  of the **original** executables by 1e-12 changes the result as much as, or
  more than, the port does (model_15: 3380 → 6874 quads).  Bitwise identity
  with the Windows build is therefore not attainable on any other compiler;
  the port reproduces the original within the original's own sensitivity.
* `scripts/setup_zhang2021_quad_official.py` builds binaries whose outputs
  are bitwise identical to the validated build (Linux x86-64, GCC 13).
  macOS/Apple-Silicon builds use the scalar `fmath` branch (bitwise equal)
  and Apple's libm; expect last-bit differences with the same chaotic
  amplification.
