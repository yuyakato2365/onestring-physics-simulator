# Official BFF backend

Version 0.2.0 does not implement a BFF-like substitute.  The
`paper_reference_bff` mode invokes the official
[GeometryCollective Boundary First Flattening](https://github.com/GeometryCollective/boundary-first-flattening)
command-line program and fails with `ReferenceBFFUnavailableError` when that
program is unavailable or rejects the input.

## Boundary policy

The OneString paper says that the flattened boundary is rectangular *where
possible*, but does not identify the exact BFF boundary-data call used by the
authors.  This is `UNSPECIFIED_IN_PAPER`.

The official CLI's default open-surface path constructs a zero boundary scale
factor vector and calls `BFF::flatten(u, true)`.  Version 0.2.0 exposes that as:

- `automatic_reference` (default; effective policy `boundary_scale_zero`)
- `boundary_scale_zero` (the same official CLI path, named explicitly)
- `target_disk` (official `--flattenToDisk` path)

`target_rectangle` and `custom_boundary_curvature` are listed as potential BFF
policies but are not exposed faithfully by the official CLI.  Requesting either
therefore fails.  The implementation does not replace them with a rectangular
harmonic map.

## Installation

Clone the official repository, including its provided Windows binaries:

```powershell
git clone https://github.com/GeometryCollective/boundary-first-flattening.git third_party/boundary-first-flattening
$env:ONESTRING_BFF_EXECUTABLE = (Resolve-Path '.\third_party\boundary-first-flattening\binaries\windows-v1.6\bff-command-line.exe')
```

On macOS/Linux, build the official CLI following its README, then point the same
environment variable at the resulting `bff-command-line` executable.

An explicit path may instead be supplied as
`PipelineParameters(bff_executable=...)`.  At runtime the JSON diagnostics store
the executable path, version hint, SHA-256 hash, requested/effective boundary
policy, and the enclosing official Git checkout's commit SHA. A standalone
prebuilt binary with no Git metadata reports `UNSPECIFIED_BY_PREBUILT_BINARY`.

## Strict input and output checks

Before launching BFF, version 0.2.0 requires:

- finite triangle-mesh vertices;
- no zero-area triangle;
- no non-manifold edge;
- consistent face winding;
- one connected component;
- exactly one valid boundary loop;
- Euler characteristic 1 (disk topology).

No automatic hole filling, cutting, reorientation, welding, or degeneracy
repair occurs in reference mode.  The official CLI may renumber vertices, so
the wrapper reconstructs input order from corresponding output face corners and
rejects inconsistent/non-bijective mappings.

## Minimal verification command

```powershell
python scripts\verify_reference_bff.py
```

The command runs a five-vertex planar disk, checks the official golden result
up to a similarity transform, and checks exact round-trip barycentric lifting.
