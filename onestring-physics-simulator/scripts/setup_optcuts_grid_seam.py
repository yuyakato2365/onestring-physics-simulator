#!/usr/bin/env python3
"""Build an isolated OptCuts fork with native UV-grid seam penalties.

NEVER edits third_party/OptCuts_official or third_party/OptCuts. Runs the
existing clean-source setup in a separate tree, patches its verified C++ source
anchors, then rebuilds only this experimental fork.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT / "third_party" / "OptCuts_grid_seam_onestring"
MARKER = "ONESTRING_NATIVE_GRID_SEAM_V3_SAFE_RANK"

HELPER = r"""
// ONESTRING_NATIVE_GRID_SEAM_V3_SAFE_RANK
// Deliberately isolated from unmodified Official OptCuts.
static double onestringGridWeight() {
    const char* raw = std::getenv("ONESTRING_OPTCUTS_GRID_SEAM_WEIGHT");
    if (!raw || !*raw) return 0.0;
    char* end = nullptr;
    double value = std::strtod(raw, &end);
    return (end != raw && end && *end == '\0' && std::isfinite(value)
            && value >= 0.0 && value <= 100.0) ? value : 0.0;
}
static double onestringGridAngle() {
    const char* raw = std::getenv("ONESTRING_OPTCUTS_GRID_SEAM_ANGLE_DEGREES");
    if (!raw || !*raw) return 0.0;
    char* end = nullptr;
    double value = std::strtod(raw, &end);
    return (end != raw && end && *end == '\0' && std::isfinite(value))
        ? value * (3.14159265358979323846 / 180.0) : 0.0;
}
// Axis misalignment: sin^2(2 theta), i.e. 0 for 0/90 deg, 1 for 45 deg.
// Score proposal directions, NOT a new global energy. Preserve every
// original candidate and its SD/seam energyChanges used by the lambda dual
// update. V2's origEwDec <= 0 hard exclusion incorrectly erased candidate
// information and could cause repeated -DBL_MAX searches.
// The grid adjustment is bounded to 25% of an ORIGINAL POSITIVE gain;
// negative/zero gains remain unchanged for the original lambda controller.
// Scaling uses the original seam coefficient lambda_t (1-lambda_init).
// The actual global SD and bijectivity constraints remain unchanged.
static double onestringGridEdgeCost(
    const Eigen::RowVector2d& a, const Eigen::RowVector2d& b,
    double physicalLength, double virtualRadius
) {
    const double t = onestringGridAngle();
    const double c = std::cos(t), s = std::sin(t);
    const double dx = b[0] - a[0], dy = b[1] - a[1];
    const double u = c * dx + s * dy;
    const double v = -s * dx + c * dy;
    const double d2 = u * u + v * v;
    if (!(d2 > 1.0e-24) || !(virtualRadius > 0.0)) return 0.0;
    const double axisMismatch = 4.0 * u * u * v * v / (d2 * d2);
    return (physicalLength / virtualRadius) * axisMismatch;
}
// Preserve the legacy score's sign and the native dual-update information.
// Move proposals are only reprioritized when the OLD objective already
// predicts positive decrease. No formerly invalid candidate is made valid.
static double onestringGridProposalScore(
    double legacy, double lambda_t, double gridCost, bool merge
) {
    if (!(legacy > 0.0) || !std::isfinite(legacy) ||
        !(gridCost >= 0.0) || !std::isfinite(gridCost)) return legacy;
    const double beta = onestringGridWeight();
    if (!(beta > 0.0) || !(lambda_t > 0.0)) return legacy;
    const double raw = lambda_t * beta * gridCost;
    const double delta = std::min(raw, 0.25 * legacy);
    return merge ? legacy + delta : legacy - delta;
}
"""

def replace_once(data: str, needle: str, replacement: str, name: str) -> str:
    count = data.count(needle)
    if count != 1:
        raise ValueError(f"OptCuts upstream changed: {name}: {count} matches, expected exactly 1")
    return data.replace(needle, replacement, 1)


def patch_trimesh_source(source: str) -> str:
    if MARKER in source:
        raise ValueError("grid-aware C++ source already patched: rebuild clean tree first")
    source = replace_once(source, '#include <fstream>\n',
                          '#include <fstream>\n#include <cmath>\n#include <cstdlib>\n#include <algorithm>\n#include <stdexcept>\n',
                          "C++ includes")
    source = replace_once(source, 'namespace OptCuts {\n',
                          'namespace OptCuts {\n' + HELPER + '\n',
                          "OptCuts namespace")
    # Boundary split: a newly exposed seam edge incurs an additional
    # orientation cost. Its old SD local solver and orientation check are
    # unchanged. Weight=0 reproduces the original decision score exactly.
    before = (
        'const double curEwDec = (1.0 - lambda_t) * SDDec - lambda_t * seInc;\n'
        '                    if(curEwDec > maxEwDec) {'
    )
    after = (
        'const double gridCost = onestringGridEdgeCost(\n'
        '                        V.row(vI), V.row(nbVI),\n'
        '                        (V_rest.row(vI) - V_rest.row(nbVI)).norm(), virtualRadius);\n'
        '                    const double origEwDec = (1.0 - lambda_t) * SDDec - lambda_t * seInc;\n'
        '                    const double curEwDec = onestringGridProposalScore(\n'
        '                        origEwDec, lambda_t, gridCost, false);\n'
        '                    if(curEwDec > maxEwDec) {'
    )
    source = replace_once(source, before, after, "boundary split score")

    # Interior split: two new seam edges; path changes in querySplit itself.
    before = (
        'const double EwDec = (1.0 - lambda_t) * SDDec - lambda_t * seInc;\n'
        '                    if(EwDec > EwDec_max) {'
    )
    after = (
        'const double gridCost = onestringGridEdgeCost(\n'
        '                        V.row(path[0]), V.row(path[1]),\n'
        '                        (V_rest.row(path[0]) - V_rest.row(path[1])).norm(), virtualRadius)\n'
        '                        + onestringGridEdgeCost(\n'
        '                        V.row(path[1]), V.row(path[2]),\n'
        '                        (V_rest.row(path[1]) - V_rest.row(path[2])).norm(), virtualRadius);\n'
        '                    const double origEwDec = (1.0 - lambda_t) * SDDec - lambda_t * seInc;\n'
        '                    const double EwDec = onestringGridProposalScore(\n'
        '                        origEwDec, lambda_t, gridCost, false);\n'
        '                    if(EwDec > EwDec_max) {'
    )
    source = replace_once(source, before, after, "interior split score")

    # Merge removes an internal seam segment. Reward removing a non-grid
    # aligned segment. A coherent physical edge is V_rest[path[0],path[1]].
    before = (
        'return lambda_t * seDec - (1.0 - lambda_t) * SDInc;'
    )
    after = (
        'const double gridRemoved = onestringGridEdgeCost(\n'
        '                        V.row(path_max[0]), V.row(path_max[1]),\n'
        '                        (V_rest.row(path_max[0]) - V_rest.row(path_max[1])).norm(), virtualRadius);\n'
        '                    const double origEwDec = lambda_t * seDec - (1.0 - lambda_t) * SDInc;\n'
        '                    return onestringGridProposalScore(\n'
        '                        origEwDec, lambda_t, gridRemoved, true);'
    )
    source = replace_once(source, before, after, "merge score")
    # Fail closed on repeated identical all-invalid candidate slates. The
    # original optimizer can otherwise rescan thousands of vertices forever
    # when lambda does not change. Never substitute a fabricated Cut/UV.
    old_query = (
        '        EwDec_max = EwDecs[candI_max];\n'
        '        switch(operationType) {'
    )
    new_query = (
        '        EwDec_max = EwDecs[candI_max];\n'
        '        if(!propagate) {\n'
        '            static int noCandidateStreak[2] = {0, 0};\n'
        '            static double noCandidateLambda[2] = {-1.0, -1.0};\n'
        '            static int noCandidateVertexCount[2] = {-1, -1};\n'
        '            const int kind = splitInterior ? 1 : 0;\n'
        '            const bool allInvalid = EwDec_max <= -__DBL_MAX__ / 2.0;\n'
        '            if(allInvalid &&\n'
        '               std::abs(noCandidateLambda[kind] - lambda_t) < 1.e-13 &&\n'
        '               noCandidateVertexCount[kind] == V_rest.rows()) {\n'
        '                ++noCandidateStreak[kind];\n'
        '            } else {\n'
        '                noCandidateStreak[kind] = allInvalid ? 1 : 0;\n'
        '            }\n'
        '            noCandidateLambda[kind] = lambda_t;\n'
        '            noCandidateVertexCount[kind] = static_cast<int>(V_rest.rows());\n'
        '            if(noCandidateStreak[kind] >= 3) {\n'
        '                std::cerr << "[GRID-CUT-NATIVE-NO-CANDIDATES] kind="\n'
        '                    << kind << " examined=" << bestCandVerts.size()\n'
        '                    << " lambda_t=" << lambda_t\n'
        '                    << " vertices=" << V_rest.rows() << std::endl;\n'
        '                throw std::runtime_error(\n'
        '                    "GRID_CUT_NATIVE_NO_ADMISSIBLE_SPLIT_AT_FIXED_LAMBDA");\n'
        '            }\n'
        '        }\n'
        '        switch(operationType) {'
    )
    source = replace_once(source, old_query, new_query, "all-invalid candidate slate diagnostic")
    return source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--reclone", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    root = args.root.expanduser().resolve()
    if root in ((ROOT / "third_party" / "OptCuts").resolve(),
                (ROOT / "third_party" / "OptCuts_official").resolve()):
        parser.error("Refusing to patch the Official or legacy Grid-OptCuts tree")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "setup_optcuts_official.py"),
         "--root", str(root), "--jobs", str(args.jobs)]
        + (["--reclone"] if args.reclone else []),
        check=True,
    )
    file = root / "src" / "TriMesh.cpp"
    original = file.read_text()
    patched = patch_trimesh_source(original)
    file.write_text(patched)
    subprocess.run(["cmake", "--build", str(root / "build"),
                    "--config", "Release", "--parallel", str(args.jobs)],
                   check=True)
    executables = [root / "build" / "OptCuts_bin",
                   root / "build" / "Release" / "OptCuts_bin"]
    binary = next((p for p in executables if p.is_file()), None)
    if binary is None:
        raise RuntimeError(f"Patched OptCuts binary not found; checked {executables}")
    print(f"[GRID-CUT-NATIVE] compiled isolated C++ seam-scoring fork: {binary}")
    print(f"[GRID-CUT-NATIVE] source marker: {MARKER}")


if __name__ == "__main__":
    main()
