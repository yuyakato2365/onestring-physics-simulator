"""Patch the local OptCuts C++ source so scale factor participates in seam search.

This is intentionally an *internal* OptCuts modification, not an outer Python
candidate selector.  The patch changes ``src/TriMesh.cpp`` in the local OptCuts
checkout used by OneString.  Each split/merge candidate keeps OptCuts' original
seam-length + Symmetric-Dirichlet score and adds a soft reward for reducing the
OneString scale-factor violation.  The behavior is gated by environment
variables, so the ordinary OptCuts/test1 baselines remain unchanged even though
they use the same rebuilt binary.

The scale metric matches the OneString diagnostic convention as closely as is
practical inside OptCuts: for each triangle, lambda is sigma_max of the local
UV->surface differential, i.e. 1 / sigma_min(surface->UV).  A global similarity
scale is removed in log space.  The soft penalty is zero when all local scales
fit inside a multiplicative band ``scale_bound`` (default 2).
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
from typing import Final


PATCH_MARKER: Final[str] = "ONESTRING_INTERNAL_SCALE_FACTOR_PATCH_V1"

_HELPER_BLOCK = r'''

    // ONESTRING_INTERNAL_SCALE_FACTOR_PATCH_V1
    // Optional OneString-aware term used only when the environment flag is set.
    // It augments OptCuts' *internal* topology candidate score; it does not
    // replace OptCuts or select among several completed OptCuts runs.
    static bool oneStringScaleEnabled()
    {
        const char* raw = std::getenv("ONESTRING_OPTCUTS_INTERNAL_SCALE_ENABLED");
        if(!raw) { return false; }
        const std::string value(raw);
        return !(value.empty() || value == "0" || value == "false" || value == "False" || value == "off");
    }

    static double oneStringScaleEnv(const char* name, double fallback)
    {
        const char* raw = std::getenv(name);
        if(!raw) { return fallback; }
        try {
            const double value = std::stod(std::string(raw));
            return std::isfinite(value) ? value : fallback;
        }
        catch(...) {
            return fallback;
        }
    }

    static double oneStringScalePenalty(const TriMesh& mesh, double* rangeOut = NULL)
    {
        if(mesh.F.rows() == 0) {
            if(rangeOut) { *rangeOut = 1.0; }
            return 0.0;
        }

        const double bound = std::max(1.000001,
            oneStringScaleEnv("ONESTRING_OPTCUTS_INTERNAL_SCALE_BOUND", 2.0));
        const double halfBand = 0.5 * std::log(bound);
        std::vector<double> logLambda(mesh.F.rows(), 0.0);
        double minLog = __DBL_MAX__;
        double maxLog = -__DBL_MAX__;

        for(int triI = 0; triI < mesh.F.rows(); ++triI) {
            const Eigen::Vector3i tri = mesh.F.row(triI);
            const Eigen::Vector3d x3D[3] = {
                mesh.V_rest.row(tri[0]), mesh.V_rest.row(tri[1]), mesh.V_rest.row(tri[2])
            };
            const Eigen::Vector2d uv[3] = {
                mesh.V.row(tri[0]), mesh.V.row(tri[1]), mesh.V.row(tri[2])
            };
            Eigen::Matrix2d dg;
            IglUtils::computeDeformationGradient(x3D, uv, dg); // surface -> UV

            const double a = dg.col(0).squaredNorm();
            const double b = dg.col(0).dot(dg.col(1));
            const double c = dg.col(1).squaredNorm();
            const double trace = a + c;
            const double disc = std::sqrt(std::max(0.0, (a - c) * (a - c) + 4.0 * b * b));
            const double sigmaMinSq = std::max(1.0e-24, 0.5 * (trace - disc));
            const double sigmaMin = std::sqrt(sigmaMinSq);
            const double lambda = 1.0 / sigmaMin; // sigma_max(UV -> surface)
            const double ll = std::log(std::max(lambda, 1.0e-12));
            logLambda[triI] = ll;
            minLog = std::min(minLog, ll);
            maxLog = std::max(maxLog, ll);
        }

        if(rangeOut) { *rangeOut = std::exp(maxLog - minLog); }

        // Remove arbitrary global UV similarity scaling.  The midpoint of the
        // log extrema is the best center for a multiplicative hard band.
        const double center = 0.5 * (minLog + maxLog);
        double penalty = 0.0;
        double weightSum = 0.0;
        for(int triI = 0; triI < mesh.F.rows(); ++triI) {
            const double violation = std::max(0.0, std::abs(logLambda[triI] - center) - halfBand);
            const double w = (triI < mesh.triArea.size()) ? std::max(mesh.triArea[triI], 1.0e-16) : 1.0;
            penalty += w * violation * violation;
            weightSum += w;
        }
        return penalty / std::max(weightSum, 1.0e-16);
    }

    static double oneStringScaleReward(const TriMesh& before, const TriMesh& after,
                                       double* beforeRange = NULL, double* afterRange = NULL)
    {
        if(!oneStringScaleEnabled()) { return 0.0; }
        const double weight = std::max(0.0,
            oneStringScaleEnv("ONESTRING_OPTCUTS_INTERNAL_SCALE_WEIGHT", 40.0));
        const double p0 = oneStringScalePenalty(before, beforeRange);
        const double p1 = oneStringScalePenalty(after, afterRange);
        return weight * (p0 - p1);
    }
'''


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"OptCuts internal patch expected one {label}, found {count}")
    return text.replace(old, new, 1)


def patch_trimesh_cpp(source: str) -> str:
    if PATCH_MARKER in source:
        return source

    source = _replace_once(
        source,
        "#include <fstream>\n",
        "#include <fstream>\n#include <cstdlib>\n#include <cmath>\n#include <string>\n",
        "standard include insertion point",
    )
    source = _replace_once(
        source,
        "namespace OptCuts {\n",
        "namespace OptCuts {\n" + _HELPER_BLOCK,
        "namespace insertion point",
    )

    source = _replace_once(
        source,
        "                    return lambda_t * seDec - (1.0 - lambda_t) * SDInc;\n",
        "                    double objectiveDec = lambda_t * seDec - (1.0 - lambda_t) * SDInc;\n"
        "                    if(oneStringScaleEnabled()) {\n"
        "                        TriMesh candidate(*this);\n"
        "                        candidate.mergeBoundaryEdges(\n"
        "                            std::pair<int, int>(path_max[0], path_max[1]),\n"
        "                            std::pair<int, int>(path_max[1], path_max[2]), finder->second);\n"
        "                        candidate.computeFeatures();\n"
        "                        objectiveDec += oneStringScaleReward(*this, candidate);\n"
        "                    }\n"
        "                    return objectiveDec;\n",
        "merge candidate score",
    )

    source = _replace_once(
        source,
        "                    const double curEwDec = (1.0 - lambda_t) * SDDec - lambda_t * seInc;\n",
        "                    double curEwDec = (1.0 - lambda_t) * SDDec - lambda_t * seInc;\n"
        "                    if(oneStringScaleEnabled()) {\n"
        "                        TriMesh candidate(*this);\n"
        "                        candidate.splitEdgeOnBoundary(edge, newVertPosI);\n"
        "                        candidate.updateFeatures();\n"
        "                        curEwDec += oneStringScaleReward(*this, candidate);\n"
        "                    }\n",
        "boundary split candidate score",
    )

    source = _replace_once(
        source,
        "                    const double EwDec = (1.0 - lambda_t) * SDDec - lambda_t * seInc;\n",
        "                    double EwDec = (1.0 - lambda_t) * SDDec - lambda_t * seInc;\n"
        "                    if(oneStringScaleEnabled()) {\n"
        "                        TriMesh candidate(*this);\n"
        "                        candidate.cutPath(path, true, 1, newVertPos);\n"
        "                        EwDec += oneStringScaleReward(*this, candidate);\n"
        "                    }\n",
        "interior split candidate score",
    )

    source = _replace_once(
        source,
        "        if(EwDec_max > EDecThres) {\n            if(isMerge) {\n",
        "        if(EwDec_max > EDecThres) {\n"
        "            if(oneStringScaleEnabled()) {\n"
        "                double beforeRange = 1.0, afterRange = 1.0;\n"
        "                (void)oneStringScalePenalty(*this, &beforeRange);\n"
        "                TriMesh candidate(*this);\n"
        "                if(isMerge) {\n"
        "                    candidate.mergeBoundaryEdges(\n"
        "                        std::pair<int, int>(path_max[0], path_max[1]),\n"
        "                        std::pair<int, int>(path_max[1], path_max[2]), newVertPos_max.row(0));\n"
        "                    candidate.computeFeatures();\n"
        "                }\n"
        "                else if(!splitInterior) {\n"
        "                    candidate.splitEdgeOnBoundary(\n"
        "                        std::pair<int, int>(path_max[0], path_max[1]), newVertPos_max);\n"
        "                    candidate.updateFeatures();\n"
        "                }\n"
        "                else {\n"
        "                    candidate.cutPath(path_max, true, 1, newVertPos_max);\n"
        "                }\n"
        "                (void)oneStringScalePenalty(candidate, &afterRange);\n"
        "                std::cout << \"[OPTCUTS-INTERNAL-SCALE] type=\"\n"
        "                          << (isMerge ? \"merge\" : (splitInterior ? \"interior_split\" : \"boundary_split\"))\n"
        "                          << \" range=\" << beforeRange << \"->\" << afterRange\n"
        "                          << \" bound=\" << oneStringScaleEnv(\"ONESTRING_OPTCUTS_INTERNAL_SCALE_BOUND\", 2.0)\n"
        "                          << \" weight=\" << oneStringScaleEnv(\"ONESTRING_OPTCUTS_INTERNAL_SCALE_WEIGHT\", 40.0)\n"
        "                          << std::endl;\n"
        "            }\n"
        "            if(isMerge) {\n",
        "accepted split/merge diagnostic",
    )
    return source


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def optcuts_root_from_executable(executable: Path | None = None) -> Path:
    if executable is not None:
        p = Path(executable).expanduser().resolve()
        for candidate in [p.parent, *p.parents]:
            if (candidate / "src" / "TriMesh.cpp").is_file() and (candidate / "CMakeLists.txt").is_file():
                return candidate
    env = os.environ.get("ONESTRING_OPTCUTS_SOURCE_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (_project_root() / "third_party" / "OptCuts").resolve()


def apply_internal_scale_patch(root: Path) -> bool:
    cpp = root / "src" / "TriMesh.cpp"
    if not cpp.is_file():
        raise RuntimeError(f"OptCuts source not found: {cpp}")
    original = cpp.read_text(encoding="utf-8")
    patched = patch_trimesh_cpp(original)
    if patched == original:
        return False
    backup = cpp.with_suffix(".cpp.onestring_original")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
    cpp.write_text(patched, encoding="utf-8")
    return True


def ensure_internal_scale_binary(executable: Path | None = None) -> Path:
    """Patch OptCuts' C++ candidate scoring and rebuild when necessary."""
    root = optcuts_root_from_executable(executable)
    changed = apply_internal_scale_patch(root)

    build_dir = root / "build"
    binary = Path(executable).expanduser().resolve() if executable else build_dir / "OptCuts_bin"
    source = root / "src" / "TriMesh.cpp"
    needs_build = changed or not binary.is_file()
    if binary.is_file() and source.stat().st_mtime > binary.stat().st_mtime:
        needs_build = True

    if needs_build:
        if not build_dir.exists():
            subprocess.run(
                ["cmake", "-S", str(root), "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release"],
                check=True,
            )
        jobs = str(max(1, min(12, os.cpu_count() or 1)))
        subprocess.run(
            ["cmake", "--build", str(build_dir), "--config", "Release", "-j", jobs],
            check=True,
        )

    if not binary.is_file():
        release_binary = build_dir / "Release" / "OptCuts_bin"
        if release_binary.is_file():
            binary = release_binary
        else:
            raise RuntimeError(f"Patched OptCuts binary was not produced under {build_dir}")

    print(
        "[OPTCUTS-INTERNAL-SCALE-PATCH] "
        f"source={'patched' if changed else 'already_patched'} binary={binary}"
    )
    return binary


__all__ = [
    "PATCH_MARKER",
    "apply_internal_scale_patch",
    "ensure_internal_scale_binary",
    "optcuts_root_from_executable",
    "patch_trimesh_cpp",
]
