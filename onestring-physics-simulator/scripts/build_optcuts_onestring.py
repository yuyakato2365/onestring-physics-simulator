#!/usr/bin/env python3
"""Build the dedicated OneString-aware OptCuts binary.

This script restores ``src/TriMesh.cpp`` from the checked-out upstream OptCuts
commit, applies the OneString scale-aware objective directly to that C++ source,
verifies the modified objective, and builds a dedicated binary under
``third_party/OptCuts/build_onestring``.

Nothing is rewritten when Streamlit runs. This is a build-time creation of a
separate modified OptCuts binary.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess


HELPER = r'''

    // OneString: scale-factor term for OptCuts seam-topology optimization.
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

    static void oneStringScaleViolation(
        const TriMesh& mesh,
        std::vector<double>& violationSq,
        double* rangeOut = NULL)
    {
        violationSq.assign(mesh.F.rows(), 0.0);
        if(mesh.F.rows() == 0) {
            if(rangeOut) { *rangeOut = 1.0; }
            return;
        }

        const double bound = std::max(
            1.000001, oneStringScaleEnv("ONESTRING_OPTCUTS_SCALE_BOUND", 2.0));
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
            const double disc = std::sqrt(std::max(0.0, (a-c)*(a-c) + 4.0*b*b));
            const double sigmaMinSq = std::max(1.0e-24, 0.5 * (trace - disc));
            const double lambda = 1.0 / std::sqrt(sigmaMinSq); // sigma_max(UV -> surface)
            const double ll = std::log(std::max(lambda, 1.0e-12));
            logLambda[triI] = ll;
            minLog = std::min(minLog, ll);
            maxLog = std::max(maxLog, ll);
        }

        if(rangeOut) { *rangeOut = std::exp(maxLog - minLog); }

        // Remove arbitrary global UV similarity scale. The multiplicative band
        // has total width `bound`, so its log half-width is log(bound)/2.
        const double center = 0.5 * (minLog + maxLog);
        for(int triI = 0; triI < mesh.F.rows(); ++triI) {
            const double violation = std::max(
                0.0, std::abs(logLambda[triI] - center) - halfBand);
            violationSq[triI] = violation * violation;
        }
    }

    static double oneStringScalePenalty(const TriMesh& mesh, double* rangeOut = NULL)
    {
        std::vector<double> violationSq;
        oneStringScaleViolation(mesh, violationSq, rangeOut);
        if(violationSq.empty()) { return 0.0; }

        double penalty = 0.0;
        double weightSum = 0.0;
        for(int triI = 0; triI < mesh.F.rows(); ++triI) {
            const double w = (triI < mesh.triArea.size())
                ? std::max(mesh.triArea[triI], 1.0e-16) : 1.0;
            penalty += w * violationSq[triI];
            weightSum += w;
        }
        return penalty / std::max(weightSum, 1.0e-16);
    }

    static double oneStringBoundaryStressExposure(const TriMesh& mesh)
    {
        // A topology-only cut leaves the current per-triangle UV deformation
        // unchanged, so comparing scale penalty before/after the cut is nearly
        // identically zero. Instead, score how much *current* scale violation is
        // released onto chart boundaries by the candidate seam. A split creates
        // boundary edges next to stressed triangles and increases this value;
        // a merge removes such freedom and decreases it. This is a cheap proxy
        // for the scale improvement available after subsequent OptCuts UV solves.
        std::vector<double> violationSq;
        oneStringScaleViolation(mesh, violationSq, NULL);
        if(violationSq.empty()) { return 0.0; }

        typedef std::pair<int, int> EdgeKey;
        std::map<EdgeKey, int> edgeCount;
        std::map<EdgeKey, int> edgeTri;
        for(int triI = 0; triI < mesh.F.rows(); ++triI) {
            const Eigen::Vector3i tri = mesh.F.row(triI);
            for(int e = 0; e < 3; ++e) {
                int a = tri[e];
                int b = tri[(e + 1) % 3];
                if(a > b) { std::swap(a, b); }
                const EdgeKey key(a, b);
                ++edgeCount[key];
                edgeTri[key] = triI;
            }
        }

        double exposedStress = 0.0;
        double totalArea = 0.0;
        for(int triI = 0; triI < mesh.F.rows(); ++triI) {
            totalArea += (triI < mesh.triArea.size())
                ? std::max(mesh.triArea[triI], 1.0e-16) : 1.0;
        }

        for(std::map<EdgeKey, int>::const_iterator it = edgeCount.begin();
            it != edgeCount.end(); ++it) {
            if(it->second != 1) { continue; }
            const EdgeKey& edge = it->first;
            const int triI = edgeTri[edge];
            if(triI < 0 || triI >= mesh.F.rows()) { continue; }

            const Eigen::Vector3d p0 = mesh.V_rest.row(edge.first);
            const Eigen::Vector3d p1 = mesh.V_rest.row(edge.second);
            const double edgeLength = std::max((p1 - p0).norm(), 1.0e-16);
            exposedStress += edgeLength * violationSq[triI];
        }

        // Normalize by sqrt(area) so the proxy remains roughly invariant under
        // uniform scaling of the input surface while retaining seam-length
        // information inside a fixed mesh.
        return exposedStress / std::sqrt(std::max(totalArea, 1.0e-16));
    }

    static double oneStringScaleReward(const TriMesh& before, const TriMesh& candidate)
    {
        const double weight = std::max(
            0.0, oneStringScaleEnv("ONESTRING_OPTCUTS_SCALE_WEIGHT", 40.0));
        if(weight == 0.0) { return 0.0; }

        // Positive for split candidates that expose high scale-stress regions to
        // a new seam; negative for merge candidates that remove such a seam.
        const double beforeExposure = oneStringBoundaryStressExposure(before);
        const double afterExposure = oneStringBoundaryStressExposure(candidate);
        return weight * (afterExposure - beforeExposure);
    }
'''


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise SystemExit(
            f"OptCuts source mismatch while editing {label}: expected 1 exact anchor, found {count}."
        )
    return source.replace(old, new, 1)


def make_onestring_source(upstream: str) -> str:
    source = upstream
    source = replace_once(
        source,
        "#include <fstream>\n",
        "#include <fstream>\n#include <cmath>\n#include <cstdlib>\n#include <map>\n#include <string>\n",
        "standard includes",
    )
    source = replace_once(
        source,
        "namespace OptCuts {\n",
        "namespace OptCuts {\n" + HELPER,
        "OptCuts namespace",
    )

    source = replace_once(
        source,
        "                    return lambda_t * seDec - (1.0 - lambda_t) * SDInc;\n",
        "                    double objectiveDec = lambda_t * seDec - (1.0 - lambda_t) * SDInc;\n"
        "                    TriMesh candidate(*this);\n"
        "                    candidate.mergeBoundaryEdges(\n"
        "                        std::pair<int, int>(path_max[0], path_max[1]),\n"
        "                        std::pair<int, int>(path_max[1], path_max[2]), finder->second);\n"
        "                    candidate.computeFeatures();\n"
        "                    objectiveDec += oneStringScaleReward(*this, candidate);\n"
        "                    return objectiveDec;\n",
        "merge candidate objective",
    )

    source = replace_once(
        source,
        "                    const double curEwDec = (1.0 - lambda_t) * SDDec - lambda_t * seInc;\n",
        "                    double curEwDec = (1.0 - lambda_t) * SDDec - lambda_t * seInc;\n"
        "                    TriMesh candidate(*this);\n"
        "                    candidate.splitEdgeOnBoundary(edge, newVertPosI);\n"
        "                    candidate.updateFeatures();\n"
        "                    curEwDec += oneStringScaleReward(*this, candidate);\n",
        "boundary split candidate objective",
    )

    source = replace_once(
        source,
        "                    const double EwDec = (1.0 - lambda_t) * SDDec - lambda_t * seInc;\n",
        "                    double EwDec = (1.0 - lambda_t) * SDDec - lambda_t * seInc;\n"
        "                    TriMesh candidate(*this);\n"
        "                    candidate.cutPath(path, true, 1, newVertPos);\n"
        "                    candidate.updateFeatures();\n"
        "                    EwDec += oneStringScaleReward(*this, candidate);\n",
        "interior split candidate objective",
    )
    return source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--optcuts-root", type=Path, default=None,
        help="upstream OptCuts checkout (default: third_party/OptCuts)",
    )
    args = parser.parse_args()

    root = project_root()
    optcuts = (args.optcuts_root or (root / "third_party" / "OptCuts")).expanduser().resolve()
    cpp = optcuts / "src" / "TriMesh.cpp"
    if not cpp.is_file() or not (optcuts / ".git").exists():
        raise SystemExit(
            f"OptCuts checkout not found at {optcuts}. Clone https://github.com/liminchen/OptCuts there first."
        )

    # Always start from the exact source tracked by the checked-out OptCuts commit.
    upstream = subprocess.run(
        ["git", "show", "HEAD:src/TriMesh.cpp"], cwd=str(optcuts),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True,
    ).stdout
    modified = make_onestring_source(upstream)
    cpp.write_text(modified, encoding="utf-8")
    print("[OPTCUTS-ONESTRING-SOURCE] wrote scale-aware TriMesh.cpp from upstream HEAD")

    required = (
        "oneStringScalePenalty",
        "oneStringBoundaryStressExposure",
        "afterExposure - beforeExposure",
        "objectiveDec += oneStringScaleReward(*this, candidate)",
        "curEwDec += oneStringScaleReward(*this, candidate)",
        "EwDec += oneStringScaleReward(*this, candidate)",
    )
    missing = [token for token in required if token not in modified]
    if missing:
        raise SystemExit(
            "OptCuts source verification failed; scale-stress seam proxy is not present: "
            + ", ".join(missing)
        )
    print("[OPTCUTS-ONESTRING-OBJECTIVE] verified scale-stress seam proxy in TriMesh::computeLocalLDec")

    build = optcuts / "build_onestring"
    build.mkdir(parents=True, exist_ok=True)
    run([
        "cmake", "-S", str(optcuts), "-B", str(build),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
    ])
    jobs = str(max(1, min(12, os.cpu_count() or 1)))
    run(["cmake", "--build", str(build), "--config", "Release", "-j", jobs])

    candidates = [
        build / "OptCuts_bin",
        build / "OptCuts_bin.exe",
        build / "Release" / "OptCuts_bin",
        build / "Release" / "OptCuts_bin.exe",
    ]
    binary = next((p for p in candidates if p.is_file()), None)
    if binary is None:
        raise SystemExit(f"OptCuts_bin was not produced under {build}")

    stamp = root / ".onestring_optcuts_binary"
    stamp.write_text(str(binary.resolve()) + "\n", encoding="utf-8")
    print(f"[OPTCUTS-ONESTRING-BUILD] binary={binary.resolve()}")
    print(f"[OPTCUTS-ONESTRING-BUILD] stamp={stamp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
