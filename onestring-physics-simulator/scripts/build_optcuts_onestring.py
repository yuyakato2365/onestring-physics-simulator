#!/usr/bin/env python3
"""Build the dedicated OneString-aware OptCuts binary.

For optcuts_test2 this build changes OptCuts' distortion model itself from
plain Symmetric Dirichlet to:

    E_geom = E_SD + scale_weight * E_scale

Because OptCuts uses SymDirichletEnergy in both its ordinary embedding solve
and its local topology-candidate relaxation, both stages see the same
OneString-aware objective. No global optimizer is run per candidate.

The scale term contributes energy and gradient. The original SPD
Symmetric-Dirichlet Hessian is retained as an inexact-Newton preconditioner;
line search still evaluates the full augmented energy.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess


HELPER = r"""

    // OneString scale-aware extension.
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

    static double oneStringScaleWeight()
    {
        return std::max(
            0.0, oneStringScaleEnv("ONESTRING_OPTCUTS_SCALE_WEIGHT", 40.0));
    }

    static double oneStringScaleHalfBand()
    {
        const double bound = std::max(
            1.000001, oneStringScaleEnv("ONESTRING_OPTCUTS_SCALE_BOUND", 2.0));
        return 0.5 * std::log(bound);
    }

    static double oneStringLogLambdaFromUV(
        const TriMesh& data,
        int triI,
        const Eigen::Matrix<double, 3, 2>& uv)
    {
        const Eigen::Vector3i tri = data.F.row(triI);
        const Eigen::Vector3d x3D[3] = {
            data.V_rest.row(tri[0]),
            data.V_rest.row(tri[1]),
            data.V_rest.row(tri[2])
        };
        const Eigen::Vector2d uvArr[3] = {
            uv.row(0), uv.row(1), uv.row(2)
        };

        Eigen::Matrix2d dg;
        IglUtils::computeDeformationGradient(x3D, uvArr, dg); // surface -> UV

        const double a = dg.col(0).squaredNorm();
        const double b = dg.col(0).dot(dg.col(1));
        const double c = dg.col(1).squaredNorm();
        const double trace = a + c;
        const double disc = std::sqrt(
            std::max(0.0, (a-c)*(a-c) + 4.0*b*b));
        const double sigmaMinSq = std::max(
            1.0e-24, 0.5 * (trace - disc));
        const double lambda = 1.0 / std::sqrt(sigmaMinSq);
        return std::log(std::max(lambda, 1.0e-12));
    }

    static double oneStringLogLambda(const TriMesh& data, int triI)
    {
        const Eigen::Vector3i tri = data.F.row(triI);
        Eigen::Matrix<double, 3, 2> uv;
        uv.row(0) = data.V.row(tri[0]);
        uv.row(1) = data.V.row(tri[1]);
        uv.row(2) = data.V.row(tri[2]);
        return oneStringLogLambdaFromUV(data, triI, uv);
    }

    static double oneStringScaleCenter(const TriMesh& data)
    {
        if(data.F.rows() == 0) { return 0.0; }
        double minLog = __DBL_MAX__;
        double maxLog = -__DBL_MAX__;
        for(int triI = 0; triI < data.F.rows(); ++triI) {
            const double q = oneStringLogLambda(data, triI);
            minLog = std::min(minLog, q);
            maxLog = std::max(maxLog, q);
        }
        // Removes arbitrary global UV similarity scale.
        return 0.5 * (minLog + maxLog);
    }

    static double oneStringScaleResidual(
        double logLambda, double center, double halfBand)
    {
        return std::max(
            0.0, std::abs(logLambda - center) - halfBand);
    }

    static double oneStringScalePenaltyByElem(
        const TriMesh& data,
        int triI,
        double center,
        bool uniformWeight)
    {
        const double scaleWeight = oneStringScaleWeight();
        if(scaleWeight <= 0.0) { return 0.0; }

        const double q = oneStringLogLambda(data, triI);
        const double r = oneStringScaleResidual(
            q, center, oneStringScaleHalfBand());
        if(r <= 0.0) { return 0.0; }

        const double w = uniformWeight
            ? 1.0
            : data.triArea[triI] / std::max(data.surfaceArea, 1.0e-16);
        return scaleWeight * w * r * r;
    }

    static void oneStringAddScaleGradient(
        const TriMesh& data,
        Eigen::VectorXd& gradient,
        bool uniformWeight)
    {
        const double scaleWeight = oneStringScaleWeight();
        if(scaleWeight <= 0.0 || data.F.rows() == 0) { return; }

        const double center = oneStringScaleCenter(data);
        const double halfBand = oneStringScaleHalfBand();

        for(int triI = 0; triI < data.F.rows(); ++triI) {
            const Eigen::Vector3i tri = data.F.row(triI);
            Eigen::Matrix<double, 3, 2> uv;
            uv.row(0) = data.V.row(tri[0]);
            uv.row(1) = data.V.row(tri[1]);
            uv.row(2) = data.V.row(tri[2]);

            const double q = oneStringLogLambdaFromUV(data, triI, uv);
            const double delta = q - center;
            const double r = std::abs(delta) - halfBand;
            if(r <= 0.0) { continue; }

            const double sign = (delta >= 0.0) ? 1.0 : -1.0;
            const double w = uniformWeight
                ? 1.0
                : data.triArea[triI] / std::max(data.surfaceArea, 1.0e-16);

            double uvScale = 0.0;
            for(int lv = 0; lv < 3; ++lv) {
                uvScale = std::max(uvScale, uv.row(lv).norm());
            }
            const double eps = 1.0e-6 * std::max(1.0e-3, uvScale);

            for(int lv = 0; lv < 3; ++lv) {
                for(int d = 0; d < 2; ++d) {
                    Eigen::Matrix<double, 3, 2> uvPlus = uv;
                    Eigen::Matrix<double, 3, 2> uvMinus = uv;
                    uvPlus(lv, d) += eps;
                    uvMinus(lv, d) -= eps;
                    const double qPlus =
                        oneStringLogLambdaFromUV(data, triI, uvPlus);
                    const double qMinus =
                        oneStringLogLambdaFromUV(data, triI, uvMinus);
                    const double dq = (qPlus - qMinus) / (2.0 * eps);

                    gradient[2 * tri[lv] + d] +=
                        scaleWeight * w * 2.0 * r * sign * dq;
                }
            }
        }
    }
"""


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise SystemExit(
            f"OptCuts source mismatch while editing {label}: "
            f"expected 1 exact anchor, found {count}."
        )
    return source.replace(old, new, 1)


def make_onestring_symdirichlet_source(upstream: str) -> str:
    source = upstream
    source = replace_once(
        source,
        "#include <cfloat>\n",
        "#include <cfloat>\n#include <cmath>\n#include <cstdlib>\n#include <string>\n",
        "SymDirichlet standard includes",
    )
    source = replace_once(
        source,
        "namespace OptCuts {\n",
        "namespace OptCuts {\n" + HELPER,
        "SymDirichlet OptCuts namespace",
    )

    source = replace_once(
        source,
        """        energyValPerElem.resize(data.F.rows());
        for(int triI = 0; triI < data.F.rows(); triI++) {
""",
        """        energyValPerElem.resize(data.F.rows());
        const double oneStringCenter = oneStringScaleCenter(data);
        for(int triI = 0; triI < data.F.rows(); triI++) {
""",
        "per-element scale center",
    )

    anchor = """            energyValPerElem[triI] = w * (1.0 + data.triAreaSq[triI] / area_U / area_U) *
                ((U3m1.squaredNorm() * data.e0SqLen[triI] + U2m1.squaredNorm() * data.e1SqLen[triI]) / 4 / data.triAreaSq[triI] -
                U3m1.dot(U2m1) * data.e0dote1[triI] / 2 / data.triAreaSq[triI]);
"""
    replacement = anchor + """            energyValPerElem[triI] += oneStringScalePenaltyByElem(
                data, triI, oneStringCenter, uniformWeight);
"""
    source = replace_once(
        source, anchor, replacement, "per-element augmented energy"
    )

    anchor = """        energyVal = w * (1.0 + data.triAreaSq[triI] / area_U / area_U) *
        ((U3m1.squaredNorm() * data.e0SqLen[triI] + U2m1.squaredNorm() * data.e1SqLen[triI]) / 4 / data.triAreaSq[triI] -
         U3m1.dot(U2m1) * data.e0dote1[triI] / 2 / data.triAreaSq[triI]);
"""
    replacement = anchor + """        const double oneStringCenter = oneStringScaleCenter(data);
        energyVal += oneStringScalePenaltyByElem(
            data, triI, oneStringCenter, uniformWeight);
"""
    source = replace_once(
        source, anchor, replacement, "single-element augmented energy"
    )

    anchor = """        for(const auto fixedVI : data.fixedVert) {
            gradient[2 * fixedVI] = 0.0;
            gradient[2 * fixedVI + 1] = 0.0;
        }
"""
    replacement = """        oneStringAddScaleGradient(data, gradient, uniformWeight);

""" + anchor
    source = replace_once(source, anchor, replacement, "augmented gradient")
    return source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--optcuts-root", type=Path, default=None,
        help="upstream OptCuts checkout (default: third_party/OptCuts)",
    )
    args = parser.parse_args()

    root = project_root()
    optcuts = (
        args.optcuts_root or (root / "third_party" / "OptCuts")
    ).expanduser().resolve()
    tri_cpp = optcuts / "src" / "TriMesh.cpp"
    sd_cpp = optcuts / "src" / "Energy" / "SymDirichletEnergy.cpp"
    if (
        not tri_cpp.is_file()
        or not sd_cpp.is_file()
        or not (optcuts / ".git").exists()
    ):
        raise SystemExit(
            f"OptCuts checkout not found at {optcuts}. "
            "Clone https://github.com/liminchen/OptCuts there first."
        )

    upstream_tri = subprocess.run(
        ["git", "show", "HEAD:src/TriMesh.cpp"],
        cwd=str(optcuts),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    ).stdout
    upstream_sd = subprocess.run(
        ["git", "show", "HEAD:src/Energy/SymDirichletEnergy.cpp"],
        cwd=str(optcuts),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    ).stdout

    # Remove the old topology-only boundary-exposure proxy. Candidate ranking
    # now inherits the same augmented distortion through OptCuts' own local
    # SymDirichletEnergy relaxation.
    tri_cpp.write_text(upstream_tri, encoding="utf-8")
    modified_sd = make_onestring_symdirichlet_source(upstream_sd)
    sd_cpp.write_text(modified_sd, encoding="utf-8")

    required = (
        "oneStringAddScaleGradient",
        "oneStringScalePenaltyByElem",
        "energyValPerElem[triI] += oneStringScalePenaltyByElem",
        "energyVal += oneStringScalePenaltyByElem",
        "oneStringAddScaleGradient(data, gradient, uniformWeight)",
    )
    missing = [token for token in required if token not in modified_sd]
    if missing:
        raise SystemExit(
            "OptCuts source verification failed; augmented OneString geometry "
            "objective is incomplete: " + ", ".join(missing)
        )

    print(
        "[OPTCUTS-ONESTRING-SOURCE] restored upstream TriMesh.cpp; "
        "patched SymDirichletEnergy.cpp"
    )
    print(
        "[OPTCUTS-ONESTRING-OBJECTIVE] "
        "E_geom = E_SD + weight * E_scale in global and local UV solves; "
        "SD Hessian retained as inexact-Newton preconditioner"
    )

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
