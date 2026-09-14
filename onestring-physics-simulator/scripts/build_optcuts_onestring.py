#!/usr/bin/env python3
"""Build the dedicated OneString-aware OptCuts binary.

This source-level patch keeps OptCuts' original topology/parameterization
algorithm, but changes the distortion model used by both the global UV solve
and the candidate-local UV relaxation from plain Symmetric Dirichlet to

    E_geom = E_SD + scale_weight * E_scale.

Important implementation rule: the scale band is defined by a GLOBAL reference
center frozen for one global OptCuts iteration. Candidate-local optimizers reuse
that same reference instead of re-centering on their local stencil. This makes
local candidate relaxation approximate improvement of the same global
lambda_max/lambda_min objective.

The center is refreshed before each global Newton iteration (block-coordinate
update), then held fixed during its gradient + line-search evaluations. The
original SD Hessian remains an inexact-Newton SPD preconditioner.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess


HELPER = r"""

    // ------------------------------------------------------------------
    // OneString scale-aware extension.
    // ------------------------------------------------------------------
    static std::atomic<double> oneStringGlobalCenter(0.0);
    static std::atomic<double> oneStringGlobalSurfaceArea(1.0);
    static std::atomic<bool> oneStringGlobalReferenceValid(false);
    static thread_local bool oneStringDiagnosticLocalScope = false;
    static std::mutex oneStringDiagnosticMutex;

    static double oneStringScaleEnv(const char* name, double fallback)
    {
        const char* raw = std::getenv(name);
        if(!raw) { return fallback; }
        try {
            const double value = std::stod(std::string(raw));
            return std::isfinite(value) ? value : fallback;
        }
        catch(...) { return fallback; }
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

    static double oneStringScaleBound()
    {
        return std::exp(2.0 * oneStringScaleHalfBand());
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

    static double oneStringCurrentMeshCenter(const TriMesh& data)
    {
        if(data.F.rows() == 0) { return 0.0; }
        double minLog = __DBL_MAX__;
        double maxLog = -__DBL_MAX__;
        for(int triI = 0; triI < data.F.rows(); ++triI) {
            const double q = oneStringLogLambda(data, triI);
            minLog = std::min(minLog, q);
            maxLog = std::max(maxLog, q);
        }
        return 0.5 * (minLog + maxLog);
    }

    // Called only by the non-muted/global Optimizer. Local candidate optimizers
    // deliberately do NOT overwrite this value.
    void oneStringSetGlobalScaleReference(const TriMesh& data)
    {
        if(data.F.rows() == 0) { return; }
        const double center = oneStringCurrentMeshCenter(data);
        oneStringGlobalCenter.store(center, std::memory_order_relaxed);
        oneStringGlobalSurfaceArea.store(
            std::max(data.surfaceArea, 1.0e-16), std::memory_order_relaxed);
        oneStringGlobalReferenceValid.store(true, std::memory_order_release);

        static std::atomic<bool> printed(false);
        bool expected = false;
        if(printed.compare_exchange_strong(expected, true)) {
            std::cerr
                << "[OPTCUTS-SCALE-CPP-ACTIVE] weight=" << oneStringScaleWeight()
                << " bound=" << oneStringScaleBound()
                << " center_mode=global-frozen-per-iteration"
                << std::endl;
        }
    }

    void oneStringSetDiagnosticScope(bool localScope)
    {
        oneStringDiagnosticLocalScope = localScope;
    }

    static double oneStringReferenceCenter(const TriMesh& data)
    {
        if(oneStringGlobalReferenceValid.load(std::memory_order_acquire)) {
            return oneStringGlobalCenter.load(std::memory_order_relaxed);
        }
        return oneStringCurrentMeshCenter(data);
    }

    static double oneStringReferenceArea(const TriMesh& data)
    {
        if(oneStringGlobalReferenceValid.load(std::memory_order_acquire)) {
            return std::max(
                oneStringGlobalSurfaceArea.load(std::memory_order_relaxed), 1.0e-16);
        }
        return std::max(data.surfaceArea, 1.0e-16);
    }

    static double oneStringScaleResidual(
        double logLambda, double center, double halfBand)
    {
        return std::max(0.0, std::abs(logLambda - center) - halfBand);
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
            : data.triArea[triI] / oneStringReferenceArea(data);
        return scaleWeight * w * r * r;
    }

    static void oneStringTriangleScaleGradient(
        const TriMesh& data,
        int triI,
        double center,
        bool uniformWeight,
        Eigen::Matrix<double, 3, 2>& triGradient)
    {
        triGradient.setZero();
        const double scaleWeight = oneStringScaleWeight();
        if(scaleWeight <= 0.0) { return; }

        const Eigen::Vector3i tri = data.F.row(triI);
        Eigen::Matrix<double, 3, 2> uv;
        uv.row(0) = data.V.row(tri[0]);
        uv.row(1) = data.V.row(tri[1]);
        uv.row(2) = data.V.row(tri[2]);

        const double q = oneStringLogLambdaFromUV(data, triI, uv);
        const double delta = q - center;
        const double r = std::abs(delta) - oneStringScaleHalfBand();
        if(r <= 0.0) { return; }

        const double sign = (delta >= 0.0) ? 1.0 : -1.0;
        const double w = uniformWeight
            ? 1.0
            : data.triArea[triI] / oneStringReferenceArea(data);

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
                const double qPlus = oneStringLogLambdaFromUV(data, triI, uvPlus);
                const double qMinus = oneStringLogLambdaFromUV(data, triI, uvMinus);
                const double dq = (qPlus - qMinus) / (2.0 * eps);
                triGradient(lv, d) =
                    scaleWeight * w * 2.0 * r * sign * dq;
            }
        }
    }

    static void oneStringAddScaleGradient(
        const TriMesh& data,
        Eigen::VectorXd& gradient,
        bool uniformWeight)
    {
        if(oneStringScaleWeight() <= 0.0 || data.F.rows() == 0) { return; }
        const double center = oneStringReferenceCenter(data);
        for(int triI = 0; triI < data.F.rows(); ++triI) {
            const Eigen::Vector3i tri = data.F.row(triI);
            Eigen::Matrix<double, 3, 2> triGradient;
            oneStringTriangleScaleGradient(
                data, triI, center, uniformWeight, triGradient);
            for(int lv = 0; lv < 3; ++lv) {
                gradient[2 * tri[lv]] += triGradient(lv, 0);
                gradient[2 * tri[lv] + 1] += triGradient(lv, 1);
            }
        }
    }

    static void oneStringAddScaleLocalGradient(
        const TriMesh& data,
        Eigen::MatrixXd& localGradients)
    {
        if(oneStringScaleWeight() <= 0.0 || data.F.rows() == 0) { return; }
        const double center = oneStringReferenceCenter(data);
        for(int triI = 0; triI < data.F.rows(); ++triI) {
            Eigen::Matrix<double, 3, 2> triGradient;
            oneStringTriangleScaleGradient(data, triI, center, false, triGradient);
            localGradients.block<3, 2>(triI * 3, 0) += triGradient;
        }
    }

    static double oneStringSDEnergy(const TriMesh& data, bool uniformWeight)
    {
        const double normalizer_div = std::max(data.surfaceArea, 1.0e-16);
        double total = 0.0;
        for(int triI = 0; triI < data.F.rows(); ++triI) {
            const Eigen::Vector3i triVInd = data.F.row(triI);
            const Eigen::RowVector2d U1 = data.V.row(triVInd[0]);
            const Eigen::RowVector2d U2 = data.V.row(triVInd[1]);
            const Eigen::RowVector2d U3 = data.V.row(triVInd[2]);
            const Eigen::RowVector2d U2m1 = U2 - U1;
            const Eigen::RowVector2d U3m1 = U3 - U1;
            const double area_U = 0.5 *
                (U2m1[0] * U3m1[1] - U2m1[1] * U3m1[0]);
            if(std::abs(area_U) < 1.0e-16) { continue; }
            const double w = uniformWeight
                ? 1.0 : data.triArea[triI] / normalizer_div;
            total += w *
                (1.0 + data.triAreaSq[triI] / area_U / area_U) *
                ((U3m1.squaredNorm() * data.e0SqLen[triI] +
                  U2m1.squaredNorm() * data.e1SqLen[triI]) /
                     4.0 / data.triAreaSq[triI] -
                 U3m1.dot(U2m1) * data.e0dote1[triI] /
                     2.0 / data.triAreaSq[triI]);
        }
        return total;
    }

    static void oneStringScaleStats(
        const TriMesh& data,
        bool uniformWeight,
        double& rawScaleEnergy,
        int& violating,
        double& scaleRange)
    {
        rawScaleEnergy = 0.0;
        violating = 0;
        scaleRange = 1.0;
        if(data.F.rows() == 0) { return; }

        const double center = oneStringReferenceCenter(data);
        const double halfBand = oneStringScaleHalfBand();
        double minLog = __DBL_MAX__;
        double maxLog = -__DBL_MAX__;
        for(int triI = 0; triI < data.F.rows(); ++triI) {
            const double q = oneStringLogLambda(data, triI);
            minLog = std::min(minLog, q);
            maxLog = std::max(maxLog, q);
            const double r = oneStringScaleResidual(q, center, halfBand);
            if(r > 0.0) { ++violating; }
            const double w = uniformWeight
                ? 1.0 : data.triArea[triI] / oneStringReferenceArea(data);
            rawScaleEnergy += w * r * r;
        }
        scaleRange = std::exp(maxLog - minLog);
    }

    static void oneStringLogDiagnostics(
        const TriMesh& data,
        bool uniformWeight,
        const Eigen::VectorXd& sdGradient,
        const Eigen::VectorXd& scaleGradient)
    {
        const char* path = std::getenv("ONESTRING_OPTCUTS_SCALE_DIAG_PATH");
        if(!path || !*path) { return; }

        static std::atomic<long> globalCalls(0);
        static std::atomic<long> localCalls(0);
        const bool local = oneStringDiagnosticLocalScope;
        const long callIndex = local ? ++localCalls : ++globalCalls;
        if(local && !(callIndex <= 25 || (callIndex % 500) == 0)) { return; }

        double rawScaleEnergy = 0.0;
        double scaleRange = 1.0;
        int violating = 0;
        oneStringScaleStats(
            data, uniformWeight, rawScaleEnergy, violating, scaleRange);
        const double sdEnergy = oneStringSDEnergy(data, uniformWeight);
        const double weightedScaleEnergy = oneStringScaleWeight() * rawScaleEnergy;
        const double energyRatio = weightedScaleEnergy /
            std::max(std::abs(sdEnergy), 1.0e-30);
        const double sdGradNorm = sdGradient.norm();
        const double scaleGradNorm = scaleGradient.norm();
        const double gradRatio = scaleGradNorm /
            std::max(sdGradNorm, 1.0e-30);

        std::lock_guard<std::mutex> guard(oneStringDiagnosticMutex);
        bool writeHeader = false;
        {
            std::ifstream check(path);
            writeHeader = !check.good() ||
                check.peek() == std::ifstream::traits_type::eof();
        }
        std::ofstream out(path, std::ios::app);
        if(!out.good()) {
            static bool warned = false;
            if(!warned) {
                warned = true;
                std::cerr << "[OPTCUTS-SCALE-DIAG-CPP-ERROR] cannot open "
                          << path << std::endl;
            }
            return;
        }
        if(writeHeader) {
            out << "scope,call,n_vertices,n_faces,sd_energy,scale_energy_raw,"
                   "scale_energy_weighted,scale_to_sd,sd_grad_norm,"
                   "scale_grad_norm,scale_grad_to_sd_grad,violating,"
                   "scale_range,scale_weight,reference_center\n";
        }
        out << (local ? "local" : "global") << ','
            << callIndex << ',' << data.V.rows() << ',' << data.F.rows() << ','
            << std::setprecision(17)
            << sdEnergy << ',' << rawScaleEnergy << ','
            << weightedScaleEnergy << ',' << energyRatio << ','
            << sdGradNorm << ',' << scaleGradNorm << ',' << gradRatio << ','
            << violating << ',' << scaleRange << ',' << oneStringScaleWeight() << ','
            << oneStringReferenceCenter(data) << '\n';
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
        "#include <cfloat>\n#include <cmath>\n#include <cstdlib>\n#include <string>\n#include <iomanip>\n#include <atomic>\n#include <mutex>\n#include <iostream>\n",
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
        const double oneStringCenter = oneStringReferenceCenter(data);
        for(int triI = 0; triI < data.F.rows(); triI++) {
""",
        "per-element shared scale center",
    )

    anchor = """            energyValPerElem[triI] = w * (1.0 + data.triAreaSq[triI] / area_U / area_U) *
                ((U3m1.squaredNorm() * data.e0SqLen[triI] + U2m1.squaredNorm() * data.e1SqLen[triI]) / 4 / data.triAreaSq[triI] -
                U3m1.dot(U2m1) * data.e0dote1[triI] / 2 / data.triAreaSq[triI]);
"""
    source = replace_once(
        source,
        anchor,
        anchor + """            energyValPerElem[triI] += oneStringScalePenaltyByElem(
                data, triI, oneStringCenter, uniformWeight);
""",
        "per-element augmented energy",
    )

    anchor = """        energyVal = w * (1.0 + data.triAreaSq[triI] / area_U / area_U) *
        ((U3m1.squaredNorm() * data.e0SqLen[triI] + U2m1.squaredNorm() * data.e1SqLen[triI]) / 4 / data.triAreaSq[triI] -
         U3m1.dot(U2m1) * data.e0dote1[triI] / 2 / data.triAreaSq[triI]);
"""
    source = replace_once(
        source,
        anchor,
        anchor + """        const double oneStringCenter = oneStringReferenceCenter(data);
        energyVal += oneStringScalePenaltyByElem(
            data, triI, oneStringCenter, uniformWeight);
""",
        "single-element augmented energy",
    )

    # Candidate discovery uses computeLocalGradient -> computeDivGradPerVert.
    # Add the scale term there too, so candidate generation is not SD-only.
    local_grad_end = """            localGradients.row(startRowI + 2) = w * (dLeft3 * rightTerm + dRight3 * leftTerm);
        }
    }
"""
    source = replace_once(
        source,
        local_grad_end,
        """            localGradients.row(startRowI + 2) = w * (dLeft3 * rightTerm + dRight3 * leftTerm);
        }
        oneStringAddScaleLocalGradient(data, localGradients);
    }
""",
        "candidate-discovery augmented local gradient",
    )

    anchor = """        for(const auto fixedVI : data.fixedVert) {
            gradient[2 * fixedVI] = 0.0;
            gradient[2 * fixedVI + 1] = 0.0;
        }
"""
    replacement = """        Eigen::VectorXd oneStringSDGradient = gradient;
        Eigen::VectorXd oneStringScaleGradient = Eigen::VectorXd::Zero(gradient.size());
        oneStringAddScaleGradient(data, oneStringScaleGradient, uniformWeight);
        for(const auto fixedVI : data.fixedVert) {
            oneStringSDGradient[2 * fixedVI] = 0.0;
            oneStringSDGradient[2 * fixedVI + 1] = 0.0;
            oneStringScaleGradient[2 * fixedVI] = 0.0;
            oneStringScaleGradient[2 * fixedVI + 1] = 0.0;
        }
        gradient += oneStringScaleGradient;
        oneStringLogDiagnostics(
            data, uniformWeight, oneStringSDGradient, oneStringScaleGradient);

""" + anchor
    source = replace_once(
        source, anchor, replacement, "augmented gradient + diagnostics"
    )
    return source


def make_onestring_optimizer_source(upstream: str) -> str:
    source = upstream
    source = replace_once(
        source,
        "namespace OptCuts {\n",
        """namespace OptCuts {
    // Implemented by the OneString patch in SymDirichletEnergy.cpp.
    void oneStringSetGlobalScaleReference(const TriMesh& data);
    void oneStringSetDiagnosticScope(bool localScope);
""",
        "Optimizer OneString declarations",
    )

    # Initial global energy must use a valid global reference too.
    source = replace_once(
        source,
        """        computeEnergyVal(result, scaffold, lastEnergyVal);
        if(!mute) {
""",
        """        if(!mute) {
            oneStringSetGlobalScaleReference(result);
        }
        computeEnergyVal(result, scaffold, lastEnergyVal);
        if(!mute) {
""",
        "Optimizer precompute global scale reference",
    )

    # The non-muted optimizer is the real/global solve. Candidate-local
    # Optimizers are constructed mute=true in TriMesh.cpp. Refresh the global
    # center once per global Newton iteration and freeze it for the ensuing
    # gradient + line-search. Local solves inherit that reference unchanged.
    source = replace_once(
        source,
        """            if(!mute) { timer.start(1); }
            computeGradient(result, scaffold, gradient);
""",
        """            if(!mute) { timer.start(1); }
            oneStringSetDiagnosticScope(mute);
            if(!mute) {
                oneStringSetGlobalScaleReference(result);
            }
            computeGradient(result, scaffold, gradient);
""",
        "Optimizer solve shared scale reference",
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
    optcuts = (
        args.optcuts_root or (root / "third_party" / "OptCuts")
    ).expanduser().resolve()
    tri_cpp = optcuts / "src" / "TriMesh.cpp"
    optimizer_cpp = optcuts / "src" / "Optimizer.cpp"
    sd_cpp = optcuts / "src" / "Energy" / "SymDirichletEnergy.cpp"
    if (
        not tri_cpp.is_file()
        or not optimizer_cpp.is_file()
        or not sd_cpp.is_file()
        or not (optcuts / ".git").exists()
    ):
        raise SystemExit(
            f"OptCuts checkout not found at {optcuts}. "
            "Clone https://github.com/liminchen/OptCuts there first."
        )

    def upstream(path: str) -> str:
        return subprocess.run(
            ["git", "show", f"HEAD:{path}"],
            cwd=str(optcuts),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        ).stdout

    upstream_tri = upstream("src/TriMesh.cpp")
    upstream_optimizer = upstream("src/Optimizer.cpp")
    upstream_sd = upstream("src/Energy/SymDirichletEnergy.cpp")

    # Remove the old topology-only proxy. Keep original topology operations,
    # but make their local UV relaxation and candidate discovery use the same
    # augmented distortion objective as the global solve.
    tri_cpp.write_text(upstream_tri, encoding="utf-8")
    modified_optimizer = make_onestring_optimizer_source(upstream_optimizer)
    modified_sd = make_onestring_symdirichlet_source(upstream_sd)
    optimizer_cpp.write_text(modified_optimizer, encoding="utf-8")
    sd_cpp.write_text(modified_sd, encoding="utf-8")

    required_sd = (
        "oneStringSetGlobalScaleReference",
        "oneStringReferenceCenter",
        "oneStringAddScaleGradient",
        "oneStringAddScaleLocalGradient",
        "oneStringScalePenaltyByElem",
        "oneStringLogDiagnostics",
        "gradient += oneStringScaleGradient",
        "oneStringAddScaleLocalGradient(data, localGradients)",
    )
    required_optimizer = (
        "oneStringSetGlobalScaleReference(result)",
        "oneStringSetDiagnosticScope(mute)",
    )
    missing = [token for token in required_sd if token not in modified_sd]
    missing += [token for token in required_optimizer if token not in modified_optimizer]
    if missing:
        raise SystemExit(
            "OptCuts source verification failed; shared OneString objective is "
            "incomplete: " + ", ".join(missing)
        )

    print(
        "[OPTCUTS-ONESTRING-SOURCE] restored upstream TriMesh.cpp; patched "
        "Optimizer.cpp + SymDirichletEnergy.cpp"
    )
    print(
        "[OPTCUTS-ONESTRING-OBJECTIVE] E_geom = E_SD + weight * E_scale; "
        "global center frozen per global Newton iteration and reused by local candidates"
    )
    print(
        "[OPTCUTS-ONESTRING-CANDIDATES] scale gradient added to "
        "computeLocalGradient candidate discovery"
    )
    print(
        "[OPTCUTS-ONESTRING-DIAGNOSTICS] scope comes from Optimizer mute/global state, "
        "not mesh face count"
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
