#!/usr/bin/env python3
"""Build and force-enable the OneString scale-sum objective in OptCuts.

This script makes the OneString scale objective an explicit physical-mesh term:

    E_physical = lambda_SD * E_SD + weight * SUM_i residual_i^2

The scale term is deliberately OUTSIDE OptCuts' SD multiplier. In the normal
OneString test2 configuration ``lambda_init=0.999``, OptCuts starts with
``lambda_SD = 1 - lambda_init = 0.001``; putting E_scale inside the SD term
would therefore weaken it by 1000x, which is not the intended objective.

The scale term currently contributes energy + gradient, while OptCuts' SD
Hessian is used as an inexact-Newton SPD preconditioner.  For the physical
(non-muted) optimizer we therefore MUST NOT also shrink that preconditioner by
lambda_SD=0.001: doing so magnifies the scale-gradient Newton direction by
roughly 1000x and sends OptCuts into repeated line-search backtracking.  The
physical preconditioner uses at least 1.0 * H_SD; muted candidate-local solves
keep the original OptCuts Hessian scaling (their energy parameter is normally
1.0 anyway).

The built binary is launched through a tiny runner generated next to it.  The
runner exports the diagnostic/activation paths itself before exec'ing the real
binary.  This removes any dependence on Streamlit/Python environment
propagation for the hard runtime proof.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise SystemExit(
            f"OneString force-enable source mismatch for {label}: "
            f"expected exactly one anchor, found {count}."
        )
    return source.replace(old, new, 1)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    optcuts = root / "third_party" / "OptCuts"
    base_builder = root / "scripts" / "build_optcuts_onestring.py"

    run([sys.executable, str(base_builder)], cwd=root)

    optimizer_cpp = optcuts / "src" / "Optimizer.cpp"
    sd_cpp = optcuts / "src" / "Energy" / "SymDirichletEnergy.cpp"
    if not optimizer_cpp.is_file() or not sd_cpp.is_file():
        raise SystemExit("Generated OptCuts C++ sources were not found after base build.")

    optimizer = optimizer_cpp.read_text(encoding="utf-8")
    sd = sd_cpp.read_text(encoding="utf-8")

    # Keep the scale term out of bulk SymDirichlet evaluation.  The single-
    # element term is intentionally retained for TriMesh local-candidate scores,
    # and computeLocalGradient keeps the scale contribution for candidate
    # discovery.
    sd = replace_once(
        sd,
        """            energyValPerElem[triI] += oneStringScalePenaltyByElem(\n                data, triI, oneStringCenter, uniformWeight);\n""",
        "",
        "remove scale from bulk SymDirichlet energy",
    )
    sd = replace_once(
        sd,
        """        Eigen::VectorXd oneStringSDGradient = gradient;\n        Eigen::VectorXd oneStringScaleGradient = Eigen::VectorXd::Zero(gradient.size());\n        oneStringAddScaleGradient(data, oneStringScaleGradient, uniformWeight);\n        for(const auto fixedVI : data.fixedVert) {\n            oneStringSDGradient[2 * fixedVI] = 0.0;\n            oneStringSDGradient[2 * fixedVI + 1] = 0.0;\n            oneStringScaleGradient[2 * fixedVI] = 0.0;\n            oneStringScaleGradient[2 * fixedVI + 1] = 0.0;\n        }\n        gradient += oneStringScaleGradient;\n        oneStringLogDiagnostics(\n            data, uniformWeight, oneStringSDGradient, oneStringScaleGradient);\n\n""",
        "",
        "remove scale from bulk SymDirichlet gradient",
    )

    namespace_close = "\n}\n"
    if not sd.endswith(namespace_close):
        raise SystemExit("Unexpected SymDirichletEnergy.cpp namespace ending.")
    exported = r'''

    double oneStringPhysicalScaleEnergy(const TriMesh& data)
    {
        if(oneStringScaleWeight() <= 0.0 || data.F.rows() == 0) { return 0.0; }
        const double center = oneStringReferenceCenter(data);
        double energy = 0.0;
        for(int triI = 0; triI < data.F.rows(); ++triI) {
            energy += oneStringScalePenaltyByElem(data, triI, center, false);
        }
        return energy;
    }

    void oneStringPhysicalScaleGradient(
        const TriMesh& data, Eigen::VectorXd& gradient)
    {
        gradient = Eigen::VectorXd::Zero(data.V.rows() * 2);
        oneStringAddScaleGradient(data, gradient, false);
        for(const auto fixedVI : data.fixedVert) {
            gradient[2 * fixedVI] = 0.0;
            gradient[2 * fixedVI + 1] = 0.0;
        }
    }

    void oneStringPhysicalScaleDiagnostics(
        const TriMesh& data,
        const Eigen::VectorXd& sdGradient,
        const Eigen::VectorXd& scaleGradient)
    {
        oneStringLogDiagnostics(data, false, sdGradient, scaleGradient);
    }
'''
    sd = sd[:-len(namespace_close)] + exported + namespace_close

    decl_anchor = """    void oneStringSetGlobalScaleReference(const TriMesh& data);\n    void oneStringSetDiagnosticScope(bool localScope);\n"""
    decl_replacement = decl_anchor + """    double oneStringPhysicalScaleEnergy(const TriMesh& data);\n    void oneStringPhysicalScaleGradient(const TriMesh& data, Eigen::VectorXd& gradient);\n    void oneStringPhysicalScaleDiagnostics(\n        const TriMesh& data,\n        const Eigen::VectorXd& sdGradient,\n        const Eigen::VectorXd& scaleGradient);\n"""
    optimizer = replace_once(
        optimizer, decl_anchor, decl_replacement,
        "Optimizer OneString physical objective declarations",
    )

    energy_anchor = """        energyTerms[0]->computeEnergyVal(data, energyVal_ET[0]);\n        energyVal = energyParams[0] * energyVal_ET[0];\n"""
    energy_replacement = """        energyTerms[0]->computeEnergyVal(data, energyVal_ET[0]);\n        const double oneStringScaleEnergy = oneStringPhysicalScaleEnergy(data);\n        energyVal = energyParams[0] * energyVal_ET[0] + oneStringScaleEnergy;\n"""
    optimizer = replace_once(
        optimizer, energy_anchor, energy_replacement,
        "physical Optimizer energy injection",
    )

    grad_anchor = """        energyTerms[0]->computeGradient(data, gradient_ET[0]);\n        gradient = energyParams[0] * gradient_ET[0];\n"""
    grad_replacement = """        energyTerms[0]->computeGradient(data, gradient_ET[0]);\n        Eigen::VectorXd oneStringSDGradient = gradient_ET[0];\n        Eigen::VectorXd oneStringScaleGradient;\n        oneStringPhysicalScaleGradient(data, oneStringScaleGradient);\n        oneStringPhysicalScaleDiagnostics(\n            data, oneStringSDGradient, oneStringScaleGradient);\n        gradient = energyParams[0] * gradient_ET[0] + oneStringScaleGradient;\n"""
    optimizer = replace_once(
        optimizer, grad_anchor, grad_replacement,
        "physical Optimizer gradient injection",
    )

    # OptCuts multiplies its SD Hessian by energyParams.  With lambda_init=.999,
    # that makes H ~= .001 H_SD while the newly added scale gradient is NOT
    # multiplied by .001.  The resulting Newton direction is about 1000x too
    # large and spends its time backtracking.  Keep H_SD as an SPD inexact-
    # Newton preconditioner at full strength for the physical/global optimizer.
    # Candidate-local optimizers are mute=true, so their original scaling is
    # untouched.
    dense_hessian_anchor = """            energyTerms[0]->computeHessian(data, Hessian);\n            Hessian *= energyParams[0];\n"""
    dense_hessian_replacement = """            energyTerms[0]->computeHessian(data, Hessian);\n            const double oneStringHessianScale =\n                mute ? energyParams[0] : ((energyParams[0] < 1.0) ? 1.0 : energyParams[0]);\n            Hessian *= oneStringHessianScale;\n"""
    optimizer = replace_once(
        optimizer, dense_hessian_anchor, dense_hessian_replacement,
        "dense physical Hessian stabilization",
    )

    sparse_hessian_anchor = """                energyTerms[eI]->computeHessian(data, &V, &I, &J);\n                V *= energyParams[eI];\n"""
    sparse_hessian_replacement = """                energyTerms[eI]->computeHessian(data, &V, &I, &J);\n                const double oneStringHessianScale =\n                    mute ? energyParams[eI] : ((energyParams[eI] < 1.0) ? 1.0 : energyParams[eI]);\n                V *= oneStringHessianScale;\n"""
    optimizer = replace_once(
        optimizer, sparse_hessian_anchor, sparse_hessian_replacement,
        "sparse physical Hessian stabilization",
    )

    # This marker sits in the exact physical energy path.  The generated runner
    # below always exports ACTIVE_PATH before exec, so failure to produce the
    # file now really means this objective path did not execute.
    marker_anchor = """        const double oneStringScaleEnergy = oneStringPhysicalScaleEnergy(data);\n        energyVal = energyParams[0] * energyVal_ET[0] + oneStringScaleEnergy;\n"""
    marker_replacement = marker_anchor + """        static std::atomic<bool> oneStringPhysicalObjectivePrinted(false);\n        bool oneStringExpected = false;\n        if(oneStringPhysicalObjectivePrinted.compare_exchange_strong(\n               oneStringExpected, true)) {\n            std::cerr\n                << \"[OPTCUTS-ONESTRING-PHYSICAL-OBJECTIVE-ACTIVE] scale_energy=\"\n                << oneStringScaleEnergy\n                << \" sd_multiplier=\" << energyParams[0]\n                << \" hessian_preconditioner_scale=\"\n                << ((energyParams[0] < 1.0) ? 1.0 : energyParams[0])\n                << \" model=sum_residual_squared_no_area_normalization\"\n                << std::endl;\n            const char* activePath = std::getenv(\n                \"ONESTRING_OPTCUTS_ACTIVE_PATH\");\n            if(activePath && *activePath) {\n                std::ofstream active(activePath, std::ios::trunc);\n                if(active.good()) {\n                    active << \"active\\n\"\n                           << \"scale_energy=\" << oneStringScaleEnergy << \"\\n\"\n                           << \"sd_multiplier=\" << energyParams[0] << \"\\n\"\n                           << \"hessian_preconditioner_scale=\"\n                           << ((energyParams[0] < 1.0) ? 1.0 : energyParams[0])\n                           << \"\\n\";\n                }\n                else {\n                    std::cerr << \"[OPTCUTS-ONESTRING-ACTIVE-FILE-ERROR] path=\"\n                              << activePath << std::endl;\n                }\n            }\n            else {\n                std::cerr << \"[OPTCUTS-ONESTRING-ACTIVE-PATH-MISSING]\" << std::endl;\n            }\n        }\n"""
    optimizer = replace_once(
        optimizer, marker_anchor, marker_replacement,
        "physical objective runtime marker",
    )

    optimizer = replace_once(
        optimizer,
        "#include <fstream>\n",
        "#include <fstream>\n#include <atomic>\n#include <cstdlib>\n#include <iostream>\n",
        "Optimizer runtime marker includes",
    )

    optimizer_cpp.write_text(optimizer, encoding="utf-8")
    sd_cpp.write_text(sd, encoding="utf-8")

    required_optimizer = (
        "oneStringPhysicalScaleEnergy(data)",
        "energyVal = energyParams[0] * energyVal_ET[0] + oneStringScaleEnergy",
        "gradient = energyParams[0] * gradient_ET[0] + oneStringScaleGradient",
        "const double oneStringHessianScale =",
        "hessian_preconditioner_scale=",
        "[OPTCUTS-ONESTRING-PHYSICAL-OBJECTIVE-ACTIVE]",
        "ONESTRING_OPTCUTS_ACTIVE_PATH",
    )
    required_sd = (
        "double oneStringPhysicalScaleEnergy(const TriMesh& data)",
        "void oneStringPhysicalScaleGradient(",
        "oneStringAddScaleLocalGradient(data, localGradients)",
        "energyVal += oneStringScalePenaltyByElem(",
    )
    missing = [x for x in required_optimizer if x not in optimizer]
    missing += [x for x in required_sd if x not in sd]
    if missing:
        raise SystemExit("Force-enable verification failed: " + ", ".join(missing))

    build = optcuts / "build_onestring"
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

    # Deterministic launcher: set critical env vars in the child process itself.
    # This fixes the observed case where the modified binary ran successfully but
    # getenv(ACTIVE_PATH) did not lead to an activation file under Streamlit.
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    active_path = logs / "optcuts_scale_objective.active"
    diag_path = logs / "optcuts_scale_objective.csv"
    runner = build / "OptCuts_onestring_runner"
    runner.write_text(
        "#!/bin/sh\n"
        "set -eu\n"
        f"export ONESTRING_OPTCUTS_ACTIVE_PATH='{active_path}'\n"
        f"export ONESTRING_OPTCUTS_SCALE_DIAG_PATH='{diag_path}'\n"
        "export ONESTRING_OPTCUTS_SCALE_BOUND=\"${ONESTRING_OPTCUTS_SCALE_BOUND:-2.0}\"\n"
        "export ONESTRING_OPTCUTS_SCALE_WEIGHT=\"${ONESTRING_OPTCUTS_SCALE_WEIGHT:-1.0}\"\n"
        f"exec '{binary.resolve()}' \"$@\"\n",
        encoding="utf-8",
    )
    runner.chmod(0o755)

    (root / ".onestring_optcuts_binary").write_text(
        str(runner.resolve()) + "\n", encoding="utf-8"
    )
    print(
        "[OPTCUTS-ONESTRING-FORCE-ENABLED] physical objective is now "
        "lambda_SD * E_SD + weight * SUM(residual^2); scale is NOT multiplied "
        "by lambda_SD; physical SD Hessian preconditioner is floored at 1.0; "
        "candidate-local Hessian scaling remains original; scaffold remains SD-only"
    )
    print(f"[OPTCUTS-ONESTRING-BUILD] binary={binary.resolve()}")
    print(f"[OPTCUTS-ONESTRING-RUNNER] runner={runner.resolve()}")
    print(f"[OPTCUTS-ONESTRING-RUNNER] active_proof={active_path}")
    print(f"[OPTCUTS-ONESTRING-RUNNER] diagnostics={diag_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
