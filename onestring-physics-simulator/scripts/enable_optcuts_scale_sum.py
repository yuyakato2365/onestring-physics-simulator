#!/usr/bin/env python3
"""Build and force-enable the OneString scale-sum objective in OptCuts.

This is intentionally more explicit than the original source patch:

* physical OptCuts energy is augmented in Optimizer::computeEnergyVal;
* physical OptCuts gradient is augmented in Optimizer::computeGradient;
* scaffold/air-mesh energy remains plain Symmetric Dirichlet;
* candidate discovery still receives the scale gradient through
  SymDirichletEnergy::computeLocalGradient;
* candidate-local optimization uses the exact same physical objective;
* E_scale is an UNNORMALIZED sum of per-triangle violation residuals.

The existing build_optcuts_onestring.py is first used to restore/patch the
upstream checkout.  This script then rewires the generated C++ so the scale
term is injected at the Optimizer level instead of relying on the physical and
scaffold calls sharing SymDirichletEnergy.
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

    # Start from the repository-tracked generated patch every time.
    run([sys.executable, str(base_builder)], cwd=root)

    optimizer_cpp = optcuts / "src" / "Optimizer.cpp"
    sd_cpp = optcuts / "src" / "Energy" / "SymDirichletEnergy.cpp"
    if not optimizer_cpp.is_file() or not sd_cpp.is_file():
        raise SystemExit("Generated OptCuts C++ sources were not found after base build.")

    optimizer = optimizer_cpp.read_text(encoding="utf-8")
    sd = sd_cpp.read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # 1. SymDirichletEnergy remains responsible for SD itself.
    #    Keep scale in getEnergyValByElemID because TriMesh's local-candidate
    #    baseline explicitly calls that method before applying its historical
    #    A_global/A_local bookkeeping. Remove scale from bulk energy and bulk
    #    gradient so Optimizer can add it exactly once to the physical mesh.
    # ------------------------------------------------------------------
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

    # Export explicit physical-mesh scale functions. They call the same helper
    # used by candidate scoring, including A_global/A_local compensation for
    # local candidate optimizers. No scaffold object is ever passed here.
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

    # ------------------------------------------------------------------
    # 2. Optimizer: add scale energy/gradient explicitly to the PHYSICAL data.
    #    Scaffold computation later in these functions still constructs a
    #    fresh SymDirichletEnergy and therefore never sees E_scale.
    # ------------------------------------------------------------------
    decl_anchor = """    void oneStringSetGlobalScaleReference(const TriMesh& data);\n    void oneStringSetDiagnosticScope(bool localScope);\n"""
    decl_replacement = decl_anchor + """    double oneStringPhysicalScaleEnergy(const TriMesh& data);\n    void oneStringPhysicalScaleGradient(const TriMesh& data, Eigen::VectorXd& gradient);\n    void oneStringPhysicalScaleDiagnostics(\n        const TriMesh& data,\n        const Eigen::VectorXd& sdGradient,\n        const Eigen::VectorXd& scaleGradient);\n"""
    optimizer = replace_once(
        optimizer, decl_anchor, decl_replacement,
        "Optimizer OneString physical objective declarations",
    )

    energy_anchor = """        energyTerms[0]->computeEnergyVal(data, energyVal_ET[0]);\n        energyVal = energyParams[0] * energyVal_ET[0];\n"""
    energy_replacement = """        energyTerms[0]->computeEnergyVal(data, energyVal_ET[0]);\n        const double oneStringScaleEnergy = oneStringPhysicalScaleEnergy(data);\n        energyVal_ET[0] += oneStringScaleEnergy;\n        energyVal = energyParams[0] * energyVal_ET[0];\n"""
    optimizer = replace_once(
        optimizer, energy_anchor, energy_replacement,
        "physical Optimizer energy injection",
    )

    grad_anchor = """        energyTerms[0]->computeGradient(data, gradient_ET[0]);\n        gradient = energyParams[0] * gradient_ET[0];\n"""
    grad_replacement = """        energyTerms[0]->computeGradient(data, gradient_ET[0]);\n        Eigen::VectorXd oneStringSDGradient = gradient_ET[0];\n        Eigen::VectorXd oneStringScaleGradient;\n        oneStringPhysicalScaleGradient(data, oneStringScaleGradient);\n        gradient_ET[0] += oneStringScaleGradient;\n        oneStringPhysicalScaleDiagnostics(\n            data, oneStringSDGradient, oneStringScaleGradient);\n        gradient = energyParams[0] * gradient_ET[0];\n"""
    optimizer = replace_once(
        optimizer, grad_anchor, grad_replacement,
        "physical Optimizer gradient injection",
    )

    # Runtime marker is emitted from the exact physical energy path, not merely
    # from initialization. This makes it impossible to silently fall back to SD.
    marker_anchor = """        const double oneStringScaleEnergy = oneStringPhysicalScaleEnergy(data);\n        energyVal_ET[0] += oneStringScaleEnergy;\n"""
    marker_replacement = marker_anchor + """        static std::atomic<bool> oneStringPhysicalObjectivePrinted(false);\n        bool oneStringExpected = false;\n        if(oneStringPhysicalObjectivePrinted.compare_exchange_strong(\n               oneStringExpected, true)) {\n            std::cerr\n                << \"[OPTCUTS-ONESTRING-PHYSICAL-OBJECTIVE-ACTIVE] scale_energy=\"\n                << oneStringScaleEnergy\n                << \" model=sum_residual_squared_no_area_normalization\"\n                << std::endl;\n        }\n"""
    optimizer = replace_once(
        optimizer, marker_anchor, marker_replacement,
        "physical objective runtime marker",
    )

    # Optimizer.cpp now uses atomic/iostream directly for the hard marker.
    optimizer = replace_once(
        optimizer,
        "#include <fstream>\n",
        "#include <fstream>\n#include <atomic>\n#include <iostream>\n",
        "Optimizer runtime marker includes",
    )

    optimizer_cpp.write_text(optimizer, encoding="utf-8")
    sd_cpp.write_text(sd, encoding="utf-8")

    required_optimizer = (
        "oneStringPhysicalScaleEnergy(data)",
        "oneStringPhysicalScaleGradient(data, oneStringScaleGradient)",
        "[OPTCUTS-ONESTRING-PHYSICAL-OBJECTIVE-ACTIVE]",
    )
    required_sd = (
        "double oneStringPhysicalScaleEnergy(const TriMesh& data)",
        "void oneStringPhysicalScaleGradient(",
        "oneStringAddScaleLocalGradient(data, localGradients)",
        # candidate baseline must still contain the single-element scale term
        "energyVal += oneStringScalePenaltyByElem(",
    )
    missing = [x for x in required_optimizer if x not in optimizer]
    missing += [x for x in required_sd if x not in sd]
    if missing:
        raise SystemExit("Force-enable verification failed: " + ", ".join(missing))

    # Rebuild after the explicit physical-objective rewrite.
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

    (root / ".onestring_optcuts_binary").write_text(
        str(binary.resolve()) + "\n", encoding="utf-8"
    )
    print(
        "[OPTCUTS-ONESTRING-FORCE-ENABLED] physical Optimizer now evaluates "
        "E_SD + weight * SUM(residual^2); scaffold remains SD-only"
    )
    print(f"[OPTCUTS-ONESTRING-BUILD] binary={binary.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
