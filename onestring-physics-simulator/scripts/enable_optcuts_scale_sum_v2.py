#!/usr/bin/env python3
"""Force-enable the OneString scale objective with the correct OptCuts weighting.

This script builds on ``enable_optcuts_scale_sum.py`` and fixes one crucial
problem: OptCuts multiplies the Symmetric-Dirichlet term by ``energyParams[0]``
(which starts near 1-lambda_init, e.g. 0.001).  The fabrication scale penalty
must NOT be multiplied by that SD Lagrange multiplier.

The physical objective therefore becomes exactly

    E = lambda_SD * E_SD + mu * SUM_i residual_i^2 + E_scaffold

and the physical gradient becomes

    g = lambda_SD * g_SD + mu * g_scale + g_scaffold.

The candidate-local optimizer uses lambda_SD=1, so it receives the same
unnormalized scale-sum objective without any special-case weakening.

OptCuts' distortion controller expects ``getLastEnergyVal(true) / lambda_SD``
to recover plain SD energy.  Because the total objective now also contains the
OneString term, this script subtracts the current OneString scale energy in
``getLastEnergyVal(true)`` before returning the value used by that controller.
This keeps the original OptCuts distortion-bound machinery semantically intact.
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
            f"OptCuts v2 source mismatch for {label}: expected exactly one anchor, found {count}."
        )
    return source.replace(old, new, 1)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    optcuts = root / "third_party" / "OptCuts"
    v1 = root / "scripts" / "enable_optcuts_scale_sum.py"

    # Produce the explicit Optimizer-level physical scale objective first.
    run([sys.executable, str(v1)], cwd=root)

    optimizer_cpp = optcuts / "src" / "Optimizer.cpp"
    if not optimizer_cpp.is_file():
        raise SystemExit(f"Missing generated source: {optimizer_cpp}")
    optimizer = optimizer_cpp.read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # Critical correction:
    # v1 put E_scale inside energyVal_ET[0], so OptCuts multiplied it by
    # energyParams[0] together with SD.  With lambda_init=0.999 that makes the
    # nominal mu=1 scale objective roughly 1000x weaker than intended.
    # Keep energyVal_ET[0] as pure SD and add E_scale after the SD multiplier.
    # ------------------------------------------------------------------
    optimizer = replace_once(
        optimizer,
        """        const double oneStringScaleEnergy = oneStringPhysicalScaleEnergy(data);\n        energyVal_ET[0] += oneStringScaleEnergy;\n""",
        """        const double oneStringScaleEnergy = oneStringPhysicalScaleEnergy(data);\n""",
        "remove scale from SD energy bucket",
    )
    optimizer = replace_once(
        optimizer,
        """        energyVal = energyParams[0] * energyVal_ET[0];\n""",
        """        energyVal = energyParams[0] * energyVal_ET[0] + oneStringScaleEnergy;\n""",
        "add scale outside SD Lagrange multiplier",
    )

    # Same correction for the gradient: leave gradient_ET[0] as pure SD so
    # OptCuts bookkeeping remains valid, and add scale after SD weighting.
    optimizer = replace_once(
        optimizer,
        """        gradient_ET[0] += oneStringScaleGradient;\n        oneStringPhysicalScaleDiagnostics(\n            data, oneStringSDGradient, oneStringScaleGradient);\n        gradient = energyParams[0] * gradient_ET[0];\n""",
        """        oneStringPhysicalScaleDiagnostics(\n            data, oneStringSDGradient, oneStringScaleGradient);\n        gradient = energyParams[0] * gradient_ET[0] + oneStringScaleGradient;\n""",
        "add scale gradient outside SD Lagrange multiplier",
    )

    # Preserve original OptCuts distortion-controller semantics.  The controller
    # asks for physical energy excluding scaffold and divides by lambda_SD to
    # estimate E_SD.  Subtract OneString E_scale first so that value stays SD.
    optimizer = replace_once(
        optimizer,
        """    double Optimizer::getLastEnergyVal(bool excludeScaffold) const\n    {\n        return ((excludeScaffold && scaffolding) ?\n                (lastEnergyVal - energyVal_scaffold) :\n                lastEnergyVal);\n    }\n""",
        """    double Optimizer::getLastEnergyVal(bool excludeScaffold) const\n    {\n        if(excludeScaffold) {\n            const double physical = scaffolding\n                ? (lastEnergyVal - energyVal_scaffold)\n                : lastEnergyVal;\n            return physical - oneStringPhysicalScaleEnergy(result);\n        }\n        return lastEnergyVal;\n    }\n""",
        "keep distortion controller scale-free",
    )

    # Add an unmistakable build-time token describing the actual formula.
    marker = "[OPTCUTS-ONESTRING-V2-ACTIVE] E=lambda_SD*E_SD + weight*SUM(residual^2) + E_scaffold"
    if marker not in optimizer:
        # Reuse the existing one-time runtime marker block and append the exact
        # corrected objective line immediately after it is emitted.
        anchor = """                << \" model=sum_residual_squared_no_area_normalization\"\n                << std::endl;\n"""
        replacement = anchor + """            std::cerr\n                << \"[OPTCUTS-ONESTRING-V2-ACTIVE] E=lambda_SD*E_SD + weight*SUM(residual^2) + E_scaffold\"\n                << std::endl;\n"""
        optimizer = replace_once(
            optimizer, anchor, replacement, "v2 runtime objective marker"
        )

    required = (
        "energyVal = energyParams[0] * energyVal_ET[0] + oneStringScaleEnergy;",
        "gradient = energyParams[0] * gradient_ET[0] + oneStringScaleGradient;",
        "return physical - oneStringPhysicalScaleEnergy(result);",
        "[OPTCUTS-ONESTRING-V2-ACTIVE]",
    )
    missing = [token for token in required if token not in optimizer]
    forbidden = (
        "energyVal_ET[0] += oneStringScaleEnergy;",
        "gradient_ET[0] += oneStringScaleGradient;",
    )
    bad = [token for token in forbidden if token in optimizer]
    if missing or bad:
        raise SystemExit(
            "OptCuts v2 verification failed. missing=" + repr(missing) + " forbidden=" + repr(bad)
        )

    optimizer_cpp.write_text(optimizer, encoding="utf-8")

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
        "[OPTCUTS-ONESTRING-V2-FORCE-ENABLED] "
        "E=lambda_SD*E_SD + weight*SUM(residual^2); "
        "scale is NOT multiplied by lambda_SD; scaffold remains SD-only"
    )
    print(f"[OPTCUTS-ONESTRING-BUILD] binary={binary.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
