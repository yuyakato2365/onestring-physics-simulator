#!/usr/bin/env python3
"""Canonical OneString Grid-OptCuts V4 patch entrypoint.

This is the only patch entrypoint setup_optcuts.py calls. Grid bijectivity is
part of the core main.cpp patch itself; there is no separately-wired bijectivity
fragment anymore.
"""
from __future__ import annotations

from pathlib import Path

from patch_optcuts_native_grid_v4_final import apply_native_grid_patch as _apply_core_native_grid_patch
from patch_optcuts_native_grid_v4_perf import apply_native_grid_perf_patch
from patch_optcuts_native_grid_v4_diagnostics import apply_native_grid_diagnostics
from patch_optcuts_native_grid_v4_trial_relax import apply_trial_relax_patch

CANONICAL_PATCH_VERSION = "4.6-cheap-adaptive-trials-authoritative-seams"
NATIVE_RUNTIME_MARKER = "[ONESTRING-GRID] native_candidate_search enabled version=4"
GRID_BIJECTIVITY_RUNTIME_MARKER = "[ONESTRING-GRID] global_bijectivity_scaffold=enabled"

_REQUIRED_TRIMESH_MARKERS = (
    "ONESTRING_GRID_NATIVE_V4",
    "ONESTRING_GRID_NATIVE_V4_FAST_SEARCH",
    "ONESTRING_GRID_NATIVE_V4_ORIGINAL_LOCAL_PROPOSAL",
    "ONESTRING_GRID_NATIVE_V4_DEFER_TRIAL_VALIDATE",
    "ONESTRING_GRID_NATIVE_V4_CHEAP_TRIAL_AUDIT",
    "ONESTRING_GRID_NATIVE_V4_ADAPTIVE_TRIAL_BUDGET",
)
_REQUIRED_MAIN_MARKERS = (
    "native_candidate_search enabled version=4",
    "ONESTRING_GRID_NATIVE_V4_BIJECTIVITY_SCAFFOLD",
    "global_bijectivity_scaffold=",
    "ONESTRING_GRID_BIJECTIVITY_SCAFFOLD_DISABLED",
    "ONESTRING_GRID_NATIVE_V4_SEAM_SIDECAR",
    "final_seam_sidecar=",
)
_REQUIRED_OPTIMIZER_MARKERS = (
    "ONESTRING_GRID_NATIVE_V4_ZERO_LOCKED_SEARCH_DIR",
    "ONESTRING_GRID_NATIVE_V4_HARD_LOCK_STEP_FORWARD",
)


def _require_markers(path: Path, markers: tuple[str, ...]) -> None:
    text = path.read_text(encoding="utf-8")
    missing = [marker for marker in markers if marker not in text]
    if missing:
        raise RuntimeError(
            f"Consolidated Grid-OptCuts V4 source verification failed for {path}: "
            f"missing markers={missing}"
        )


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one anchor, found {count}")
    return text.replace(old, new, 1)


def _patch_candidate_cost(root: Path) -> bool:
    """Keep exact topology validation but remove redundant global SD rescoring.

    The proposal-aware V4 ranks feasible Grid embeddings using OptCuts' original
    local proposal score.  Recomputing total Symmetric Dirichlet before/after
    every trial therefore adds O(F) work twice per trial without affecting the
    selected candidate.  Large meshes also need a smaller exact-trial budget.
    """
    path = root / "src" / "TriMesh.cpp"
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    changed = False

    if "ONESTRING_GRID_NATIVE_V4_CHEAP_TRIAL_AUDIT" not in text:
        old = '''    const double before = oneStringTotalSD(mesh);\n    const double after = oneStringTotalSD(trial);\n    if(!std::isfinite(before) || !std::isfinite(after)) return -DBL_MAX;\n    const double sdDec = before - after;\n    energyChanges.first = -sdDec;\n    energyChanges.second = seamIncrease;\n    // Sign is intentionally NOT used to decide whether the topology is useful:\n    // the trial has not yet undergone OptCuts relaxation.  Returning a finite\n    // value simply means the exact Grid topology is immediately valid.\n    return (1.0 - lambda_t) * sdDec - lambda_t * seamIncrease;\n'''
        new = '''    // ONESTRING_GRID_NATIVE_V4_CHEAP_TRIAL_AUDIT\n    // Feasibility only: candidate ranking uses OptCuts' original local proposal\n    // score, so a full-mesh SD before/after pass here is redundant and extremely\n    // expensive on production meshes. Keep the exact topology/inversion/Grid\n    // checks above, then return a finite sentinel.\n    (void)lambda_t;\n    energyChanges.first = 0.0;\n    energyChanges.second = seamIncrease;\n    return 0.0;\n'''
        text = _replace_once(text, old, new, "Grid trial cheap feasibility audit")
        changed = True

    if "ONESTRING_GRID_NATIVE_V4_ADAPTIVE_TRIAL_BUDGET" not in text:
        old = '''    const int targetCap = std::max(1, static_cast<int>(std::llround(\n        oneStringGridEnvDouble("ONESTRING_OPTCUTS_GRID_TARGETS_PER_VERTEX", 4.0))));\n    const int maxTrials = std::max(1, static_cast<int>(std::llround(\n        oneStringGridEnvDouble("ONESTRING_OPTCUTS_GRID_MAX_EXACT_TRIALS", 4.0))));\n'''
        new = '''    // ONESTRING_GRID_NATIVE_V4_ADAPTIVE_TRIAL_BUDGET\n    // Exact trial cuts copy and validate the current mesh.  Preserve exhaustive\n    // small-mesh behaviour, but automatically cap work for large production\n    // surfaces unless the user explicitly overrides the environment variables.\n    const double defaultTargetCap = mesh.F.rows() >= 8000 ? 2.0 :\n                                    (mesh.F.rows() >= 3000 ? 3.0 : 4.0);\n    const double defaultMaxTrials = mesh.F.rows() >= 3000 ? 1.0 :\n                                    (mesh.F.rows() >= 1000 ? 2.0 : 4.0);\n    const int targetCap = std::max(1, static_cast<int>(std::llround(\n        oneStringGridEnvDouble("ONESTRING_OPTCUTS_GRID_TARGETS_PER_VERTEX", defaultTargetCap))));\n    const int maxTrials = std::max(1, static_cast<int>(std::llround(\n        oneStringGridEnvDouble("ONESTRING_OPTCUTS_GRID_MAX_EXACT_TRIALS", defaultMaxTrials))));\n'''
        text = _replace_once(text, old, new, "Grid adaptive exact-trial budget")
        changed = True

    if changed:
        path.write_text(text, encoding="utf-8")
        print(f"Applied cheap/adaptive Grid candidate audit: {path}")
    else:
        print("Cheap/adaptive Grid candidate audit already present.")
    return changed


def _patch_optimizer_hard_grid_locks(root: Path) -> bool:
    """Make Grid locks a hard optimizer invariant, not only a solver hint."""
    path = root / "src" / "Optimizer.cpp"
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    if all(marker in text for marker in _REQUIRED_OPTIMIZER_MARKERS):
        print("Grid-OptCuts hard optimizer locks already present.")
        return False

    text = _replace_once(
        text,
        """        else {\n            linSysSolver->solve(minusG, searchDir);\n        }\n        if(!mute) { timer_step.stop(); }\n        \n        fractureInitiated = false;\n""",
        """        else {\n            linSysSolver->solve(minusG, searchDir);\n        }\n        if(!mute) { timer_step.stop(); }\n\n        // ONESTRING_GRID_NATIVE_V4_ZERO_LOCKED_SEARCH_DIR\n        for(const int vI : result.oneStringGridLockedVert) {\n            if(vI >= 0 && vI < result.V.rows() && vI * 2 + 1 < searchDir.size()) {\n                searchDir[vI * 2] = 0.0;\n                searchDir[vI * 2 + 1] = 0.0;\n            }\n        }\n        \n        fractureInitiated = false;\n""",
        "Optimizer hard-zero Grid lock search direction",
    )

    text = _replace_once(
        text,
        """        for(int vI = 0; vI < data.V.rows(); vI++) {\n            data.V(vI, 0) = dataV0(vI, 0) + stepSize * searchDir[vI * 2];\n            data.V(vI, 1) = dataV0(vI, 1) + stepSize * searchDir[vI * 2 + 1];\n        }\n""",
        """        for(int vI = 0; vI < data.V.rows(); vI++) {\n            // ONESTRING_GRID_NATIVE_V4_HARD_LOCK_STEP_FORWARD\n            if(data.oneStringGridLockedVert.find(vI) != data.oneStringGridLockedVert.end()) {\n                data.V.row(vI) = dataV0.row(vI);\n                continue;\n            }\n            data.V(vI, 0) = dataV0(vI, 0) + stepSize * searchDir[vI * 2];\n            data.V(vI, 1) = dataV0(vI, 1) + stepSize * searchDir[vI * 2 + 1];\n        }\n""",
        "Optimizer hard-skip Grid lock stepForward",
    )

    path.write_text(text, encoding="utf-8")
    print(f"Applied hard Grid seam/junction optimizer locks: {path}")
    return True


def _patch_authoritative_grid_seam_sidecar(root: Path) -> bool:
    """Export actual final cohesive seam sides from C++ instead of re-inferring them from OBJ."""
    path = root / "src" / "main.cpp"
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    if "ONESTRING_GRID_NATIVE_V4_SEAM_SIDECAR" in text:
        print("Grid-OptCuts authoritative seam sidecar already present.")
        return False

    old = """    if(writeMesh) {\n        triSoup[channel_result]->saveAsMesh(outputFolderPath + infoName + \"_mesh.obj\", F);\n        triSoup[channel_result]->saveAsMesh(outputFolderPath + infoName + \"_mesh_normalizedUV.obj\", F, true);\n    }\n"""
    new = """    if(writeMesh) {\n        // ONESTRING_GRID_NATIVE_V4_SEAM_SIDECAR\n        // Export the actual cohesive seam topology before OBJ gluing.  Python must\n        // consume this file instead of trying to infer cuts from v/vt indices.\n        if(oneStringMainGridEnabled()) {\n            const OptCuts::TriMesh* gridMesh = triSoup[channel_result];\n            if(gridMesh->boundaryEdge.size() != gridMesh->cohE.rows()) {\n                throw std::runtime_error(\"ONESTRING_GRID_FINAL_COHESIVE_METADATA_MISMATCH\");\n            }\n            const double h = oneStringMainGridH();\n            const double angle = oneStringMainGridEnvDouble(\"ONESTRING_OPTCUTS_GRID_ANGLE_RAD\", 0.0);\n            const double phaseU = oneStringMainGridEnvDouble(\"ONESTRING_OPTCUTS_GRID_PHASE_U\", 0.0);\n            const double phaseV = oneStringMainGridEnvDouble(\"ONESTRING_OPTCUTS_GRID_PHASE_V\", 0.0);\n            const double c = std::cos(angle), s = std::sin(angle);\n            const double tol = 1.0e-7;\n            auto gridUnits = [&](const Eigen::RowVector2d& p) {\n                return Eigen::Vector2d(\n                    (p[0] * c + p[1] * s - phaseU) / h,\n                    (-p[0] * s + p[1] * c - phaseV) / h\n                );\n            };\n            auto isGridPoint = [&](const Eigen::RowVector2d& p) {\n                const Eigen::Vector2d g = gridUnits(p);\n                return std::abs(g[0] - std::round(g[0])) <= tol &&\n                       std::abs(g[1] - std::round(g[1])) <= tol;\n            };\n            auto isGridEdge = [&](const Eigen::RowVector2d& a, const Eigen::RowVector2d& b) {\n                if(!isGridPoint(a) || !isGridPoint(b) || (a - b).norm() <= h * tol) return false;\n                const Eigen::Vector2d ga = gridUnits(a), gb = gridUnits(b);\n                return std::abs(std::round(ga[0]) - std::round(gb[0])) <= tol ||\n                       std::abs(std::round(ga[1]) - std::round(gb[1])) <= tol;\n            };\n\n            const std::string seamPath = outputFolderPath + infoName + \"_grid_seams.txt\";\n            std::ofstream seamFile(seamPath);\n            if(!seamFile.is_open()) {\n                throw std::runtime_error(\"ONESTRING_GRID_FINAL_SEAM_SIDECAR_OPEN_FAILED\");\n            }\n            seamFile.precision(17);\n            int seamSideCount = 0;\n            for(int cohI = 0; cohI < gridMesh->cohE.rows(); ++cohI) {\n                if(gridMesh->boundaryEdge[cohI]) continue;\n                const Eigen::RowVector4i e = gridMesh->cohE.row(cohI);\n                if(e.minCoeff() < 0 || e.maxCoeff() >= gridMesh->V.rows()) {\n                    throw std::runtime_error(\"ONESTRING_GRID_FINAL_COHESIVE_INDEX_INVALID\");\n                }\n                for(int side = 0; side < 2; ++side) {\n                    const int k = side * 2;\n                    const Eigen::RowVector2d a = gridMesh->V.row(e[k]);\n                    const Eigen::RowVector2d b = gridMesh->V.row(e[k + 1]);\n                    if(!isGridEdge(a, b)) {\n                        std::cerr << \"[ONESTRING-GRID] final_off_lattice_cohesive_side coh=\" << cohI\n                                  << \" side=\" << side << \" a=\" << a << \" b=\" << b << std::endl;\n                        throw std::runtime_error(\"ONESTRING_GRID_FINAL_COHESIVE_SEAM_OFF_LATTICE\");\n                    }\n                    seamFile << a[0] << ' ' << a[1] << ' ' << b[0] << ' ' << b[1] << '\\n';\n                    ++seamSideCount;\n                }\n            }\n            seamFile.close();\n            std::cout << \"[ONESTRING-GRID] final_seam_sidecar=\" << seamPath\n                      << \" sides=\" << seamSideCount << std::endl;\n        }\n        triSoup[channel_result]->saveAsMesh(outputFolderPath + infoName + \"_mesh.obj\", F);\n        triSoup[channel_result]->saveAsMesh(outputFolderPath + infoName + \"_mesh_normalizedUV.obj\", F, true);\n    }\n"""
    text = _replace_once(text, old, new, "main.cpp authoritative Grid seam sidecar")
    path.write_text(text, encoding="utf-8")
    print(f"Applied authoritative Grid seam sidecar export: {path}")
    return True


def apply_grid_optcuts_v4(root: Path) -> bool:
    root = root.expanduser().resolve()
    changed = False

    changed = bool(_apply_core_native_grid_patch(root)) or changed
    changed = bool(apply_native_grid_perf_patch(root)) or changed
    changed = bool(apply_native_grid_diagnostics(root)) or changed
    changed = bool(apply_trial_relax_patch(root)) or changed
    changed = bool(_patch_candidate_cost(root)) or changed
    changed = bool(_patch_optimizer_hard_grid_locks(root)) or changed
    changed = bool(_patch_authoritative_grid_seam_sidecar(root)) or changed

    _require_markers(root / "src" / "TriMesh.cpp", _REQUIRED_TRIMESH_MARKERS)
    _require_markers(root / "src" / "main.cpp", _REQUIRED_MAIN_MARKERS)
    _require_markers(root / "src" / "Optimizer.cpp", _REQUIRED_OPTIMIZER_MARKERS)

    print(
        "Verified consolidated Grid-OptCuts V4 source contract "
        f"({CANONICAL_PATCH_VERSION})."
    )
    return changed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    apply_grid_optcuts_v4(args.root)
