#!/usr/bin/env python3
"""Robust function-scoped installer for OneString native Grid-OptCuts V4.

The algorithmic implementation (candidate generation/scoring helpers) lives in
``patch_optcuts_native_grid.py``.  This installer deliberately applies those
changes only inside the intended official OptCuts functions, so repeated source
patterns elsewhere in TriMesh.cpp cannot be modified accidentally.
"""
from __future__ import annotations

from pathlib import Path

from patch_optcuts_native_grid import (
    BACKUP_SUFFIX,
    MAIN_HELPERS,
    MARKER,
    OLD_MARKERS,
    TRIMESH_HELPERS,
)


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one anchor, found {count}")
    return text.replace(old, new, 1)


def _function_bounds(text: str, signature: str, next_signature: str, label: str) -> tuple[int, int]:
    start = text.find(signature)
    if start < 0:
        raise RuntimeError(f"{label}: function signature not found: {signature!r}")
    end = text.find(next_signature, start + len(signature))
    if end < 0:
        raise RuntimeError(f"{label}: next function signature not found: {next_signature!r}")
    return start, end


def _replace_in_function(
    text: str,
    signature: str,
    next_signature: str,
    old: str,
    new: str,
    label: str,
) -> str:
    start, end = _function_bounds(text, signature, next_signature, label)
    body = text[start:end]
    count = body.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one function-local anchor, found {count}")
    body = body.replace(old, new, 1)
    return text[:start] + body + text[end:]


def _backup_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + BACKUP_SUFFIX)


def _restore_if_patched(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    patched = (
        MARKER in text
        or any(marker in text for marker in OLD_MARKERS)
        or "ONESTRING_GRID_NATIVE_V4_MAIN" in text
        or "oneStringGridLockedVert" in text
    )
    if not patched:
        return
    backup = _backup_path(path)
    if not backup.is_file():
        raise RuntimeError(
            f"{path} already contains a OneString Grid patch, but pristine backup {backup} is missing. "
            "Restore the official OptCuts checkout or delete/reclone third_party/OptCuts before rebuilding."
        )
    path.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Restored pristine OptCuts source: {path}")


def _backup_once(path: Path) -> None:
    backup = _backup_path(path)
    if not backup.exists():
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")


def _patch_header(root: Path) -> None:
    path = root / "src" / "TriMesh.hpp"
    _restore_if_patched(path)
    _backup_once(path)
    text = path.read_text(encoding="utf-8")
    text = _replace_once(
        text,
        "        std::set<int> fixedVert; // for linear solve\n        Eigen::Matrix<double, 2, 3> bbox;",
        "        std::set<int> fixedVert; // for linear solve\n"
        "        // OneString native Grid-OptCuts: persistent fabrication-grid seam/junction locks.\n"
        "        std::set<int> oneStringGridLockedVert;\n"
        "        Eigen::Matrix<double, 2, 3> bbox;",
        "TriMesh.hpp grid lock member",
    )
    path.write_text(text, encoding="utf-8")


def _patch_trimesh(root: Path) -> None:
    path = root / "src" / "TriMesh.cpp"
    _restore_if_patched(path)
    _backup_once(path)
    text = path.read_text(encoding="utf-8")

    text = _replace_once(
        text,
        "extern int inSplitTotalAmt;\n\nnamespace OptCuts {",
        "extern int inSplitTotalAmt;\n" + TRIMESH_HELPERS + "\nnamespace OptCuts {",
        "TriMesh.cpp helper insertion",
    )

    text = _replace_in_function(
        text,
        "    void TriMesh::computeFeatures(bool multiComp, bool resetFixedV)",
        "    void TriMesh::updateFeatures(void)",
        """        if(resetFixedV) {\n            fixedVert.clear();\n            fixedVert.insert(0);\n        }\n""",
        """        if(resetFixedV) {\n            fixedVert.clear();\n            fixedVert.insert(0);\n        }\n        if(oneStringGridEnabled()) {\n            fixedVert.insert(oneStringGridLockedVert.begin(), oneStringGridLockedVert.end());\n        }\n""",
        "computeFeatures persistent Grid locks",
    )

    text = _replace_in_function(
        text,
        "    void TriMesh::resetFixedVert(const std::set<int>& p_fixedVert)",
        "    void TriMesh::buildCohEfromRecord",
        """        fixedVert = p_fixedVert;\n        computeLaplacianMtr();\n""",
        """        fixedVert = p_fixedVert;\n        if(oneStringGridEnabled()) {\n            fixedVert.insert(oneStringGridLockedVert.begin(), oneStringGridLockedVert.end());\n        }\n        computeLaplacianMtr();\n""",
        "resetFixedVert persistent Grid locks",
    )

    text = _replace_in_function(
        text,
        "    bool TriMesh::splitOrMerge(double lambda_t, double EDecThres, bool propagate, bool splitInterior,",
        "    void TriMesh::onePointCut(int vI)",
        """        if(splitInterior) {\n            querySplit(lambda_t, propagate, splitInterior,\n                       EwDec_max, path_max, newVertPos_max,\n                       energyChanes_split);\n        }\n        else {\n            double EwDec_max_split, EwDec_max_merge;\n""",
        """        if(splitInterior) {\n            querySplit(lambda_t, propagate, splitInterior,\n                       EwDec_max, path_max, newVertPos_max,\n                       energyChanes_split);\n        }\n        else if(oneStringGridEnabled()) {\n            // Native Grid-OptCuts V4 is deliberately split-only. An unconstrained\n            // merge can destroy the fixed fabrication lattice.\n            querySplit(lambda_t, propagate, splitInterior,\n                       EwDec_max, path_max, newVertPos_max,\n                       energyChanes_split);\n            isMerge = false;\n        }\n        else {\n            double EwDec_max_split, EwDec_max_merge;\n""",
        "splitOrMerge disable merge in Grid mode",
    )

    text = _replace_in_function(
        text,
        "    double TriMesh::computeLocalLDec(int vI, double lambda_t,",
        "    double TriMesh::computeLocalEdDec_inSplit",
        """    {\n        if(!path_max.empty()) {\n            // merge query\n""",
        """    {\n        if(oneStringGridEnabled() && path_max.empty()) {\n            return oneStringGridComputeLocalLDec(*this, vI, lambda_t, path_max, newVertPos_max, energyChanges_max);\n        }\n\n        if(!path_max.empty()) {\n            // merge query\n""",
        "computeLocalLDec Grid candidate dispatch",
    )

    # cutPath: paired interior Grid seam A-B-C / A-D-C.
    text = _replace_in_function(
        text,
        "    int TriMesh::cutPath(std::vector<int> path, bool makeCoh, int changePos,",
        "    void TriMesh::computeSeamScore",
        """        if(changePos) {\n            assert((changePos == 1) && \"right now only support change 1\"); //!!! still only allow 1?\n            assert(newVertPos.cols() == 2);\n            assert(changePos * 2 == newVertPos.rows());\n        }\n""",
        """        const bool oneStringGridEncodedInterior = oneStringGridEnabled() && changePos &&\n            newVertPos.cols() == 2 && newVertPos.rows() == 4;\n        if(changePos) {\n            assert((changePos == 1) && \"right now only support change 1\");\n            assert(newVertPos.cols() == 2);\n            assert(oneStringGridEncodedInterior || changePos * 2 == newVertPos.rows());\n            if(oneStringGridEncodedInterior && path.size() != 3) {\n                throw std::runtime_error(\"ONESTRING_GRID_INTERIOR_PATH_SIZE_UNSUPPORTED\");\n            }\n        }\n""",
        "cutPath paired Grid encoding",
    )
    text = _replace_in_function(
        text,
        "    int TriMesh::cutPath(std::vector<int> path, bool makeCoh, int changePos,",
        "    void TriMesh::computeSeamScore",
        """            // path is interior\n            assert(path.size() >= 3);\n            \n            std::vector<int> tri_left;\n""",
        """            // path is interior\n            assert(path.size() >= 3);\n            if(oneStringGridEncodedInterior) {\n                V.row(path[0]) = newVertPos.row(2);\n                V.row(path[2]) = newVertPos.row(3);\n            }\n            \n            std::vector<int> tri_left;\n""",
        "cutPath apply Grid endpoints",
    )
    text = _replace_in_function(
        text,
        "    int TriMesh::cutPath(std::vector<int> path, bool makeCoh, int changePos,",
        "    void TriMesh::computeSeamScore",
        """            computeFeatures(); //TODO: only update locally\n            \n            for(int vI = 2; vI + 1 < path.size(); vI++) {\n""",
        """            computeFeatures(); //TODO: only update locally\n            if(oneStringGridEncodedInterior) {\n                oneStringLockGridVertices(*this, {path[0], path[1], path[2], nV});\n                if(!checkInversion(true) || !oneStringAllCohesiveSeamSidesGridAligned(*this)) {\n                    throw std::runtime_error(\"ONESTRING_GRID_APPLIED_INTERIOR_CUT_INVALID\");\n                }\n            }\n            \n            for(int vI = 2; vI + 1 < path.size(); vI++) {\n""",
        "cutPath lock paired Grid seam",
    )

    # splitEdgeOnBoundary: extend an existing Grid seam without collapsing its two UV sides.
    text = _replace_in_function(
        text,
        "    void TriMesh::splitEdgeOnBoundary(const std::pair<int, int>& edge,",
        "    void TriMesh::mergeBoundaryEdges",
        """        else {\n            assert(isBoundaryVert(edge.second) && \"Input edge must attach mesh boundary!\");\n            \n            vI_boundary = edge.second;\n            vI_interior = edge.first;\n        }\n        \n        fracTail.erase(vI_boundary);\n""",
        """        else {\n            assert(isBoundaryVert(edge.second) && \"Input edge must attach mesh boundary!\");\n            \n            vI_boundary = edge.second;\n            vI_interior = edge.first;\n        }\n        const bool oneStringGridEncodedBoundary = oneStringGridEnabled() && changeVertPos &&\n            !duplicateBoth && newVertPos.cols() == 2 && newVertPos.rows() == 3;\n        if(oneStringGridEnabled() && changeVertPos && !duplicateBoth &&\n           newVertPos.rows() != 2 && !oneStringGridEncodedBoundary) {\n            throw std::runtime_error(\"ONESTRING_GRID_BOUNDARY_ENCODING_SIZE\");\n        }\n        if(oneStringGridEncodedBoundary) {\n            V.row(vI_interior) = newVertPos.row(2);\n        }\n        \n        fracTail.erase(vI_boundary);\n""",
        "splitEdgeOnBoundary decode Grid target",
    )
    text = _replace_in_function(
        text,
        "    void TriMesh::splitEdgeOnBoundary(const std::pair<int, int>& edge,",
        "    void TriMesh::mergeBoundaryEdges",
        """        int nV = static_cast<int>(V_rest.rows());\n        V_rest.conservativeResize(nV + 1, 3);\n""",
        """        int nV = static_cast<int>(V_rest.rows());\n        const int oneStringGridBoundaryDuplicate = nV;\n        V_rest.conservativeResize(nV + 1, 3);\n""",
        "splitEdgeOnBoundary capture duplicate id",
    )
    text = _replace_in_function(
        text,
        "    void TriMesh::splitEdgeOnBoundary(const std::pair<int, int>& edge,",
        "    void TriMesh::mergeBoundaryEdges",
        """        if(duplicateBoth) {\n            int nV = static_cast<int>(V_rest.rows());\n""",
        """        if(duplicateBoth) {\n            int nV = static_cast<int>(V_rest.rows());\n""",
        "splitEdgeOnBoundary duplicateBoth structural assertion",
    )
    start, end = _function_bounds(
        text,
        "    void TriMesh::splitEdgeOnBoundary(const std::pair<int, int>& edge,",
        "    void TriMesh::mergeBoundaryEdges",
        "splitEdgeOnBoundary final Grid lock",
    )
    body = text[start:end]
    closing = "    }\n    \n"
    if not body.endswith(closing):
        raise RuntimeError("splitEdgeOnBoundary final Grid lock: unexpected function ending")
    body = body[:-len(closing)] + """        if(oneStringGridEncodedBoundary) {\n            oneStringGridLockedVert.insert(vI_boundary);\n            oneStringGridLockedVert.insert(oneStringGridBoundaryDuplicate);\n            oneStringGridLockedVert.insert(vI_interior);\n            fixedVert.insert(vI_boundary);\n            fixedVert.insert(oneStringGridBoundaryDuplicate);\n            fixedVert.insert(vI_interior);\n            if(!checkInversion(true)) {\n                throw std::runtime_error(\"ONESTRING_GRID_APPLIED_BOUNDARY_CUT_INVERTED\");\n            }\n        }\n    }\n    \n"""
    text = text[:start] + body + text[end:]

    path.write_text(text, encoding="utf-8")
    print(f"Applied function-scoped OneString native Grid-OptCuts V4 patch: {path}")


def _patch_main(root: Path) -> None:
    path = root / "src" / "main.cpp"
    _restore_if_patched(path)
    _backup_once(path)
    text = path.read_text(encoding="utf-8")

    text = _replace_once(
        text,
        """#include <fstream>\n#include <string>\n#include <ctime>\n\n\nEigen::MatrixXd V, UV, N;""",
        """#include <fstream>\n#include <string>\n#include <ctime>\n#include <cstdlib>\n#include <cmath>\n#include <stdexcept>\n\n""" + MAIN_HELPERS + "\nEigen::MatrixXd V, UV, N;",
        "main.cpp Grid helper insertion",
    )

    text = _replace_once(
        text,
        """            Eigen::MatrixXd bnd_uv;\n            OptCuts::IglUtils::map_vertices_to_circle(temp.V_rest,\n                                        bnd_stacked.tail(bnd_all[longest_bnd_id].size()),\n                                        bnd_uv);\n            double xOffset = componentI % UVGridDim * 2.1, yOffset = componentI / UVGridDim * 2.1;\n            for(int bnd_uvI = 0; bnd_uvI < bnd_uv.rows(); bnd_uvI++) {\n                bnd_uv(bnd_uvI, 0) += xOffset;\n                bnd_uv(bnd_uvI, 1) += yOffset;\n            }\n""",
        """            Eigen::MatrixXd bnd_uv;\n            if(oneStringMainGridEnabled() && temp.initSeams.rows() > 0) {\n                if(n_components != 1) {\n                    throw std::runtime_error(\"ONESTRING_GRID_INITIAL_MULTICOMP_UNSUPPORTED\");\n                }\n                bnd_uv = oneStringMainInitialGridBoundary(temp, bnd_all[longest_bnd_id]);\n            }\n            else {\n                OptCuts::IglUtils::map_vertices_to_circle(temp.V_rest,\n                                            bnd_stacked.tail(bnd_all[longest_bnd_id].size()),\n                                            bnd_uv);\n                double xOffset = componentI % UVGridDim * 2.1, yOffset = componentI / UVGridDim * 2.1;\n                for(int bnd_uvI = 0; bnd_uvI < bnd_uv.rows(); bnd_uvI++) {\n                    bnd_uv(bnd_uvI, 0) += xOffset;\n                    bnd_uv(bnd_uvI, 1) += yOffset;\n                }\n            }\n""",
        "main.cpp initial Grid boundary",
    )

    text = _replace_once(
        text,
        """        triSoup.emplace_back(new OptCuts::TriMesh(V, F, UV_Tutte, temp.F, false));\n        outputFolderPath += meshName + \"_Tutte_\" + OptCuts::IglUtils::rtos(lambda_init) + \"_\" + OptCuts::IglUtils::rtos(testID) +\n""",
        """        if(oneStringMainGridEnabled() && temp.initSeams.rows() > 0) {\n            OptCuts::TriMesh probe(V, F, UV_Tutte, temp.F, false);\n            if(!probe.checkInversion(true)) {\n                oneStringMainReflectFabricationV(UV_Tutte);\n                OptCuts::TriMesh reflectedProbe(V, F, UV_Tutte, temp.F, false);\n                if(!reflectedProbe.checkInversion(true)) {\n                    throw std::runtime_error(\"ONESTRING_GRID_INITIAL_HARMONIC_INVERTED\");\n                }\n            }\n        }\n        OptCuts::TriMesh* oneStringInitialMesh = new OptCuts::TriMesh(V, F, UV_Tutte, temp.F, false);\n        if(oneStringMainGridEnabled() && temp.initSeams.rows() > 0) {\n            std::set<int> fixed = oneStringInitialMesh->fixedVert;\n            for(int i = 0; i < bnd_stacked.size(); ++i) {\n                oneStringInitialMesh->oneStringGridLockedVert.insert(bnd_stacked[i]);\n                fixed.insert(bnd_stacked[i]);\n            }\n            oneStringInitialMesh->resetFixedVert(fixed);\n        }\n        triSoup.emplace_back(oneStringInitialMesh);\n        outputFolderPath += meshName + \"_Tutte_\" + OptCuts::IglUtils::rtos(lambda_init) + \"_\" + OptCuts::IglUtils::rtos(testID) +\n""",
        "main.cpp lock initial Grid boundary",
    )

    text = _replace_once(
        text,
        """    optimizer = new OptCuts::Optimizer(*triSoup[0], energyTerms, energyParams, 0, false, bijectiveParam && !rand1PInitCut); // for random one point initial cut, don't need air meshes in the beginning since it's impossible for a quad to intersect itself\n    \n    optimizer->precompute();\n""",
        """    if(oneStringMainGridEnabled()) {\n        std::cout << \"[ONESTRING-GRID] native_candidate_search enabled version=4 h=\"\n                  << oneStringMainGridH() << std::endl;\n    }\n    optimizer = new OptCuts::Optimizer(*triSoup[0], energyTerms, energyParams, 0, false, bijectiveParam && !rand1PInitCut); // for random one point initial cut, don't need air meshes in the beginning since it's impossible for a quad to intersect itself\n    \n    optimizer->precompute();\n""",
        "main.cpp native V4 marker",
    )

    # V4 is split-only. If the distortion bound is already met, stop. If not,
    # keep the OptCuts dual update but select only already-evaluated Grid splits.
    text = _replace_once(
        text,
        """    double measure_bound = E_SD;\n    const double eps_lambda = std::min(1.0e-3, std::abs(updateLambda(measure_bound) - energyParams[0]));\n    \n    //TODO?: stop when first violates bounds from feasible, don't go to best feasible. check after each merge whether distortion is violated\n""",
        """    double measure_bound = E_SD;\n    const double eps_lambda = std::min(1.0e-3, std::abs(updateLambda(measure_bound) - energyParams[0]));\n\n    if(oneStringMainGridEnabled()) {\n        if(measure_bound <= upperBound) {\n            optimizer->updateEnergyData(true, false, false);\n            return false;\n        }\n        energyParams[0] = updateLambda(measure_bound);\n        energyParams[0] = std::max(eps_lambda, std::min(1.0 - eps_lambda, energyParams[0]));\n        if(checkConvergence) {\n            int bestB = -1, bestI = -1;\n            double changeB = __DBL_MAX__, changeI = __DBL_MAX__;\n            for(int attempt = 0; attempt < 64; ++attempt) {\n                bestB = energyChanges_bSplit.empty() ? -1 :\n                    computeBestCand(energyChanges_bSplit, 1.0 - energyParams[0], changeB);\n                bestI = energyChanges_iSplit.empty() ? -1 :\n                    computeBestCand(energyChanges_iSplit, 1.0 - energyParams[0], changeI);\n                if((bestB >= 0 && changeB <= 0.0) || (bestI >= 0 && changeI <= 0.0)) break;\n                const double next = updateLambda(measure_bound, energyParams[0]);\n                if(std::abs(next - energyParams[0]) <= 1.0e-12) break;\n                energyParams[0] = std::max(eps_lambda, std::min(1.0 - eps_lambda, next));\n            }\n            const bool useB = bestB >= 0 && changeB <= 0.0 &&\n                (bestI < 0 || changeI > 0.0 || changeB <= changeI);\n            const bool useI = bestI >= 0 && changeI <= 0.0 && !useB;\n            if(useB) {\n                opType_queried = 0;\n                path_queried = paths_bSplit[bestB];\n                newVertPos_queried = newVertPoses_bSplit[bestB];\n            }\n            else if(useI) {\n                opType_queried = 1;\n                path_queried = paths_iSplit[bestI];\n                newVertPos_queried = newVertPoses_iSplit[bestI];\n            }\n            else {\n                std::cout << \"[ONESTRING-GRID] no feasible Grid split remains before requested distortion bound\" << std::endl;\n                optimizer->updateEnergyData(true, false, false);\n                return false;\n            }\n        }\n        optimizer->updateEnergyData(true, false, false);\n        return true;\n    }\n    \n    //TODO?: stop when first violates bounds from feasible, don't go to best feasible. check after each merge whether distortion is violated\n""",
        "main.cpp Grid split-only dual update",
    )

    path.write_text(text, encoding="utf-8")
    print(f"Applied function-scoped OneString native Grid-OptCuts V4 patch: {path}")


def apply_native_grid_patch(root: Path) -> bool:
    root = root.expanduser().resolve()
    _patch_header(root)
    _patch_trimesh(root)
    _patch_main(root)
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    apply_native_grid_patch(args.root)
