#!/usr/bin/env python3
"""Final installer entrypoint for OneString native Grid-OptCuts V4.

TriMesh.hpp/TriMesh.cpp are patched by the function-scoped V4 installer. This
module owns the official main.cpp edits with anchors verified against the pinned
OptCuts source used by OneString.
"""
from __future__ import annotations

from pathlib import Path

from patch_optcuts_native_grid import MAIN_HELPERS
from patch_optcuts_native_grid_v4 import (
    _backup_once,
    _patch_header,
    _patch_trimesh,
    _replace_once,
    _restore_if_patched,
)


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

    # Official no-input-UV initialization maps the cut boundary to a circle.
    # Grid V4 replaces only that boundary target; the harmonic solve itself is
    # still the authors' implementation.
    text = _replace_once(
        text,
        """            Eigen::MatrixXd bnd_uv;\n            igl::map_vertices_to_circle(temp.V_rest,\n                                        bnd_stacked.tail(bnd_all[longest_bnd_id].size()),\n                                        bnd_uv);\n            double xOffset = componentI % UVGridDim * 2.1, yOffset = componentI / UVGridDim * 2.1;\n            for(int bnd_uvI = 0; bnd_uvI < bnd_uv.rows(); bnd_uvI++) {\n                bnd_uv(bnd_uvI, 0) += xOffset;\n                bnd_uv(bnd_uvI, 1) += yOffset;\n            }\n""",
        """            Eigen::MatrixXd bnd_uv;\n            if(oneStringMainGridEnabled() && temp.initSeams.rows() > 0) {\n                if(n_components != 1) {\n                    throw std::runtime_error(\"ONESTRING_GRID_INITIAL_MULTICOMP_UNSUPPORTED\");\n                }\n                bnd_uv = oneStringMainInitialGridBoundary(temp, bnd_all[longest_bnd_id]);\n            }\n            else {\n                igl::map_vertices_to_circle(temp.V_rest,\n                                            bnd_stacked.tail(bnd_all[longest_bnd_id].size()),\n                                            bnd_uv);\n                double xOffset = componentI % UVGridDim * 2.1, yOffset = componentI / UVGridDim * 2.1;\n                for(int bnd_uvI = 0; bnd_uvI < bnd_uv.rows(); bnd_uvI++) {\n                    bnd_uv(bnd_uvI, 0) += xOffset;\n                    bnd_uv(bnd_uvI, 1) += yOffset;\n                }\n            }\n""",
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
        "main.cpp V4 runtime marker",
    )

    # The authors' dual update switches between split and merge. V4 deliberately
    # has no unconstrained merge, so when the bound is met it stops; otherwise it
    # updates lambda and, on the re-query pass, chooses only recorded Grid splits.
    text = _replace_once(
        text,
        """    double measure_bound = E_SD;\n    const double eps_lambda = std::min(1.0e-3, std::abs(updateLambda(measure_bound) - energyParams[0]));\n    \n    //TODO?: stop when first violates bounds from feasible, don't go to best feasible. check after each merge whether distortion is violated\n""",
        """    double measure_bound = E_SD;\n    const double eps_lambda = std::min(1.0e-3, std::abs(updateLambda(measure_bound) - energyParams[0]));\n\n    if(oneStringMainGridEnabled()) {\n        if(measure_bound <= upperBound) {\n            optimizer->updateEnergyData(true, false, false);\n            return false;\n        }\n        energyParams[0] = updateLambda(measure_bound);\n        energyParams[0] = std::max(eps_lambda, std::min(1.0 - eps_lambda, energyParams[0]));\n        if(checkConvergence) {\n            int bestB = -1, bestI = -1;\n            double changeB = __DBL_MAX__, changeI = __DBL_MAX__;\n            for(int attempt = 0; attempt < 64; ++attempt) {\n                bestB = energyChanges_bSplit.empty() ? -1 : computeBestCand(energyChanges_bSplit, 1.0 - energyParams[0], changeB);\n                bestI = energyChanges_iSplit.empty() ? -1 : computeBestCand(energyChanges_iSplit, 1.0 - energyParams[0], changeI);\n                if((bestB >= 0 && changeB <= 0.0) || (bestI >= 0 && changeI <= 0.0)) break;\n                const double next = updateLambda(measure_bound, energyParams[0]);\n                if(std::abs(next - energyParams[0]) <= 1.0e-12) break;\n                energyParams[0] = std::max(eps_lambda, std::min(1.0 - eps_lambda, next));\n            }\n            const bool useB = bestB >= 0 && changeB <= 0.0 &&\n                (bestI < 0 || changeI > 0.0 || changeB <= changeI);\n            const bool useI = bestI >= 0 && changeI <= 0.0 && !useB;\n            if(useB) {\n                opType_queried = 0;\n                path_queried = paths_bSplit[bestB];\n                newVertPos_queried = newVertPoses_bSplit[bestB];\n            }\n            else if(useI) {\n                opType_queried = 1;\n                path_queried = paths_iSplit[bestI];\n                newVertPos_queried = newVertPoses_iSplit[bestI];\n            }\n            else {\n                std::cout << \"[ONESTRING-GRID] no feasible Grid split remains before requested distortion bound\" << std::endl;\n                optimizer->updateEnergyData(true, false, false);\n                return false;\n            }\n        }\n        optimizer->updateEnergyData(true, false, false);\n        return true;\n    }\n    \n    //TODO?: stop when first violates bounds from feasible, don't go to best feasible. check after each merge whether distortion is violated\n""",
        "main.cpp Grid split-only dual update",
    )

    path.write_text(text, encoding="utf-8")
    print(f"Applied verified Grid-OptCuts V4 main patch: {path}")


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
