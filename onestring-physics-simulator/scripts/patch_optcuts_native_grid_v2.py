#!/usr/bin/env python3
"""Second-stage native Grid-OptCuts patch.

V1 constrains OptCuts split candidates.  V2 closes two correctness gaps:
* the genus-0 initial topological cut is followed by Tutte mapping in upstream
  OptCuts, so V2 constructs a *grid-feasible initial UV state* before the main
  OptCuts loop starts; and
* accepted interior/boundary cuts are checked immediately after the real
  topology mutation, not only on the candidate copy.

The initial-state solve is not a final-seam post-process. It occurs before the
first OptCuts topology iteration and defines the feasible starting point of the
constrained optimization problem.
"""
from __future__ import annotations

from pathlib import Path

from patch_optcuts_native_grid import apply_native_grid_patch

MARKER = "ONESTRING_GRID_NATIVE_V2"


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one anchor, found {count}")
    return text.replace(old, new, 1)


def _patch_header(root: Path) -> bool:
    path = root / "src" / "TriMesh.hpp"
    text = path.read_text(encoding="utf-8")
    if "oneStringInitializeGridSeams" in text:
        return False
    text = _replace_once(
        text,
        "        void highCurvOnePointCut(void);\n",
        "        void highCurvOnePointCut(void);\n"
        "        // OneString native Grid-OptCuts: establish the grid-feasible initial UV state.\n"
        "        bool oneStringInitializeGridSeams(void);\n",
        "declare native grid initializer",
    )
    backup = path.with_suffix(path.suffix + ".onestring-grid-native-v2-backup")
    if not backup.exists():
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    path.write_text(text, encoding="utf-8")
    return True


def _patch_cpp(root: Path) -> bool:
    path = root / "src" / "TriMesh.cpp"
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        return False
    if "ONESTRING_GRID_NATIVE_V1" not in text:
        raise RuntimeError("native Grid-OptCuts V1 must be applied before V2")

    initializer = r'''
    // ONESTRING_GRID_NATIVE_V2
    bool TriMesh::oneStringInitializeGridSeams(void)
    {
        if(!oneStringGridEnabled()) return true;

        // Native V1 intentionally supports the genus-0 two-edge seed first.
        // Higher-genus cut_to_disk seeds have a larger seam graph and require a
        // separate constrained graph initializer rather than an implicit fallback.
        if(cohE.rows() != 2) {
            std::cout << "[ONESTRING-GRID] unsupported_initial_seam_graph cohesive_edges="
                      << cohE.rows() << " (native V2 expects the genus-0 two-edge seed)" << std::endl;
            return false;
        }

        std::vector<int> seamIds;
        for(int e = 0; e < cohE.rows(); ++e) {
            for(int k = 0; k < 4; ++k) {
                const int id = cohE(e, k);
                if(id < 0) continue;
                if(std::find(seamIds.begin(), seamIds.end(), id) == seamIds.end()) seamIds.emplace_back(id);
            }
        }
        if(seamIds.size() < 4) return false;

        // Group UV copies that are the same physical surface vertex.  Duplicated
        // seam copies have identical V_rest coordinates even though UV ids differ.
        std::vector<std::vector<int>> groups;
        const double xyzTol = std::max(1.0e-12, avgEdgeLen * 1.0e-9);
        for(const int id : seamIds) {
            int group = -1;
            for(int g = 0; g < static_cast<int>(groups.size()); ++g) {
                if((V_rest.row(id) - V_rest.row(groups[g][0])).norm() <= xyzTol) {
                    group = g;
                    break;
                }
            }
            if(group < 0) {
                groups.push_back(std::vector<int>(1, id));
            }
            else {
                groups[group].push_back(id);
            }
        }
        if(groups.size() != 3) {
            std::cout << "[ONESTRING-GRID] initial_seed_physical_vertex_count=" << groups.size()
                      << " expected=3" << std::endl;
            return false;
        }

        std::vector<std::set<int>> graph(groups.size());
        auto groupOf = [&](int id) -> int {
            for(int g = 0; g < static_cast<int>(groups.size()); ++g) {
                if(std::find(groups[g].begin(), groups[g].end(), id) != groups[g].end()) return g;
            }
            return -1;
        };
        for(int e = 0; e < cohE.rows(); ++e) {
            const int a = groupOf(cohE(e, 0));
            const int b = groupOf(cohE(e, 1));
            if(a < 0 || b < 0 || a == b) return false;
            graph[a].insert(b);
            graph[b].insert(a);
        }
        int center = -1;
        std::vector<int> ends;
        for(int g = 0; g < static_cast<int>(graph.size()); ++g) {
            if(graph[g].size() == 2) center = g;
            else if(graph[g].size() == 1) ends.emplace_back(g);
        }
        if(center < 0 || ends.size() != 2) return false;

        auto representative = [&](int g) -> Eigen::RowVector2d {
            Eigen::RowVector2d p = Eigen::RowVector2d::Zero();
            for(const int id : groups[g]) p += V.row(id);
            return p / static_cast<double>(groups[g].size());
        };
        const Eigen::RowVector2d p0 = representative(ends[0]);
        const Eigen::RowVector2d p1 = representative(center);
        const Eigen::RowVector2d p2 = representative(ends[1]);
        const auto embeddings = oneStringGridCornerEmbeddings(p0, p1, p2);
        if(embeddings.empty()) {
            std::cout << "[ONESTRING-GRID] no_grid_embedding_for_initial_seed" << std::endl;
            return false;
        }

        SymDirichletEnergy SD;
        double bestEnergy = std::numeric_limits<double>::infinity();
        TriMesh bestMesh;
        bool found = false;
        for(const auto& gridPath : embeddings) {
            TriMesh trial = *this;
            const int orderedGroups[3] = {ends[0], center, ends[1]};
            for(int i = 0; i < 3; ++i) {
                for(const int id : groups[orderedGroups[i]]) {
                    trial.V.row(id) = gridPath.row(i);
                    trial.fixedVert.insert(id);
                }
            }
            if(!trial.checkInversion(true)) continue;

            std::vector<Energy*> terms(1, &SD);
            std::vector<double> weights(1, 1.0);
            Optimizer optimizer(trial, terms, weights, 0, true, false);
            optimizer.precompute();
            optimizer.setRelGL2Tol(1.0e-7);
            optimizer.solve(120);
            TriMesh candidate = optimizer.getResult();
            if(!candidate.checkInversion(true)) continue;

            bool latticeOK = true;
            for(int i = 0; i < 3; ++i) {
                for(const int id : groups[orderedGroups[i]]) {
                    const Eigen::Vector2d guv = oneStringToGridUnits(candidate.V.row(id));
                    const Eigen::Vector2d nearest(std::round(guv[0]), std::round(guv[1]));
                    if((guv - nearest).norm() > 1.0e-7) latticeOK = false;
                }
            }
            if(!latticeOK) continue;
            const double energy = optimizer.getLastEnergyVal(true);
            if(std::isfinite(energy) && energy < bestEnergy) {
                bestEnergy = energy;
                bestMesh = candidate;
                found = true;
            }
        }
        if(!found) {
            std::cout << "[ONESTRING-GRID] no_injective_grid_feasible_initial_parameterization" << std::endl;
            return false;
        }
        *this = bestMesh;
        initSeams = cohE;
        std::cout << "[ONESTRING-GRID] initial_grid_state_ready energy=" << bestEnergy
                  << " fixed_vertices=" << fixedVert.size() << std::endl;
        return true;
    }

'''
    text = _replace_once(
        text,
        "    void TriMesh::farthestPointCut(int p_vI)\n",
        initializer + "    void TriMesh::farthestPointCut(int p_vI)\n",
        "insert initial feasible-state solver",
    )

    # The real interior topology change must remain locally injective after all
    # vertex duplication/reindexing, not merely in the pre-cut candidate copy.
    text = _replace_once(
        text,
        "            computeFeatures(); //TODO: only update locally\n            \n            for(int vI = 2; vI + 1 < path.size(); vI++) {\n",
        "            computeFeatures(); //TODO: only update locally\n"
        "            if(oneStringGridEncoded && !checkInversion(true)) {\n"
        "                throw std::runtime_error(\"ONESTRING_GRID_APPLIED_INTERIOR_CUT_INVERTED\");\n"
        "            }\n"
        "            \n            for(int vI = 2; vI + 1 < path.size(); vI++) {\n",
        "interior post-topology inversion guard",
    )

    # After either cut type, every encoded grid target itself must still be an
    # exact lattice point.  This catches accidental row-order/index corruption.
    old_boundary_guard = '''            if(!checkInversion(true)) {\n                throw std::runtime_error("ONESTRING_GRID_APPLIED_BOUNDARY_CUT_INVERTED");\n            }\n'''
    new_boundary_guard = '''            for(int gi = 0; gi < gridPath.rows(); ++gi) {\n                const Eigen::Vector2d g = oneStringToGridUnits(gridPath.row(gi));\n                const Eigen::Vector2d nearest(std::round(g[0]), std::round(g[1]));\n                if((g - nearest).norm() > 1.0e-7) {\n                    throw std::runtime_error("ONESTRING_GRID_APPLIED_BOUNDARY_CUT_OFF_LATTICE");\n                }\n            }\n            if(!checkInversion(true)) {\n                throw std::runtime_error("ONESTRING_GRID_APPLIED_BOUNDARY_CUT_INVERTED");\n            }\n'''
    text = _replace_once(text, old_boundary_guard, new_boundary_guard, "boundary lattice guard")

    backup = path.with_suffix(path.suffix + ".onestring-grid-native-v2-backup")
    if not backup.exists():
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    path.write_text(text, encoding="utf-8")
    return True


def _patch_main(root: Path) -> bool:
    path = root / "src" / "main.cpp"
    text = path.read_text(encoding="utf-8")
    if "initial_grid_state_ready" in text or "oneStringInitializeGridSeams" in text:
        return False
    old = "        triSoup.emplace_back(new OptCuts::TriMesh(V, F, UV_Tutte, temp.F, false));\n"
    new = (
        old
        + "        if(!triSoup.back()->oneStringInitializeGridSeams()) {\n"
        + "            std::cout << \"ONESTRING_GRID_INITIALIZATION_INFEASIBLE: could not construct a grid-feasible initial parameterization\" << std::endl;\n"
        + "            return -1;\n"
        + "        }\n"
    )
    text = _replace_once(text, old, new, "activate native grid initial feasible state")
    backup = path.with_suffix(path.suffix + ".onestring-grid-native-v2-backup")
    if not backup.exists():
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    path.write_text(text, encoding="utf-8")
    return True


def apply_native_grid_patch_v2(root: Path) -> bool:
    root = root.expanduser().resolve()
    changed = bool(apply_native_grid_patch(root))
    changed = _patch_header(root) or changed
    changed = _patch_cpp(root) or changed
    changed = _patch_main(root) or changed
    return changed


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    apply_native_grid_patch_v2(args.root)
