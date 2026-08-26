#!/usr/bin/env python3
"""Make Grid-aware first-seam search practical on large meshes.

The first-seam search must still perform an actual onePointCut + Grid boundary +
harmonic feasibility check, but it should not run a full Symmetric Dirichlet
scan for every sampled initial vertex.  This patch reduces the default exact
candidate budget and ranks feasible candidates with a cheap normalized physical
seam-length proxy.
"""
from __future__ import annotations

from pathlib import Path

MARKER = "ONESTRING_GRID_NATIVE_V4_INITIAL_SEARCH_FAST"


def apply_initial_search_perf_patch(root: Path) -> bool:
    path = root / "src" / "main.cpp"
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        print("Grid first-seam performance patch already present.")
        return False

    old_budget = '''                            const double defaultCandidateBudget = temp.F.rows() >= 8000 ? 12.0 :
                                                                  (temp.F.rows() >= 3000 ? 18.0 : 32.0);'''
    new_budget = '''                            // ONESTRING_GRID_NATIVE_V4_INITIAL_SEARCH_FAST
                            // Each exact candidate performs one full harmonic solve.
                            // Keep enough spatial coverage to avoid an arbitrary first cut,
                            // but do not spend dozens of full-mesh solves before OptCuts starts.
                            const double defaultCandidateBudget = temp.F.rows() >= 8000 ? 3.0 :
                                                                  (temp.F.rows() >= 3000 ? 4.0 : 6.0);'''
    if text.count(old_budget) != 1:
        raise RuntimeError(f"initial Grid candidate budget anchor count={text.count(old_budget)}")
    text = text.replace(old_budget, new_budget, 1)

    old_score = '''                                    OptCuts::SymDirichletEnergy sd;
                                    double totalSD = 0.0;
                                    bool finiteSD = true;
                                    for(int triI = 0; triI < probe.F.rows(); ++triI) {
                                        double elem = 0.0;
                                        sd.getEnergyValByElemID(probe, triI, elem);
                                        if(!std::isfinite(elem)) { finiteSD = false; break; }
                                        totalSD += elem;
                                    }
                                    if(!finiteSD) continue;
                                    double seamLength = 0.0;
                                    for(int cohI = 0; cohI < trial.initSeams.rows(); ++cohI) {
                                        const Eigen::RowVector4i e = trial.initSeams.row(cohI);
                                        if(e[0] >= 0 && e[1] >= 0 && e[0] < trial.V_rest.rows() && e[1] < trial.V_rest.rows()) {
                                            seamLength += (trial.V_rest.row(e[0]) - trial.V_rest.row(e[1])).norm();
                                        }
                                    }
                                    const double avgSD = totalSD / std::max<Eigen::Index>(1, probe.F.rows());
                                    const double score = avgSD + 1.0e-3 * seamLength /
                                        std::max(1.0e-12, probe.virtualRadius);'''
    new_score = '''                                    // The expensive part above is the exact feasibility
                                    // check (actual cut + harmonic map + inversion + authoritative
                                    // cohesive Grid alignment).  Do not additionally scan every
                                    // triangle for SD for every initial candidate.  Among feasible
                                    // cuts, prefer the shorter physical seam as a cheap proxy; the
                                    // ordinary OptCuts optimizer will handle distortion globally.
                                    double seamLength = 0.0;
                                    for(int cohI = 0; cohI < trial.initSeams.rows(); ++cohI) {
                                        const Eigen::RowVector4i e = trial.initSeams.row(cohI);
                                        if(e[0] >= 0 && e[1] >= 0 && e[0] < trial.V_rest.rows() && e[1] < trial.V_rest.rows()) {
                                            seamLength += (trial.V_rest.row(e[0]) - trial.V_rest.row(e[1])).norm();
                                        }
                                    }
                                    const double score = seamLength /
                                        std::max(1.0e-12, probe.virtualRadius);'''
    if text.count(old_score) != 1:
        raise RuntimeError(f"initial Grid expensive score anchor count={text.count(old_score)}")
    text = text.replace(old_score, new_score, 1)

    path.write_text(text, encoding="utf-8")
    print(f"Applied fast Grid first-seam search: {path}")
    return True


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("root", type=Path)
    a = p.parse_args(); apply_initial_search_perf_patch(a.root.resolve())
