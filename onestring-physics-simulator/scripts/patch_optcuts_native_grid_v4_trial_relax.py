#!/usr/bin/env python3
"""Validate Grid cuts and make the first cut obey the same Grid-feasible policy."""
from __future__ import annotations
from pathlib import Path

TRI_MARKER = "ONESTRING_GRID_NATIVE_V4_DEFER_TRIAL_VALIDATE"
MAIN_MARKER = "ONESTRING_GRID_NATIVE_V4_BIJECTIVITY_SCAFFOLD"
INITIAL_SEARCH_MARKER = "ONESTRING_GRID_NATIVE_V4_INITIAL_CANDIDATE_SEARCH"


def apply_trial_relax_patch(root: Path) -> bool:
    changed = False

    tri_path = root / "src" / "TriMesh.cpp"
    text = tri_path.read_text(encoding="utf-8")
    if TRI_MARKER not in text:
        old_interior = '''            if(oneStringGridEncodedInterior) {
                oneStringLockGridVertices(*this, {path[0], path[1], path[2], nV});
                if(!checkInversion(true) || !oneStringAllCohesiveSeamSidesGridAligned(*this)) {
                    throw std::runtime_error("ONESTRING_GRID_APPLIED_INTERIOR_CUT_INVALID");
                }
            }
'''
        new_interior = '''            if(oneStringGridEncodedInterior) {
                oneStringLockGridVertices(*this, {path[0], path[1], path[2], nV});
                // ONESTRING_GRID_NATIVE_V4_DEFER_TRIAL_VALIDATE
                if(allowCutThrough &&
                   (!checkInversion(true) || !oneStringAllCohesiveSeamSidesGridAligned(*this))) {
                    throw std::runtime_error("ONESTRING_GRID_APPLIED_INTERIOR_CUT_INVALID");
                }
            }
'''
        if text.count(old_interior) != 1:
            raise RuntimeError(f"interior Grid validation anchor count={text.count(old_interior)}")
        text = text.replace(old_interior, new_interior, 1)

        old_boundary = '''        if(oneStringGridEncodedBoundary) {
            oneStringGridLockedVert.insert(vI_boundary);
            oneStringGridLockedVert.insert(oneStringGridBoundaryDuplicate);
            oneStringGridLockedVert.insert(vI_interior);
            fixedVert.insert(vI_boundary);
            fixedVert.insert(oneStringGridBoundaryDuplicate);
            fixedVert.insert(vI_interior);
            if(!checkInversion(true)) {
                throw std::runtime_error("ONESTRING_GRID_APPLIED_BOUNDARY_CUT_INVERTED");
            }
        }
'''
        new_boundary = '''        if(oneStringGridEncodedBoundary) {
            oneStringGridLockedVert.insert(vI_boundary);
            oneStringGridLockedVert.insert(oneStringGridBoundaryDuplicate);
            oneStringGridLockedVert.insert(vI_interior);
            fixedVert.insert(vI_boundary);
            fixedVert.insert(oneStringGridBoundaryDuplicate);
            fixedVert.insert(vI_interior);
            if(allowCutThrough &&
               (!checkInversion(true) || !oneStringAllCohesiveSeamSidesGridAligned(*this))) {
                throw std::runtime_error("ONESTRING_GRID_APPLIED_BOUNDARY_CUT_INVALID");
            }
        }
'''
        if text.count(old_boundary) != 1:
            raise RuntimeError(f"boundary Grid validation anchor count={text.count(old_boundary)}")
        text = text.replace(old_boundary, new_boundary, 1)
        tri_path.write_text(text, encoding="utf-8")
        print(f"Applied Grid-OptCuts deferred trial/accepted-cut validation: {tri_path}")
        changed = True

    main_path = root / "src" / "main.cpp"
    main = main_path.read_text(encoding="utf-8")

    # First seam: do not accept OptCuts' arbitrary first vertex and only snap its
    # finished boundary afterwards. Trial multiple physical one-point cuts, put
    # each resulting 4-vertex boundary on the fixed Grid, harmonic-map it, reject
    # inversion/off-Grid cohesive topology, and choose the lowest-SD feasible one.
    if INITIAL_SEARCH_MARKER not in main:
        old = '''                    case 0:
                        temp.onePointCut(F_component[componentI](0, 0));
                        rand1PInitCut = (n_components == 1);
                        break;
'''
        new = '''                    case 0:
                        if(oneStringMainGridEnabled()) {
                            // ONESTRING_GRID_NATIVE_V4_INITIAL_CANDIDATE_SEARCH
                            if(n_components != 1) {
                                throw std::runtime_error("ONESTRING_GRID_INITIAL_MULTICOMP_UNSUPPORTED");
                            }
                            const int nVInitial = temp.V_rest.rows();
                            const double defaultCandidateBudget = temp.F.rows() >= 8000 ? 12.0 :
                                                                  (temp.F.rows() >= 3000 ? 18.0 : 32.0);
                            const int candidateBudget = std::max(1, std::min(nVInitial,
                                static_cast<int>(std::llround(oneStringMainGridEnvDouble(
                                    "ONESTRING_OPTCUTS_GRID_INITIAL_CANDIDATES", defaultCandidateBudget)))));
                            std::vector<int> initialCandidates;
                            initialCandidates.reserve(candidateBudget + 1);
                            initialCandidates.push_back(F_component[componentI](0, 0));
                            for(int sampleI = 0; sampleI < candidateBudget; ++sampleI) {
                                const int vI = std::min(nVInitial - 1,
                                    static_cast<int>((static_cast<long long>(sampleI) * nVInitial) /
                                                     std::max(1, candidateBudget)));
                                if(std::find(initialCandidates.begin(), initialCandidates.end(), vI) == initialCandidates.end()) {
                                    initialCandidates.push_back(vI);
                                }
                            }

                            bool foundInitial = false;
                            double bestInitialScore = std::numeric_limits<double>::infinity();
                            int bestInitialVertex = -1;
                            OptCuts::TriMesh bestInitialMesh;
                            int testedInitial = 0;
                            int feasibleInitial = 0;

                            const double gridH = oneStringMainGridH();
                            const double gridAngle = oneStringMainGridEnvDouble("ONESTRING_OPTCUTS_GRID_ANGLE_RAD", 0.0);
                            const double gridPhaseU = oneStringMainGridEnvDouble("ONESTRING_OPTCUTS_GRID_PHASE_U", 0.0);
                            const double gridPhaseV = oneStringMainGridEnvDouble("ONESTRING_OPTCUTS_GRID_PHASE_V", 0.0);
                            const double gc = std::cos(gridAngle), gs = std::sin(gridAngle);
                            const double gridTol = 1.0e-7;
                            auto gridUnits = [&](const Eigen::RowVector2d& p) {
                                return Eigen::Vector2d(
                                    (p[0] * gc + p[1] * gs - gridPhaseU) / gridH,
                                    (-p[0] * gs + p[1] * gc - gridPhaseV) / gridH
                                );
                            };
                            auto gridPoint = [&](const Eigen::RowVector2d& p) {
                                const Eigen::Vector2d g = gridUnits(p);
                                return std::abs(g[0] - std::round(g[0])) <= gridTol &&
                                       std::abs(g[1] - std::round(g[1])) <= gridTol;
                            };
                            auto gridEdge = [&](const Eigen::RowVector2d& a, const Eigen::RowVector2d& b) {
                                if(!gridPoint(a) || !gridPoint(b) || (a - b).norm() <= gridH * gridTol) return false;
                                const Eigen::Vector2d ga = gridUnits(a), gb = gridUnits(b);
                                return std::abs(std::round(ga[0]) - std::round(gb[0])) <= gridTol ||
                                       std::abs(std::round(ga[1]) - std::round(gb[1])) <= gridTol;
                            };

                            for(const int candidateVI : initialCandidates) {
                                if(candidateVI < 0 || candidateVI >= nVInitial || temp.vNeighbor[candidateVI].size() < 3) continue;
                                ++testedInitial;
                                try {
                                    OptCuts::TriMesh trial = temp;
                                    trial.onePointCut(candidateVI);
                                    std::vector<std::vector<int>> trialLoops;
                                    igl::boundary_loop(trial.F, trialLoops);
                                    if(trialLoops.size() != 1 || trialLoops[0].size() != 4) continue;

                                    Eigen::VectorXi trialBoundary = Eigen::VectorXi::Map(
                                        trialLoops[0].data(), trialLoops[0].size());
                                    Eigen::MatrixXd trialBoundaryUV = oneStringMainInitialGridBoundary(trial, trialLoops[0]);
                                    Eigen::SparseMatrix<double> trialA, trialM;
                                    OptCuts::IglUtils::computeUniformLaplacian(trial.F, trialA);
                                    Eigen::MatrixXd trialUV;
                                    igl::harmonic(trialA, trialM, trialBoundary, trialBoundaryUV, 1, trialUV);

                                    OptCuts::TriMesh probe(V, F, trialUV, trial.F, false);
                                    if(!probe.checkInversion(true)) {
                                        oneStringMainReflectFabricationV(trialUV);
                                        probe = OptCuts::TriMesh(V, F, trialUV, trial.F, false);
                                        if(!probe.checkInversion(true)) continue;
                                    }
                                    if(probe.boundaryEdge.size() != probe.cohE.rows()) continue;
                                    bool seamOK = true;
                                    for(int cohI = 0; cohI < probe.cohE.rows() && seamOK; ++cohI) {
                                        if(probe.boundaryEdge[cohI]) continue;
                                        const Eigen::RowVector4i e = probe.cohE.row(cohI);
                                        if(e.minCoeff() < 0 || e.maxCoeff() >= probe.V.rows()) { seamOK = false; break; }
                                        seamOK = gridEdge(probe.V.row(e[0]), probe.V.row(e[1])) &&
                                                 gridEdge(probe.V.row(e[2]), probe.V.row(e[3]));
                                    }
                                    if(!seamOK) continue;

                                    OptCuts::SymDirichletEnergy sd;
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
                                        std::max(1.0e-12, probe.virtualRadius);
                                    ++feasibleInitial;
                                    if(score < bestInitialScore) {
                                        bestInitialScore = score;
                                        bestInitialVertex = candidateVI;
                                        // Preserve the exact cut topology/rest metadata from the
                                        // trial, but commit the Grid-feasible harmonic UV that was
                                        // actually validated above.  The previous code stored the
                                        // pre-Grid trial UV here, which made coh=0 drift off lattice.
                                        trial.V = trialUV;
                                        bestInitialMesh = trial;
                                        foundInitial = true;
                                    }
                                }
                                catch(...) {
                                    continue;
                                }
                            }
                            if(!foundInitial) {
                                throw std::runtime_error("ONESTRING_GRID_NO_FEASIBLE_INITIAL_CUT");
                            }
                            temp = bestInitialMesh;
                            rand1PInitCut = false;
                            std::cout << "[ONESTRING-GRID] initial_candidate_search selected_vertex="
                                      << bestInitialVertex << " tested=" << testedInitial
                                      << " feasible=" << feasibleInitial
                                      << " score=" << bestInitialScore << std::endl;
                        }
                        else {
                            temp.onePointCut(F_component[componentI](0, 0));
                            rand1PInitCut = (n_components == 1);
                        }
                        break;
'''
        if main.count(old) != 1:
            raise RuntimeError(f"Grid initial candidate-search anchor count={main.count(old)}")
        main = main.replace(old, new, 1)
        changed = True
        print(f"Applied Grid-aware first-seam candidate search: {main_path}")

    # Older patch variants inserted the scaffold marker directly at the first cut.
    # Current canonical main.cpp owns the authoritative scaffold decision later at
    # Optimizer construction, so only patch this legacy path when still present.
    if MAIN_MARKER not in main:
        old = '''                        temp.onePointCut(F_component[componentI](0, 0));
                        rand1PInitCut = (n_components == 1);
                        break;
'''
        new = '''                        temp.onePointCut(F_component[componentI](0, 0));
                        if(oneStringMainGridEnabled()) {
                            rand1PInitCut = false;
                        }
                        else {
                            rand1PInitCut = (n_components == 1);
                        }
                        break;
'''
        if main.count(old) == 1:
            main = main.replace(old, new, 1)
            changed = True

    if changed:
        main_path.write_text(main, encoding="utf-8")

    return changed


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("root", type=Path)
    a = p.parse_args(); apply_trial_relax_patch(a.root.resolve())
