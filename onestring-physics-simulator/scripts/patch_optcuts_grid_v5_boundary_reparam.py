#!/usr/bin/env python3
"""V5: treat Grid seams as UV boundary conditions, not surface-space Grid cuts.

OptCuts still proposes a physical cut topology on the source surface S. Once a
cut topology is considered, its two UV boundary copies are assigned H/V lattice
coordinates in Omega. Those seam coordinates are Dirichlet boundary conditions
for the parameterization itself: only the interior UV is re-solved.

Candidate enumeration stays cheap. A transiently inverted UV immediately after
inserting a cut is *not* rejected, because it has not yet been reparameterized
under the new boundary conditions. At a topology decision point, only a small
global shortlist of the best OptCuts proposals receives the expensive harmonic
boundary-constrained solve. The first feasible proposal is then committed and
subsequent Symmetric Dirichlet optimization keeps its seam vertices fixed.

This is deliberately not post-hoc snapping.
"""
from __future__ import annotations

from pathlib import Path

MARKER = "ONESTRING_GRID_NATIVE_V5_BOUNDARY_REPARAM"
SHORTLIST_MARKER = "ONESTRING_GRID_NATIVE_V5_REPARAM_SHORTLIST"


def _bounds(text: str, signature: str, next_signature: str, label: str) -> tuple[int, int]:
    start = text.find(signature)
    if start < 0:
        raise RuntimeError(f"{label}: signature not found: {signature!r}")
    end = text.find(next_signature, start + len(signature))
    if end < 0:
        raise RuntimeError(f"{label}: next signature not found: {next_signature!r}")
    return start, end


def _replace_in_function(
    text: str,
    signature: str,
    next_signature: str,
    old: str,
    new: str,
    label: str,
) -> str:
    start, end = _bounds(text, signature, next_signature, label)
    body = text[start:end]
    count = body.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one function-local anchor, found {count}")
    return text[:start] + body.replace(old, new, 1) + text[end:]


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one anchor, found {count}")
    return text.replace(old, new, 1)


def apply_boundary_reparameterization_patch(root: Path) -> bool:
    root = root.expanduser().resolve()
    header = root / "src" / "TriMesh.hpp"
    tri = root / "src" / "TriMesh.cpp"
    optimizer = root / "src" / "Optimizer.cpp"
    main_path = root / "src" / "main.cpp"
    for path in (header, tri, optimizer, main_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    changed = False

    # Public API so Optimizer and the final topology-decision shortlist can invoke
    # the same constrained Omega mapping.
    h = header.read_text(encoding="utf-8")
    if MARKER not in h:
        anchor = '''        int cutPath(std::vector<int> path, bool makeCoh = false, int changePos = 0,\n                     const Eigen::MatrixXd& newVertPos = Eigen::MatrixXd(), bool allowCutThrough = true);\n'''
        replacement = anchor + '''        // ONESTRING_GRID_NATIVE_V5_BOUNDARY_REPARAM\n        bool oneStringReparameterizeGridBoundary(void);\n'''
        if h.count(anchor) != 1:
            raise RuntimeError(
                f"TriMesh.hpp V5 API anchor: expected exactly one anchor, found {h.count(anchor)}"
            )
        h = h.replace(anchor, replacement, 1)
        header.write_text(h, encoding="utf-8")
        changed = True

    t = tri.read_text(encoding="utf-8")
    if MARKER not in t:
        method = r'''
    // ONESTRING_GRID_NATIVE_V5_BOUNDARY_REPARAM
    bool TriMesh::oneStringReparameterizeGridBoundary(void)
    {
        if(!oneStringGridEnabled()) return true;
        if(boundaryEdge.size() != cohE.rows()) return false;

        // C is a source-surface cut topology. Its two UV copies Phi(C) must
        // already be represented by H/V Grid edges before solving. Their current
        // Grid coordinates are the Dirichlet data; they are not snapped later.
        std::set<int> seamVerts;
        for(int cohI = 0; cohI < cohE.rows(); ++cohI) {
            if(boundaryEdge[cohI]) continue;
            const Eigen::RowVector4i e = cohE.row(cohI);
            if(e.minCoeff() < 0 || e.maxCoeff() >= V.rows()) return false;
            if(!oneStringIsGridEdge(V.row(e[0]), V.row(e[1])) ||
               !oneStringIsGridEdge(V.row(e[2]), V.row(e[3]))) return false;
            for(int k = 0; k < 4; ++k) seamVerts.insert(e[k]);
        }
        // Open source surfaces can legitimately have no cut seam yet.
        if(seamVerts.empty()) return true;

        // Fix the complete current UV boundary during the harmonic solve. For a
        // closed source surface this boundary is the cut graph itself. For an
        // open source surface it also preserves the true source boundary.
        std::vector<int> boundaryVerts;
        boundaryVerts.reserve(V.rows());
        for(int vI = 0; vI < V.rows(); ++vI) {
            if(isBoundaryVert(vI)) boundaryVerts.emplace_back(vI);
        }
        if(boundaryVerts.empty()) return false;

        Eigen::VectorXi b(static_cast<int>(boundaryVerts.size()));
        Eigen::MatrixXd bc(static_cast<int>(boundaryVerts.size()), 2);
        for(int i = 0; i < static_cast<int>(boundaryVerts.size()); ++i) {
            b[i] = boundaryVerts[i];
            bc.row(i) = V.row(boundaryVerts[i]);
        }

        Eigen::SparseMatrix<double> A, M;
        OptCuts::IglUtils::computeUniformLaplacian(F, A);
        Eigen::MatrixXd constrainedUV;
        igl::harmonic(A, M, b, bc, 1, constrainedUV);
        if(constrainedUV.rows() != V.rows() || constrainedUV.cols() != 2 ||
           !constrainedUV.allFinite()) return false;

        // Boundary coordinates are a fabrication invariant. Re-impose them
        // exactly instead of accepting linear-solver roundoff.
        for(int i = 0; i < static_cast<int>(boundaryVerts.size()); ++i) {
            constrainedUV.row(boundaryVerts[i]) = bc.row(i);
        }

        const Eigen::MatrixXd oldV = V;
        V = constrainedUV;
        for(const int vI : seamVerts) {
            oneStringGridLockedVert.insert(vI);
            fixedVert.insert(vI);
        }
        computeLaplacianMtr();

        if(!checkInversion(true) || !oneStringAllCohesiveSeamSidesGridAligned(*this)) {
            V = oldV;
            computeLaplacianMtr();
            return false;
        }

        std::cout << "[ONESTRING-GRID] boundary_constrained_reparameterization seam_vertices="
                  << seamVerts.size() << " boundary_vertices=" << boundaryVerts.size()
                  << std::endl;
        return true;
    }
'''
        anchor = "    void TriMesh::computeSeamScore(Eigen::VectorXd& seamScore) const\n"
        if t.count(anchor) != 1:
            raise RuntimeError(
                f"TriMesh.cpp V5 method insertion: expected exactly one anchor, found {t.count(anchor)}"
            )
        t = t.replace(anchor, method + "\n" + anchor, 1)

        # V4 audited a candidate before the new Grid boundary had been used to
        # solve the interior. That rejects exactly the candidates V5 is intended
        # to rescue. Keep topology/old-lock/Grid-edge checks cheap here; perform
        # the inversion test only after the constrained solve in the global
        # shortlist below.
        diagnostic_inversion = (
            'if(!trial.checkInversion(true)) { std::cout << "[ONESTRING-GRID-TRIAL-REJECT] '
            'inversion_after_relax" << std::endl; return -DBL_MAX; }'
        )
        plain_inversion = "if(!trial.checkInversion(true)) return -DBL_MAX;"
        if diagnostic_inversion in t:
            t = t.replace(
                diagnostic_inversion,
                '// V5: transient candidate inversion is checked after boundary-constrained reparameterization.',
                1,
            )
        elif plain_inversion in t:
            t = t.replace(
                plain_inversion,
                '// V5: transient candidate inversion is checked after boundary-constrained reparameterization.',
                1,
            )
        else:
            raise RuntimeError("TriMesh.cpp V5 deferred-inversion anchor not found")

        # Direct splitOrMerge path: if this path is used outside main.cpp's
        # queried-operation route, an accepted topology still gets the exact same
        # boundary-constrained Omega solve.
        t = _replace_in_function(
            t,
            "    bool TriMesh::splitOrMerge(double lambda_t, double EDecThres, bool propagate, bool splitInterior,",
            "    void TriMesh::onePointCut(int vI)",
            """            return true;\n""",
            """            if(oneStringGridEnabled() && !isMerge &&\n               !oneStringReparameterizeGridBoundary()) {\n                throw std::runtime_error(\"ONESTRING_GRID_BOUNDARY_CONSTRAINED_REPARAM_FAILED\");\n            }\n            return true;\n""",
            "TriMesh::splitOrMerge V5 constrained mapping",
        )
        tri.write_text(t, encoding="utf-8")
        changed = True

    o = optimizer.read_text(encoding="utf-8")
    if MARKER not in o:
        # Every Optimizer run starts from a parameterization whose seam is an
        # explicit UV boundary condition. This also re-derives persistent seam
        # locks from authoritative cohesive topology after mesh copying.
        o = _replace_in_function(
            o,
            "    void Optimizer::precompute(void)",
            "    int Optimizer::solve(int maxIter)",
            """        result = data0;\n""",
            """        result = data0;\n        // ONESTRING_GRID_NATIVE_V5_BOUNDARY_REPARAM\n        if(!result.oneStringReparameterizeGridBoundary()) {\n            throw std::runtime_error(\"ONESTRING_GRID_INITIAL_BOUNDARY_CONSTRAINED_REPARAM_FAILED\");\n        }\n""",
            "Optimizer::precompute V5 constrained mapping",
        )

        # createFracture(opType, ...) directly commits the already-selected split.
        # Reparameterize after the complete topology edit and before rebuilding
        # scaffold/Hessian. This is the authoritative committed state.
        o = _replace_in_function(
            o,
            "    bool Optimizer::createFracture(int opType, const std::vector<int>& path, const Eigen::MatrixXd& newVertPos, bool allowPropagate)",
            "    bool Optimizer::createFracture(double stressThres, int propType, bool allowPropagate, bool allowInSplit)",
            """        timer.stop();\n        \n        if(scaffolding) {\n""",
            """        timer.stop();\n\n        if(!result.oneStringReparameterizeGridBoundary()) {\n            throw std::runtime_error(\"ONESTRING_GRID_BOUNDARY_CONSTRAINED_REPARAM_FAILED_AFTER_TOPOLOGY\");\n        }\n        \n        if(scaffolding) {\n""",
            "Optimizer::createFracture V5 constrained mapping",
        )

        o = _replace_in_function(
            o,
            "    void Optimizer::setConfig(const TriMesh& config, int iterNum, int p_topoIter)",
            "    void Optimizer::setPropagateFracture(bool p_prop)",
            """        result = config;\n""",
            """        result = config;\n        if(!result.oneStringReparameterizeGridBoundary()) {\n            throw std::runtime_error(\"ONESTRING_GRID_CONFIG_BOUNDARY_CONSTRAINED_REPARAM_FAILED\");\n        }\n""",
            "Optimizer::setConfig V5 constrained mapping",
        )
        optimizer.write_text(o, encoding="utf-8")
        changed = True

    # V4 chose the best candidate only from the local OptCuts score. V5 keeps that
    # score as the cheap ranking oracle, but before committing a topology it
    # actually constructs the cut, fixes Phi(C) to the Grid, re-solves the
    # interior and checks bijectivity. Only a tiny GLOBAL shortlist receives this
    # expensive solve, so production meshes do not pay one harmonic solve per
    # candidate vertex/per iteration.
    m = main_path.read_text(encoding="utf-8")
    if SHORTLIST_MARKER not in m:
        old = '''            const bool useB = bestB >= 0 && changeB <= 0.0 &&
                (bestI < 0 || changeI > 0.0 || changeB <= changeI);
            const bool useI = bestI >= 0 && changeI <= 0.0 && !useB;
            if(useB) {
                opType_queried = 0;
                path_queried = paths_bSplit[bestB];
                newVertPos_queried = newVertPoses_bSplit[bestB];
            }
            else if(useI) {
                opType_queried = 1;
                path_queried = paths_iSplit[bestI];
                newVertPos_queried = newVertPoses_iSplit[bestI];
            }
            else {
                std::cout << "[ONESTRING-GRID] no feasible Grid split remains before requested distortion bound" << std::endl;
                optimizer->updateEnergyData(true, false, false);
                return false;
            }
'''
        new = '''            // ONESTRING_GRID_NATIVE_V5_REPARAM_SHORTLIST
            struct OneStringGridRankedSplit {
                int opType;
                int index;
                double weightedChange;
            };
            std::vector<OneStringGridRankedSplit> rankedGridSplits;
            const double cutLambda = 1.0 - energyParams[0];
            auto appendRanked = [&](int opType,
                                    const std::vector<std::pair<double, double>>& changes) {
                for(int i = 0; i < static_cast<int>(changes.size()); ++i) {
                    if(changes[i].first == __DBL_MAX__ || changes[i].second == __DBL_MAX__) continue;
                    const double weighted = changes[i].first * (1.0 - cutLambda) +
                                            changes[i].second * cutLambda;
                    if(std::isfinite(weighted) && weighted <= 0.0) {
                        rankedGridSplits.push_back({opType, i, weighted});
                    }
                }
            };
            appendRanked(0, energyChanges_bSplit);
            appendRanked(1, energyChanges_iSplit);
            std::sort(rankedGridSplits.begin(), rankedGridSplits.end(),
                      [](const OneStringGridRankedSplit& a, const OneStringGridRankedSplit& b) {
                          return a.weightedChange < b.weightedChange;
                      });

            const int faceCount = optimizer->getResult().F.rows();
            const double defaultShortlist = faceCount >= 8000 ? 2.0 :
                                            (faceCount >= 3000 ? 3.0 : 8.0);
            const int shortlistCap = std::max(1, static_cast<int>(std::llround(
                oneStringMainGridEnvDouble("ONESTRING_OPTCUTS_GRID_REPARAM_TRIALS", defaultShortlist))));
            const int exactCount = std::min<int>(shortlistCap, rankedGridSplits.size());

            bool foundConstrainedSplit = false;
            int selectedOpType = -1;
            int selectedIndex = -1;
            for(int rankI = 0; rankI < exactCount; ++rankI) {
                const auto& cand = rankedGridSplits[rankI];
                try {
                    OptCuts::TriMesh trial = optimizer->getResult();
                    if(cand.opType == 0) {
                        if(cand.index < 0 || cand.index >= static_cast<int>(paths_bSplit.size()) ||
                           cand.index >= static_cast<int>(newVertPoses_bSplit.size()) ||
                           paths_bSplit[cand.index].size() != 2) continue;
                        trial.splitEdgeOnBoundary(
                            std::pair<int, int>(paths_bSplit[cand.index][0], paths_bSplit[cand.index][1]),
                            newVertPoses_bSplit[cand.index], true, false);
                        trial.updateFeatures();
                    }
                    else {
                        if(cand.index < 0 || cand.index >= static_cast<int>(paths_iSplit.size()) ||
                           cand.index >= static_cast<int>(newVertPoses_iSplit.size()) ||
                           paths_iSplit[cand.index].size() != 3) continue;
                        trial.cutPath(paths_iSplit[cand.index], true, 1,
                                      newVertPoses_iSplit[cand.index], false);
                    }

                    // This is the actual feasibility test for the new method:
                    // topology C is fixed on S, Phi(C) is a Grid boundary in
                    // Omega, and only interior UV is solved. Inversion is tested
                    // after that solve, not before it.
                    if(!trial.oneStringReparameterizeGridBoundary()) {
                        std::cout << "[ONESTRING-GRID] constrained_candidate_reject rank="
                                  << rankI << " op=" << cand.opType << " reason=reparameterization"
                                  << std::endl;
                        continue;
                    }

                    selectedOpType = cand.opType;
                    selectedIndex = cand.index;
                    foundConstrainedSplit = true;
                    std::cout << "[ONESTRING-GRID] constrained_candidate_accept rank="
                              << rankI << " op=" << cand.opType
                              << " weighted_change=" << cand.weightedChange << std::endl;
                    break;
                }
                catch(const std::exception& e) {
                    std::cout << "[ONESTRING-GRID] constrained_candidate_reject rank="
                              << rankI << " op=" << cand.opType
                              << " reason=" << e.what() << std::endl;
                }
            }

            if(foundConstrainedSplit && selectedOpType == 0) {
                opType_queried = 0;
                path_queried = paths_bSplit[selectedIndex];
                newVertPos_queried = newVertPoses_bSplit[selectedIndex];
            }
            else if(foundConstrainedSplit && selectedOpType == 1) {
                opType_queried = 1;
                path_queried = paths_iSplit[selectedIndex];
                newVertPos_queried = newVertPoses_iSplit[selectedIndex];
            }
            else {
                std::cout << "[ONESTRING-GRID] no feasible boundary-constrained Grid split remains before requested distortion bound"
                          << std::endl;
                optimizer->updateEnergyData(true, false, false);
                return false;
            }
'''
        m = _replace_once(m, old, new, "main.cpp V5 global constrained shortlist")
        main_path.write_text(m, encoding="utf-8")
        changed = True

    return changed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    apply_boundary_reparameterization_patch(args.root)
