#!/usr/bin/env python3
"""Fast native Grid-OptCuts V4 candidate search.

The previous V4 variants either:
- ran a full harmonic solve for every Grid embedding (far too slow), or
- scored the unrelaxed snapped embedding directly (fast, but systematically
  rejected useful cuts because a newly cut topology needs relaxation).

This patch keeps OptCuts' own *local* split proposal/evaluation as the distortion
oracle, then restricts that proposed physical cut topology to fabrication-grid
embeddings.  In other words, OptCuts still decides which local topological split
is useful, but a split is returned only if an H/V fixed-h Grid realization of
that exact topology passes an actual trial cut, inversion checks, persistent
Grid locks and cohesive-seam alignment.

No full-mesh solve is performed during candidate enumeration.  After a Grid cut
is accepted, the ordinary OptCuts optimizer globally relaxes the new topology.
"""
from __future__ import annotations

from pathlib import Path

MARKER = "ONESTRING_GRID_NATIVE_V4_FAST_SEARCH"
PROPOSAL_MARKER = "ONESTRING_GRID_NATIVE_V4_ORIGINAL_LOCAL_PROPOSAL"
OLD_RELAX_MARKER = "ONESTRING_GRID_NATIVE_V4_RELAXED_TRIAL_SCORE"
LOCAL_SCORE_MARKER = "ONESTRING_GRID_NATIVE_V4_LOCAL_TRIAL_SCORE"

LOCAL_SCORE = r'''double oneStringScoreTrial(
    const OptCuts::TriMesh& mesh,
    const OptCuts::TriMesh& trial,
    double lambda_t,
    double seamIncrease,
    std::pair<double, double>& energyChanges)
{
    // ONESTRING_GRID_NATIVE_V4_LOCAL_TRIAL_SCORE
    // Feasibility audit only.  The Grid candidate's objective score is supplied
    // by OptCuts' original local split optimizer in oneStringGridComputeLocalLDec.
    if(!oneStringLockedPreserved(mesh, trial)) return -DBL_MAX;
    if(!oneStringAllLockedOnGrid(trial)) return -DBL_MAX;
    if(!oneStringAllCohesiveSeamSidesGridAligned(trial)) return -DBL_MAX;
    if(!trial.checkInversion(true)) return -DBL_MAX;

    const double before = oneStringTotalSD(mesh);
    const double after = oneStringTotalSD(trial);
    if(!std::isfinite(before) || !std::isfinite(after)) return -DBL_MAX;
    const double sdDec = before - after;
    energyChanges.first = -sdDec;
    energyChanges.second = seamIncrease;
    // Sign is intentionally NOT used to decide whether the topology is useful:
    // the trial has not yet undergone OptCuts relaxation.  Returning a finite
    // value simply means the exact Grid topology is immediately valid.
    return (1.0 - lambda_t) * sdDec - lambda_t * seamIncrease;
}'''

FAST_FUNCTION = r'''// ONESTRING_GRID_NATIVE_V4_ORIGINAL_LOCAL_PROPOSAL
// querySplit evaluates vertices in parallel, so the recursion-bypass flag must
// be thread-local.  It lets Grid-OptCuts ask the unmodified OptCuts
// computeLocalLDec() for its local distortion-driven topology proposal without
// recursively entering the Grid wrapper.
thread_local bool oneStringGridOriginalProposalPass = false;

bool oneStringGridOriginalProposalActive()
{
    return oneStringGridOriginalProposalPass;
}

double oneStringGetOriginalLocalProposal(
    const OptCuts::TriMesh& mesh,
    int vI,
    double lambda_t,
    std::vector<int>& path,
    Eigen::MatrixXd& newVertPos,
    std::pair<double, double>& energyChanges)
{
    const bool previous = oneStringGridOriginalProposalPass;
    oneStringGridOriginalProposalPass = true;
    try {
        const double score = mesh.computeLocalLDec(
            vI, lambda_t, path, newVertPos, energyChanges);
        oneStringGridOriginalProposalPass = previous;
        return score;
    }
    catch(...) {
        oneStringGridOriginalProposalPass = previous;
        throw;
    }
}

double oneStringGridComputeLocalLDec(
    const OptCuts::TriMesh& mesh,
    int vI,
    double lambda_t,
    std::vector<int>& path_max,
    Eigen::MatrixXd& newVertPos_max,
    std::pair<double, double>& energyChanges_max)
{
    path_max.clear();
    newVertPos_max.resize(0, 2);
    energyChanges_max = std::pair<double, double>(DBL_MAX, DBL_MAX);

    // ONESTRING_GRID_NATIVE_V4_FAST_SEARCH
    std::vector<int> proposalPath;
    Eigen::MatrixXd proposalPos;
    std::pair<double, double> proposalChanges(DBL_MAX, DBL_MAX);
    double proposalScore = -DBL_MAX;
    try {
        proposalScore = oneStringGetOriginalLocalProposal(
            mesh, vI, lambda_t, proposalPath, proposalPos, proposalChanges);
    }
    catch(...) {
        return -DBL_MAX;
    }
    if(!std::isfinite(proposalScore) || proposalScore <= 0.0 || proposalPath.empty()) {
        return -DBL_MAX;
    }

    const int targetCap = std::max(1, static_cast<int>(std::llround(
        oneStringGridEnvDouble("ONESTRING_OPTCUTS_GRID_TARGETS_PER_VERTEX", 4.0))));
    const int maxTrials = std::max(1, static_cast<int>(std::llround(
        oneStringGridEnvDouble("ONESTRING_OPTCUTS_GRID_MAX_EXACT_TRIALS", 4.0))));
    const double penaltyFraction = std::max(0.0,
        oneStringGridEnvDouble("ONESTRING_OPTCUTS_GRID_SNAP_PENALTY_FRACTION", 0.01));

    auto cappedTargets = [&](int vertexI) {
        std::vector<Eigen::RowVector2d> result = oneStringVertexGridTargets(mesh, vertexI);
        if(static_cast<int>(result.size()) > targetCap) result.resize(targetCap);
        return result;
    };
    auto proposalAdjustedScore = [&](double proxy) {
        // Relative penalty: Grid displacement breaks ties/prevents wild snapping
        // without destroying the scale of OptCuts' own energy decrease.
        return proposalScore - penaltyFraction * std::abs(proposalScore) * proxy;
    };

    double best = -DBL_MAX;

    if(mesh.isBoundaryVert(vI)) {
        if(proposalPath.size() != 2) return -DBL_MAX;
        int boundaryVI = -1, interiorVI = -1;
        if(mesh.isBoundaryVert(proposalPath[0]) && !mesh.isBoundaryVert(proposalPath[1])) {
            boundaryVI = proposalPath[0]; interiorVI = proposalPath[1];
        }
        else if(mesh.isBoundaryVert(proposalPath[1]) && !mesh.isBoundaryVert(proposalPath[0])) {
            boundaryVI = proposalPath[1]; interiorVI = proposalPath[0];
        }
        else {
            return -DBL_MAX;
        }
        if(boundaryVI != vI) return -DBL_MAX;

        struct Candidate {
            Eigen::RowVector2d A, B, C;
            double proxy;
        };
        std::vector<Candidate> candidates;
        const auto targetsA = cappedTargets(boundaryVI);
        const auto targetsC = cappedTargets(interiorVI);
        const auto targetsB = oneStringNearbyGridPoints(mesh.V.row(boundaryVI), targetCap);
        if(targetsA.empty() || targetsC.empty() || targetsB.empty()) return -DBL_MAX;

        for(const auto& A : targetsA) {
            for(const auto& C : targetsC) {
                if(!oneStringIsGridEdge(A, C)) continue;
                for(const auto& B : targetsB) {
                    if(oneStringSameGridPoint(A, B) || oneStringSameGridPoint(B, C)) continue;
                    if(!oneStringIsGridEdge(B, C)) continue;
                    double proxy = oneStringSnapSteps(mesh.V.row(interiorVI), C);
                    if(proposalPos.rows() >= 2) {
                        proxy += (A - proposalPos.row(0)).norm() / oneStringGridH();
                        proxy += (B - proposalPos.row(1)).norm() / oneStringGridH();
                    }
                    else {
                        proxy += oneStringSnapSteps(mesh.V.row(boundaryVI), A);
                        proxy += oneStringSnapSteps(mesh.V.row(boundaryVI), B);
                    }
                    candidates.push_back({A, B, C, proxy});
                }
            }
        }
        std::sort(candidates.begin(), candidates.end(), [](const Candidate& lhs, const Candidate& rhs) {
            return lhs.proxy < rhs.proxy;
        });

        const double seamInc = proposalChanges.second;
        const int trialCount = std::min<int>(maxTrials, candidates.size());
        for(int ci = 0; ci < trialCount; ++ci) {
            const Candidate& cand = candidates[ci];
            double feasibilityScore = -DBL_MAX;
            Eigen::MatrixXd encoded;
            std::pair<double, double> auditChanges;
            if(!oneStringTryBoundaryGridCut(
                    mesh, boundaryVI, interiorVI, cand.A, cand.B, cand.C,
                    lambda_t, seamInc, feasibilityScore, encoded, auditChanges)) continue;
            const double score = proposalAdjustedScore(cand.proxy);
            if(score > best) {
                best = score;
                path_max = {boundaryVI, interiorVI};
                newVertPos_max = encoded;
                energyChanges_max = proposalChanges;
            }
        }
        return best;
    }

    // Native V4 represents an interior split as one physical two-edge path and
    // two H/V UV boundary copies A-B-C and A-D-C.
    if(proposalPath.size() != 3 || proposalPath[1] != vI) return -DBL_MAX;
    const int a = proposalPath[0];
    const int c = proposalPath[2];
    if(mesh.isBoundaryVert(a) || mesh.isBoundaryVert(c)) return -DBL_MAX;
    if(mesh.oneStringGridLockedVert.find(vI) != mesh.oneStringGridLockedVert.end()) return -DBL_MAX;

    struct Candidate {
        Eigen::RowVector2d A, B, D, C;
        int swapSides;
        double proxy;
    };
    std::vector<Candidate> candidates;
    const auto targetsA = cappedTargets(a);
    const auto targetsC = cappedTargets(c);
    if(targetsA.empty() || targetsC.empty()) return -DBL_MAX;

    for(const auto& A : targetsA) {
        const OneStringGridIJ ia = oneStringNearestGridIJ(A);
        for(const auto& C : targetsC) {
            const OneStringGridIJ ic = oneStringNearestGridIJ(C);
            if(ia.i == ic.i || ia.j == ic.j) continue;
            const Eigen::RowVector2d B = oneStringFromGridIJ(ic.i, ia.j);
            const Eigen::RowVector2d D = oneStringFromGridIJ(ia.i, ic.j);
            if(oneStringSnapSteps(mesh.V.row(vI), B) > oneStringGridMaxSnapSteps() + 1.0e-9) continue;
            if(oneStringSnapSteps(mesh.V.row(vI), D) > oneStringGridMaxSnapSteps() + 1.0e-9) continue;
            if(!oneStringIsGridEdge(A, B) || !oneStringIsGridEdge(B, C) ||
               !oneStringIsGridEdge(A, D) || !oneStringIsGridEdge(D, C)) continue;

            double proxy0 = oneStringSnapSteps(mesh.V.row(a), A) +
                            oneStringSnapSteps(mesh.V.row(c), C);
            double proxy1 = proxy0;
            if(proposalPos.rows() >= 2) {
                proxy0 += (B - proposalPos.row(0)).norm() / oneStringGridH();
                proxy0 += (D - proposalPos.row(1)).norm() / oneStringGridH();
                proxy1 += (D - proposalPos.row(0)).norm() / oneStringGridH();
                proxy1 += (B - proposalPos.row(1)).norm() / oneStringGridH();
            }
            else {
                proxy0 += oneStringSnapSteps(mesh.V.row(vI), B) + oneStringSnapSteps(mesh.V.row(vI), D);
                proxy1 = proxy0;
            }
            candidates.push_back({A, B, D, C, 0, proxy0});
            candidates.push_back({A, B, D, C, 1, proxy1});
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& lhs, const Candidate& rhs) {
        return lhs.proxy < rhs.proxy;
    });

    const double seamInc = proposalChanges.second;
    const int trialCount = std::min<int>(maxTrials, candidates.size());
    for(int ci = 0; ci < trialCount; ++ci) {
        const Candidate& cand = candidates[ci];
        const Eigen::RowVector2d side0 = cand.swapSides ? cand.D : cand.B;
        const Eigen::RowVector2d side1 = cand.swapSides ? cand.B : cand.D;
        double feasibilityScore = -DBL_MAX;
        Eigen::MatrixXd encoded;
        std::pair<double, double> auditChanges;
        if(!oneStringTryInteriorGridCut(
                mesh, proposalPath, cand.A, side0, side1, cand.C,
                lambda_t, seamInc, feasibilityScore, encoded, auditChanges)) continue;
        const double score = proposalAdjustedScore(cand.proxy);
        if(score > best) {
            best = score;
            path_max = proposalPath;
            newVertPos_max = encoded;
            energyChanges_max = proposalChanges;
        }
    }
    return best;
}'''


def _replace_function_before(text: str, signature: str, next_signature: str, replacement: str) -> str:
    start = text.find(signature)
    if start < 0:
        raise RuntimeError(f"Grid-OptCuts helper was not found: {signature}")
    end = text.find(next_signature, start)
    if end < 0:
        raise RuntimeError(f"Grid-OptCuts following helper was not found: {next_signature}")
    return text[:start] + replacement + "\n\n" + text[end:]


def apply_native_grid_perf_patch(root: Path) -> bool:
    path = root / "src" / "TriMesh.cpp"
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    changed = False

    if LOCAL_SCORE_MARKER not in text:
        if OLD_RELAX_MARKER in text:
            text = _replace_function_before(
                text, "bool oneStringRelaxGridTrialUV(",
                "bool oneStringTryInteriorGridCut(", LOCAL_SCORE)
        else:
            text = _replace_function_before(
                text, "double oneStringScoreTrial(",
                "bool oneStringTryInteriorGridCut(", LOCAL_SCORE)
        changed = True

    # Install the proposal-aware Grid helper.  On a fresh checkout the old V4
    # helper is present; on an upgraded checkout replace any previous fast helper.
    start = text.find("double oneStringGridComputeLocalLDec(")
    if start < 0:
        raise RuntimeError("Grid-OptCuts V4 computeLocalLDec helper was not found")
    helper_start = start
    existing_proposal = text.rfind("// ONESTRING_GRID_NATIVE_V4_ORIGINAL_LOCAL_PROPOSAL", 0, start)
    if existing_proposal >= 0:
        helper_start = existing_proposal
    anon_end = text.find("\n\n} // anonymous namespace", start)
    if anon_end < 0:
        raise RuntimeError("Grid-OptCuts V4 helper namespace end was not found")
    function_end = text.rfind("\n}", start, anon_end)
    if function_end < 0:
        raise RuntimeError("Grid-OptCuts V4 computeLocalLDec function end was not found")
    function_end += 2
    current_block = text[helper_start:function_end]
    if PROPOSAL_MARKER not in current_block:
        text = text[:helper_start] + FAST_FUNCTION + text[function_end:]
        changed = True

    # The original OptCuts method is still present.  Add a thread-local bypass so
    # the helper above can ask it for the distortion-driven local proposal.
    old_dispatch = '''        if(oneStringGridEnabled() && path_max.empty()) {
            return oneStringGridComputeLocalLDec(*this, vI, lambda_t, path_max, newVertPos_max, energyChanges_max);
        }
'''
    new_dispatch = '''        if(oneStringGridEnabled() && path_max.empty() && !oneStringGridOriginalProposalActive()) {
            return oneStringGridComputeLocalLDec(*this, vI, lambda_t, path_max, newVertPos_max, energyChanges_max);
        }
'''
    if old_dispatch in text:
        text = text.replace(old_dispatch, new_dispatch, 1)
        changed = True
    elif new_dispatch not in text:
        raise RuntimeError("Grid-OptCuts computeLocalLDec dispatch anchor was not found")

    if changed:
        path.write_text(text, encoding="utf-8")
        print(f"Applied OptCuts-local-proposal Grid search: {path}")
    else:
        print("OptCuts-local-proposal Grid search already present.")
    return changed


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    apply_native_grid_perf_patch(args.root.resolve())
