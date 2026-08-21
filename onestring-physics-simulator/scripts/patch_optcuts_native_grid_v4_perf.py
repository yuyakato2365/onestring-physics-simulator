#!/usr/bin/env python3
"""Bound and accelerate OneString native Grid-OptCuts V4 candidate scoring.

Grid-OptCuts must search only fabrication-feasible cuts, but candidate scoring
cannot solve a full-mesh harmonic parameterization for every lattice proposal.
That made even a 42-vertex test mesh take minutes.

This patch therefore uses a strict two-stage policy:
1. enumerate/rank nearby Grid-feasible H/V candidates with cheap geometric
   proxies and keep only a bounded number per source vertex;
2. for those candidates, perform the ACTUAL topology cut, validate inversion,
   persistent Grid locks and cohesive-seam H/V alignment, then evaluate the
   immediate Symmetric Dirichlet objective.

No full-mesh solve is performed inside candidate enumeration.  Once a candidate
is accepted, the ordinary OptCuts optimizer performs the expensive global UV
relaxation exactly once for the accepted topology.  This keeps the search space
faithful to Grid-OptCuts while avoiding O(candidate_count) global factorizations.
"""
from __future__ import annotations

from pathlib import Path

MARKER = "ONESTRING_GRID_NATIVE_V4_FAST_SEARCH"
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
    // This function is intentionally cheap.  Candidate enumeration must never
    // run a global harmonic/OptCuts solve.  The accepted topology is globally
    // relaxed by the ordinary OptCuts optimizer after split selection.
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
    return (1.0 - lambda_t) * sdDec - lambda_t * seamIncrease;
}'''

FAST_FUNCTION = r'''double oneStringGridComputeLocalLDec(
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
    double best = -DBL_MAX;

    // ONESTRING_GRID_NATIVE_V4_FAST_SEARCH
    const int targetCap = std::max(1, static_cast<int>(std::llround(
        oneStringGridEnvDouble("ONESTRING_OPTCUTS_GRID_TARGETS_PER_VERTEX", 4.0))));
    const int maxTrials = std::max(1, static_cast<int>(std::llround(
        oneStringGridEnvDouble("ONESTRING_OPTCUTS_GRID_MAX_EXACT_TRIALS", 4.0))));

    auto cappedTargets = [&](int vertexI) {
        std::vector<Eigen::RowVector2d> result = oneStringVertexGridTargets(mesh, vertexI);
        if(static_cast<int>(result.size()) > targetCap) result.resize(targetCap);
        return result;
    };

    if(mesh.isBoundaryVert(vI)) {
        struct Candidate {
            int nbVI;
            Eigen::RowVector2d A;
            Eigen::RowVector2d B;
            Eigen::RowVector2d C;
            double seamInc;
            double proxy;
        };
        std::vector<Candidate> candidates;
        const auto targetsA = cappedTargets(vI);
        if(targetsA.empty()) return -DBL_MAX;

        for(const int nbVI : mesh.vNeighbor[vI]) {
            if(mesh.isBoundaryVert(nbVI)) continue;
            if(mesh.edge2Tri.find(std::pair<int, int>(vI, nbVI)) == mesh.edge2Tri.end() ||
               mesh.edge2Tri.find(std::pair<int, int>(nbVI, vI)) == mesh.edge2Tri.end()) continue;
            const auto targetsC = cappedTargets(nbVI);
            std::vector<Eigen::RowVector2d> targetsB = oneStringNearbyGridPoints(mesh.V.row(vI), targetCap);
            const double seamInc = (mesh.V_rest.row(vI) - mesh.V_rest.row(nbVI)).norm() /
                mesh.virtualRadius * (mesh.vertWeight[vI] + mesh.vertWeight[nbVI]) / 2.0;

            for(const auto& A : targetsA) {
                for(const auto& C : targetsC) {
                    if(!oneStringIsGridEdge(A, C)) continue;
                    for(const auto& B : targetsB) {
                        if(oneStringSameGridPoint(A, B) || oneStringSameGridPoint(B, C)) continue;
                        if(!oneStringIsGridEdge(B, C)) continue;
                        const double proxy =
                            oneStringSnapSteps(mesh.V.row(vI), A) +
                            oneStringSnapSteps(mesh.V.row(vI), B) +
                            oneStringSnapSteps(mesh.V.row(nbVI), C) +
                            0.05 * seamInc;
                        candidates.push_back({nbVI, A, B, C, seamInc, proxy});
                    }
                }
            }
        }

        std::sort(candidates.begin(), candidates.end(), [](const Candidate& lhs, const Candidate& rhs) {
            return lhs.proxy < rhs.proxy;
        });
        const int trialCount = std::min<int>(maxTrials, candidates.size());
        for(int ci = 0; ci < trialCount; ++ci) {
            const Candidate& cand = candidates[ci];
            double score = -DBL_MAX;
            Eigen::MatrixXd encoded;
            std::pair<double, double> changes;
            if(!oneStringTryBoundaryGridCut(mesh, vI, cand.nbVI, cand.A, cand.B, cand.C,
                                           lambda_t, cand.seamInc, score, encoded, changes)) continue;
            if(score > best) {
                best = score;
                path_max = {vI, cand.nbVI};
                newVertPos_max = encoded;
                energyChanges_max = changes;
            }
        }
        return best;
    }

    for(const int nbVI : mesh.vNeighbor[vI]) {
        if(mesh.isBoundaryVert(nbVI)) return -DBL_MAX;
    }
    if(mesh.oneStringGridLockedVert.find(vI) != mesh.oneStringGridLockedVert.end()) return -DBL_MAX;

    struct Candidate {
        int a;
        int c;
        Eigen::RowVector2d A;
        Eigen::RowVector2d B;
        Eigen::RowVector2d D;
        Eigen::RowVector2d C;
        int swapSides;
        double seamInc;
        double proxy;
    };
    std::vector<Candidate> candidates;

    for(const auto& pair : mesh.validSplit[vI]) {
        const int a = pair.first;
        const int c = pair.second;
        if(a >= c) continue;
        const auto targetsA = cappedTargets(a);
        const auto targetsC = cappedTargets(c);
        if(targetsA.empty() || targetsC.empty()) continue;
        const double seamInc = (
            (mesh.V_rest.row(a) - mesh.V_rest.row(vI)).norm() * (mesh.vertWeight[a] + mesh.vertWeight[vI]) +
            (mesh.V_rest.row(vI) - mesh.V_rest.row(c)).norm() * (mesh.vertWeight[vI] + mesh.vertWeight[c])
        ) / mesh.virtualRadius / 2.0;

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

                const double proxy =
                    oneStringSnapSteps(mesh.V.row(a), A) +
                    oneStringSnapSteps(mesh.V.row(c), C) +
                    0.5 * (oneStringSnapSteps(mesh.V.row(vI), B) +
                           oneStringSnapSteps(mesh.V.row(vI), D)) +
                    0.05 * seamInc;
                candidates.push_back({a, c, A, B, D, C, 0, seamInc, proxy});
                candidates.push_back({a, c, A, B, D, C, 1, seamInc, proxy + 1.0e-9});
            }
        }
    }

    std::sort(candidates.begin(), candidates.end(), [](const Candidate& lhs, const Candidate& rhs) {
        return lhs.proxy < rhs.proxy;
    });
    const int trialCount = std::min<int>(maxTrials, candidates.size());
    for(int ci = 0; ci < trialCount; ++ci) {
        const Candidate& cand = candidates[ci];
        const Eigen::RowVector2d side0 = cand.swapSides ? cand.D : cand.B;
        const Eigen::RowVector2d side1 = cand.swapSides ? cand.B : cand.D;
        const std::vector<int> path = {cand.a, vI, cand.c};
        double score = -DBL_MAX;
        Eigen::MatrixXd encoded;
        std::pair<double, double> changes;
        if(!oneStringTryInteriorGridCut(mesh, path, cand.A, side0, side1, cand.C,
                                       lambda_t, cand.seamInc, score, encoded, changes)) continue;
        if(score > best) {
            best = score;
            path_max = path;
            newVertPos_max = encoded;
            energyChanges_max = changes;
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
        # Upgrade an already-patched V4 checkout as well as a fresh V4 source.
        if OLD_RELAX_MARKER in text:
            text = _replace_function_before(
                text,
                "bool oneStringRelaxGridTrialUV(",
                "bool oneStringTryInteriorGridCut(",
                LOCAL_SCORE,
            )
        else:
            text = _replace_function_before(
                text,
                "double oneStringScoreTrial(",
                "bool oneStringTryInteriorGridCut(",
                LOCAL_SCORE,
            )
        changed = True

    if MARKER not in text:
        start = text.find("double oneStringGridComputeLocalLDec(")
        if start < 0:
            raise RuntimeError("Grid-OptCuts V4 computeLocalLDec helper was not found")
        anon_end = text.find("\n\n} // anonymous namespace", start)
        if anon_end < 0:
            raise RuntimeError("Grid-OptCuts V4 helper namespace end was not found")
        function_end = text.rfind("\n}", start, anon_end)
        if function_end < 0:
            raise RuntimeError("Grid-OptCuts V4 computeLocalLDec function end was not found")
        function_end += 2
        text = text[:start] + FAST_FUNCTION + text[function_end:]
        changed = True

    if changed:
        path.write_text(text, encoding="utf-8")
        print(f"Applied bounded local-score Grid-OptCuts search: {path}")
    else:
        print("Grid-OptCuts V4 bounded local-score search already present.")
    return changed


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    apply_native_grid_perf_patch(args.root.resolve())
