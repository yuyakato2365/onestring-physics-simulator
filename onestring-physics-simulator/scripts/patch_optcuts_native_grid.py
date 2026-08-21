#!/usr/bin/env python3
"""Patch official OptCuts into OneString native Grid-OptCuts.

The modification is inside OptCuts' topology search, never a completed-seam
post-process. In native grid mode every split candidate must first be realizable
on one fixed orthogonal h-lattice. The candidate is then *actually cut at those
lattice coordinates* on a trial mesh and scored by the resulting Symmetric
Dirichlet decrease plus the ordinary OptCuts seam-length term. Accepted seam
vertices become permanent lattice locks.

Native V3 still uses OptCuts' existing source-mesh edges as topological cut
candidates. It does not yet insert arbitrary new 3D surface vertices at grid-line
/ source-triangle intersections. This limits candidate resolution, not the grid
constraint of accepted cuts.
"""
from __future__ import annotations

from pathlib import Path

MARKER = "ONESTRING_GRID_NATIVE_V3"
OLD_MARKERS = ("ONESTRING_GRID_NATIVE_V1", "ONESTRING_GRID_NATIVE_V2")

HELPERS = r'''
// ONESTRING_GRID_NATIVE_V3
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <limits>
#include <stdexcept>
#include <string>
#include <algorithm>

namespace {

bool oneStringGridEnabled()
{
    const char* raw = std::getenv("ONESTRING_OPTCUTS_GRID_NATIVE");
    if(!raw) return false;
    const std::string value(raw);
    return (value == "1") || (value == "true") || (value == "TRUE") || (value == "on");
}

double oneStringGridEnvDouble(const char* key, double fallback)
{
    const char* raw = std::getenv(key);
    if(!raw) return fallback;
    try { return std::stod(std::string(raw)); }
    catch(...) { return fallback; }
}

double oneStringGridH()
{
    return std::max(1.0e-10, oneStringGridEnvDouble("ONESTRING_OPTCUTS_GRID_H", 0.1));
}

double oneStringGridAngle()
{
    return oneStringGridEnvDouble("ONESTRING_OPTCUTS_GRID_ANGLE_RAD", 0.0);
}

double oneStringGridPhaseU()
{
    return oneStringGridEnvDouble("ONESTRING_OPTCUTS_GRID_PHASE_U", 0.0);
}

double oneStringGridPhaseV()
{
    return oneStringGridEnvDouble("ONESTRING_OPTCUTS_GRID_PHASE_V", 0.0);
}

double oneStringGridMaxSnapSteps()
{
    return std::max(0.5, oneStringGridEnvDouble("ONESTRING_OPTCUTS_GRID_MAX_SNAP_STEPS", 2.0));
}

Eigen::Vector2d oneStringToGridUnits(const Eigen::RowVector2d& p)
{
    const double a = oneStringGridAngle();
    const Eigen::Vector2d u(std::cos(a), std::sin(a));
    const Eigen::Vector2d v(-std::sin(a), std::cos(a));
    const Eigen::Vector2d q = p.transpose();
    const double h = oneStringGridH();
    return Eigen::Vector2d(
        (q.dot(u) - oneStringGridPhaseU()) / h,
        (q.dot(v) - oneStringGridPhaseV()) / h
    );
}

Eigen::RowVector2d oneStringFromGridIndex(long long i, long long j)
{
    const double a = oneStringGridAngle();
    const Eigen::Vector2d u(std::cos(a), std::sin(a));
    const Eigen::Vector2d v(-std::sin(a), std::cos(a));
    const double h = oneStringGridH();
    const Eigen::Vector2d q =
        (oneStringGridPhaseU() + static_cast<double>(i) * h) * u +
        (oneStringGridPhaseV() + static_cast<double>(j) * h) * v;
    return q.transpose();
}

long long oneStringRoundIndex(double x)
{
    return static_cast<long long>(std::llround(x));
}

double oneStringEmbeddingMaxSnapSteps(const Eigen::MatrixXd& original, const Eigen::MatrixXd& target)
{
    const double h = oneStringGridH();
    double result = 0.0;
    for(int i = 0; i < original.rows(); ++i) {
        result = std::max(result, (original.row(i) - target.row(i)).norm() / h);
    }
    return result;
}

void oneStringPushUnique(std::vector<Eigen::MatrixXd>& out, const Eigen::MatrixXd& candidate)
{
    for(const auto& existing : out) {
        if(existing.rows() == candidate.rows() && (existing - candidate).norm() <= 1.0e-12) return;
    }
    out.emplace_back(candidate);
}

std::vector<Eigen::MatrixXd> oneStringGridEdgeEmbeddings(
    const Eigen::RowVector2d& p0, const Eigen::RowVector2d& p1)
{
    const Eigen::Vector2d g0 = oneStringToGridUnits(p0);
    const Eigen::Vector2d g1 = oneStringToGridUnits(p1);
    const long long i0 = oneStringRoundIndex(g0[0]);
    const long long i1 = oneStringRoundIndex(g1[0]);
    const long long j0 = oneStringRoundIndex(g0[1]);
    const long long j1 = oneStringRoundIndex(g1[1]);
    std::vector<Eigen::MatrixXd> out;
    Eigen::MatrixXd p(2, 2); p << p0, p1;

    for(long long dj = -1; dj <= 1; ++dj) {
        const long long j = oneStringRoundIndex(0.5 * (g0[1] + g1[1])) + dj;
        long long a = i0, b = i1;
        if(a == b) b += (g1[0] >= g0[0]) ? 1 : -1;
        Eigen::MatrixXd q(2, 2);
        q.row(0) = oneStringFromGridIndex(a, j);
        q.row(1) = oneStringFromGridIndex(b, j);
        if(oneStringEmbeddingMaxSnapSteps(p, q) <= oneStringGridMaxSnapSteps()) oneStringPushUnique(out, q);
    }

    for(long long di = -1; di <= 1; ++di) {
        const long long i = oneStringRoundIndex(0.5 * (g0[0] + g1[0])) + di;
        long long a = j0, b = j1;
        if(a == b) b += (g1[1] >= g0[1]) ? 1 : -1;
        Eigen::MatrixXd q(2, 2);
        q.row(0) = oneStringFromGridIndex(i, a);
        q.row(1) = oneStringFromGridIndex(i, b);
        if(oneStringEmbeddingMaxSnapSteps(p, q) <= oneStringGridMaxSnapSteps()) oneStringPushUnique(out, q);
    }
    return out;
}

std::vector<Eigen::MatrixXd> oneStringGridCornerEmbeddings(
    const Eigen::RowVector2d& p0,
    const Eigen::RowVector2d& p1,
    const Eigen::RowVector2d& p2)
{
    const Eigen::Vector2d g0 = oneStringToGridUnits(p0);
    const Eigen::Vector2d g1 = oneStringToGridUnits(p1);
    const Eigen::Vector2d g2 = oneStringToGridUnits(p2);
    long long i0 = oneStringRoundIndex(g0[0]), j0 = oneStringRoundIndex(g0[1]);
    long long i1 = oneStringRoundIndex(g1[0]), j1 = oneStringRoundIndex(g1[1]);
    long long i2 = oneStringRoundIndex(g2[0]), j2 = oneStringRoundIndex(g2[1]);
    std::vector<Eigen::MatrixXd> out;
    Eigen::MatrixXd p(3, 2); p << p0, p1, p2;

    auto accept = [&](long long a0, long long b0, long long a1, long long b1, long long a2, long long b2) {
        if((a0 == a1 && b0 == b1) || (a1 == a2 && b1 == b2)) return;
        Eigen::MatrixXd q(3, 2);
        q.row(0) = oneStringFromGridIndex(a0, b0);
        q.row(1) = oneStringFromGridIndex(a1, b1);
        q.row(2) = oneStringFromGridIndex(a2, b2);
        if(oneStringEmbeddingMaxSnapSteps(p, q) <= oneStringGridMaxSnapSteps()) oneStringPushUnique(out, q);
    };

    for(long long dj = -1; dj <= 1; ++dj) {
        const long long j = oneStringRoundIndex((g0[1] + g1[1] + g2[1]) / 3.0) + dj;
        long long a0 = i0, a1 = i1, a2 = i2;
        if(a1 == a0) a1 += (g1[0] >= g0[0]) ? 1 : -1;
        if(a2 == a1) a2 += (g2[0] >= g1[0]) ? 1 : -1;
        if((a1 - a0) * (a2 - a1) > 0) accept(a0, j, a1, j, a2, j);
    }

    for(long long di = -1; di <= 1; ++di) {
        const long long i = oneStringRoundIndex((g0[0] + g1[0] + g2[0]) / 3.0) + di;
        long long b0 = j0, b1 = j1, b2 = j2;
        if(b1 == b0) b1 += (g1[1] >= g0[1]) ? 1 : -1;
        if(b2 == b1) b2 += (g2[1] >= g1[1]) ? 1 : -1;
        if((b1 - b0) * (b2 - b1) > 0) accept(i, b0, i, b1, i, b2);
    }

    {
        long long ai0 = i0, aj0 = j0, ai2 = i2, aj2 = j2;
        if(ai2 == ai0) ai2 += (g2[0] >= g0[0]) ? 1 : -1;
        if(aj2 == aj0) aj2 += (g2[1] >= g0[1]) ? 1 : -1;
        accept(ai0, aj0, ai2, aj0, ai2, aj2); // H -> V
        accept(ai0, aj0, ai0, aj2, ai2, aj2); // V -> H
    }
    return out;
}

bool oneStringGridPreservesLocked(
    const OptCuts::TriMesh& mesh,
    const std::vector<int>& path,
    const Eigen::MatrixXd& target)
{
    if(static_cast<int>(path.size()) != target.rows()) return false;
    const double tol = std::max(1.0e-10, 1.0e-7 * oneStringGridH());
    for(int i = 0; i < static_cast<int>(path.size()); ++i) {
        if(mesh.oneStringGridLockedVert.find(path[i]) == mesh.oneStringGridLockedVert.end()) continue;
        if((mesh.V.row(path[i]) - target.row(i)).norm() > tol) return false;
    }
    return true;
}

Eigen::MatrixXd oneStringTrialInteriorEncoding(const Eigen::MatrixXd& gridPath)
{
    Eigen::MatrixXd encoded(5, 2);
    encoded.row(0) = gridPath.row(1);
    encoded.row(1) = gridPath.row(1);
    encoded.bottomRows(3) = gridPath;
    return encoded;
}

Eigen::MatrixXd oneStringTrialBoundaryEncoding(
    const OptCuts::TriMesh& mesh,
    const std::vector<int>& path,
    const Eigen::MatrixXd& gridPath)
{
    const bool cutThrough = mesh.isBoundaryVert(path[0]) && mesh.isBoundaryVert(path[1]);
    Eigen::MatrixXd encoded(cutThrough ? 6 : 4, 2);
    encoded.setZero();
    encoded.row(0) = gridPath.row(0);
    encoded.row(1) = gridPath.row(0);
    if(cutThrough) {
        encoded.row(2) = gridPath.row(1);
        encoded.row(3) = gridPath.row(1);
    }
    encoded.bottomRows(2) = gridPath;
    return encoded;
}

double oneStringGridTotalSD(const OptCuts::TriMesh& mesh)
{
    OptCuts::SymDirichletEnergy sd;
    double total = 0.0;
    for(int triI = 0; triI < mesh.F.rows(); ++triI) {
        double e = 0.0;
        sd.getEnergyValByElemID(mesh, triI, e);
        if(!std::isfinite(e)) return std::numeric_limits<double>::infinity();
        total += e;
    }
    return total;
}

// Exact hard-Grid candidate score used by native V3.  This performs the same
// topological split that would actually be accepted, at the exact candidate
// lattice coordinates, then measures the real SD change. No free-OptCuts UV
// result is used as a proxy for the constrained candidate.
double oneStringGridActualCutSDDec(
    const OptCuts::TriMesh& mesh,
    const std::vector<int>& path,
    const Eigen::MatrixXd& gridPath,
    bool boundarySplit)
{
    if(!oneStringGridPreservesLocked(mesh, path, gridPath)) return -__DBL_MAX__;
    try {
        const double before = oneStringGridTotalSD(mesh);
        if(!std::isfinite(before)) return -__DBL_MAX__;
        OptCuts::TriMesh trial = mesh;
        if(boundarySplit) {
            trial.splitEdgeOnBoundary(
                std::pair<int, int>(path[0], path[1]),
                oneStringTrialBoundaryEncoding(mesh, path, gridPath), true, true);
            trial.updateFeatures();
        }
        else {
            trial.cutPath(path, true, 1, oneStringTrialInteriorEncoding(gridPath), false);
        }
        if(!trial.checkInversion(true)) return -__DBL_MAX__;
        const double after = oneStringGridTotalSD(trial);
        if(!std::isfinite(after)) return -__DBL_MAX__;
        return before - after;
    }
    catch(...) {
        return -__DBL_MAX__;
    }
}

bool oneStringGridActualCutFeasible(
    const OptCuts::TriMesh& mesh,
    const std::vector<int>& path,
    const Eigen::MatrixXd& gridPath,
    bool boundarySplit)
{
    return oneStringGridActualCutSDDec(mesh, path, gridPath, boundarySplit) > -__DBL_MAX__ / 2.0;
}

} // anonymous namespace
'''


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one anchor, found {count}")
    return text.replace(old, new, 1)


def _restore_old_native_if_needed(root: Path) -> None:
    cpp = root / "src" / "TriMesh.cpp"
    if not cpp.is_file():
        return
    text = cpp.read_text(encoding="utf-8")
    if MARKER in text or not any(old in text for old in OLD_MARKERS):
        return
    backup = cpp.with_suffix(cpp.suffix + ".onestring-grid-native-backup")
    if not backup.is_file():
        raise RuntimeError(
            "Local OptCuts contains an older OneString native Grid patch but its backup is missing. "
            "Restore upstream TriMesh.cpp and rerun setup_optcuts.py."
        )
    cpp.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
    print("Restored upstream TriMesh.cpp from old native Grid backup before applying V3.")


def patch_tri_mesh_header(root: Path) -> bool:
    path = root / "src" / "TriMesh.hpp"
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    if "oneStringGridLockedVert" in text:
        return False
    text = replace_once(
        text,
        "        std::set<int> fixedVert; // for linear solve\n",
        "        std::set<int> fixedVert; // for linear solve\n"
        "        std::set<int> oneStringGridLockedVert; // OneString native Grid seam/junction locks\n",
        "add native Grid locked-vertex state",
    )
    backup = path.with_suffix(path.suffix + ".onestring-grid-native-backup")
    if not backup.exists():
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    path.write_text(text, encoding="utf-8")
    print(f"Applied OneString native Grid state patch: {path}")
    return True


def patch_tri_mesh(root: Path) -> bool:
    path = root / "src" / "TriMesh.cpp"
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        print("OneString native Grid-OptCuts V3 patch already present.")
        return False

    text = replace_once(
        text,
        "extern int inSplitTotalAmt;\n\nnamespace OptCuts {",
        "extern int inSplitTotalAmt;\n" + HELPERS + "\nnamespace OptCuts {",
        "insert helper block",
    )

    text = replace_once(
        text,
        "        if(resetFixedV) {\n            fixedVert.clear();\n            fixedVert.insert(0);\n        }\n",
        "        if(resetFixedV) {\n            fixedVert.clear();\n            fixedVert.insert(0);\n            if(oneStringGridEnabled()) {\n"
        "                fixedVert.insert(oneStringGridLockedVert.begin(), oneStringGridLockedVert.end());\n"
        "            }\n        }\n",
        "preserve native Grid locks when rebuilding fixed set",
    )

    text = replace_once(
        text,
        "        else {\n            double EwDec_max_split, EwDec_max_merge;\n",
        "        else if(oneStringGridEnabled()) {\n"
        "            querySplit(lambda_t, propagate, splitInterior,\n"
        "                       EwDec_max, path_max, newVertPos_max, energyChanes_split);\n"
        "            isMerge = false;\n"
        "        }\n"
        "        else {\n            double EwDec_max_split, EwDec_max_merge;\n",
        "disable merge in native grid mode",
    )

    old_one_point = '''    void TriMesh::onePointCut(int vI)\n    {\n        assert((vI >= 0) && (vI < V_rest.rows()));\n        std::vector<int> path(vNeighbor[vI].begin(), vNeighbor[vI].end());\n        assert(path.size() >= 3);\n        path[1] = vI;\n        path.resize(3);\n        \n        bool makeCoh = true;\n        if(!makeCoh) {\n            for(int pI = 0; pI + 1 < path.size(); pI++) {\n                initSeamLen += (V_rest.row(path[pI]) - V_rest.row(path[pI + 1])).norm();\n            }\n        }\n        \n        cutPath(path, makeCoh);\n        \n        if(makeCoh) {\n            initSeams = cohE;\n        }\n    }\n'''
    new_one_point = '''    void TriMesh::onePointCut(int vI)\n    {\n        assert((vI >= 0) && (vI < V_rest.rows()));\n        bool makeCoh = true;\n        if(oneStringGridEnabled()) {\n            double bestSDDec = -__DBL_MAX__;\n            std::vector<int> bestPath;\n            Eigen::MatrixXd bestGrid;\n            std::vector<int> neighbors(vNeighbor[vI].begin(), vNeighbor[vI].end());\n            for(int a = 0; a < static_cast<int>(neighbors.size()); ++a) {\n                for(int b = a + 1; b < static_cast<int>(neighbors.size()); ++b) {\n                    std::vector<int> path = {neighbors[a], vI, neighbors[b]};\n                    const auto embeddings = oneStringGridCornerEmbeddings(V.row(path[0]), V.row(path[1]), V.row(path[2]));\n                    for(const auto& gridPath : embeddings) {\n                        const double sdDec = oneStringGridActualCutSDDec(*this, path, gridPath, false);\n                        if(sdDec > bestSDDec) {\n                            bestSDDec = sdDec;\n                            bestPath = path;\n                            bestGrid = gridPath;\n                        }\n                    }\n                }\n            }\n            if(bestPath.empty() || bestSDDec <= -__DBL_MAX__ / 2.0) {\n                throw std::runtime_error("ONESTRING_GRID_INITIAL_CUT_INFEASIBLE");\n            }\n            cutPath(bestPath, makeCoh, 1, oneStringTrialInteriorEncoding(bestGrid));\n            std::cout << "[ONESTRING-GRID] initial_cut constrained_SD_dec=" << bestSDDec << std::endl;\n            if(makeCoh) initSeams = cohE;\n            return;\n        }\n\n        std::vector<int> path(vNeighbor[vI].begin(), vNeighbor[vI].end());\n        assert(path.size() >= 3);\n        path[1] = vI;\n        path.resize(3);\n        if(!makeCoh) {\n            for(int pI = 0; pI + 1 < path.size(); pI++) {\n                initSeamLen += (V_rest.row(path[pI]) - V_rest.row(path[pI + 1])).norm();\n            }\n        }\n        cutPath(path, makeCoh);\n        if(makeCoh) initSeams = cohE;\n    }\n'''
    text = replace_once(text, old_one_point, new_one_point, "grid initial one-point cut")

    old_boundary = '''                    Eigen::MatrixXd newVertPosI;\n                    const double SDDec = queryLocalEdDec_bSplit(edge, newVertPosI);\n                    \n                    const double seInc = (V_rest.row(vI) - V_rest.row(nbVI)).norm() /\n                        virtualRadius * (vertWeight[vI] + vertWeight[nbVI]) / 2.0;\n                    const double curEwDec = (1.0 - lambda_t) * SDDec - lambda_t * seInc;\n                    if(curEwDec > maxEwDec) {\n                        maxEwDec = curEwDec;\n                        path_max[0] = vI;\n                        path_max[1] = nbVI;\n                        newVertPos_max = newVertPosI;\n                        energyChanges_max.first = -SDDec;\n                        energyChanges_max.second = seInc;\n                    }\n'''
    new_boundary = '''                    Eigen::MatrixXd newVertPosI;\n                    const double seInc = (V_rest.row(vI) - V_rest.row(nbVI)).norm() /\n                        virtualRadius * (vertWeight[vI] + vertWeight[nbVI]) / 2.0;\n\n                    if(oneStringGridEnabled()) {\n                        std::vector<int> gridPathIDs = {vI, nbVI};\n                        const auto embeddings = oneStringGridEdgeEmbeddings(V.row(vI), V.row(nbVI));\n                        for(const auto& gridPath : embeddings) {\n                            const double constrainedSDDec = oneStringGridActualCutSDDec(*this, gridPathIDs, gridPath, true);\n                            if(constrainedSDDec <= -__DBL_MAX__ / 2.0) continue;\n                            const double curEwDec = (1.0 - lambda_t) * constrainedSDDec - lambda_t * seInc;\n                            if(curEwDec > maxEwDec) {\n                                maxEwDec = curEwDec;\n                                path_max[0] = vI;\n                                path_max[1] = nbVI;\n                                newVertPos_max = oneStringTrialBoundaryEncoding(*this, gridPathIDs, gridPath);\n                                energyChanges_max.first = -constrainedSDDec;\n                                energyChanges_max.second = seInc;\n                            }\n                        }\n                    }\n                    else {\n                        const double SDDec = queryLocalEdDec_bSplit(edge, newVertPosI);\n                        const double curEwDec = (1.0 - lambda_t) * SDDec - lambda_t * seInc;\n                        if(curEwDec > maxEwDec) {\n                            maxEwDec = curEwDec;\n                            path_max[0] = vI;\n                            path_max[1] = nbVI;\n                            newVertPos_max = newVertPosI;\n                            energyChanges_max.first = -SDDec;\n                            energyChanges_max.second = seInc;\n                        }\n                    }\n'''
    text = replace_once(text, old_boundary, new_boundary, "grid boundary split scoring")

    old_interior = '''                    SDDec += computeLocalEdDec_inSplit(umbrella, freeVert, path, newVertPos);\n                    //TODO: share local mesh before split, also for boundary splits\n                    \n                    const double seInc = ((V_rest.row(path[0]) - V_rest.row(path[1])).norm() * (vertWeight[path[0]] + vertWeight[path[1]]) +\n                                          (V_rest.row(path[1]) - V_rest.row(path[2])).norm() * (vertWeight[path[1]] + vertWeight[path[2]])) / virtualRadius / 2.0;\n                    const double EwDec = (1.0 - lambda_t) * SDDec - lambda_t * seInc;\n                    if(EwDec > EwDec_max) {\n                        EwDec_max = EwDec;\n                        newVertPos_max = newVertPos;\n                        path_max = path;\n                        energyChanges_max.first = -SDDec;\n                        energyChanges_max.second = seInc;\n                    }\n'''
    new_interior = '''                    const double seInc = ((V_rest.row(path[0]) - V_rest.row(path[1])).norm() * (vertWeight[path[0]] + vertWeight[path[1]]) +\n                                          (V_rest.row(path[1]) - V_rest.row(path[2])).norm() * (vertWeight[path[1]] + vertWeight[path[2]])) / virtualRadius / 2.0;\n\n                    if(oneStringGridEnabled()) {\n                        const auto embeddings = oneStringGridCornerEmbeddings(V.row(path[0]), V.row(path[1]), V.row(path[2]));\n                        for(const auto& gridPath : embeddings) {\n                            const double constrainedSDDec = oneStringGridActualCutSDDec(*this, path, gridPath, false);\n                            if(constrainedSDDec <= -__DBL_MAX__ / 2.0) continue;\n                            const double EwDec = (1.0 - lambda_t) * constrainedSDDec - lambda_t * seInc;\n                            if(EwDec > EwDec_max) {\n                                EwDec_max = EwDec;\n                                newVertPos_max = oneStringTrialInteriorEncoding(gridPath);\n                                path_max = path;\n                                energyChanges_max.first = -constrainedSDDec;\n                                energyChanges_max.second = seInc;\n                            }\n                        }\n                    }\n                    else {\n                        SDDec += computeLocalEdDec_inSplit(umbrella, freeVert, path, newVertPos);\n                        const double EwDec = (1.0 - lambda_t) * SDDec - lambda_t * seInc;\n                        if(EwDec > EwDec_max) {\n                            EwDec_max = EwDec;\n                            newVertPos_max = newVertPos;\n                            path_max = path;\n                            energyChanges_max.first = -SDDec;\n                            energyChanges_max.second = seInc;\n                        }\n                    }\n'''
    text = replace_once(text, old_interior, new_interior, "grid interior split scoring")

    text = replace_once(
        text,
        '''        if(changePos) {\n            assert((changePos == 1) && "right now only support change 1"); //!!! still only allow 1?\n            assert(newVertPos.cols() == 2);\n            assert(changePos * 2 == newVertPos.rows());\n        }\n''',
        '''        const bool oneStringGridEncoded = oneStringGridEnabled() && changePos && newVertPos.cols() == 2 && newVertPos.rows() >= 5;\n        if(changePos) {\n            assert((changePos == 1) && "right now only support change 1");\n            assert(newVertPos.cols() == 2);\n            assert(oneStringGridEncoded || (changePos * 2 == newVertPos.rows()));\n        }\n''',
        "allow encoded interior grid target",
    )

    text = replace_once(
        text,
        '''            if(changePos) {\n                V.row(nV) = newVertPos.block(0, 0, 1, 2);\n                V.row(path[1]) = newVertPos.block(1, 0, 1, 2);\n            }\n            else {\n                V.row(nV) = V.row(path[1]);\n            }\n''',
        '''            if(changePos) {\n                V.row(nV) = newVertPos.block(0, 0, 1, 2);\n                V.row(path[1]) = newVertPos.block(1, 0, 1, 2);\n                if(oneStringGridEncoded) {\n                    const Eigen::MatrixXd gridPath = newVertPos.bottomRows(3);\n                    V.row(path[0]) = gridPath.row(0);\n                    V.row(path[1]) = gridPath.row(1);\n                    V.row(nV) = gridPath.row(1);\n                    V.row(path[2]) = gridPath.row(2);\n                    for(const int lockVI : {path[0], path[1], path[2], nV}) {\n                        fixedVert.insert(lockVI);\n                        oneStringGridLockedVert.insert(lockVI);\n                    }\n                }\n            }\n            else {\n                V.row(nV) = V.row(path[1]);\n            }\n''',
        "apply and lock interior grid target",
    )

    text = replace_once(
        text,
        '''        return cuts_made;\n    }\n    \n    void TriMesh::computeSeamScore''',
        '''        if(oneStringGridEncoded && !checkInversion(true)) {\n            throw std::runtime_error("ONESTRING_GRID_APPLIED_INTERIOR_CUT_INVERTED");\n        }\n        return cuts_made;\n    }\n    \n    void TriMesh::computeSeamScore''',
        "post-cut interior inversion guard",
    )

    text = replace_once(
        text,
        '''                if(changeVertPos) {\n                    assert(newVertPos.rows() == 4);\n                }\n                duplicateBoth = true;\n''',
        '''                if(changeVertPos) {\n                    assert(newVertPos.rows() == 4 || (oneStringGridEnabled() && newVertPos.rows() == 6));\n                }\n                duplicateBoth = true;\n''',
        "allow encoded boundary cut-through target",
    )

    text = replace_once(
        text,
        '''        fracTail.erase(vI_boundary);\n''',
        '''        const int oneStringGridVertexBase = static_cast<int>(V_rest.rows());\n        const int oneStringGridBaseRows = duplicateBoth ? 4 : 2;\n        const bool oneStringGridBoundaryEncoded = oneStringGridEnabled() && changeVertPos &&\n            newVertPos.cols() == 2 && newVertPos.rows() == oneStringGridBaseRows + 2;\n        fracTail.erase(vI_boundary);\n''',
        "capture boundary grid encoding",
    )

    old_tail = '''            }\n        }\n    }\n    \n    void TriMesh::mergeBoundaryEdges'''
    new_tail = '''            }\n        }\n\n        if(oneStringGridBoundaryEncoded) {\n            const Eigen::MatrixXd gridPath = newVertPos.bottomRows(2);\n            const Eigen::RowVector2d qBoundary = (edge.first == vI_boundary) ? gridPath.row(0) : gridPath.row(1);\n            const Eigen::RowVector2d qInterior = (edge.first == vI_boundary) ? gridPath.row(1) : gridPath.row(0);\n            const int boundaryDuplicate = oneStringGridVertexBase;\n            V.row(vI_boundary) = qBoundary;\n            V.row(boundaryDuplicate) = qBoundary;\n            V.row(vI_interior) = qInterior;\n            for(const int lockVI : {vI_boundary, boundaryDuplicate, vI_interior}) {\n                fixedVert.insert(lockVI);\n                oneStringGridLockedVert.insert(lockVI);\n            }\n            if(duplicateBoth) {\n                const int interiorDuplicate = oneStringGridVertexBase + 1;\n                V.row(interiorDuplicate) = qInterior;\n                fixedVert.insert(interiorDuplicate);\n                oneStringGridLockedVert.insert(interiorDuplicate);\n            }\n            if(!checkInversion(true)) {\n                throw std::runtime_error("ONESTRING_GRID_APPLIED_BOUNDARY_CUT_INVERTED");\n            }\n        }\n    }\n    \n    void TriMesh::mergeBoundaryEdges'''
    text = replace_once(text, old_tail, new_tail, "finalize and lock boundary grid assignment")

    text = replace_once(
        text,
        '''    void TriMesh::computeFeatures(bool multiComp, bool resetFixedV)\n    {\n''',
        '''    void TriMesh::computeFeatures(bool multiComp, bool resetFixedV)\n    {\n        static bool oneStringGridPrinted = false;\n        if(oneStringGridEnabled() && !oneStringGridPrinted) {\n            std::cout << "[ONESTRING-GRID] native_candidate_search enabled version=3 h=" << oneStringGridH()\n                      << " angle_rad=" << oneStringGridAngle() << std::endl;\n            oneStringGridPrinted = true;\n        }\n''',
        "native grid marker",
    )

    backup = path.with_suffix(path.suffix + ".onestring-grid-native-backup")
    if not backup.exists():
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    path.write_text(text, encoding="utf-8")
    print(f"Applied OneString native Grid-OptCuts V3 patch: {path}")
    return True


def apply_native_grid_patch(root: Path) -> bool:
    _restore_old_native_if_needed(root)
    changed_h = patch_tri_mesh_header(root)
    changed_c = patch_tri_mesh(root)
    return changed_h or changed_c


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    apply_native_grid_patch(args.root.expanduser().resolve())
