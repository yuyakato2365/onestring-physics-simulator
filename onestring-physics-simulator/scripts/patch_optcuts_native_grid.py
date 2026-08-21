#!/usr/bin/env python3
"""Patch the local official OptCuts checkout with OneString native grid cuts.

This patch changes OptCuts' *candidate search space*.  It does not post-process a
finished OptCuts seam.  In native grid mode every split candidate must first
admit an embedding on one fixed orthogonal h-lattice (H, V, H->V, or V->H).
The candidate is scored with the ordinary OptCuts local SD decrease minus the
SD cost caused by moving the cut path onto that lattice.  The selected cut is
then applied at the exact lattice coordinates and its seam vertices are added to
TriMesh::fixedVert, so later UV optimization cannot move the seam off-grid.

Limit of this first native implementation: OptCuts still chooses cut topology on
its existing triangle-mesh edges.  It does not yet insert new surface vertices at
arbitrary grid-line/triangle intersections.  Thus it is a genuine constrained
OptCuts search, but its topological candidate resolution is bounded by the input
surface triangulation.
"""
from __future__ import annotations

from pathlib import Path

MARKER = "ONESTRING_GRID_NATIVE_V1"

HELPERS = r'''
// ONESTRING_GRID_NATIVE_V1
#include <cstdlib>
#include <cmath>
#include <limits>
#include <stdexcept>

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
    if(oneStringEmbeddingMaxSnapSteps(candidate, candidate) > 1.0e-12) return;
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

    // Horizontal candidate: same lattice row, non-zero integer x extent.
    for(long long dj = -1; dj <= 1; ++dj) {
        const long long j = oneStringRoundIndex(0.5 * (g0[1] + g1[1])) + dj;
        long long a = i0, b = i1;
        if(a == b) b += (g1[0] >= g0[0]) ? 1 : -1;
        Eigen::MatrixXd q(2, 2);
        q.row(0) = oneStringFromGridIndex(a, j);
        q.row(1) = oneStringFromGridIndex(b, j);
        Eigen::MatrixXd p(2, 2); p << p0, p1;
        if(oneStringEmbeddingMaxSnapSteps(p, q) <= oneStringGridMaxSnapSteps()) oneStringPushUnique(out, q);
    }

    // Vertical candidate: same lattice column, non-zero integer y extent.
    for(long long di = -1; di <= 1; ++di) {
        const long long i = oneStringRoundIndex(0.5 * (g0[0] + g1[0])) + di;
        long long a = j0, b = j1;
        if(a == b) b += (g1[1] >= g0[1]) ? 1 : -1;
        Eigen::MatrixXd q(2, 2);
        q.row(0) = oneStringFromGridIndex(i, a);
        q.row(1) = oneStringFromGridIndex(i, b);
        Eigen::MatrixXd p(2, 2); p << p0, p1;
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
        Eigen::MatrixXd q(3, 2);
        q.row(0) = oneStringFromGridIndex(a0, b0);
        q.row(1) = oneStringFromGridIndex(a1, b1);
        q.row(2) = oneStringFromGridIndex(a2, b2);
        if((q.row(1) - q.row(0)).norm() < 0.5 * oneStringGridH()) return;
        if((q.row(2) - q.row(1)).norm() < 0.5 * oneStringGridH()) return;
        if(oneStringEmbeddingMaxSnapSteps(p, q) <= oneStringGridMaxSnapSteps()) oneStringPushUnique(out, q);
    };

    // H-H straight run.
    for(long long dj = -1; dj <= 1; ++dj) {
        const long long j = oneStringRoundIndex((g0[1] + g1[1] + g2[1]) / 3.0) + dj;
        long long a0 = i0, a1 = i1, a2 = i2;
        if(a1 == a0) a1 += (g1[0] >= g0[0]) ? 1 : -1;
        if(a2 == a1) a2 += (g2[0] >= g1[0]) ? 1 : -1;
        accept(a0, j, a1, j, a2, j);
    }

    // V-V straight run.
    for(long long di = -1; di <= 1; ++di) {
        const long long i = oneStringRoundIndex((g0[0] + g1[0] + g2[0]) / 3.0) + di;
        long long b0 = j0, b1 = j1, b2 = j2;
        if(b1 == b0) b1 += (g1[1] >= g0[1]) ? 1 : -1;
        if(b2 == b1) b2 += (g2[1] >= g1[1]) ? 1 : -1;
        accept(i, b0, i, b1, i, b2);
    }

    // H -> V: bend is the exact lattice intersection (x from end, y from start).
    {
        long long ai0 = i0, aj0 = j0, ai2 = i2, aj2 = j2;
        if(ai2 == ai0) ai2 += (g2[0] >= g0[0]) ? 1 : -1;
        if(aj2 == aj0) aj2 += (g2[1] >= g0[1]) ? 1 : -1;
        accept(ai0, aj0, ai2, aj0, ai2, aj2);
    }

    // V -> H: bend is the complementary lattice intersection.
    {
        long long ai0 = i0, aj0 = j0, ai2 = i2, aj2 = j2;
        if(ai2 == ai0) ai2 += (g2[0] >= g0[0]) ? 1 : -1;
        if(aj2 == aj0) aj2 += (g2[1] >= g0[1]) ? 1 : -1;
        accept(ai0, aj0, ai0, aj2, ai2, aj2);
    }
    return out;
}

double oneStringGridSnapSDIncrease(
    const OptCuts::TriMesh& mesh,
    const std::vector<int>& path,
    const Eigen::MatrixXd& target)
{
    if(static_cast<int>(path.size()) != target.rows()) return std::numeric_limits<double>::infinity();
    OptCuts::TriMesh trial = mesh;
    for(int i = 0; i < target.rows(); ++i) trial.V.row(path[i]) = target.row(i);
    if(!trial.checkInversion(true)) return std::numeric_limits<double>::infinity();

    OptCuts::SymDirichletEnergy sd;
    double before = 0.0, after = 0.0;
    for(int triI = 0; triI < mesh.F.rows(); ++triI) {
        double eb = 0.0, ea = 0.0;
        sd.getEnergyValByElemID(mesh, triI, eb);
        sd.getEnergyValByElemID(trial, triI, ea);
        before += eb;
        after += ea;
    }
    const double inc = (after - before) / std::max(1.0, mesh.surfaceArea);
    return std::max(0.0, inc);
}

Eigen::MatrixXd oneStringEncodeInteriorCut(
    const Eigen::MatrixXd& optcutsPos,
    const Eigen::MatrixXd& gridPath)
{
    Eigen::MatrixXd encoded(optcutsPos.rows() + 3, 2);
    encoded.topRows(optcutsPos.rows()) = optcutsPos;
    encoded.bottomRows(3) = gridPath;
    return encoded;
}

Eigen::MatrixXd oneStringEncodeBoundaryCut(
    const Eigen::MatrixXd& optcutsPos,
    const Eigen::MatrixXd& gridPath)
{
    Eigen::MatrixXd encoded(optcutsPos.rows() + 2, 2);
    encoded.topRows(optcutsPos.rows()) = optcutsPos;
    encoded.bottomRows(2) = gridPath;
    return encoded;
}

} // anonymous namespace
'''


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one anchor, found {count}")
    return text.replace(old, new, 1)


def patch_tri_mesh(root: Path) -> bool:
    path = root / "src" / "TriMesh.cpp"
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        print("OneString native Grid-OptCuts patch already present.")
        return False

    text = replace_once(
        text,
        "extern int inSplitTotalAmt;\n\nnamespace OptCuts {",
        "extern int inSplitTotalAmt;\n" + HELPERS + "\nnamespace OptCuts {",
        "insert helper block",
    )

    # Grid mode never lets the unconstrained merge branch destroy a lattice seam.
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

    # Initial one-point cut must itself come from the same lattice candidate set.
    old_one_point = '''    void TriMesh::onePointCut(int vI)\n    {\n        assert((vI >= 0) && (vI < V_rest.rows()));\n        std::vector<int> path(vNeighbor[vI].begin(), vNeighbor[vI].end());\n        assert(path.size() >= 3);\n        path[1] = vI;\n        path.resize(3);\n        \n        bool makeCoh = true;\n        if(!makeCoh) {\n            for(int pI = 0; pI + 1 < path.size(); pI++) {\n                initSeamLen += (V_rest.row(path[pI]) - V_rest.row(path[pI + 1])).norm();\n            }\n        }\n        \n        cutPath(path, makeCoh);\n        \n        if(makeCoh) {\n            initSeams = cohE;\n        }\n    }\n'''
    new_one_point = '''    void TriMesh::onePointCut(int vI)\n    {\n        assert((vI >= 0) && (vI < V_rest.rows()));\n        bool makeCoh = true;\n        if(oneStringGridEnabled()) {\n            double bestCost = std::numeric_limits<double>::infinity();\n            std::vector<int> bestPath;\n            Eigen::MatrixXd bestGrid;\n            std::vector<int> neighbors(vNeighbor[vI].begin(), vNeighbor[vI].end());\n            for(int a = 0; a < static_cast<int>(neighbors.size()); ++a) {\n                for(int b = a + 1; b < static_cast<int>(neighbors.size()); ++b) {\n                    std::vector<int> path = {neighbors[a], vI, neighbors[b]};\n                    const auto embeddings = oneStringGridCornerEmbeddings(V.row(path[0]), V.row(path[1]), V.row(path[2]));\n                    for(const auto& gridPath : embeddings) {\n                        const double cost = oneStringGridSnapSDIncrease(*this, path, gridPath);\n                        if(std::isfinite(cost) && cost < bestCost) {\n                            bestCost = cost;\n                            bestPath = path;\n                            bestGrid = gridPath;\n                        }\n                    }\n                }\n            }\n            if(bestPath.empty()) {\n                throw std::runtime_error("ONESTRING_GRID_INITIAL_CUT_INFEASIBLE");\n            }\n            Eigen::MatrixXd encoded(5, 2);\n            encoded.row(0) = bestGrid.row(1);\n            encoded.row(1) = bestGrid.row(1);\n            encoded.bottomRows(3) = bestGrid;\n            cutPath(bestPath, makeCoh, 1, encoded);\n            std::cout << "[ONESTRING-GRID] initial_cut snap_SD_cost=" << bestCost << std::endl;\n            if(makeCoh) initSeams = cohE;\n            return;\n        }\n\n        std::vector<int> path(vNeighbor[vI].begin(), vNeighbor[vI].end());\n        assert(path.size() >= 3);\n        path[1] = vI;\n        path.resize(3);\n        if(!makeCoh) {\n            for(int pI = 0; pI + 1 < path.size(); pI++) {\n                initSeamLen += (V_rest.row(path[pI]) - V_rest.row(path[pI + 1])).norm();\n            }\n        }\n        cutPath(path, makeCoh);\n        if(makeCoh) initSeams = cohE;\n    }\n'''
    text = replace_once(text, old_one_point, new_one_point, "grid initial one-point cut")

    # Boundary split: ordinary OptCuts topology proposal, but only lattice-embeddable
    # paths survive and their snap distortion is included in the score.
    old_boundary = '''                    Eigen::MatrixXd newVertPosI;\n                    const double SDDec = queryLocalEdDec_bSplit(edge, newVertPosI);\n                    \n                    const double seInc = (V_rest.row(vI) - V_rest.row(nbVI)).norm() /\n                        virtualRadius * (vertWeight[vI] + vertWeight[nbVI]) / 2.0;\n                    const double curEwDec = (1.0 - lambda_t) * SDDec - lambda_t * seInc;\n                    if(curEwDec > maxEwDec) {\n                        maxEwDec = curEwDec;\n                        path_max[0] = vI;\n                        path_max[1] = nbVI;\n                        newVertPos_max = newVertPosI;\n                        energyChanges_max.first = -SDDec;\n                        energyChanges_max.second = seInc;\n                    }\n'''
    new_boundary = '''                    Eigen::MatrixXd newVertPosI;\n                    const double SDDec = queryLocalEdDec_bSplit(edge, newVertPosI);\n                    const double seInc = (V_rest.row(vI) - V_rest.row(nbVI)).norm() /\n                        virtualRadius * (vertWeight[vI] + vertWeight[nbVI]) / 2.0;\n\n                    if(oneStringGridEnabled()) {\n                        std::vector<int> gridPathIDs = {vI, nbVI};\n                        const auto embeddings = oneStringGridEdgeEmbeddings(V.row(vI), V.row(nbVI));\n                        for(const auto& gridPath : embeddings) {\n                            const double snapSDInc = oneStringGridSnapSDIncrease(*this, gridPathIDs, gridPath);\n                            if(!std::isfinite(snapSDInc)) continue;\n                            const double constrainedSDDec = SDDec - snapSDInc;\n                            const double curEwDec = (1.0 - lambda_t) * constrainedSDDec - lambda_t * seInc;\n                            if(curEwDec > maxEwDec) {\n                                maxEwDec = curEwDec;\n                                path_max[0] = vI;\n                                path_max[1] = nbVI;\n                                newVertPos_max = oneStringEncodeBoundaryCut(newVertPosI, gridPath);\n                                energyChanges_max.first = -constrainedSDDec;\n                                energyChanges_max.second = seInc;\n                            }\n                        }\n                    }\n                    else {\n                        const double curEwDec = (1.0 - lambda_t) * SDDec - lambda_t * seInc;\n                        if(curEwDec > maxEwDec) {\n                            maxEwDec = curEwDec;\n                            path_max[0] = vI;\n                            path_max[1] = nbVI;\n                            newVertPos_max = newVertPosI;\n                            energyChanges_max.first = -SDDec;\n                            energyChanges_max.second = seInc;\n                        }\n                    }\n'''
    text = replace_once(text, old_boundary, new_boundary, "grid boundary split scoring")

    old_interior = '''                    SDDec += computeLocalEdDec_inSplit(umbrella, freeVert, path, newVertPos);\n                    //TODO: share local mesh before split, also for boundary splits\n                    \n                    const double seInc = ((V_rest.row(path[0]) - V_rest.row(path[1])).norm() * (vertWeight[path[0]] + vertWeight[path[1]]) +\n                                          (V_rest.row(path[1]) - V_rest.row(path[2])).norm() * (vertWeight[path[1]] + vertWeight[path[2]])) / virtualRadius / 2.0;\n                    const double EwDec = (1.0 - lambda_t) * SDDec - lambda_t * seInc;\n                    if(EwDec > EwDec_max) {\n                        EwDec_max = EwDec;\n                        newVertPos_max = newVertPos;\n                        path_max = path;\n                        energyChanges_max.first = -SDDec;\n                        energyChanges_max.second = seInc;\n                    }\n'''
    new_interior = '''                    SDDec += computeLocalEdDec_inSplit(umbrella, freeVert, path, newVertPos);\n                    //TODO: share local mesh before split, also for boundary splits\n                    const double seInc = ((V_rest.row(path[0]) - V_rest.row(path[1])).norm() * (vertWeight[path[0]] + vertWeight[path[1]]) +\n                                          (V_rest.row(path[1]) - V_rest.row(path[2])).norm() * (vertWeight[path[1]] + vertWeight[path[2]])) / virtualRadius / 2.0;\n\n                    if(oneStringGridEnabled()) {\n                        const auto embeddings = oneStringGridCornerEmbeddings(V.row(path[0]), V.row(path[1]), V.row(path[2]));\n                        for(const auto& gridPath : embeddings) {\n                            const double snapSDInc = oneStringGridSnapSDIncrease(*this, path, gridPath);\n                            if(!std::isfinite(snapSDInc)) continue;\n                            const double constrainedSDDec = SDDec - snapSDInc;\n                            const double EwDec = (1.0 - lambda_t) * constrainedSDDec - lambda_t * seInc;\n                            if(EwDec > EwDec_max) {\n                                EwDec_max = EwDec;\n                                newVertPos_max = oneStringEncodeInteriorCut(newVertPos, gridPath);\n                                path_max = path;\n                                energyChanges_max.first = -constrainedSDDec;\n                                energyChanges_max.second = seInc;\n                            }\n                        }\n                    }\n                    else {\n                        const double EwDec = (1.0 - lambda_t) * SDDec - lambda_t * seInc;\n                        if(EwDec > EwDec_max) {\n                            EwDec_max = EwDec;\n                            newVertPos_max = newVertPos;\n                            path_max = path;\n                            energyChanges_max.first = -SDDec;\n                            energyChanges_max.second = seInc;\n                        }\n                    }\n'''
    text = replace_once(text, old_interior, new_interior, "grid interior split scoring")

    # Interior application: encoded bottom 3 rows are the exact lattice path.
    text = replace_once(
        text,
        '''        if(changePos) {\n            assert((changePos == 1) && "right now only support change 1"); //!!! still only allow 1?\n            assert(newVertPos.cols() == 2);\n            assert(changePos * 2 == newVertPos.rows());\n        }\n''',
        '''        const bool oneStringGridEncoded = oneStringGridEnabled() && changePos && newVertPos.cols() == 2 && newVertPos.rows() >= 5;\n        if(changePos) {\n            assert((changePos == 1) && "right now only support change 1");\n            assert(newVertPos.cols() == 2);\n            assert(oneStringGridEncoded || (changePos * 2 == newVertPos.rows()));\n        }\n''',
        "allow encoded interior grid target",
    )

    text = replace_once(
        text,
        '''            if(changePos) {\n                V.row(nV) = newVertPos.block(0, 0, 1, 2);\n                V.row(path[1]) = newVertPos.block(1, 0, 1, 2);\n            }\n            else {\n                V.row(nV) = V.row(path[1]);\n            }\n''',
        '''            if(changePos) {\n                V.row(nV) = newVertPos.block(0, 0, 1, 2);\n                V.row(path[1]) = newVertPos.block(1, 0, 1, 2);\n                if(oneStringGridEncoded) {\n                    const Eigen::MatrixXd gridPath = newVertPos.bottomRows(3);\n                    V.row(path[0]) = gridPath.row(0);\n                    V.row(path[1]) = gridPath.row(1);\n                    V.row(nV) = gridPath.row(1);\n                    V.row(path[2]) = gridPath.row(2);\n                    fixedVert.insert(path[0]);\n                    fixedVert.insert(path[1]);\n                    fixedVert.insert(path[2]);\n                    fixedVert.insert(nV);\n                }\n            }\n            else {\n                V.row(nV) = V.row(path[1]);\n            }\n''',
        "apply interior grid target",
    )

    # Boundary application: permit 2 extra rows and lock original+duplicated seam vertices.
    text = replace_once(
        text,
        '''                if(changeVertPos) {\n                    assert(newVertPos.rows() == 4);\n                }\n                duplicateBoth = true;\n''',
        '''                if(changeVertPos) {\n                    assert(newVertPos.rows() == 4 || (oneStringGridEnabled() && newVertPos.rows() == 6));\n                }\n                duplicateBoth = true;\n''',
        "allow encoded boundary cut-through target",
    )

    # Save the vertex count before duplicates so we can identify both seam copies.
    text = replace_once(
        text,
        '''        fracTail.erase(vI_boundary);\n''',
        '''        const int oneStringGridVertexBase = static_cast<int>(V_rest.rows());\n        const int oneStringGridBaseRows = duplicateBoth ? 4 : 2;\n        const bool oneStringGridBoundaryEncoded = oneStringGridEnabled() && changeVertPos &&\n            newVertPos.cols() == 2 && newVertPos.rows() == oneStringGridBaseRows + 2;\n        fracTail.erase(vI_boundary);\n''',
        "capture boundary grid encoding",
    )

    # Append final exact lattice assignment at end of splitEdgeOnBoundary.
    boundary_end = '''        if(duplicateBoth) {\n            int nV = static_cast<int>(V_rest.rows());\n'''
    # We insert at the unique function-closing sequence immediately before mergeBoundaryEdges.
    old_tail = '''            }\n        }\n    }\n    \n    void TriMesh::mergeBoundaryEdges'''
    new_tail = '''            }\n        }\n\n        if(oneStringGridBoundaryEncoded) {\n            const Eigen::MatrixXd gridPath = newVertPos.bottomRows(2);\n            const Eigen::RowVector2d qBoundary = (edge.first == vI_boundary) ? gridPath.row(0) : gridPath.row(1);\n            const Eigen::RowVector2d qInterior = (edge.first == vI_boundary) ? gridPath.row(1) : gridPath.row(0);\n            const int boundaryDuplicate = oneStringGridVertexBase;\n            V.row(vI_boundary) = qBoundary;\n            V.row(boundaryDuplicate) = qBoundary;\n            V.row(vI_interior) = qInterior;\n            fixedVert.insert(vI_boundary);\n            fixedVert.insert(boundaryDuplicate);\n            fixedVert.insert(vI_interior);\n            if(duplicateBoth) {\n                const int interiorDuplicate = oneStringGridVertexBase + 1;\n                V.row(interiorDuplicate) = qInterior;\n                fixedVert.insert(interiorDuplicate);\n            }\n            if(!checkInversion(true)) {\n                throw std::runtime_error("ONESTRING_GRID_APPLIED_BOUNDARY_CUT_INVERTED");\n            }\n        }\n    }\n    \n    void TriMesh::mergeBoundaryEdges'''
    text = replace_once(text, old_tail, new_tail, "finalize boundary grid assignment")

    # Emit a marker in every native run so Python can refuse an unpatched binary.
    text = replace_once(
        text,
        '''    void TriMesh::computeFeatures(bool multiComp, bool resetFixedV)\n    {\n''',
        '''    void TriMesh::computeFeatures(bool multiComp, bool resetFixedV)\n    {\n        static bool oneStringGridPrinted = false;\n        if(oneStringGridEnabled() && !oneStringGridPrinted) {\n            std::cout << "[ONESTRING-GRID] native_candidate_search enabled h=" << oneStringGridH()\n                      << " angle_rad=" << oneStringGridAngle() << std::endl;\n            oneStringGridPrinted = true;\n        }\n''',
        "native grid marker",
    )

    backup = path.with_suffix(path.suffix + ".onestring-grid-native-backup")
    if not backup.exists():
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    path.write_text(text, encoding="utf-8")
    print(f"Applied OneString native Grid-OptCuts patch: {path}")
    return True


def apply_native_grid_patch(root: Path) -> bool:
    return patch_tri_mesh(root)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    apply_native_grid_patch(args.root.expanduser().resolve())
