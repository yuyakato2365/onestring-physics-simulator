#!/usr/bin/env python3
"""Patch official OptCuts into OneString native Grid-OptCuts V4.

V4 implements the user's intended algorithmic model: the OptCuts topology search
itself is restricted to fabrication-grid cuts.  A finished free OptCuts seam is
never snapped afterward.

Core invariants in native grid mode:
- fixed lattice spacing h and one global orthogonal frame;
- accepted physical seam sides are H/V grid-edge paths;
- an interior 2-edge physical cut is represented by TWO UV boundary paths
  A-B-C and A-D-C (paired rectangle corners), not coincident zero-width copies;
- boundary propagation is trialled directly on the grid and accepted only if
  every cohesive seam side remains grid aligned;
- candidates are scored by the Symmetric Dirichlet energy of the ACTUAL trial
  cut, plus the ordinary OptCuts seam-length term;
- accepted seam/junction UV vertices are persistent solver locks;
- merge is disabled in this first native version rather than allowing an
  unconstrained merge to destroy the fabrication lattice;
- for a closed genus-0 input, the initial two-edge cut is parameterized with a
  grid-rectangle boundary before the OptCuts optimizer starts.

Topology-resolution limitation: V4 still uses OptCuts' existing source triangle
mesh edges as physical cut candidates.  It does not yet insert arbitrary new 3D
surface vertices at grid-line/source-triangle intersections.
"""
from __future__ import annotations

from pathlib import Path

MARKER = "ONESTRING_GRID_NATIVE_V4"
OLD_MARKERS = (
    "ONESTRING_GRID_NATIVE_V1",
    "ONESTRING_GRID_NATIVE_V2",
    "ONESTRING_GRID_NATIVE_V3",
)
BACKUP_SUFFIX = ".onestring-grid-native-backup"

TRIMESH_HELPERS = r'''
// ONESTRING_GRID_NATIVE_V4
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <limits>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <vector>

namespace {

bool oneStringGridEnabled()
{
    const char* raw = std::getenv("ONESTRING_OPTCUTS_GRID_NATIVE");
    if(!raw) return false;
    const std::string value(raw);
    return value == "1" || value == "true" || value == "TRUE" || value == "on";
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

struct OneStringGridIJ {
    long long i = 0;
    long long j = 0;
};

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

OneStringGridIJ oneStringNearestGridIJ(const Eigen::RowVector2d& p)
{
    const Eigen::Vector2d g = oneStringToGridUnits(p);
    OneStringGridIJ out;
    out.i = static_cast<long long>(std::llround(g[0]));
    out.j = static_cast<long long>(std::llround(g[1]));
    return out;
}

Eigen::RowVector2d oneStringFromGridIJ(long long i, long long j)
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

bool oneStringIsGridPoint(const Eigen::RowVector2d& p)
{
    const Eigen::Vector2d g = oneStringToGridUnits(p);
    const double tol = 1.0e-7;
    return std::abs(g[0] - std::round(g[0])) <= tol &&
           std::abs(g[1] - std::round(g[1])) <= tol;
}

bool oneStringSameGridPoint(const Eigen::RowVector2d& a, const Eigen::RowVector2d& b)
{
    return (a - b).norm() <= std::max(1.0e-10, oneStringGridH() * 1.0e-7);
}

bool oneStringIsGridEdge(const Eigen::RowVector2d& a, const Eigen::RowVector2d& b)
{
    if(!oneStringIsGridPoint(a) || !oneStringIsGridPoint(b) || oneStringSameGridPoint(a, b)) return false;
    const OneStringGridIJ ia = oneStringNearestGridIJ(a);
    const OneStringGridIJ ib = oneStringNearestGridIJ(b);
    return ia.i == ib.i || ia.j == ib.j;
}

double oneStringSnapSteps(const Eigen::RowVector2d& from, const Eigen::RowVector2d& to)
{
    return (from - to).norm() / oneStringGridH();
}

std::vector<Eigen::RowVector2d> oneStringNearbyGridPoints(
    const Eigen::RowVector2d& p,
    int maxCount = 9)
{
    const Eigen::Vector2d g = oneStringToGridUnits(p);
    const long long ci = static_cast<long long>(std::llround(g[0]));
    const long long cj = static_cast<long long>(std::llround(g[1]));
    const int radius = std::max(1, static_cast<int>(std::ceil(oneStringGridMaxSnapSteps())));
    std::vector<std::pair<double, Eigen::RowVector2d>> ranked;
    for(int di = -radius; di <= radius; ++di) {
        for(int dj = -radius; dj <= radius; ++dj) {
            const Eigen::RowVector2d q = oneStringFromGridIJ(ci + di, cj + dj);
            const double d = oneStringSnapSteps(p, q);
            if(d <= oneStringGridMaxSnapSteps() + 1.0e-9) ranked.emplace_back(d, q);
        }
    }
    std::sort(ranked.begin(), ranked.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.first < rhs.first;
    });
    std::vector<Eigen::RowVector2d> out;
    for(const auto& item : ranked) {
        bool duplicate = false;
        for(const auto& q : out) {
            if(oneStringSameGridPoint(q, item.second)) { duplicate = true; break; }
        }
        if(!duplicate) out.emplace_back(item.second);
        if(static_cast<int>(out.size()) >= maxCount) break;
    }
    return out;
}

std::vector<Eigen::RowVector2d> oneStringVertexGridTargets(const OptCuts::TriMesh& mesh, int vI)
{
    if(mesh.oneStringGridLockedVert.find(vI) != mesh.oneStringGridLockedVert.end()) {
        if(oneStringIsGridPoint(mesh.V.row(vI))) return {mesh.V.row(vI)};
        return {};
    }
    return oneStringNearbyGridPoints(mesh.V.row(vI));
}

double oneStringTotalSD(const OptCuts::TriMesh& mesh)
{
    OptCuts::SymDirichletEnergy sd;
    double value = 0.0;
    for(int triI = 0; triI < mesh.F.rows(); ++triI) {
        double e = 0.0;
        sd.getEnergyValByElemID(mesh, triI, e);
        if(!std::isfinite(e)) return std::numeric_limits<double>::infinity();
        value += e;
    }
    return value;
}

bool oneStringLockedPreserved(const OptCuts::TriMesh& before, const OptCuts::TriMesh& after)
{
    const double tol = std::max(1.0e-10, oneStringGridH() * 1.0e-7);
    for(const int vI : before.oneStringGridLockedVert) {
        if(vI < 0 || vI >= before.V.rows() || vI >= after.V.rows()) return false;
        if((before.V.row(vI) - after.V.row(vI)).norm() > tol) return false;
    }
    return true;
}

bool oneStringAllLockedOnGrid(const OptCuts::TriMesh& mesh)
{
    for(const int vI : mesh.oneStringGridLockedVert) {
        if(vI < 0 || vI >= mesh.V.rows() || !oneStringIsGridPoint(mesh.V.row(vI))) return false;
    }
    return true;
}

bool oneStringAllCohesiveSeamSidesGridAligned(const OptCuts::TriMesh& mesh)
{
    if(mesh.boundaryEdge.size() != mesh.cohE.rows()) return false;
    for(int cohI = 0; cohI < mesh.cohE.rows(); ++cohI) {
        // boundaryEdge==1 is a true source-surface boundary; it is not a cut seam.
        if(mesh.boundaryEdge[cohI]) continue;
        const Eigen::RowVector4i e = mesh.cohE.row(cohI);
        if(e.minCoeff() < 0) continue;
        if(!oneStringIsGridEdge(mesh.V.row(e[0]), mesh.V.row(e[1]))) return false;
        if(!oneStringIsGridEdge(mesh.V.row(e[2]), mesh.V.row(e[3]))) return false;
    }
    return true;
}

void oneStringLockGridVertices(OptCuts::TriMesh& mesh, const std::vector<int>& ids)
{
    for(const int vI : ids) {
        if(vI < 0 || vI >= mesh.V.rows()) continue;
        mesh.oneStringGridLockedVert.insert(vI);
        mesh.fixedVert.insert(vI);
    }
    mesh.computeLaplacianMtr();
}

double oneStringScoreTrial(
    const OptCuts::TriMesh& mesh,
    const OptCuts::TriMesh& trial,
    double lambda_t,
    double seamIncrease,
    std::pair<double, double>& energyChanges)
{
    if(!trial.checkInversion(true)) return -DBL_MAX;
    if(!oneStringLockedPreserved(mesh, trial)) return -DBL_MAX;
    if(!oneStringAllLockedOnGrid(trial)) return -DBL_MAX;
    if(!oneStringAllCohesiveSeamSidesGridAligned(trial)) return -DBL_MAX;
    const double before = oneStringTotalSD(mesh);
    const double after = oneStringTotalSD(trial);
    if(!std::isfinite(before) || !std::isfinite(after)) return -DBL_MAX;
    const double sdDec = before - after;
    energyChanges.first = -sdDec;
    energyChanges.second = seamIncrease;
    return (1.0 - lambda_t) * sdDec - lambda_t * seamIncrease;
}

bool oneStringTryInteriorGridCut(
    const OptCuts::TriMesh& mesh,
    const std::vector<int>& path,
    const Eigen::RowVector2d& A,
    const Eigen::RowVector2d& B,
    const Eigen::RowVector2d& D,
    const Eigen::RowVector2d& C,
    double lambda_t,
    double seamIncrease,
    double& score,
    Eigen::MatrixXd& encoded,
    std::pair<double, double>& energyChanges)
{
    encoded.resize(4, 2);
    // cutPath creates a duplicate of path[1].  row0 is the duplicate side,
    // row1 the original side; rows2/3 are the shared physical endpoints.
    encoded.row(0) = B;
    encoded.row(1) = D;
    encoded.row(2) = A;
    encoded.row(3) = C;
    try {
        OptCuts::TriMesh trial = mesh;
        trial.cutPath(path, true, 1, encoded, false);
        score = oneStringScoreTrial(mesh, trial, lambda_t, seamIncrease, energyChanges);
        return score != -DBL_MAX;
    }
    catch(...) {
        score = -DBL_MAX;
        return false;
    }
}

bool oneStringTryBoundaryGridCut(
    const OptCuts::TriMesh& mesh,
    int boundaryVI,
    int interiorVI,
    const Eigen::RowVector2d& A,
    const Eigen::RowVector2d& B,
    const Eigen::RowVector2d& C,
    double lambda_t,
    double seamIncrease,
    double& score,
    Eigen::MatrixXd& encoded,
    std::pair<double, double>& energyChanges)
{
    encoded.resize(3, 2);
    // A and B are the two UV copies of the old fracture-tail vertex after
    // extension; C is the new shared physical tip.
    encoded.row(0) = A;
    encoded.row(1) = B;
    encoded.row(2) = C;
    try {
        OptCuts::TriMesh trial = mesh;
        trial.splitEdgeOnBoundary(std::pair<int, int>(boundaryVI, interiorVI), encoded, true, false);
        trial.updateFeatures();
        score = oneStringScoreTrial(mesh, trial, lambda_t, seamIncrease, energyChanges);
        return score != -DBL_MAX;
    }
    catch(...) {
        score = -DBL_MAX;
        return false;
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
    double best = -DBL_MAX;

    if(mesh.isBoundaryVert(vI)) {
        const auto targetsA = oneStringVertexGridTargets(mesh, vI);
        if(targetsA.empty()) return -DBL_MAX;
        for(const int nbVI : mesh.vNeighbor[vI]) {
            if(mesh.isBoundaryVert(nbVI)) continue; // native V4 does not cut through two existing boundaries
            if(mesh.edge2Tri.find(std::pair<int, int>(vI, nbVI)) == mesh.edge2Tri.end() ||
               mesh.edge2Tri.find(std::pair<int, int>(nbVI, vI)) == mesh.edge2Tri.end()) continue;
            const auto targetsC = oneStringVertexGridTargets(mesh, nbVI);
            const auto targetsB = oneStringNearbyGridPoints(mesh.V.row(vI));
            const double seamInc = (mesh.V_rest.row(vI) - mesh.V_rest.row(nbVI)).norm() /
                mesh.virtualRadius * (mesh.vertWeight[vI] + mesh.vertWeight[nbVI]) / 2.0;
            for(const auto& A : targetsA) {
                for(const auto& C : targetsC) {
                    if(!oneStringIsGridEdge(A, C)) continue;
                    for(const auto& B : targetsB) {
                        if(oneStringSameGridPoint(A, B) || oneStringSameGridPoint(B, C)) continue;
                        if(!oneStringIsGridEdge(B, C)) continue;
                        double score = -DBL_MAX;
                        Eigen::MatrixXd encoded;
                        std::pair<double, double> changes;
                        if(!oneStringTryBoundaryGridCut(mesh, vI, nbVI, A, B, C,
                                                       lambda_t, seamInc, score, encoded, changes)) continue;
                        if(score > best) {
                            best = score;
                            path_max = {vI, nbVI};
                            newVertPos_max = encoded;
                            energyChanges_max = changes;
                        }
                    }
                }
            }
        }
        return best;
    }

    // Match the original OptCuts rule: do not create an interior split touching
    // an already existing boundary.  Such growth is handled by boundary propagation.
    for(const int nbVI : mesh.vNeighbor[vI]) {
        if(mesh.isBoundaryVert(nbVI)) return -DBL_MAX;
    }
    if(mesh.oneStringGridLockedVert.find(vI) != mesh.oneStringGridLockedVert.end()) return -DBL_MAX;

    for(const auto& pair : mesh.validSplit[vI]) {
        const int a = pair.first;
        const int c = pair.second;
        if(a >= c) continue; // validSplit stores both orientations
        const auto targetsA = oneStringVertexGridTargets(mesh, a);
        const auto targetsC = oneStringVertexGridTargets(mesh, c);
        if(targetsA.empty() || targetsC.empty()) continue;
        const double seamInc = (
            (mesh.V_rest.row(a) - mesh.V_rest.row(vI)).norm() * (mesh.vertWeight[a] + mesh.vertWeight[vI]) +
            (mesh.V_rest.row(vI) - mesh.V_rest.row(c)).norm() * (mesh.vertWeight[vI] + mesh.vertWeight[c])
        ) / mesh.virtualRadius / 2.0;

        for(const auto& A : targetsA) {
            const OneStringGridIJ ia = oneStringNearestGridIJ(A);
            for(const auto& C : targetsC) {
                const OneStringGridIJ ic = oneStringNearestGridIJ(C);
                // A and C must differ in both lattice coordinates so the two
                // 2-segment boundary copies can form a non-degenerate rectangle.
                if(ia.i == ic.i || ia.j == ic.j) continue;
                const Eigen::RowVector2d B = oneStringFromGridIJ(ic.i, ia.j);
                const Eigen::RowVector2d D = oneStringFromGridIJ(ia.i, ic.j);
                if(oneStringSnapSteps(mesh.V.row(vI), B) > oneStringGridMaxSnapSteps() + 1.0e-9) continue;
                if(oneStringSnapSteps(mesh.V.row(vI), D) > oneStringGridMaxSnapSteps() + 1.0e-9) continue;
                if(!oneStringIsGridEdge(A, B) || !oneStringIsGridEdge(B, C) ||
                   !oneStringIsGridEdge(A, D) || !oneStringIsGridEdge(D, C)) continue;

                const std::vector<int> path = {a, vI, c};
                for(int swapSides = 0; swapSides < 2; ++swapSides) {
                    const Eigen::RowVector2d side0 = swapSides ? D : B;
                    const Eigen::RowVector2d side1 = swapSides ? B : D;
                    double score = -DBL_MAX;
                    Eigen::MatrixXd encoded;
                    std::pair<double, double> changes;
                    if(!oneStringTryInteriorGridCut(mesh, path, A, side0, side1, C,
                                                   lambda_t, seamInc, score, encoded, changes)) continue;
                    if(score > best) {
                        best = score;
                        path_max = path;
                        newVertPos_max = encoded;
                        energyChanges_max = changes;
                    }
                }
            }
        }
    }
    return best;
}

} // anonymous namespace
'''

MAIN_HELPERS = r'''
// ONESTRING_GRID_NATIVE_V4_MAIN
namespace {

bool oneStringMainGridEnabled()
{
    const char* raw = std::getenv("ONESTRING_OPTCUTS_GRID_NATIVE");
    if(!raw) return false;
    const std::string value(raw);
    return value == "1" || value == "true" || value == "TRUE" || value == "on";
}

double oneStringMainGridEnvDouble(const char* key, double fallback)
{
    const char* raw = std::getenv(key);
    if(!raw) return fallback;
    try { return std::stod(std::string(raw)); }
    catch(...) { return fallback; }
}

double oneStringMainGridH()
{
    return std::max(1.0e-10, oneStringMainGridEnvDouble("ONESTRING_OPTCUTS_GRID_H", 0.1));
}

Eigen::RowVector2d oneStringMainFromGridIJ(long long i, long long j)
{
    const double a = oneStringMainGridEnvDouble("ONESTRING_OPTCUTS_GRID_ANGLE_RAD", 0.0);
    const Eigen::Vector2d u(std::cos(a), std::sin(a));
    const Eigen::Vector2d v(-std::sin(a), std::cos(a));
    const double h = oneStringMainGridH();
    const double pu = oneStringMainGridEnvDouble("ONESTRING_OPTCUTS_GRID_PHASE_U", 0.0);
    const double pv = oneStringMainGridEnvDouble("ONESTRING_OPTCUTS_GRID_PHASE_V", 0.0);
    return ((pu + static_cast<double>(i) * h) * u +
            (pv + static_cast<double>(j) * h) * v).transpose();
}

Eigen::MatrixXd oneStringMainInitialGridBoundary(
    const OptCuts::TriMesh& temp,
    const std::vector<int>& bnd)
{
    if(bnd.size() != 4) {
        throw std::runtime_error(
            "ONESTRING_GRID_INITIAL_BOUNDARY_UNSUPPORTED: native V4 expects the random two-edge "
            "initial closed-surface cut to produce exactly four UV boundary vertices"
        );
    }
    const double h = oneStringMainGridH();
    auto length = [&](int a, int b) {
        return (temp.V_rest.row(bnd[a]) - temp.V_rest.row(bnd[b])).norm();
    };
    const double wPhysical = 0.5 * (length(0, 1) + length(2, 3));
    const double hPhysical = 0.5 * (length(1, 2) + length(3, 0));
    const long long wi = std::max<long long>(1, static_cast<long long>(std::llround(wPhysical / h)));
    const long long hi = std::max<long long>(1, static_cast<long long>(std::llround(hPhysical / h)));
    Eigen::MatrixXd uv(4, 2);
    uv.row(0) = oneStringMainFromGridIJ(0, 0);
    uv.row(1) = oneStringMainFromGridIJ(wi, 0);
    uv.row(2) = oneStringMainFromGridIJ(wi, hi);
    uv.row(3) = oneStringMainFromGridIJ(0, hi);
    return uv;
}

void oneStringMainReflectFabricationV(Eigen::MatrixXd& uv)
{
    const double a = oneStringMainGridEnvDouble("ONESTRING_OPTCUTS_GRID_ANGLE_RAD", 0.0);
    const Eigen::Vector2d v(-std::sin(a), std::cos(a));
    const double pv = oneStringMainGridEnvDouble("ONESTRING_OPTCUTS_GRID_PHASE_V", 0.0);
    for(int r = 0; r < uv.rows(); ++r) {
        Eigen::Vector2d q = uv.row(r).transpose();
        const double coord = q.dot(v);
        q -= 2.0 * (coord - pv) * v;
        uv.row(r) = q.transpose();
    }
}

} // anonymous namespace
'''


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one anchor, found {count}")
    return text.replace(old, new, 1)


def restore_old_patch(path: Path) -> None:
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    if not any(marker in text for marker in OLD_MARKERS):
        return
    backup = path.with_suffix(path.suffix + BACKUP_SUFFIX)
    if not backup.is_file():
        raise RuntimeError(
            f"{path} contains an older OneString Grid-OptCuts patch but {backup} is missing; "
            "restore the official OptCuts file before applying V4"
        )
    path.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Restored pristine OptCuts source before V4 patch: {path}")


def backup_once(path: Path) -> None:
    backup = path.with_suffix(path.suffix + BACKUP_SUFFIX)
    if not backup.exists():
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")


def patch_header(root: Path) -> bool:
    path = root / "src" / "TriMesh.hpp"
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    if "oneStringGridLockedVert" in text:
        return False
    backup_once(path)
    text = replace_once(
        text,
        "        std::set<int> fixedVert; // for linear solve\n        Eigen::Matrix<double, 2, 3> bbox;",
        "        std::set<int> fixedVert; // for linear solve\n"
        "        // OneString native Grid-OptCuts: persistent fabrication-grid seam/junction locks.\n"
        "        std::set<int> oneStringGridLockedVert;\n"
        "        Eigen::Matrix<double, 2, 3> bbox;",
        "TriMesh grid lock member",
    )
    path.write_text(text, encoding="utf-8")
    print(f"Patched Grid-OptCuts lock state: {path}")
    return True


def patch_trimesh(root: Path) -> bool:
    path = root / "src" / "TriMesh.cpp"
    if not path.is_file():
        raise FileNotFoundError(path)
    restore_old_patch(path)
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        print("OneString native Grid-OptCuts V4 TriMesh patch already present.")
        return False
    backup_once(path)

    text = replace_once(
        text,
        "extern int inSplitTotalAmt;\n\nnamespace OptCuts {",
        "extern int inSplitTotalAmt;\n" + TRIMESH_HELPERS + "\nnamespace OptCuts {",
        "insert V4 helper block",
    )

    text = replace_once(
        text,
        '''        if(resetFixedV) {\n            fixedVert.clear();\n            fixedVert.insert(0);\n        }\n''',
        '''        if(resetFixedV) {\n            fixedVert.clear();\n            fixedVert.insert(0);\n        }\n        if(oneStringGridEnabled()) {\n            fixedVert.insert(oneStringGridLockedVert.begin(), oneStringGridLockedVert.end());\n        }\n''',
        "preserve grid locks in computeFeatures",
    )

    text = replace_once(
        text,
        '''        fixedVert = p_fixedVert;\n        computeLaplacianMtr();\n''',
        '''        fixedVert = p_fixedVert;\n        if(oneStringGridEnabled()) {\n            fixedVert.insert(oneStringGridLockedVert.begin(), oneStringGridLockedVert.end());\n        }\n        computeLaplacianMtr();\n''',
        "preserve grid locks in resetFixedVert",
    )

    text = replace_once(
        text,
        '''        if(!path_max.empty()) {\n            // merge query\n''',
        '''        if(oneStringGridEnabled() && path_max.empty()) {\n            return oneStringGridComputeLocalLDec(*this, vI, lambda_t, path_max, newVertPos_max, energyChanges_max);\n        }\n\n        if(!path_max.empty()) {\n            // merge query\n''',
        "dispatch Grid split candidate search",
    )

    text = replace_once(
        text,
        '''        if(splitInterior) {\n            querySplit(lambda_t, propagate, splitInterior,\n                       EwDec_max, path_max, newVertPos_max,\n                       energyChanes_split);\n        }\n        else {\n            double EwDec_max_split, EwDec_max_merge;\n''',
        '''        if(splitInterior) {\n            querySplit(lambda_t, propagate, splitInterior,\n                       EwDec_max, path_max, newVertPos_max,\n                       energyChanes_split);\n        }\n        else if(oneStringGridEnabled()) {\n            // Native Grid-OptCuts V4 is deliberately split-only.  An unconstrained\n            // merge could move or remove persistent lattice seam vertices.\n            querySplit(lambda_t, propagate, splitInterior,\n                       EwDec_max, path_max, newVertPos_max,\n                       energyChanes_split);\n            isMerge = false;\n        }\n        else {\n            double EwDec_max_split, EwDec_max_merge;\n''',
        "disable unconstrained merge in Grid mode",
    )

    text = replace_once(
        text,
        '''        if(changePos) {\n            assert((changePos == 1) && "right now only support change 1"); //!!! still only allow 1?\n            assert(newVertPos.cols() == 2);\n            assert(changePos * 2 == newVertPos.rows());\n        }\n''',
        '''        const bool oneStringGridEncodedInterior = oneStringGridEnabled() && changePos &&\n            newVertPos.cols() == 2 && newVertPos.rows() == 4;\n        if(changePos) {\n            assert((changePos == 1) && "right now only support change 1");\n            assert(newVertPos.cols() == 2);\n            assert(oneStringGridEncodedInterior || changePos * 2 == newVertPos.rows());\n            if(oneStringGridEncodedInterior && path.size() != 3) {\n                throw std::runtime_error("ONESTRING_GRID_INTERIOR_PATH_SIZE_UNSUPPORTED");\n            }\n        }\n''',
        "allow paired Grid interior encoding",
    )

    text = replace_once(
        text,
        '''            // path is interior\n            assert(path.size() >= 3);\n            \n            std::vector<int> tri_left;\n''',
        '''            // path is interior\n            assert(path.size() >= 3);\n            if(oneStringGridEncodedInterior) {\n                V.row(path[0]) = newVertPos.row(2);\n                V.row(path[2]) = newVertPos.row(3);\n            }\n            \n            std::vector<int> tri_left;\n''',
        "apply Grid interior endpoint targets",
    )

    text = replace_once(
        text,
        '''            computeFeatures(); //TODO: only update locally\n            \n            for(int vI = 2; vI + 1 < path.size(); vI++) {\n''',
        '''            computeFeatures(); //TODO: only update locally\n            if(oneStringGridEncodedInterior) {\n                oneStringLockGridVertices(*this, {path[0], path[1], path[2], nV});\n                if(!checkInversion(true) || !oneStringAllCohesiveSeamSidesGridAligned(*this)) {\n                    throw std::runtime_error("ONESTRING_GRID_APPLIED_INTERIOR_CUT_INVALID");\n                }\n            }\n            \n            for(int vI = 2; vI + 1 < path.size(); vI++) {\n''',
        "lock and validate applied Grid interior cut",
    )

    text = replace_once(
        text,
        '''        if(isBoundaryVert(edge.first)) {\n            if(allowCutThrough && isBoundaryVert(edge.second)) {\n                if(changeVertPos) {\n                    assert(newVertPos.rows() == 4);\n                }\n                duplicateBoth = true;\n            }\n        }\n''',
        '''        if(isBoundaryVert(edge.first)) {\n            if(allowCutThrough && isBoundaryVert(edge.second)) {\n                if(changeVertPos) {\n                    assert(newVertPos.rows() == 4);\n                }\n                duplicateBoth = true;\n            }\n        }\n''',
        "boundary split anchor",
    )
    # Insert Grid boundary encoding immediately after boundary/interior endpoint resolution.
    text = replace_once(
        text,
        '''        else {\n            assert(isBoundaryVert(edge.second) && "Input edge must attach mesh boundary!");\n            \n            vI_boundary = edge.second;\n            vI_interior = edge.first;\n        }\n        \n        fracTail.erase(vI_boundary);\n''',
        '''        else {\n            assert(isBoundaryVert(edge.second) && "Input edge must attach mesh boundary!");\n            \n            vI_boundary = edge.second;\n            vI_interior = edge.first;\n        }\n        const bool oneStringGridEncodedBoundary = oneStringGridEnabled() && changeVertPos &&\n            !duplicateBoth && newVertPos.cols() == 2 && newVertPos.rows() == 3;\n        if(oneStringGridEnabled() && changeVertPos && !duplicateBoth &&\n           newVertPos.rows() != 2 && !oneStringGridEncodedBoundary) {\n            throw std::runtime_error("ONESTRING_GRID_BOUNDARY_ENCODING_SIZE");\n        }\n        if(oneStringGridEncodedBoundary) {\n            V.row(vI_interior) = newVertPos.row(2);\n        }\n        \n        fracTail.erase(vI_boundary);\n''',
        "decode Grid boundary target",
    )

    text = replace_once(
        text,
        '''        int nV = static_cast<int>(V_rest.rows());\n        V_rest.conservativeResize(nV + 1, 3);\n''',
        '''        int nV = static_cast<int>(V_rest.rows());\n        const int oneStringGridBoundaryDuplicate = nV;\n        V_rest.conservativeResize(nV + 1, 3);\n''',
        "capture Grid boundary duplicate id",
    )

    text = replace_once(
        text,
        '''            }\n        }\n    }\n    \n    void TriMesh::mergeBoundaryEdges''',
        '''            }\n        }\n        if(oneStringGridEncodedBoundary) {\n            oneStringGridLockedVert.insert(vI_boundary);\n            oneStringGridLockedVert.insert(oneStringGridBoundaryDuplicate);\n            oneStringGridLockedVert.insert(vI_interior);\n            fixedVert.insert(vI_boundary);\n            fixedVert.insert(oneStringGridBoundaryDuplicate);\n            fixedVert.insert(vI_interior);\n            if(!checkInversion(true)) {\n                throw std::runtime_error("ONESTRING_GRID_APPLIED_BOUNDARY_CUT_INVERTED");\n            }\n        }\n    }\n    \n    void TriMesh::mergeBoundaryEdges''',
        "lock applied Grid boundary cut",
    )

    path.write_text(text, encoding="utf-8")
    print(f"Applied OneString native Grid-OptCuts V4 TriMesh patch: {path}")
    return True


def patch_main(root: Path) -> bool:
    path = root / "src" / "main.cpp"
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    if "ONESTRING_GRID_NATIVE_V4_MAIN" in text:
        print("OneString native Grid-OptCuts V4 main patch already present.")
        return False
    backup_once(path)

    text = replace_once(
        text,
        '''#include <fstream>\n#include <string>\n#include <ctime>\n\n\nEigen::MatrixXd V, UV, N;''',
        '''#include <fstream>\n#include <string>\n#include <ctime>\n#include <cstdlib>\n#include <cmath>\n#include <stdexcept>\n\n''' + MAIN_HELPERS + '''\nEigen::MatrixXd V, UV, N;''',
        "insert Grid main helpers",
    )

    text = replace_once(
        text,
        '''            Eigen::MatrixXd bnd_uv;\n            OptCuts::IglUtils::map_vertices_to_circle(temp.V_rest,\n                                        bnd_stacked.tail(bnd_all[longest_bnd_id].size()),\n                                        bnd_uv);\n            double xOffset = componentI % UVGridDim * 2.1, yOffset = componentI / UVGridDim * 2.1;\n            for(int bnd_uvI = 0; bnd_uvI < bnd_uv.rows(); bnd_uvI++) {\n                bnd_uv(bnd_uvI, 0) += xOffset;\n                bnd_uv(bnd_uvI, 1) += yOffset;\n            }\n''',
        '''            Eigen::MatrixXd bnd_uv;\n            if(oneStringMainGridEnabled() && temp.initSeams.rows() > 0) {\n                if(n_components != 1) {\n                    throw std::runtime_error("ONESTRING_GRID_INITIAL_MULTICOMP_UNSUPPORTED");\n                }\n                bnd_uv = oneStringMainInitialGridBoundary(temp, bnd_all[longest_bnd_id]);\n            }\n            else {\n                OptCuts::IglUtils::map_vertices_to_circle(temp.V_rest,\n                                            bnd_stacked.tail(bnd_all[longest_bnd_id].size()),\n                                            bnd_uv);\n                double xOffset = componentI % UVGridDim * 2.1, yOffset = componentI / UVGridDim * 2.1;\n                for(int bnd_uvI = 0; bnd_uvI < bnd_uv.rows(); bnd_uvI++) {\n                    bnd_uv(bnd_uvI, 0) += xOffset;\n                    bnd_uv(bnd_uvI, 1) += yOffset;\n                }\n            }\n''',
        "Grid initial boundary instead of circle",
    )

    text = replace_once(
        text,
        '''        triSoup.emplace_back(new OptCuts::TriMesh(V, F, UV_Tutte, temp.F, false));\n        outputFolderPath += meshName + "_Tutte_" + OptCuts::IglUtils::rtos(lambda_init) + "_" + OptCuts::IglUtils::rtos(testID) +\n''',
        '''        if(oneStringMainGridEnabled() && temp.initSeams.rows() > 0) {\n            OptCuts::TriMesh probe(V, F, UV_Tutte, temp.F, false);\n            if(!probe.checkInversion(true)) {\n                // A global reflection reverses orientation while preserving the same\n                // h-lattice (reflection is about the configured fabrication-v phase).\n                oneStringMainReflectFabricationV(UV_Tutte);\n                OptCuts::TriMesh reflectedProbe(V, F, UV_Tutte, temp.F, false);\n                if(!reflectedProbe.checkInversion(true)) {\n                    throw std::runtime_error("ONESTRING_GRID_INITIAL_HARMONIC_INVERTED");\n                }\n            }\n        }\n        OptCuts::TriMesh* oneStringInitialMesh = new OptCuts::TriMesh(V, F, UV_Tutte, temp.F, false);\n        if(oneStringMainGridEnabled() && temp.initSeams.rows() > 0) {\n            std::set<int> fixed = oneStringInitialMesh->fixedVert;\n            for(int i = 0; i < bnd_stacked.size(); ++i) {\n                oneStringInitialMesh->oneStringGridLockedVert.insert(bnd_stacked[i]);\n                fixed.insert(bnd_stacked[i]);\n            }\n            oneStringInitialMesh->resetFixedVert(fixed);\n        }\n        triSoup.emplace_back(oneStringInitialMesh);\n        outputFolderPath += meshName + "_Tutte_" + OptCuts::IglUtils::rtos(lambda_init) + "_" + OptCuts::IglUtils::rtos(testID) +\n''',
        "lock Grid initial seam after harmonic map",
    )

    text = replace_once(
        text,
        '''    optimizer = new OptCuts::Optimizer(*triSoup[0], energyTerms, energyParams, 0, false, bijectiveParam && !rand1PInitCut); // for random one point initial cut, don't need air meshes in the beginning since it's impossible for a quad to intersect itself\n    \n    optimizer->precompute();\n''',
        '''    if(oneStringMainGridEnabled()) {\n        std::cout << "[ONESTRING-GRID] native_candidate_search enabled version=4 h="\n                  << oneStringMainGridH() << std::endl;\n    }\n    optimizer = new OptCuts::Optimizer(*triSoup[0], energyTerms, energyParams, 0, false, bijectiveParam && !rand1PInitCut); // for random one point initial cut, don't need air meshes in the beginning since it's impossible for a quad to intersect itself\n    \n    optimizer->precompute();\n''',
        "emit native V4 marker",
    )

    text = replace_once(
        text,
        '''    double measure_bound = E_SD;\n    const double eps_lambda = std::min(1.0e-3, std::abs(updateLambda(measure_bound) - energyParams[0]));\n    \n    //TODO?: stop when first violates bounds from feasible, don't go to best feasible. check after each merge whether distortion is violated\n''',
        '''    double measure_bound = E_SD;\n    const double eps_lambda = std::min(1.0e-3, std::abs(updateLambda(measure_bound) - energyParams[0]));\n\n    if(oneStringMainGridEnabled()) {\n        // Grid V4 is split-only.  Keep OptCuts' dual weight update, but never\n        // request an unconstrained merge.  When the distortion bound is not met\n        // and normal boundary/interior queries both stalled, select the best\n        // already-evaluated feasible Grid split under the updated lambda.\n        if(checkConvergence && measure_bound <= upperBound) {\n            optimizer->updateEnergyData(true, false, false);\n            return false;\n        }\n        energyParams[0] = updateLambda(measure_bound);\n        energyParams[0] = std::max(eps_lambda, std::min(1.0 - eps_lambda, energyParams[0]));\n        if(checkConvergence && measure_bound > upperBound) {\n            int bestB = -1, bestI = -1;\n            double changeB = __DBL_MAX__, changeI = __DBL_MAX__;\n            for(int attempt = 0; attempt < 64; ++attempt) {\n                bestB = energyChanges_bSplit.empty() ? -1 :\n                    computeBestCand(energyChanges_bSplit, 1.0 - energyParams[0], changeB);\n                bestI = energyChanges_iSplit.empty() ? -1 :\n                    computeBestCand(energyChanges_iSplit, 1.0 - energyParams[0], changeI);\n                if((bestB >= 0 && changeB <= 0.0) || (bestI >= 0 && changeI <= 0.0)) break;\n                const double next = updateLambda(measure_bound, energyParams[0]);\n                if(std::abs(next - energyParams[0]) <= 1.0e-12) break;\n                energyParams[0] = std::max(eps_lambda, std::min(1.0 - eps_lambda, next));\n            }\n            const bool useB = bestB >= 0 && changeB <= 0.0 &&\n                (bestI < 0 || changeI > 0.0 || changeB <= changeI);\n            const bool useI = bestI >= 0 && changeI <= 0.0 && !useB;\n            if(useB) {\n                opType_queried = 0;\n                path_queried = paths_bSplit[bestB];\n                newVertPos_queried = newVertPoses_bSplit[bestB];\n            }\n            else if(useI) {\n                opType_queried = 1;\n                path_queried = paths_iSplit[bestI];\n                newVertPos_queried = newVertPoses_iSplit[bestI];\n            }\n            else {\n                std::cout << "[ONESTRING-GRID] no feasible Grid split remains before requested distortion bound" << std::endl;\n                optimizer->updateEnergyData(true, false, false);\n                return false;\n            }\n        }\n        optimizer->updateEnergyData(true, false, false);\n        return true;\n    }\n    \n    //TODO?: stop when first violates bounds from feasible, don't go to best feasible. check after each merge whether distortion is violated\n''',
        "Grid split-only dual update",
    )

    path.write_text(text, encoding="utf-8")
    print(f"Applied OneString native Grid-OptCuts V4 main patch: {path}")
    return True


def apply_native_grid_patch(root: Path) -> bool:
    root = root.expanduser().resolve()
    changed = False
    changed |= patch_header(root)
    changed |= patch_trimesh(root)
    changed |= patch_main(root)
    return changed


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    apply_native_grid_patch(args.root)
