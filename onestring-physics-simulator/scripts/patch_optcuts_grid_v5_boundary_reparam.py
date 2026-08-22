#!/usr/bin/env python3
"""V5: treat Grid seams as UV boundary conditions, not surface-space Grid cuts.

OptCuts still proposes a physical cut topology on the source surface.  Once a
cut is accepted, its two UV boundary copies are already assigned H/V lattice
coordinates by the native Grid candidate encoder.  V5 then freezes the complete
current UV boundary, re-solves only the interior parameterization, and keeps all
cohesive-seam vertices fixed during subsequent Symmetric Dirichlet optimization.

This is deliberately *not* post-hoc snapping: the Grid seam is a Dirichlet
boundary condition of the parameterization solve itself.
"""
from __future__ import annotations

from pathlib import Path

MARKER = "ONESTRING_GRID_NATIVE_V5_BOUNDARY_REPARAM"


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


def apply_boundary_reparameterization_patch(root: Path) -> bool:
    root = root.expanduser().resolve()
    header = root / "src" / "TriMesh.hpp"
    tri = root / "src" / "TriMesh.cpp"
    optimizer = root / "src" / "Optimizer.cpp"
    for path in (header, tri, optimizer):
        if not path.is_file():
            raise FileNotFoundError(path)

    changed = False

    # Public API so Optimizer can invoke the constrained mapping after it copies
    # the initial mesh and after an accepted topology operation.
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

        // The source-surface seam topology is represented by cohesive edge pairs.
        // Their UV copies must already lie on H/V Grid edges before solving.
        // Those seam coordinates become hard Dirichlet data; they are never
        // snapped after the parameterization has been computed.
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

        // Fix the complete current UV boundary during the harmonic solve.  This
        // keeps true source boundaries stable as well, while only cohesive seam
        // vertices are made persistent Grid locks for later SD optimization.
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

        // Re-impose boundary values exactly rather than relying on solver
        // roundoff.  This is important because Grid membership is a fabrication
        // invariant, not a soft numerical preference.
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

        # Direct splitOrMerge path: only after the selected topology operation is
        # committed.  Candidate enumeration remains cheap.
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
        # explicit UV boundary condition.  This also re-derives persistent seam
        # locks from the authoritative cohesive topology after mesh copying.
        o = _replace_in_function(
            o,
            "    void Optimizer::precompute(void)",
            "    int Optimizer::solve(int maxIter)",
            """        result = data0;\n""",
            """        result = data0;\n        // ONESTRING_GRID_NATIVE_V5_BOUNDARY_REPARAM\n        if(!result.oneStringReparameterizeGridBoundary()) {\n            throw std::runtime_error(\"ONESTRING_GRID_INITIAL_BOUNDARY_CONSTRAINED_REPARAM_FAILED\");\n        }\n""",
            "Optimizer::precompute V5 constrained mapping",
        )

        # createFracture(opType, ...) bypasses TriMesh::splitOrMerge and directly
        # commits the already-selected split. Reparameterize once, after the full
        # topology edit and feature update, before rebuilding the scaffold/Hessian.
        o = _replace_in_function(
            o,
            "    bool Optimizer::createFracture(int opType, const std::vector<int>& path, const Eigen::MatrixXd& newVertPos, bool allowPropagate)",
            "    bool Optimizer::createFracture(double stressThres, int propType, bool allowPropagate, bool allowInSplit)",
            """        timer.stop();\n        \n        if(scaffolding) {\n""",
            """        timer.stop();\n\n        if(!result.oneStringReparameterizeGridBoundary()) {\n            throw std::runtime_error(\"ONESTRING_GRID_BOUNDARY_CONSTRAINED_REPARAM_FAILED_AFTER_TOPOLOGY\");\n        }\n        \n        if(scaffolding) {\n""",
            "Optimizer::createFracture V5 constrained mapping",
        )

        # Restoring a saved/configured topology must obey the same contract.
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

    return changed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    apply_boundary_reparameterization_patch(args.root)
