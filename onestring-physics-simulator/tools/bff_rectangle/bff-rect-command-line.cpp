// Rectangle-constrained Boundary First Flattening CLI (independent adapter).
//
// This program LINKs the unmodified official GeometryCollective BFF library.
// BFF::flatten(k, false) treats k as prescribed boundary EXTERIOR angles.
// Exactly 4 boundary vertices receive pi/2 and every other vertex receives
// zero, yielding a rectangular boundary up to floating-point tolerance.
// This is not a harmonic/LSCM boundary substitution.
//
// Source API: https://github.com/GeometryCollective/boundary-first-flattening
// OneString adapter under the parent project's license.
#include <cstdint>
#include "bff/project/Bff.h"
#include "bff/mesh/MeshIO.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace bff;

int main(int argc, char** argv) {
    const bool customCornerPositions =
        argc == 17 && std::string(argv[4]) == "--corner-xyz";
    if ((argc != 4 && argc != 6 && !customCornerPositions) ||
        std::string(argv[3]) != "--writeOnlyUVs" ||
        (argc == 6 && std::string(argv[4]) != "--corner-phase")) {
        std::cerr << "Usage: bff-rect-command-line input.obj output.obj --writeOnlyUVs "
                  << "[--corner-phase frac | --corner-xyz x0 y0 z0 x1 y1 z1 "
                  << "x2 y2 z2 x3 y3 z3]\\n";
        return EXIT_FAILURE;
    }
    // The original quarter-arclength behavior remains the default.
    const double cornerPhase = argc == 6 ? std::atof(argv[5]) : 0.0;
    if (!std::isfinite(cornerPhase) || cornerPhase < 0.0 || cornerPhase >= 0.25) {
        std::cerr << "--corner-phase must be in [0, 0.25)\\n";
        return EXIT_FAILURE;
    }
    std::string error;
    Model model;
    if (!MeshIO::read(argv[1], model, error)) {
        std::cerr << "BFF input failed: " << error << "\n";
        return EXIT_FAILURE;
    }
    if (model.size() != 1 || model[0].boundaries.size() != 1) {
        std::cerr << "Rectangular BFF requires exactly one open disk component\n";
        return EXIT_FAILURE;
    }
    Mesh& mesh = model[0];
    BFF bff(mesh);
    std::vector<WedgeCIter> boundary;
    for (WedgeCIter w : mesh.cutBoundary()) {
        boundary.push_back(w);
    }
    const int n = static_cast<int>(boundary.size());
    if (n < 4 || bff.data->bN != n) {
        std::cerr << "Rectangular BFF requires at least four distinct boundary wedges\n";
        return EXIT_FAILURE;
    }

    // Choose four ordered corner vertices at approximately 0, 25, 50,
    // and 75 percent of the 3D boundary arclength. Ensure all distinct
    // even on meshes with nonuniform boundary vertex density.
    std::vector<double> cumulative(n+1, 0.0);
    for (int i=0; i<n; ++i) {
        Vector a = boundary[i]->vertex()->position;
        Vector b = boundary[(i+1)%n]->vertex()->position;
        const double length = (b-a).norm();
        if (!std::isfinite(length) || length <= 0.0) {
            std::cerr << "Degenerate BFF boundary edge\n";
            return EXIT_FAILURE;
        }
        cumulative[i+1] = cumulative[i] + length;
    }
    const double perimeter = cumulative[n];
    if (!std::isfinite(perimeter) || perimeter <= 0.0) {
        std::cerr << "Invalid BFF boundary perimeter\n";
        return EXIT_FAILURE;
    }
    std::vector<int> corners(4,0);
    if (customCornerPositions) {
        // Candidate corners come from PCA extrema on the actual 3D source
        // boundary. Passing xyz avoids relying on the OBJ/BFF wedge index
        // ordering, which may differ across implementations.
        for (int k=0; k<4; ++k) {
            double values[3];
            for (int d=0; d<3; ++d) {
                char* end=nullptr;
                values[d]=std::strtod(argv[5+3*k+d],&end);
                if (!end || *end!='\0' || !std::isfinite(values[d])) {
                    std::cerr << "Invalid explicit boundary corner coordinate\n";
                    return EXIT_FAILURE;
                }
            }
            int best=-1;
            double minDist=std::numeric_limits<double>::infinity();
            for (int i=0; i<n; ++i) {
                // MeshIO::read() normalizes 3D positions (unit radius)
                // internally. Convert back to the ORIGINAL OBJ coordinates
                // before matching the source-boundary corner sent by Python.
                Vector v=boundary[i]->vertex()->position * mesh.radius + mesh.cm;
                double dist=0.;
                for (int d=0; d<3; ++d) {
                    const double delta=v[d]-values[d];
                    dist+=delta*delta;
                }
                if (dist<minDist) { minDist=dist; best=i; }
            }
            if (best<0 || minDist>1e-12*perimeter*perimeter) {
                std::cerr << "Explicit corner is not an original boundary vertex\n";
                return EXIT_FAILURE;
            }
            corners[k]=best;
        }
        std::sort(corners.begin(),corners.end());
        if (std::unique(corners.begin(),corners.end())!=corners.end()) {
            std::cerr << "Explicit rectangle corners are not distinct\n";
            return EXIT_FAILURE;
        }
    } else {
        int previous=-1;
        for (int k=0; k<4; ++k) {
            int best=-1;
            double best_error=std::numeric_limits<double>::infinity();
            const int last_allowed=n-(4-k);
            for (int i=previous+1; i<=last_allowed; ++i) {
                const double discrepancy=std::abs(cumulative[i]-perimeter*(cornerPhase+k/4.0));
                if (discrepancy<best_error) {
                    best_error=discrepancy;
                    best=i;
                }
            }
            if (best<0) {
                std::cerr << "Cannot select four noncoincident rectangle corners\n";
                return EXIT_FAILURE;
            }
            corners[k]=best;
            previous=best;
        }
    }

    // BFF::flatten(..., false) uses its protected meanScaling gauge,
    // initialized only by the scale-factor overload. Call that official
    // overload once first; this avoids uninitialized state or poking
    // protected library fields, without modifying upstream source.
    DenseMatrix zeroScale(bff.data->bN);
    if (!bff.flatten(zeroScale, true)) {
        std::cerr << "Official BFF gauge initialization failed\\n";
        return EXIT_FAILURE;
    }
    DenseMatrix exterior(bff.data->bN);
    for (int k=0; k<4; ++k) {
        const int idx=bff.data->bIndex[boundary[corners[k]]];
        exterior(idx) = 0.5 * M_PI;
    }
    if (!bff.flatten(exterior, false)) {
        std::cerr << "Official BFF prescribed exterior-angle solve failed\n";
        return EXIT_FAILURE;
    }
    // Do not request PCA reorientation or an additional UV map:
    // only the conformal flattening returned by BFF::flatten is used.
    // IMPORTANT: upstream MeshIO::writeOBJ uses std::to_string(uv.x/y),
    // rounding UV vertices to ~6 decimals. Small valid UV triangles can
    // thereby become EXACTLY collinear or change orientation in the OBJ
    // consumed by Python (e.g. source face 275). Preserve the same upstream
    // packing, UV ordering and face indices, but serialize with round-trip
    // double precision. The BFF solve and UV geometry are NOT modified.
    std::vector<Vector> originalUvIslandCenters, newUvIslandCenters;
    std::vector<uint8_t> isUvIslandFlipped;
    Vector modelMinBounds, modelMaxBounds;
    const std::vector<uint8_t> isSurfaceMappedToSphere(1, 0);
    MeshIO::packUvs(model, 1.0, isSurfaceMappedToSphere,
                    originalUvIslandCenters, newUvIslandCenters,
                    isUvIslandFlipped, modelMinBounds, modelMaxBounds);
    std::vector<Vector> positions, uvs;
    std::vector<int> vIndices, uvIndices, indicesOffset;
    MeshIO::collectModelUvs(
        model, false, isSurfaceMappedToSphere,
        originalUvIslandCenters, newUvIslandCenters, isUvIslandFlipped,
        modelMinBounds, modelMaxBounds, positions, uvs, vIndices,
        uvIndices, indicesOffset);
    std::ofstream out(argv[2]);
    if (!out.is_open() || uvs.empty() || indicesOffset.size() < 2) {
        std::cerr << "Failed to prepare full-precision BFF rectangle OBJ\n";
        return EXIT_FAILURE;
    }
    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (const Vector& uv : uvs) {
        if (!std::isfinite(uv.x) || !std::isfinite(uv.y)) {
            std::cerr << "Official BFF produced non-finite UVs\n";
            return EXIT_FAILURE;
        }
        out << "v " << uv.x << " " << uv.y << " 0\n";
    }
    for (size_t face=0; face+1<indicesOffset.size(); ++face) {
        out << "f";
        for (int j=indicesOffset[face]; j<indicesOffset[face+1]; ++j) {
            if (uvIndices[j]<0 || static_cast<size_t>(uvIndices[j])>=uvs.size()) {
                std::cerr << "Invalid BFF UV face index\n";
                return EXIT_FAILURE;
            }
            out << " " << (uvIndices[j]+1);
        }
        out << "\n";
    }
    out.close();
    if (!out.good()) {
        std::cerr << "Writing full-precision BFF rectangle OBJ failed\n";
        return EXIT_FAILURE;
    }
    std::cerr << "BFF_RECTANGLE_OUTPUT_PRECISION="
              << std::numeric_limits<double>::max_digits10 << "\n";
    std::cerr << "BFF_RECTANGLE_CORNERS="
              << corners[0] << "," << corners[1] << ","
              << corners[2] << "," << corners[3] << "\n";
    std::cerr << "BFF_RECTANGLE_CORNER_PHASE=" << cornerPhase << "\\n";
    return EXIT_SUCCESS;
}
