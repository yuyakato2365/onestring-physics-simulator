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
#include "bff/project/Bff.h"
#include "bff/mesh/MeshIO.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace bff;

int main(int argc, char** argv) {
    if (argc != 4 || std::string(argv[3]) != "--writeOnlyUVs") {
        std::cerr << "Usage: bff-rect-command-line input.obj output.obj --writeOnlyUVs\n";
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
    int previous=0;
    for (int k=1; k<4; ++k) {
        int best=-1;
        double best_error=std::numeric_limits<double>::infinity();
        const int last_allowed=n-(4-k);
        for (int i=previous+1; i<=last_allowed; ++i) {
            const double discrepancy=std::abs(cumulative[i]-perimeter*k/4.0);
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

    // The original BFF::flatten(..., false) consults meanScaling to
    // fix its additive scale gauge. The headless CLI has no GUI state:
    // initialize it explicitly rather than using an uninitialized double.
    bff.meanScaling=0.0;
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
    if (!MeshIO::write(argv[2], model, std::vector<uint8_t>(1,0),
                       false, true, 1.0)) {
        std::cerr << "Failed to write BFF rectangle output\n";
        return EXIT_FAILURE;
    }
    std::cerr << "BFF_RECTANGLE_CORNERS="
              << corners[0] << "," << corners[1] << ","
              << corners[2] << "," << corners[3] << "\n";
    return EXIT_SUCCESS;
}
