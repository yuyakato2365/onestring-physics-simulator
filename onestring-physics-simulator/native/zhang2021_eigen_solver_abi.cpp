// Sparse linear-system solver backend for the ORIGINAL Zhang et al. (CAD 2021)
// executable. Original quad meshing/optimization algorithm is unchanged.
// This is an independent numerical backend, NOT PARDISO, and never imports,
// patches, decrypts, or circumvents proprietary PARDISO binaries/licenses.
//
// The authors' Segment.exe imports precisely 'pardisoinit' and 'pardiso'
// from libpardiso600-WIN-X86-64.dll. At explicit user opt-in, a separate
// runtime copy may replace that binary with this independently compiled
// Eigen SparseLU implementation. Do not modify/distribute the original
// authors' PARDISO DLL.
//
// Original class passes the 6-argument PARDISO 6 init and 17-argument
// PARDISO 6 solve signatures; DO NOT accidentally use the different Intel
// MKL 3/16-argument signatures.
// SPDX-License-Identifier: MIT (this integration shim only)
#define NOMINMAX
#include <Eigen/SparseLU>
#include <Eigen/SparseCore>
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <map>
#include <memory>
#include <new>
#include <utility>
#include <vector>

using SpMat = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;
using Triplet = Eigen::Triplet<double>;
using DenseMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

struct SolverState {
    SpMat matrix;
    Eigen::SparseLU<SpMat, Eigen::COLAMDOrdering<int>> lu;
    int dimension = -1;
    bool analyzed = false;
    bool factorized = false;
};

// Caller owns 64 pointer slots. The authors' program zero-initializes them
// in pardiso_init(), so state slot 0 is unused prior to first phase 11.
static SolverState *state(void **pt) {
    return pt ? static_cast<SolverState *>(pt[0]) : nullptr;
}
static void set_error(int *error, int code) {
    if (error) *error = code;
}
static bool fill_symmetric_matrix(SolverState &s, int n, const double *values,
                                  const int *row_starts, const int *col_indices,
                                  int mtype) {
    if (n <= 0 || !values || !row_starts || !col_indices || n > 250000) return false;
    // Original Segment's sparse index arrays are converted to 1-based before
    // every phase 11. We reject malformed bounds instead of guessing indexing.
    if (row_starts[0] != 1) return false;
    const int nnz = row_starts[n] - 1;
    if (nnz <= 0 || nnz > 30000000) return false;
    std::map<std::pair<int,int>, std::pair<double,double>> upper_lower;
    std::vector<Triplet> general;
    general.reserve(static_cast<size_t>(nnz));
    for (int i = 0; i < n; ++i) {
        int begin = row_starts[i] - 1, end = row_starts[i+1] - 1;
        if (begin < 0 || end < begin || end > nnz) return false;
        for (int k = begin; k < end; ++k) {
            int j = col_indices[k] - 1;
            double val = values[k];
            if (j < 0 || j >= n || !std::isfinite(val)) return false;
            if (mtype == -2 || mtype == 2 || mtype == 1) {
                auto pair = std::make_pair(std::min(i,j), std::max(i,j));
                auto &item = upper_lower[pair];
                if (i <= j) item.first += val;
                else item.second += val;
            } else {
                general.emplace_back(i,j,val);
            }
        }
    }
    if (mtype == -2 || mtype == 2 || mtype == 1) {
        general.reserve(upper_lower.size() * 2);
        for (const auto &entry : upper_lower) {
            int i = entry.first.first, j = entry.first.second;
            double first = entry.second.first;
            double second = entry.second.second;
            if (i == j) {
                general.emplace_back(i,j,first+second);
            } else {
                // If the input CSR already contains both halves, they must
                // agree. Never double-count symmetric off-diagonal entries.
                const bool has_upper = std::abs(first) > 0;
                const bool has_lower = std::abs(second) > 0;
                if (has_upper && has_lower &&
                    std::abs(first-second) > 1e-7 * std::max({1.,std::abs(first),std::abs(second)}))
                    return false;
                double value = has_upper ? first : second;
                if (value != 0) {
                    general.emplace_back(i,j,value);
                    general.emplace_back(j,i,value);
                }
            }
        }
    }
    s.matrix.resize(n,n);
    s.matrix.setFromTriplets(general.begin(),general.end());
    s.matrix.makeCompressed();
    s.dimension = n;
    return s.matrix.nonZeros() > 0;
}
extern "C" __declspec(dllexport)
void pardisoinit(void *pt_raw, int *mtype, int *solver,
                 int *iparm, double *dparm, int *error) noexcept {
    set_error(error, 0);
    if (!pt_raw || !mtype || !iparm || !dparm || !error) {
        set_error(error, -100);
        return;
    }
    try {
        auto pt = static_cast<void **>(pt_raw);
        std::fill(pt,pt+64,nullptr);
        std::fill(iparm,iparm+64,0);
        std::fill(dparm,dparm+64,0.0);
        iparm[0] = 1;
        if (solver) *solver = 0;
        if (*mtype != -2 && *mtype != 2 && *mtype != 1 && *mtype != 11) {
            *error = -101;
            return;
        }
    } catch (...) {
        set_error(error, -102);
    }
}
extern "C" __declspec(dllexport)
void pardiso(void *pt_raw, int *maxfct, int *mnum, int *mtype, int *phase,
             int *n, double *a, int *ia, int *ja, int *perm, int *nrhs,
             int *iparm, int *msglvl, double *b, double *x,
             int *error, double *dparm) noexcept {
    set_error(error,0);
    if (!pt_raw || !phase || !error) {
        set_error(error,-100);
        return;
    }
    auto pt = static_cast<void **>(pt_raw);
    try {
        if (*phase == -1) {
            delete state(pt);
            pt[0] = nullptr;
            return;
        }
        if (*phase == 0) {
            if (auto s = state(pt)) s->factorized = false;
            return;
        }
        if (!mtype || !n || !nrhs || !iparm || !ia || !ja || !a) {
            *error = -100;
            return;
        }
        if (*phase == 11) {
            if (state(pt)) {
                delete state(pt);
                pt[0] = nullptr;
            }
            auto s = std::make_unique<SolverState>();
            if (!fill_symmetric_matrix(*s,*n,a,ia,ja,*mtype)) {
                *error = -103;
                return;
            }
            s->lu.analyzePattern(s->matrix);
            if (s->lu.info()!=Eigen::Success) {
                *error = -104;
                return;
            }
            s->analyzed = true;
            pt[0] = s.release();
        } else if (*phase == 22) {
            auto s = state(pt);
            if (!s || !s->analyzed || s->dimension != *n) {
                *error = -105;
                return;
            }
            s->lu.factorize(s->matrix);
            if (s->lu.info()!=Eigen::Success) {
                *error = -106;
                return;
            }
            s->factorized = true;
        } else if (*phase == 33) {
            auto s = state(pt);
            if (!s || !s->factorized || s->dimension != *n ||
                    *nrhs <= 0 || *nrhs > 512 || !b || !x) {
                *error = -107;
                return;
            }
            const Eigen::Map<const DenseMat> rhs(b,*n,*nrhs);
            DenseMat solution = s->lu.solve(rhs);
            if (s->lu.info()!=Eigen::Success || !solution.allFinite()) {
                *error = -108;
                return;
            }
            const double den = std::max(1.,rhs.norm());
            const double relative_residual = (s->matrix*solution-rhs).norm()/den;
            if (!std::isfinite(relative_residual) || relative_residual > 1e-6) {
                *error = -109;
                return;
            }
            std::copy(solution.data(),solution.data()+solution.size(),x);
        } else {
            *error = -110;
        }
    } catch (...) {
        // No C++ exception may propagate across the old DLL's C ABI.
        set_error(error,-111);
    }
}
