// kernels.hpp -- the arithmetic of the solver on one rank's block.
//
// Every routine works on a sub-range [i0,i1) x [j0,j1) x [k0,k1) of the local
// interior so that the "overlap" exchange mode can update the deep interior while
// halo messages are in flight and then finish the outer layer afterwards.
// The OpenMP pragmas are ignored when compiled without -fopenmp, which is how the
// serial and pure-MPI binaries are produced from the same source.
//
// Two families of stencil kernels:
//   plain  (jacobi_sweep, apply_A, residual): constant coefficients, every cell is
//          an unknown -- the manufactured Poisson problem used for the benchmarks.
//   masked (jacobi_sweep_m, apply_A_m, residual_m): the satellite gap filling. A
//          byte mask marks each cell unknown / observed / land (gridio.hpp). Only
//          unknown cells are updated; land neighbours are skipped (zero-flux
//          coast), so the diagonal is the sum of the weights of the active
//          neighbours and is recomputed on the fly from the neighbours' mask bytes.
#pragma once
#include <cstdint>
#include "common.hpp"

namespace poisson {

struct Range { int i0, i1, j0, j1, k0, k1; };
inline Range full(const Local& L) { return {1, L.nx + 1, 1, L.ny + 1, 1, L.nz + 1}; }

// One Jacobi sweep: u_new = (f + sum_d ihd2*(u_old[-d] + u_old[+d])) / diag
inline void jacobi_sweep(const Local& L, const double* __restrict u, double* __restrict un,
                         const double* __restrict f, Range r) {
    const long sy = L.sy, sz = L.sz;
    const double ihx2 = L.ihx2, ihy2 = L.ihy2, ihz2 = L.ihz2, inv_diag = 1.0 / L.diag;
#pragma omp parallel for collapse(2) schedule(static)
    for (int k = r.k0; k < r.k1; ++k)
        for (int j = r.j0; j < r.j1; ++j) {
            const size_t row = L.idx(0, j, k);
#pragma omp simd
            for (int i = r.i0; i < r.i1; ++i) {
                const size_t c = row + i;
                un[c] = (f[c] + ihx2 * (u[c - 1] + u[c + 1])
                              + ihy2 * (u[c - sy] + u[c + sy])
                              + ihz2 * (u[c - sz] + u[c + sz])) * inv_diag;
            }
        }
}

// Matrix-free operator: Ap = diag*p - sum_d ihd2*(p[-d] + p[+d])   (this is -laplace_h p)
inline void apply_A(const Local& L, const double* __restrict p, double* __restrict Ap, Range r) {
    const long sy = L.sy, sz = L.sz;
    const double ihx2 = L.ihx2, ihy2 = L.ihy2, ihz2 = L.ihz2, diag = L.diag;
#pragma omp parallel for collapse(2) schedule(static)
    for (int k = r.k0; k < r.k1; ++k)
        for (int j = r.j0; j < r.j1; ++j) {
            const size_t row = L.idx(0, j, k);
#pragma omp simd
            for (int i = r.i0; i < r.i1; ++i) {
                const size_t c = row + i;
                Ap[c] = diag * p[c] - ihx2 * (p[c - 1] + p[c + 1])
                                    - ihy2 * (p[c - sy] + p[c + sy])
                                    - ihz2 * (p[c - sz] + p[c + sz]);
            }
        }
}

// Residual r = f - A u over the whole interior, returning the local sum of squares.
inline double residual(const Local& L, const double* __restrict u, const double* __restrict f,
                       double* __restrict res) {
    const long sy = L.sy, sz = L.sz;
    const double ihx2 = L.ihx2, ihy2 = L.ihy2, ihz2 = L.ihz2, diag = L.diag;
    double sum = 0.0;
#pragma omp parallel for collapse(2) schedule(static) reduction(+:sum)
    for (int k = 1; k <= L.nz; ++k)
        for (int j = 1; j <= L.ny; ++j) {
            const size_t row = L.idx(0, j, k);
            for (int i = 1; i <= L.nx; ++i) {
                const size_t c = row + i;
                const double Au = diag * u[c] - ihx2 * (u[c - 1] + u[c + 1])
                                              - ihy2 * (u[c - sy] + u[c + sy])
                                              - ihz2 * (u[c - sz] + u[c + sz]);
                res[c] = f[c] - Au;
                sum += res[c] * res[c];
            }
        }
    return sum;
}

// ---- masked kernels (satellite gap filling) --------------------------------
// Mask codes are those of gridio.hpp: 0 unknown, 1 observed, 2 land, 3 hidden.
// Unknown and hidden cells are solved; observed and land cells keep the value 0.

// Weighted sum over the active (non-land) neighbours and the matching diagonal.
#define POISSON_MASKED_NEIGHBOURS(m, v, c, sy, sz, ihx2, ihy2, ihz2, s, D)      \
    double s = 0.0, D = 0.0;                                                    \
    if (m[c - 1]  != 2) { D += ihx2; s += ihx2 * v[c - 1];  }                   \
    if (m[c + 1]  != 2) { D += ihx2; s += ihx2 * v[c + 1];  }                   \
    if (m[c - sy] != 2) { D += ihy2; s += ihy2 * v[c - sy]; }                   \
    if (m[c + sy] != 2) { D += ihy2; s += ihy2 * v[c + sy]; }                   \
    if (ihz2 > 0.0) {                                                           \
        if (m[c - sz] != 2) { D += ihz2; s += ihz2 * v[c - sz]; }               \
        if (m[c + sz] != 2) { D += ihz2; s += ihz2 * v[c + sz]; }               \
    }

inline void jacobi_sweep_m(const Local& L, const uint8_t* __restrict m, const double* __restrict u,
                           double* __restrict un, const double* __restrict f, Range r) {
    const long sy = L.sy, sz = L.sz;
    const double ihx2 = L.ihx2, ihy2 = L.ihy2, ihz2 = L.ihz2;
#pragma omp parallel for collapse(2) schedule(static)
    for (int k = r.k0; k < r.k1; ++k)
        for (int j = r.j0; j < r.j1; ++j) {
            const size_t row = L.idx(0, j, k);
            for (int i = r.i0; i < r.i1; ++i) {
                const size_t c = row + i;
                if (m[c] != 0 && m[c] != 3) continue;          // observed or land: stays 0
                POISSON_MASKED_NEIGHBOURS(m, u, c, sy, sz, ihx2, ihy2, ihz2, s, D)
                un[c] = D > 0.0 ? (f[c] + s) / D : 0.0;
            }
        }
}

inline void apply_A_m(const Local& L, const uint8_t* __restrict m, const double* __restrict p,
                      double* __restrict Ap, Range r) {
    const long sy = L.sy, sz = L.sz;
    const double ihx2 = L.ihx2, ihy2 = L.ihy2, ihz2 = L.ihz2;
#pragma omp parallel for collapse(2) schedule(static)
    for (int k = r.k0; k < r.k1; ++k)
        for (int j = r.j0; j < r.j1; ++j) {
            const size_t row = L.idx(0, j, k);
            for (int i = r.i0; i < r.i1; ++i) {
                const size_t c = row + i;
                if (m[c] != 0 && m[c] != 3) { Ap[c] = 0.0; continue; }
                POISSON_MASKED_NEIGHBOURS(m, p, c, sy, sz, ihx2, ihy2, ihz2, s, D)
                Ap[c] = D > 0.0 ? D * p[c] - s : p[c];          // isolated pond: identity row
            }
        }
}

inline double residual_m(const Local& L, const uint8_t* __restrict m, const double* __restrict u,
                         const double* __restrict f, double* __restrict res) {
    const long sy = L.sy, sz = L.sz;
    const double ihx2 = L.ihx2, ihy2 = L.ihy2, ihz2 = L.ihz2;
    double sum = 0.0;
#pragma omp parallel for collapse(2) schedule(static) reduction(+:sum)
    for (int k = 1; k <= L.nz; ++k)
        for (int j = 1; j <= L.ny; ++j) {
            const size_t row = L.idx(0, j, k);
            for (int i = 1; i <= L.nx; ++i) {
                const size_t c = row + i;
                if (m[c] != 0 && m[c] != 3) { res[c] = 0.0; continue; }
                POISSON_MASKED_NEIGHBOURS(m, u, c, sy, sz, ihx2, ihy2, ihz2, s, D)
                const double Au = D > 0.0 ? D * u[c] - s : u[c];
                res[c] = f[c] - Au;
                sum += res[c] * res[c];
            }
        }
    return sum;
}
#undef POISSON_MASKED_NEIGHBOURS

// ---- vector operations (shared by both problems) ---------------------------

// Local dot product over the interior.
inline double dot(const Local& L, const double* __restrict a, const double* __restrict b) {
    double sum = 0.0;
#pragma omp parallel for collapse(2) schedule(static) reduction(+:sum)
    for (int k = 1; k <= L.nz; ++k)
        for (int j = 1; j <= L.ny; ++j) {
            const size_t row = L.idx(0, j, k);
            for (int i = 1; i <= L.nx; ++i) sum += a[row + i] * b[row + i];
        }
    return sum;
}

// y = y + alpha * x over the interior.
inline void axpy(const Local& L, double alpha, const double* __restrict x, double* __restrict y) {
#pragma omp parallel for collapse(2) schedule(static)
    for (int k = 1; k <= L.nz; ++k)
        for (int j = 1; j <= L.ny; ++j) {
            const size_t row = L.idx(0, j, k);
#pragma omp simd
            for (int i = 1; i <= L.nx; ++i) y[row + i] += alpha * x[row + i];
        }
}

// p = r + beta * p over the interior (CG direction update).
inline void xpay(const Local& L, const double* __restrict r, double beta, double* __restrict p) {
#pragma omp parallel for collapse(2) schedule(static)
    for (int k = 1; k <= L.nz; ++k)
        for (int j = 1; j <= L.ny; ++j) {
            const size_t row = L.idx(0, j, k);
#pragma omp simd
            for (int i = 1; i <= L.nx; ++i) p[row + i] = r[row + i] + beta * p[row + i];
        }
}

// Fill f with the right-hand side and zero everything else (u = 0 initial guess, zero halos).
inline void init_problem(const Params& P, const Local& L, double* f) {
#pragma omp parallel for collapse(2) schedule(static)
    for (int k = 1; k <= L.nz; ++k)
        for (int j = 1; j <= L.ny; ++j)
            for (int i = 1; i <= L.nx; ++i)
                f[L.idx(i, j, k)] = rhs_at(P, L, L.ox + i, L.oy + j, L.oz + k);
}

// Local error sums against the manufactured solution: returns {sum of squares, max abs}.
inline void error_norms(const Params& P, const Local& L, const double* u, double& sumsq, double& maxabs) {
    double s = 0.0, m = 0.0;
#pragma omp parallel for collapse(2) schedule(static) reduction(+:s) reduction(max:m)
    for (int k = 1; k <= L.nz; ++k)
        for (int j = 1; j <= L.ny; ++j)
            for (int i = 1; i <= L.nx; ++i) {
                const double e = u[L.idx(i, j, k)] - exact_at(P, L, L.ox + i, L.oy + j, L.oz + k);
                s += e * e;
                if (std::fabs(e) > m) m = std::fabs(e);
            }
    sumsq = s; maxabs = m;
}

// Helper used by the overlap mode: apply `fn` to the outer layer of the interior,
// i.e. the cells that touch a halo. The six slabs are chosen so that no cell is
// visited twice. `fn(Range)` is any of the range kernels above.
// 2D or 3D is taken from L.dim, not from nz: a 3D block can be one layer thick
// (for example 16 ranks along z on a 16^3 grid), and every one of its cells then
// touches a z-halo, so none of them may be updated before the messages arrive.
template <class Fn>
inline void for_each_boundary_slab(const Local& L, Fn fn) {
    const int nx = L.nx, ny = L.ny, nz = L.nz;
    // x-faces: full y,z extent
    fn(Range{1, 2, 1, ny + 1, 1, nz + 1});
    if (nx > 1) fn(Range{nx, nx + 1, 1, ny + 1, 1, nz + 1});
    if (nx > 2) {
        // y-faces: x in [2,nx)
        fn(Range{2, nx, 1, 2, 1, nz + 1});
        if (ny > 1) fn(Range{2, nx, ny, ny + 1, 1, nz + 1});
        if (ny > 2 && L.dim == 3) {
            // z-faces: x in [2,nx), y in [2,ny)
            fn(Range{2, nx, 2, ny, 1, 2});
            if (nz > 1) fn(Range{2, nx, 2, ny, nz, nz + 1});
        }
    }
}
// The deep interior, i.e. cells that do not touch any halo. Empty if the block is too thin
// (in 3D that includes any block with fewer than 3 layers in some direction).
inline Range deep_interior(const Local& L) {
    if (L.dim == 2) return Range{2, L.nx, 2, L.ny, 1, 2};      // 2D: a single z-layer, no z-halo
    return Range{2, L.nx, 2, L.ny, 2, L.nz};
}
inline bool has_deep_interior(const Local& L) {
    Range r = deep_interior(L);
    return r.i1 > r.i0 && r.j1 > r.j0 && r.k1 > r.k0;
}

} // namespace poisson
