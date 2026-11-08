// gpu_kernels.cuh -- CUDA versions of the stencil kernels in kernels.hpp.
//
// Thread mapping: a 3D grid of 3D blocks covers the sub-range [i0,i1)x[j0,j1)x[k0,k1)
// of the local block; thread (x,y,z) handles cell (i0+x, j0+y, k0+z). Because i is
// the contiguous direction, consecutive threads in x touch consecutive addresses,
// which gives coalesced memory access.
// Reductions (dot, residual) use a grid-stride loop: a fixed number of threads
// each sums a strided subset, then adds its partial sum to one device double with
// atomicAdd. This is simple, deterministic in cost, and needs no shared memory.
#pragma once
#include "cuda_compat.hpp"

namespace gpu {

struct Geo {                     // plain-old-data copy of what the kernels need from Local
    int nx, ny, nz; long sy, sz;
    double ihx2, ihy2, ihz2, diag;
};
struct Box { int i0, i1, j0, j1, k0, k1; };

__device__ inline size_t cell(const Geo& g, int i, int j, int k) { return (size_t)k * g.sz + (size_t)j * g.sy + (size_t)i; }

__global__ void k_jacobi(Geo g, Box b, const double* u, double* un, const double* f) {
    int i = b.i0 + blockIdx.x * blockDim.x + threadIdx.x;
    int j = b.j0 + blockIdx.y * blockDim.y + threadIdx.y;
    int k = b.k0 + blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= b.i1 || j >= b.j1 || k >= b.k1) return;
    size_t c = cell(g, i, j, k);
    un[c] = (f[c] + g.ihx2 * (u[c - 1] + u[c + 1]) + g.ihy2 * (u[c - g.sy] + u[c + g.sy])
                  + g.ihz2 * (u[c - g.sz] + u[c + g.sz])) / g.diag;
}

__global__ void k_applyA(Geo g, Box b, const double* p, double* Ap) {
    int i = b.i0 + blockIdx.x * blockDim.x + threadIdx.x;
    int j = b.j0 + blockIdx.y * blockDim.y + threadIdx.y;
    int k = b.k0 + blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= b.i1 || j >= b.j1 || k >= b.k1) return;
    size_t c = cell(g, i, j, k);
    Ap[c] = g.diag * p[c] - g.ihx2 * (p[c - 1] + p[c + 1]) - g.ihy2 * (p[c - g.sy] + p[c + g.sy])
                          - g.ihz2 * (p[c - g.sz] + p[c + g.sz]);
}

// y += alpha x  over the interior
__global__ void k_axpy(Geo g, double alpha, const double* x, double* y) {
    int i = 1 + blockIdx.x * blockDim.x + threadIdx.x;
    int j = 1 + blockIdx.y * blockDim.y + threadIdx.y;
    int k = 1 + blockIdx.z * blockDim.z + threadIdx.z;
    if (i > g.nx || j > g.ny || k > g.nz) return;
    size_t c = cell(g, i, j, k);
    y[c] += alpha * x[c];
}
// p = r + beta p  over the interior
__global__ void k_xpay(Geo g, const double* r, double beta, double* p) {
    int i = 1 + blockIdx.x * blockDim.x + threadIdx.x;
    int j = 1 + blockIdx.y * blockDim.y + threadIdx.y;
    int k = 1 + blockIdx.z * blockDim.z + threadIdx.z;
    if (i > g.nx || j > g.ny || k > g.nz) return;
    size_t c = cell(g, i, j, k);
    p[c] = r[c] + beta * p[c];
}

// Grid-stride dot product over the interior; result accumulated into *out (must be zeroed first).
__global__ void k_dot(Geo g, const double* a, const double* b, double* out) {
    const long n = (long)g.nx * g.ny * g.nz;
    const long stride = (long)gridDim.x * blockDim.x;
    double s = 0.0;
    for (long t = blockIdx.x * blockDim.x + threadIdx.x; t < n; t += stride) {
        int i = 1 + (int)(t % g.nx);
        long rest = t / g.nx;
        int j = 1 + (int)(rest % g.ny);
        int k = 1 + (int)(rest / g.ny);
        size_t c = cell(g, i, j, k);
        s += a[c] * b[c];
    }
    atomicAdd(out, s);
}

// r = f - A u over the interior, sum of squares into *out.
__global__ void k_residual(Geo g, const double* u, const double* f, double* r, double* out) {
    const long n = (long)g.nx * g.ny * g.nz;
    const long stride = (long)gridDim.x * blockDim.x;
    double s = 0.0;
    for (long t = blockIdx.x * blockDim.x + threadIdx.x; t < n; t += stride) {
        int i = 1 + (int)(t % g.nx);
        long rest = t / g.nx;
        int j = 1 + (int)(rest % g.ny);
        int k = 1 + (int)(rest / g.ny);
        size_t c = cell(g, i, j, k);
        double Au = g.diag * u[c] - g.ihx2 * (u[c - 1] + u[c + 1]) - g.ihy2 * (u[c - g.sy] + u[c + g.sy])
                                  - g.ihz2 * (u[c - g.sz] + u[c + g.sz]);
        r[c] = f[c] - Au;
        s += r[c] * r[c];
    }
    atomicAdd(out, s);
}

// Faces. dim d in {0,1,2}, plane index pos along d. The face is the interior of the
// other two dimensions, flattened as t = a + A*b where (a,b) are those two indices.
__device__ inline void face_cell(const Geo& g, int d, int pos, long t, int& i, int& j, int& k) {
    if (d == 0)      { int A = g.ny; i = pos;              j = 1 + (int)(t % A); k = 1 + (int)(t / A); }
    else if (d == 1) { int A = g.nx; i = 1 + (int)(t % A); j = pos;              k = 1 + (int)(t / A); }
    else             { int A = g.nx; i = 1 + (int)(t % A); j = 1 + (int)(t / A); k = pos; }
}
__host__ __device__ inline long face_size(const Geo& g, int d) {
    return d == 0 ? (long)g.ny * g.nz : d == 1 ? (long)g.nx * g.nz : (long)g.nx * g.ny;
}
__global__ void k_pack(Geo g, int d, int pos, const double* u, double* buf) {
    long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= face_size(g, d)) return;
    int i, j, k; face_cell(g, d, pos, t, i, j, k);
    buf[t] = u[cell(g, i, j, k)];
}
__global__ void k_unpack(Geo g, int d, int pos, const double* buf, double* u) {
    long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= face_size(g, d)) return;
    int i, j, k; face_cell(g, d, pos, t, i, j, k);
    u[cell(g, i, j, k)] = buf[t];
}

// ---- masked kernels for the satellite gap filling (see kernels.hpp) -------
// Mask codes: 0 unknown, 1 observed, 2 land, 3 hidden. Only 0 and 3 are updated.
__device__ inline void masked_neighbours(const Geo& g, const unsigned char* m, const double* v, size_t c, double& s, double& D) {
    s = 0.0; D = 0.0;
    if (m[c - 1]    != 2) { D += g.ihx2; s += g.ihx2 * v[c - 1];    }
    if (m[c + 1]    != 2) { D += g.ihx2; s += g.ihx2 * v[c + 1];    }
    if (m[c - g.sy] != 2) { D += g.ihy2; s += g.ihy2 * v[c - g.sy]; }
    if (m[c + g.sy] != 2) { D += g.ihy2; s += g.ihy2 * v[c + g.sy]; }
    if (g.ihz2 > 0.0) {
        if (m[c - g.sz] != 2) { D += g.ihz2; s += g.ihz2 * v[c - g.sz]; }
        if (m[c + g.sz] != 2) { D += g.ihz2; s += g.ihz2 * v[c + g.sz]; }
    }
}
__global__ void k_jacobi_m(Geo g, Box b, const unsigned char* m, const double* u, double* un, const double* f) {
    int i = b.i0 + blockIdx.x * blockDim.x + threadIdx.x;
    int j = b.j0 + blockIdx.y * blockDim.y + threadIdx.y;
    int k = b.k0 + blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= b.i1 || j >= b.j1 || k >= b.k1) return;
    size_t c = cell(g, i, j, k);
    if (m[c] != 0 && m[c] != 3) return;
    double s, D; masked_neighbours(g, m, u, c, s, D);
    un[c] = D > 0.0 ? (f[c] + s) / D : 0.0;
}
__global__ void k_applyA_m(Geo g, Box b, const unsigned char* m, const double* p, double* Ap) {
    int i = b.i0 + blockIdx.x * blockDim.x + threadIdx.x;
    int j = b.j0 + blockIdx.y * blockDim.y + threadIdx.y;
    int k = b.k0 + blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= b.i1 || j >= b.j1 || k >= b.k1) return;
    size_t c = cell(g, i, j, k);
    if (m[c] != 0 && m[c] != 3) { Ap[c] = 0.0; return; }
    double s, D; masked_neighbours(g, m, p, c, s, D);
    Ap[c] = D > 0.0 ? D * p[c] - s : p[c];
}
__global__ void k_residual_m(Geo g, const unsigned char* m, const double* u, const double* f, double* r, double* out) {
    const long n = (long)g.nx * g.ny * g.nz;
    const long stride = (long)gridDim.x * blockDim.x;
    double sum = 0.0;
    for (long t = blockIdx.x * blockDim.x + threadIdx.x; t < n; t += stride) {
        int i = 1 + (int)(t % g.nx);
        long rest = t / g.nx;
        int j = 1 + (int)(rest % g.ny);
        int k = 1 + (int)(rest / g.ny);
        size_t c = cell(g, i, j, k);
        if (m[c] != 0 && m[c] != 3) { r[c] = 0.0; continue; }
        double s, D; masked_neighbours(g, m, u, c, s, D);
        double Au = D > 0.0 ? D * u[c] - s : u[c];
        r[c] = f[c] - Au;
        sum += r[c] * r[c];
    }
    atomicAdd(out, sum);
}

// ---- host-side launch helpers -------------------------------------------
inline dim3 box_block() { return dim3(64, 4, 1); }
inline dim3 box_grid(const Box& b) {
    dim3 t = box_block();
    return dim3((b.i1 - b.i0 + t.x - 1) / t.x, (b.j1 - b.j0 + t.y - 1) / t.y, (b.k1 - b.k0 + t.z - 1) / t.z);
}
inline Box interior(const Geo& g) { return Box{1, g.nx + 1, 1, g.ny + 1, 1, g.nz + 1}; }
inline bool box_empty(const Box& b) { return b.i1 <= b.i0 || b.j1 <= b.j0 || b.k1 <= b.k0; }
constexpr int REDUCE_BLOCKS = 1024, REDUCE_THREADS = 256;

} // namespace gpu
