// cuda_compat.hpp -- lets the CUDA source be compiled and *run* on a machine
// without a GPU, purely to test its logic.
//
// Under nvcc (__CUDACC__ defined) this header is almost empty: KERNEL_LAUNCH
// expands to the usual <<<grid, block>>> launch. Without nvcc it provides a tiny
// sequential emulation: the kernel function is called once per (block, thread)
// pair with the threadIdx/blockIdx globals set accordingly, cudaMalloc becomes
// malloc, cudaMemcpy becomes memcpy, and atomicAdd is a plain addition.
// The emulation is slow and single-threaded; it exists only so the MPI+CUDA
// driver can be validated against the CPU builds before a GPU is available.
#pragma once
#include <cstdlib>
#include <cstring>
#include <cstdio>

// CUDA_CHECK stops the job on any error. It calls poisson::fail (common.hpp), which
// the MPI driver points at MPI_Abort, so one failing rank stops all of them.
#include "common.hpp"
#define CUDA_CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); poisson::fail(4); } } while (0)

#ifdef __CUDACC__
#include <cuda_runtime.h>
// A launch with a bad configuration fails silently until the next synchronising
// call; cudaGetLastError right after the launch reports it at the line that caused it.
#define KERNEL_LAUNCH(kernel, grid, block, ...) do { kernel<<<grid, block>>>(__VA_ARGS__); CUDA_CHECK(cudaGetLastError()); } while (0)
#define POISSON_CUDA_REAL 1
#else
#define POISSON_CUDA_REAL 0
#define __global__
#define __device__
#define __host__
struct dim3 { unsigned x, y, z; dim3(unsigned a = 1, unsigned b = 1, unsigned c = 1) : x(a), y(b), z(c) {} };
struct uint3 { unsigned x, y, z; };
inline uint3 threadIdx, blockIdx;
inline dim3 blockDim, gridDim;
typedef int cudaError_t;
const cudaError_t cudaSuccess = 0;
enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice, cudaMemcpyDefault };
inline cudaError_t cudaMalloc(void** p, size_t n) { *p = std::malloc(n); return *p ? cudaSuccess : 1; }
template <class T> inline cudaError_t cudaMalloc(T** p, size_t n) { return cudaMalloc((void**)p, n); }
inline cudaError_t cudaMallocHost(void** p, size_t n) { return cudaMalloc(p, n); }
template <class T> inline cudaError_t cudaMallocHost(T** p, size_t n) { return cudaMalloc((void**)p, n); }
inline cudaError_t cudaFree(void* p) { std::free(p); return cudaSuccess; }
inline cudaError_t cudaFreeHost(void* p) { std::free(p); return cudaSuccess; }
inline cudaError_t cudaMemcpy(void* d, const void* s, size_t n, cudaMemcpyKind) { std::memcpy(d, s, n); return cudaSuccess; }
inline cudaError_t cudaMemset(void* d, int v, size_t n) { std::memset(d, v, n); return cudaSuccess; }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
inline cudaError_t cudaGetDeviceCount(int* n) { *n = 1; return cudaSuccess; }
inline cudaError_t cudaSetDevice(int) { return cudaSuccess; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline const char* cudaGetErrorString(cudaError_t) { return "emulated"; }
inline double atomicAdd(double* addr, double v) { double old = *addr; *addr += v; return old; }
template <class Fn> inline void emul_launch(dim3 grid, dim3 block, Fn fn) {
    gridDim = grid; blockDim = block;
    for (unsigned bz = 0; bz < grid.z; ++bz) for (unsigned by = 0; by < grid.y; ++by) for (unsigned bx = 0; bx < grid.x; ++bx) {
        blockIdx = {bx, by, bz};
        for (unsigned tz = 0; tz < block.z; ++tz) for (unsigned ty = 0; ty < block.y; ++ty) for (unsigned tx = 0; tx < block.x; ++tx) {
            threadIdx = {tx, ty, tz};
            fn();
        }
    }
}
#define KERNEL_LAUNCH(kernel, grid, block, ...) do { emul_launch(grid, block, [&]() { kernel(__VA_ARGS__); }); CUDA_CHECK(cudaGetLastError()); } while (0)
#endif
