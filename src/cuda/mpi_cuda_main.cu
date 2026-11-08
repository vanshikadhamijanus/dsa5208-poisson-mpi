// mpi_cuda_main.cu -- MPI + CUDA driver: one GPU per MPI rank.
//
// The decomposition is the same Cartesian split as the CPU code (decomp.hpp).
// Each rank keeps its block in device memory. Halo exchange: faces are packed
// into contiguous device buffers by a kernel, copied to pinned host buffers,
// sent with MPI_Isend/Irecv, copied back and unpacked. With --cuda-aware the
// host staging is skipped and MPI is handed device pointers directly.
// "overlap" mode launches the deep-interior kernel while the messages are in
// flight. Reductions happen on the GPU and only one double per rank goes to
// MPI_Allreduce.
// With --data the satellite gap-filling problem is solved: the pixel mask is
// uploaded once and the masked kernels (gpu_kernels.cuh) replace the plain ones.
#include <mpi.h>
#include <unistd.h>
#include <vector>
#include "cuda_compat.hpp"
#include "gpu_kernels.cuh"
#include "decomp.hpp"
#include "gridio.hpp"

using namespace poisson;
using gpu::Geo; using gpu::Box;

static Geo make_geo(const Local& L) { return Geo{L.nx, L.ny, L.nz, L.sy, L.sz, L.ihx2, L.ihy2, L.ihz2, L.diag}; }
static Box to_box(Range r) { return Box{r.i0, r.i1, r.j0, r.j1, r.k0, r.k1}; }

struct GpuHalo {
    double *d_send[3][2] = {}, *d_recv[3][2] = {}, *h_send[3][2] = {}, *h_recv[3][2] = {};
    long size[3] = {0, 0, 0};
    MPI_Request req[12]; int nreq = 0;
};

static void halo_alloc(GpuHalo& H, const Geo& g, int ndims) {
    for (int d = 0; d < ndims; ++d) {
        H.size[d] = gpu::face_size(g, d);
        for (int s = 0; s < 2; ++s) {
            CUDA_CHECK(cudaMalloc(&H.d_send[d][s], H.size[d] * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&H.d_recv[d][s], H.size[d] * sizeof(double)));
            CUDA_CHECK(cudaMallocHost(&H.h_send[d][s], H.size[d] * sizeof(double)));
            CUDA_CHECK(cudaMallocHost(&H.h_recv[d][s], H.size[d] * sizeof(double)));
        }
    }
}
static void halo_free(GpuHalo& H, int ndims) {
    for (int d = 0; d < ndims; ++d) for (int s = 0; s < 2; ++s) {
        cudaFree(H.d_send[d][s]); cudaFree(H.d_recv[d][s]); cudaFreeHost(H.h_send[d][s]); cudaFreeHost(H.h_recv[d][s]);
    }
}

// Pack faces, start the sends/receives. Side 0 = lo face (plane 1) goes to lo neighbour,
// side 1 = hi face (plane n_d) goes to hi neighbour. Tags match decomp.hpp.
static void halo_start(const Decomp& D, const Geo& g, GpuHalo& H, double* u, bool cuda_aware) {
    const int n[3] = {g.nx, g.ny, g.nz};
    H.nreq = 0;
    for (int d = 0; d < D.ndims_active; ++d) {
        int threads = 256, blocks = (int)((H.size[d] + threads - 1) / threads);
        if (D.lo[d] != MPI_PROC_NULL) KERNEL_LAUNCH(gpu::k_pack, dim3(blocks), dim3(threads), g, d, 1,    u, H.d_send[d][0]);
        if (D.hi[d] != MPI_PROC_NULL) KERNEL_LAUNCH(gpu::k_pack, dim3(blocks), dim3(threads), g, d, n[d], u, H.d_send[d][1]);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int d = 0; d < D.ndims_active; ++d) {
        double* slo = cuda_aware ? H.d_send[d][0] : H.h_send[d][0];
        double* shi = cuda_aware ? H.d_send[d][1] : H.h_send[d][1];
        double* rlo = cuda_aware ? H.d_recv[d][0] : H.h_recv[d][0];
        double* rhi = cuda_aware ? H.d_recv[d][1] : H.h_recv[d][1];
        if (!cuda_aware) {
            if (D.lo[d] != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpy(slo, H.d_send[d][0], H.size[d] * sizeof(double), cudaMemcpyDeviceToHost));
            if (D.hi[d] != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpy(shi, H.d_send[d][1], H.size[d] * sizeof(double), cudaMemcpyDeviceToHost));
        }
        MPI_Irecv(rlo, (int)H.size[d], MPI_DOUBLE, D.lo[d], 10 + d, D.cart, &H.req[H.nreq++]);
        MPI_Irecv(rhi, (int)H.size[d], MPI_DOUBLE, D.hi[d], 20 + d, D.cart, &H.req[H.nreq++]);
        MPI_Isend(shi, (int)H.size[d], MPI_DOUBLE, D.hi[d], 10 + d, D.cart, &H.req[H.nreq++]);
        MPI_Isend(slo, (int)H.size[d], MPI_DOUBLE, D.lo[d], 20 + d, D.cart, &H.req[H.nreq++]);
    }
}
// Wait, copy received faces to the device and unpack them into the halo planes.
static void halo_finish(const Decomp& D, const Geo& g, GpuHalo& H, double* u, bool cuda_aware) {
    const int n[3] = {g.nx, g.ny, g.nz};
    MPI_Waitall(H.nreq, H.req, MPI_STATUSES_IGNORE);
    for (int d = 0; d < D.ndims_active; ++d) {
        int threads = 256, blocks = (int)((H.size[d] + threads - 1) / threads);
        if (D.lo[d] != MPI_PROC_NULL) {
            if (!cuda_aware) CUDA_CHECK(cudaMemcpy(H.d_recv[d][0], H.h_recv[d][0], H.size[d] * sizeof(double), cudaMemcpyHostToDevice));
            KERNEL_LAUNCH(gpu::k_unpack, dim3(blocks), dim3(threads), g, d, 0, H.d_recv[d][0], u);
        }
        if (D.hi[d] != MPI_PROC_NULL) {
            if (!cuda_aware) CUDA_CHECK(cudaMemcpy(H.d_recv[d][1], H.h_recv[d][1], H.size[d] * sizeof(double), cudaMemcpyHostToDevice));
            KERNEL_LAUNCH(gpu::k_unpack, dim3(blocks), dim3(threads), g, d, n[d] + 1, H.d_recv[d][1], u);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

struct GpuTimers { double compute = 0, halo = 0, allreduce = 0; };

// Global reduction helper: GPU partial sum -> one MPI_Allreduce.
static double gpu_dot(const Decomp& D, const Geo& g, const double* a, const double* b, double* d_scalar, GpuTimers& T) {
    double t0 = MPI_Wtime();
    CUDA_CHECK(cudaMemset(d_scalar, 0, sizeof(double)));
    KERNEL_LAUNCH(gpu::k_dot, dim3(gpu::REDUCE_BLOCKS), dim3(gpu::REDUCE_THREADS), g, a, b, d_scalar);
    double local; CUDA_CHECK(cudaMemcpy(&local, d_scalar, sizeof(double), cudaMemcpyDeviceToHost));
    T.compute += MPI_Wtime() - t0;
    t0 = MPI_Wtime();
    double global; MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, D.cart);
    T.allreduce += MPI_Wtime() - t0;
    return global;
}

// Residual sum of squares on the device (plain or masked), then one MPI_Allreduce.
static double gpu_residual(const Decomp& D, const Geo& g, const unsigned char* d_m, const double* u, const double* f,
                           double* scratch, double* d_scalar, GpuTimers& T) {
    double t0 = MPI_Wtime();
    CUDA_CHECK(cudaMemset(d_scalar, 0, sizeof(double)));
    if (d_m) KERNEL_LAUNCH(gpu::k_residual_m, dim3(gpu::REDUCE_BLOCKS), dim3(gpu::REDUCE_THREADS), g, d_m, u, f, scratch, d_scalar);
    else     KERNEL_LAUNCH(gpu::k_residual,   dim3(gpu::REDUCE_BLOCKS), dim3(gpu::REDUCE_THREADS), g, u, f, scratch, d_scalar);
    double local; CUDA_CHECK(cudaMemcpy(&local, d_scalar, sizeof(double), cudaMemcpyDeviceToHost));
    T.compute += MPI_Wtime() - t0;
    t0 = MPI_Wtime();
    double global; MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, D.cart);
    T.allreduce += MPI_Wtime() - t0;
    return global;
}

// Exchange halos of u, then run `launch(box)` over the interior (optionally overlapped).
template <class Launch>
static void exchange_and_apply(const Decomp& D, const Geo& g, const Local& L, GpuHalo& H, double* u,
                               bool overlap, bool cuda_aware, Launch launch, GpuTimers& T) {
    double t0 = MPI_Wtime();
    halo_start(D, g, H, u, cuda_aware);
    T.halo += MPI_Wtime() - t0;
    if (overlap && has_deep_interior(L)) {
        t0 = MPI_Wtime();
        launch(to_box(deep_interior(L)));           // asynchronous: runs while messages fly
        T.compute += MPI_Wtime() - t0;
        t0 = MPI_Wtime();
        halo_finish(D, g, H, u, cuda_aware);        // includes a device sync
        T.halo += MPI_Wtime() - t0;
        t0 = MPI_Wtime();
        for_each_boundary_slab(L, [&](Range r) { launch(to_box(r)); });
        CUDA_CHECK(cudaDeviceSynchronize());
        T.compute += MPI_Wtime() - t0;
    } else {
        t0 = MPI_Wtime();
        halo_finish(D, g, H, u, cuda_aware);
        T.halo += MPI_Wtime() - t0;
        t0 = MPI_Wtime();
        launch(to_box(full(L)));
        CUDA_CHECK(cudaDeviceSynchronize());
        T.compute += MPI_Wtime() - t0;
    }
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    Params P = parse_args(argc, argv);
    if (P.data_mode()) data_configure(P);
    Decomp D = make_decomp(P);
    const Local& L = D.L;
    const Geo g = make_geo(L);

    // one GPU per rank: pick by rank within the node
    MPI_Comm node; MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
    int node_rank; MPI_Comm_rank(node, &node_rank); MPI_Comm_free(&node);
    int ndev = 1; CUDA_CHECK(cudaGetDeviceCount(&ndev));
    CUDA_CHECK(cudaSetDevice(node_rank % ndev));

    const size_t n = L.ntot, bytes = n * sizeof(double);
    std::vector<double> hf(n, 0.0), hu(n, 0.0);
    Result R; R.impl = POISSON_CUDA_REAL ? "mpi_cuda" : "mpi_cuda_emul";
    GpuTimers T;

    double t0 = MPI_Wtime();
    DataBlock B;
    unsigned char* d_m = nullptr;
    if (P.data_mode()) {
        B = load_data(P, L);
        finish_data(B, L, [&](double x) { double gsum; MPI_Allreduce(&x, &gsum, 1, MPI_DOUBLE, MPI_SUM, D.cart); return gsum; }, hf.data());
        CUDA_CHECK(cudaMalloc(&d_m, n));
        CUDA_CHECK(cudaMemcpy(d_m, B.mask.data(), n, cudaMemcpyHostToDevice));
    } else {
        init_problem(P, L, hf.data());
    }
    double *d_u, *d_un, *d_f, *d_scalar;
    CUDA_CHECK(cudaMalloc(&d_u, bytes)); CUDA_CHECK(cudaMalloc(&d_un, bytes)); CUDA_CHECK(cudaMalloc(&d_f, bytes));
    CUDA_CHECK(cudaMalloc(&d_scalar, sizeof(double)));
    CUDA_CHECK(cudaMemset(d_u, 0, bytes)); CUDA_CHECK(cudaMemset(d_un, 0, bytes));
    CUDA_CHECK(cudaMemcpy(d_f, hf.data(), bytes, cudaMemcpyHostToDevice));
    GpuHalo H; halo_alloc(H, g, D.ndims_active);
    double fnorm = std::sqrt(gpu_dot(D, g, d_f, d_f, d_scalar, T));
    if (fnorm == 0.0) fnorm = 1.0;
    T = GpuTimers{};
    R.t_setup = MPI_Wtime() - t0;

    const bool overlap = P.exchange == "overlap";
    const bool cuda_aware = P.cuda_aware;
    int check_every = P.check_every;
    if (P.solver == "jacobi" && P.tol > 0 && check_every == 0) check_every = 10;
    long it = 0; double rel = 0.0; int converged = 0;

    MPI_Barrier(D.cart);
    const double t_start = MPI_Wtime();
    if (P.solver == "jacobi") {
        double* cur = d_u; double* nxt = d_un;
        for (it = 1; it <= P.iters; ++it) {
            exchange_and_apply(D, g, L, H, cur, overlap, cuda_aware, [&](Box b) {
                if (d_m) KERNEL_LAUNCH(gpu::k_jacobi_m, gpu::box_grid(b), gpu::box_block(), g, b, d_m, cur, nxt, d_f);
                else     KERNEL_LAUNCH(gpu::k_jacobi,   gpu::box_grid(b), gpu::box_block(), g, b, cur, nxt, d_f);
            }, T);
            std::swap(cur, nxt);
            if (check_every > 0 && it % check_every == 0) {
                double tt = MPI_Wtime(); halo_start(D, g, H, cur, cuda_aware); halo_finish(D, g, H, cur, cuda_aware); T.halo += MPI_Wtime() - tt;
                rel = std::sqrt(gpu_residual(D, g, d_m, cur, d_f, nxt, d_scalar, T)) / fnorm;
                if (P.verbose && D.rank == 0) std::printf("  it %ld  rel_resid %.3e\n", it, rel);
                if (P.tol > 0 && rel < P.tol) { converged = 1; break; }
            }
        }
        if (it > P.iters) it = P.iters;
        if (cur != d_u) CUDA_CHECK(cudaMemcpy(d_u, cur, bytes, cudaMemcpyDeviceToDevice));
    } else {
        double *d_r, *d_p, *d_Ap;
        CUDA_CHECK(cudaMalloc(&d_r, bytes)); CUDA_CHECK(cudaMalloc(&d_p, bytes)); CUDA_CHECK(cudaMalloc(&d_Ap, bytes));
        CUDA_CHECK(cudaMemcpy(d_r, d_f, bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_p, d_f, bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemset(d_Ap, 0, bytes));
        double rr = gpu_dot(D, g, d_r, d_r, d_scalar, T);
        const Box I = gpu::interior(g);
        for (it = 1; it <= P.iters; ++it) {
            exchange_and_apply(D, g, L, H, d_p, overlap, cuda_aware, [&](Box b) {
                if (d_m) KERNEL_LAUNCH(gpu::k_applyA_m, gpu::box_grid(b), gpu::box_block(), g, b, d_m, d_p, d_Ap);
                else     KERNEL_LAUNCH(gpu::k_applyA,   gpu::box_grid(b), gpu::box_block(), g, b, d_p, d_Ap);
            }, T);
            const double pAp = gpu_dot(D, g, d_p, d_Ap, d_scalar, T);
            if (pAp == 0.0) { converged = rr == 0.0; break; }
            const double alpha = rr / pAp;
            double tt = MPI_Wtime();
            KERNEL_LAUNCH(gpu::k_axpy, gpu::box_grid(I), gpu::box_block(), g,  alpha, d_p,  d_u);
            KERNEL_LAUNCH(gpu::k_axpy, gpu::box_grid(I), gpu::box_block(), g, -alpha, d_Ap, d_r);
            CUDA_CHECK(cudaDeviceSynchronize());
            T.compute += MPI_Wtime() - tt;
            const double rr_new = gpu_dot(D, g, d_r, d_r, d_scalar, T);
            rel = std::sqrt(rr_new) / fnorm;
            if (P.verbose && D.rank == 0 && it % 50 == 0) std::printf("  it %ld  rel_resid %.3e\n", it, rel);
            if (P.tol > 0 && rel < P.tol) { converged = 1; break; }
            tt = MPI_Wtime();
            KERNEL_LAUNCH(gpu::k_xpay, gpu::box_grid(I), gpu::box_block(), g, d_r, rr_new / rr, d_p);
            CUDA_CHECK(cudaDeviceSynchronize());
            T.compute += MPI_Wtime() - tt;
            rr = rr_new;
        }
        if (it > P.iters) it = P.iters;
        cudaFree(d_r); cudaFree(d_p); cudaFree(d_Ap);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(D.cart);
    R.t_total = MPI_Wtime() - t_start;

    // diagnostics (not timed): true residual on the GPU, error measures on the host
    halo_start(D, g, H, d_u, cuda_aware); halo_finish(D, g, H, d_u, cuda_aware);
    GpuTimers Tdiag;
    R.rel_resid = std::sqrt(gpu_residual(D, g, d_m, d_u, d_f, d_un, d_scalar, Tdiag)) / fnorm;
    CUDA_CHECK(cudaMemcpy(hu.data(), d_u, bytes, cudaMemcpyDeviceToHost));
    auto gsum = [&](double x) { double y; MPI_Allreduce(&x, &y, 1, MPI_DOUBLE, MPI_SUM, D.cart); return y; };
    auto gmax = [&](double x) { double y; MPI_Allreduce(&x, &y, 1, MPI_DOUBLE, MPI_MAX, D.cart); return y; };
    if (P.data_mode()) {
        DataSums S = data_local_errors(B, L, hu.data(), P.eval_day >= 0 ? P.eval_day - P.day : -1);
        const double n_h = gsum(S.n_h), n_fill = gsum(S.n_fill), n_ro = gsum(S.n_refobs);
        R.n_obs = (long)gsum((double)B.n_obs); R.n_unknown = (long)gsum((double)B.n_unknown);
        R.n_hidden = (long)gsum((double)B.n_hidden); R.n_land = (long)gsum((double)B.n_land);
        R.mean_obs = B.mean;
        R.l2_err = n_h > 0 ? std::sqrt(gsum(S.se_h) / n_h) : 0.0;
        R.mae_hidden = n_h > 0 ? gsum(S.ae_h) / n_h : 0.0;
        R.linf_err = gmax(S.mx_h);
        R.rmse_baseline = n_h > 0 ? std::sqrt(gsum(S.se_base) / n_h) : 0.0;
        R.rmse_ref_fill = n_fill > 0 ? std::sqrt(gsum(S.se_ref_fill) / n_fill) : 0.0;
        R.rmse_ref_obs = n_ro > 0 ? std::sqrt(gsum(S.se_ref_obs) / n_ro) : 0.0;
        if (!P.out.empty()) {
            std::vector<float> out; compose_output(B, L, hu.data(), out);
            if (!write_grid_mpi(D, P, output_header(B, P), out) && D.rank == 0) std::fprintf(stderr, "could not write %s\n", P.out.c_str());
            else if (D.rank == 0) std::printf("filled field written to %s\n", P.out.c_str());
        }
    } else {
        double e2, einf; error_norms(P, L, hu.data(), e2, einf);
        const double cellv = L.hx * L.hy * (P.dim == 3 ? L.hz : 1.0);
        R.l2_err = std::sqrt(gsum(e2) * cellv); R.linf_err = gmax(einf);
    }
    R.iters = it; R.converged = converged;

    double loc[3] = {T.compute, T.halo, T.allreduce}, mx[3], sm[3];
    MPI_Allreduce(loc, mx, 3, MPI_DOUBLE, MPI_MAX, D.cart);
    MPI_Allreduce(loc, sm, 3, MPI_DOUBLE, MPI_SUM, D.cart);
    R.t_compute_max = mx[0]; R.t_halo_max = mx[1]; R.t_allreduce_max = mx[2];
    R.t_compute_mean = sm[0] / D.size; R.t_halo_mean = sm[1] / D.size; R.t_allreduce_mean = sm[2] / D.size;
    R.ranks = D.size; R.threads = 1; R.dims_str = dims_string(D);
    char hn[256]; gethostname(hn, sizeof hn); R.hostname = hn;
    if (D.rank == 0) { print_result(P, R); write_csv(P, R); }

    halo_free(H, D.ndims_active);
    if (d_m) cudaFree(d_m);
    cudaFree(d_u); cudaFree(d_un); cudaFree(d_f); cudaFree(d_scalar);
    free_decomp(D);
    MPI_Finalize();
    return 0;
}
