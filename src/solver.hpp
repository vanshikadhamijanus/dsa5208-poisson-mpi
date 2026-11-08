// solver.hpp -- the Jacobi and conjugate gradient iterations, written once.
//
// The loop body is identical for the serial and the MPI builds; what differs is
// how halos are refreshed and how global sums are formed. Those two operations
// are supplied by a small "Comm" policy object:
//   C.wtime()                 wall clock
//   C.exchange(u)             refresh all halos of u (blocking from the caller's view)
//   C.can_overlap()           true if start/wait may be used to hide communication
//   C.start(u) / C.wait()     split-phase exchange for the overlap mode
//   C.sum(x) / C.max(x)       global reductions
//   C.rma_register(ptrs,n)    tell a one-sided implementation which arrays get exchanged
//   C.t_halo, C.t_allreduce   accumulated communication time (filled by the policy)
//
// The same loops serve both problems. With a DataBlock (satellite gap filling)
// the masked kernels are used and the right-hand side comes from the observed
// pixels; without one, the manufactured Poisson problem is set up and the
// error against its exact solution is reported.
#pragma once
#include <algorithm>
#include <utility>
#include "common.hpp"
#include "kernels.hpp"
#include "gridio.hpp"

namespace poisson {

// Refresh halos of `u` then apply `kernel(range)` to the whole interior, hiding
// the message latency behind the deep-interior work when the policy allows it.
template <class Comm, class Kernel>
inline void exchange_and_apply(Comm& C, const Local& L, double* u, Kernel kernel, double& t_compute) {
    if (C.can_overlap() && has_deep_interior(L)) {
        C.start(u);
        double t0 = C.wtime();
        kernel(deep_interior(L));
        t_compute += C.wtime() - t0;
        C.wait();
        t0 = C.wtime();
        for_each_boundary_slab(L, kernel);
        t_compute += C.wtime() - t0;
    } else {
        C.exchange(u);
        double t0 = C.wtime();
        kernel(full(L));
        t_compute += C.wtime() - t0;
    }
}

template <class Comm>
inline Result solve(const Params& P, const Local& L, Comm& C, const char* impl, DataBlock* D = nullptr) {
    Result R;
    R.impl = impl;
    const size_t n = L.ntot;
    std::vector<double> u(n, 0.0), un(n, 0.0), f(n, 0.0);
    const uint8_t* M = D ? D->mask.data() : nullptr;

    // kernels for this problem (masked or plain)
    auto sweep = [&](const double* a, double* b, Range r) { if (M) jacobi_sweep_m(L, M, a, b, f.data(), r); else jacobi_sweep(L, a, b, f.data(), r); };
    auto applyA = [&](const double* a, double* b, Range r) { if (M) apply_A_m(L, M, a, b, r); else apply_A(L, a, b, r); };
    auto resid = [&](const double* a, double* scratch) { return M ? residual_m(L, M, a, f.data(), scratch) : residual(L, a, f.data(), scratch); };

    double t0 = C.wtime();
    if (D) finish_data(*D, L, [&](double x) { return C.sum(x); }, f.data());
    else init_problem(P, L, f.data());
    double fnorm = std::sqrt(C.sum(dot(L, f.data(), f.data())));
    if (fnorm == 0.0) fnorm = 1.0;                       // nothing to fill: avoid 0/0
    R.t_setup = C.wtime() - t0;

    double t_compute = 0.0;
    long it = 0;
    double rel = 0.0;
    int converged = 0;
    int check_every = P.check_every;
    if (P.solver == "jacobi" && P.tol > 0 && check_every == 0) check_every = 10;

    C.barrier();
    const double t_start = C.wtime();

    if (P.solver == "jacobi") {
        double* cur = u.data();
        double* nxt = un.data();
        double* bufs[2] = {cur, nxt};
        C.rma_register(bufs, 2);
        for (it = 1; it <= P.iters; ++it) {
            exchange_and_apply(C, L, cur, [&](Range r) { sweep(cur, nxt, r); }, t_compute);
            std::swap(cur, nxt);
            if (check_every > 0 && it % check_every == 0) {
                // residual needs fresh halos of the new iterate
                C.exchange(cur);
                double tt = C.wtime();
                double local = resid(cur, nxt);                     // nxt is scratch here
                t_compute += C.wtime() - tt;
                rel = std::sqrt(C.sum(local)) / fnorm;
                if (P.verbose && C.rank() == 0) std::printf("  it %ld  rel_resid %.3e\n", it, rel);
                if (P.tol > 0 && rel < P.tol) { converged = 1; break; }
            }
        }
        if (it > P.iters) it = P.iters;
        if (cur != u.data()) std::copy(cur, cur + n, u.data());
    } else { // conjugate gradient
        std::vector<double> r(n, 0.0), p(n, 0.0), Ap(n, 0.0);
        // u = 0  =>  r = f,  p = r
        std::copy(f.begin(), f.end(), r.begin());
        std::copy(f.begin(), f.end(), p.begin());
        double* pb[1] = {p.data()};
        C.rma_register(pb, 1);
        double rr = C.sum(dot(L, r.data(), r.data()));
        rel = std::sqrt(rr) / fnorm;
        for (it = 1; it <= P.iters; ++it) {
            exchange_and_apply(C, L, p.data(), [&](Range rg) { applyA(p.data(), Ap.data(), rg); }, t_compute);
            double tt = C.wtime();
            double pAp_local = dot(L, p.data(), Ap.data());
            t_compute += C.wtime() - tt;
            const double pAp = C.sum(pAp_local);
            if (pAp == 0.0) { converged = rr == 0.0; break; }   // nothing left to solve
            const double alpha = rr / pAp;
            tt = C.wtime();
            axpy(L,  alpha, p.data(),  u.data());
            axpy(L, -alpha, Ap.data(), r.data());
            double rr_local = dot(L, r.data(), r.data());
            t_compute += C.wtime() - tt;
            const double rr_new = C.sum(rr_local);
            rel = std::sqrt(rr_new) / fnorm;
            if (P.verbose && C.rank() == 0 && it % 50 == 0) std::printf("  it %ld  rel_resid %.3e\n", it, rel);
            if (P.tol > 0 && rel < P.tol) { converged = 1; break; }
            tt = C.wtime();
            xpay(L, r.data(), rr_new / rr, p.data());
            t_compute += C.wtime() - tt;
            rr = rr_new;
        }
        if (it > P.iters) it = P.iters;
    }

    C.barrier();
    R.t_total = C.wtime() - t_start;

    // Final diagnostics (not timed): true residual, then the error measure of the problem.
    C.exchange(u.data());
    const double res_local = resid(u.data(), un.data());
    R.rel_resid = std::sqrt(C.sum(res_local)) / fnorm;
    if (D) {
        DataSums S = data_local_errors(*D, L, u.data(), P.eval_day >= 0 ? P.eval_day - P.day : -1);
        const double n_h = C.sum(S.n_h), n_fill = C.sum(S.n_fill), n_ro = C.sum(S.n_refobs);
        R.n_obs = (long)C.sum((double)D->n_obs); R.n_unknown = (long)C.sum((double)D->n_unknown);
        R.n_hidden = (long)C.sum((double)D->n_hidden); R.n_land = (long)C.sum((double)D->n_land);
        R.mean_obs = D->mean;
        R.l2_err = n_h > 0 ? std::sqrt(C.sum(S.se_h) / n_h) : 0.0;        // RMSE on hidden pixels
        R.mae_hidden = n_h > 0 ? C.sum(S.ae_h) / n_h : 0.0;
        R.linf_err = C.max(S.mx_h);
        R.rmse_baseline = n_h > 0 ? std::sqrt(C.sum(S.se_base) / n_h) : 0.0;
        R.rmse_ref_fill = n_fill > 0 ? std::sqrt(C.sum(S.se_ref_fill) / n_fill) : 0.0;
        R.rmse_ref_obs = n_ro > 0 ? std::sqrt(C.sum(S.se_ref_obs) / n_ro) : 0.0;
        std::copy(u.begin(), u.end(), un.begin());                          // hand the anomaly back through D->solution
        D->solution.swap(un);
    } else {
        double e2, einf;
        error_norms(P, L, u.data(), e2, einf);
        const double cell = L.hx * L.hy * (P.dim == 3 ? L.hz : 1.0);
        R.l2_err = std::sqrt(C.sum(e2) * cell);
        R.linf_err = C.max(einf);
    }
    R.iters = it;
    R.converged = converged;

    // Timer summary across ranks.
    R.t_compute_max = C.max(t_compute);          R.t_compute_mean = C.sum(t_compute) / C.size();
    R.t_halo_max = C.max(C.t_halo);              R.t_halo_mean = C.sum(C.t_halo) / C.size();
    R.t_allreduce_max = C.max(C.t_allreduce);    R.t_allreduce_mean = C.sum(C.t_allreduce) / C.size();
    R.ranks = C.size();
    R.threads = C.threads();
    R.dims_str = C.dims_string();
    R.hostname = C.hostname();
    return R;
}

} // namespace poisson
