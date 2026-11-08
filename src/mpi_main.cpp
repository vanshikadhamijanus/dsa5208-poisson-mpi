// mpi_main.cpp -- distributed-memory driver (MPI, or MPI+OpenMP when built with -fopenmp).
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "decomp.hpp"
#include "solver.hpp"

using namespace poisson;

struct MpiComm {
    Decomp& D;
    const std::string mode;
    double t_halo = 0.0, t_allreduce = 0.0;
    MPI_Request req[12];
    int nreq = 0;
    std::vector<std::pair<double*, MPI_Win>> wins;

    MpiComm(Decomp& d, const std::string& m) : D(d), mode(m) {}
    ~MpiComm() { for (auto& w : wins) MPI_Win_free(&w.second); }

    double wtime() const { return MPI_Wtime(); }
    bool can_overlap() const { return mode == "overlap"; }
    int rank() const { return D.rank; }
    int size() const { return D.size; }
    int threads() const {
#ifdef _OPENMP
        return omp_get_max_threads();
#else
        return 1;
#endif
    }
    std::string dims_string() const { return poisson::dims_string(D); }
    std::string hostname() const { char b[256]; gethostname(b, sizeof b); return b; }
    void barrier() const { MPI_Barrier(D.cart); }

    void rma_register(double** bufs, int n) {
        if (mode != "rma") return;
        for (int i = 0; i < n; ++i) {
            MPI_Win w;
            MPI_Win_create(bufs[i], (MPI_Aint)(D.L.ntot * sizeof(double)), sizeof(double), MPI_INFO_NULL, D.cart, &w);
            wins.emplace_back(bufs[i], w);
        }
    }
    void exchange(double* u) {
        const double t0 = MPI_Wtime();
        if (mode == "sendrecv") exchange_sendrecv(D, u);
        else if (mode == "rma") {
            MPI_Win w = MPI_WIN_NULL;
            for (auto& p : wins) if (p.first == u) w = p.second;
            if (w == MPI_WIN_NULL) {               // array not registered: fall back to messages
                nreq = exchange_start(D, u, req); exchange_wait(nreq, req);
            } else { D.win = w; exchange_rma(D, u); D.win = MPI_WIN_NULL; }
        } else { nreq = exchange_start(D, u, req); exchange_wait(nreq, req); }
        t_halo += MPI_Wtime() - t0;
    }
    void start(double* u) { const double t0 = MPI_Wtime(); nreq = exchange_start(D, u, req); t_halo += MPI_Wtime() - t0; }
    void wait()           { const double t0 = MPI_Wtime(); exchange_wait(nreq, req);        t_halo += MPI_Wtime() - t0; }
    double sum(double x) { double g; const double t0 = MPI_Wtime(); MPI_Allreduce(&x, &g, 1, MPI_DOUBLE, MPI_SUM, D.cart); t_allreduce += MPI_Wtime() - t0; return g; }
    double max(double x) { double g; const double t0 = MPI_Wtime(); MPI_Allreduce(&x, &g, 1, MPI_DOUBLE, MPI_MAX, D.cart); t_allreduce += MPI_Wtime() - t0; return g; }
};

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    Params P = parse_args(argc, argv);
#ifdef _OPENMP
    if (P.threads > 0) omp_set_num_threads(P.threads);
    const char* impl = "mpi_omp";
#else
    const char* impl = "mpi";
#endif
    if (P.data_mode()) data_configure(P);                 // grid size from the file header (every rank reads it)
    Decomp D = make_decomp(P);
    {
        MpiComm C(D, P.exchange);
        Result R;
        if (P.data_mode()) {
            DataBlock B = load_data(P, D.L);
            R = solve(P, D.L, C, impl, &B);
            if (!P.out.empty()) {
                std::vector<float> out; compose_output(B, D.L, B.solution.data(), out);
                if (!write_grid_mpi(D, P, output_header(B, P), out) && D.rank == 0) std::fprintf(stderr, "could not write %s\n", P.out.c_str());
                else if (D.rank == 0) std::printf("filled field written to %s\n", P.out.c_str());
            }
        } else {
            R = solve(P, D.L, C, impl);
        }
        if (D.rank == 0) { print_result(P, R); write_csv(P, R); }
    }
    free_decomp(D);
    MPI_Finalize();
    return 0;
}
