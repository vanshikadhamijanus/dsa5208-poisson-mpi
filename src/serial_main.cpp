// serial_main.cpp -- single-process reference implementation.
// Compiled without -fopenmp it is the serial baseline; compiled with -fopenmp it
// becomes the pure shared-memory (OpenMP only) variant, with no MPI involved.
#include <chrono>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "solver.hpp"

using namespace poisson;

struct SerialComm {
    double t_halo = 0.0, t_allreduce = 0.0;
    double wtime() const { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
    void exchange(double*) {}
    bool can_overlap() const { return false; }
    void start(double*) {}
    void wait() {}
    void rma_register(double**, int) {}
    double sum(double x) const { return x; }
    double max(double x) const { return x; }
    void barrier() const {}
    int rank() const { return 0; }
    int size() const { return 1; }
    int threads() const {
#ifdef _OPENMP
        return omp_get_max_threads();
#else
        return 1;
#endif
    }
    std::string dims_string() const { return "1x1x1"; }
    std::string hostname() const { char b[256]; gethostname(b, sizeof b); return b; }
};

int main(int argc, char** argv) {
    Params P = parse_args(argc, argv);
#ifdef _OPENMP
    if (P.threads > 0) omp_set_num_threads(P.threads);
    const char* impl = "omp";
#else
    const char* impl = "serial";
#endif
    if (P.data_mode()) data_configure(P);                 // grid size from the file header
    else if (P.local_n > 0) { for (int d = 0; d < 3; ++d) P.n[d] = P.local_n; if (P.dim == 2) P.n[2] = 1; }
    Local L = make_local(P, P.n, (int)P.n[0], (int)P.n[1], (int)P.n[2], 0, 0, 0);
    SerialComm C;
    Result R;
    if (P.data_mode()) {
        DataBlock D = load_data(P, L);
        R = solve(P, L, C, impl, &D);
        if (!P.out.empty()) {
            std::vector<float> out; compose_output(D, L, D.solution.data(), out);
            if (!write_grid_serial(P.out, output_header(D, P), out)) std::perror(P.out.c_str());
            else std::printf("filled field written to %s\n", P.out.c_str());
        }
    } else {
        R = solve(P, L, C, impl);
    }
    print_result(P, R);
    write_csv(P, R);
    return 0;
}
