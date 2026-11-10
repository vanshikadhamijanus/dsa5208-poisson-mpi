// common.hpp -- problem definition, command-line parameters, and CSV output
// shared by the serial, MPI, MPI+OpenMP and MPI+CUDA drivers.
//
// Two problems share one solver:
//
// 1. Manufactured Poisson problem (validation and scaling benchmarks):
//                   -laplace(u) = f   on the unit square (dim=2) or unit cube (dim=3)
//                   u = 0 on the boundary (Dirichlet condition)
//    with the known solution u(x,y,z) = exp(x) sin(pi x) sin(pi y) [sin(pi z)]
//    f = -laplace(u) = exp(x) sin(pi y) [sin(pi z)] * ( (dim*pi^2 - 1) sin(pi x) - 2 pi cos(pi x) )
//    The exp(x) factor matters: a plain product of sines is an eigenvector of the
//    discrete Laplacian, so conjugate gradient would converge in one step and the
//    benchmark would be meaningless. With exp(x) the solution has broad spectral
//    content, CG needs O(n) iterations, and the discretisation error is O(h^2).
//
// 2. Satellite gap filling (the application, selected with --data FILE):
//    a gridded sea-surface-temperature map (2D: one day; 3D: a stack of days with
//    time as the third axis) has holes where clouds blocked the sensor. Observed
//    pixels are held fixed (Dirichlet data), land pixels are excluded (zero-flux
//    coast), and the Laplace equation laplace(u) = 0 is solved on the remaining
//    pixels: every missing pixel becomes the average of its neighbours. The grid
//    spacing is one pixel and one day; --tweight sets the weight of the time
//    neighbours relative to the space neighbours. Validation hides a fraction of
//    the observed pixels (--holdout) and measures the error of the filled values.
//
// Storage convention (used by every variant): the local block is always a 3D
// array with one layer of halo cells on every side. A 2D problem is stored as a
// 3D block whose z-extent is 1 (nz = 1) and whose z-coefficient ihz2 is 0, so
// the z-neighbours contribute nothing. Index (i,j,k) -> k*sz + j*sy + i, so i (x)
// is the contiguous (fastest) direction.
#pragma once
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>

namespace poisson {

constexpr double PI = 3.14159265358979323846;

// Error exit. The MPI drivers set fatal_hook to a function that calls MPI_Abort,
// so a rank that fails (for example on a file read) stops the whole job instead
// of leaving the other ranks waiting in their next collective call.
inline void (*fatal_hook)(int) = nullptr;
[[noreturn]] inline void fail(int code) {
    if (fatal_hook) fatal_hook(code);
    std::exit(code);
}

struct Params {
    int dim = 2;                          // 2 or 3
    long n[3] = {256, 256, 1};            // global interior points per dimension (nz=1 for 2D)
    long local_n = 0;                     // weak scaling: if >0, n_d = local_n * dims_d
    std::string solver = "jacobi";        // jacobi | cg
    std::string exchange = "isend";       // sendrecv | isend | overlap | rma
    int dims[3] = {0, 0, 0};              // process grid (0 = let MPI_Dims_create decide)
    long iters = 1000;                    // maximum iterations
    double tol = 0.0;                     // relative residual tolerance (0 = run exactly `iters` sweeps)
    int check_every = 0;                  // jacobi: compute residual every k iterations (0 = never)
    int threads = 0;                      // OpenMP threads override (0 = OMP_NUM_THREADS)
    std::string csv;                      // append a results row to this file
    std::string tag;                      // free text copied to the CSV row
    bool verbose = false;
    bool cuda_aware = false;              // mpi_cuda: pass device pointers straight to MPI
    // --- satellite gap filling -------------------------------------------------
    std::string data;                     // input grid file (see gridio.hpp); empty = manufactured problem
    std::string out;                      // write the filled field to this grid file
    std::string ref;                      // optional gap-free reference grid (same shape as data) for comparison
    double holdout = 0.0;                 // fraction of observed pixels hidden for validation
    double tweight = 1.0;                 // 3D: weight of the time neighbours relative to the space neighbours
    int day = 0;                          // 2D: which time slice of the file to use (0-based)
    int days = 0;                         // 3D: how many slices to use starting at --day (0 = all)
    unsigned seed = 1;                    // seed of the deterministic holdout selection
    int holdout_block = 1;                // side of the square patches hidden together (1 = single pixels)
    int eval_day = -1;                    // 3D: report the hidden-pixel errors for this day only (-1 = all days)
    bool data_mode() const { return !data.empty(); }
};

inline void usage(const char* prog) {
    std::fprintf(stderr,
        "usage: %s [options]\n"
        "  --dim 2|3              problem dimension (default 2)\n"
        "  --n N | --n NX,NY[,NZ] global interior grid points per dimension (default 256)\n"
        "  --local-n M            weak scaling: interior points per rank per dimension\n"
        "  --solver jacobi|cg     iterative method (default jacobi)\n"
        "  --exchange MODE        halo exchange: sendrecv|isend|overlap|rma (default isend)\n"
        "  --dims PX,PY[,PZ]      force the MPI process grid\n"
        "  --iters K              maximum iterations (default 1000)\n"
        "  --tol T                stop when ||r||/||f|| < T (default 0: fixed iterations)\n"
        "  --check-every C        jacobi: check the residual every C iterations (default 0)\n"
        "  --threads T            OpenMP threads per rank\n"
        "  --csv FILE             append one result row to FILE\n"
        "  --tag TEXT             label stored in the CSV row\n"
        "  --cuda-aware           (mpi_cuda only) use CUDA-aware MPI\n"
        "  --verbose              print per-iteration progress\n"
        "satellite gap filling (replaces the manufactured problem):\n"
        "  --data FILE            grid file written by bin/gridtool (sets the grid size)\n"
        "  --day K                2D: time slice to fill (default 0); 3D: first slice\n"
        "  --days N               3D: number of slices to use (default 0 = all remaining)\n"
        "  --tweight W            3D: weight of time neighbours vs space neighbours (default 1)\n"
        "  --holdout F            hide this fraction of observed pixels to measure the fill error\n"
        "  --holdout-block B      hide square patches of BxB pixels instead of single pixels (default 1)\n"
        "  --eval-day K           3D: measure the hidden-pixel error on day K only (default: all days)\n"
        "  --seed S               seed of the holdout selection (default 1)\n"
        "  --ref FILE             gap-free reference grid on the same grid, for comparison\n"
        "  --out FILE             write the filled field (grid file)\n", prog);
}

// Strict number parsing: the whole string must be a number, otherwise stop with
// a message. (std::atoi would turn "abc" into 0 and the run would go ahead.)
inline long to_long(const std::string& s, const char* name) {
    char* end = nullptr;
    errno = 0;
    const long v = std::strtol(s.c_str(), &end, 10);
    if (s.empty() || errno != 0 || *end != '\0') { std::fprintf(stderr, "%s: '%s' is not an integer\n", name, s.c_str()); fail(2); }
    return v;
}
inline double to_double(const std::string& s, const char* name) {
    char* end = nullptr;
    errno = 0;
    const double v = std::strtod(s.c_str(), &end);
    if (s.empty() || errno != 0 || *end != '\0' || !std::isfinite(v)) { std::fprintf(stderr, "%s: '%s' is not a number\n", name, s.c_str()); fail(2); }
    return v;
}

// "256" -> {256,256,256}; "256,128" -> {256,128,fill}. Returns how many values were given.
inline int parse_list(const std::string& s, long* out, int maxn, long fill, const char* name) {
    std::vector<long> v;
    size_t pos = 0;
    while (pos <= s.size()) {
        size_t c = s.find(',', pos);
        if (c == std::string::npos) c = s.size();
        v.push_back(to_long(s.substr(pos, c - pos), name));
        pos = c + 1;
    }
    if ((int)v.size() > maxn) { std::fprintf(stderr, "%s: at most %d values\n", name, maxn); fail(2); }
    if (v.size() == 1) { for (int d = 0; d < maxn; ++d) out[d] = v[0]; return 1; }
    for (int d = 0; d < maxn; ++d) out[d] = d < (int)v.size() ? v[d] : fill;
    return (int)v.size();
}

inline Params parse_args(int argc, char** argv) {
    Params p;
    std::string nstr, dstr;
    for (int a = 1; a < argc; ++a) {
        std::string k = argv[a];
        auto need = [&](const char* name) -> std::string {
            if (a + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", name); usage(argv[0]); fail(2); }
            return argv[++a];
        };
        auto need_l = [&](const char* name) { return to_long(need(name), name); };
        auto need_d = [&](const char* name) { return to_double(need(name), name); };
        if      (k == "--dim")         p.dim = (int)need_l("--dim");
        else if (k == "--n")           nstr = need("--n");
        else if (k == "--local-n")     p.local_n = need_l("--local-n");
        else if (k == "--solver")      p.solver = need("--solver");
        else if (k == "--exchange")    p.exchange = need("--exchange");
        else if (k == "--dims")        dstr = need("--dims");
        else if (k == "--iters")       p.iters = need_l("--iters");
        else if (k == "--tol")         p.tol = need_d("--tol");
        else if (k == "--check-every") p.check_every = (int)need_l("--check-every");
        else if (k == "--threads")     p.threads = (int)need_l("--threads");
        else if (k == "--csv")         p.csv = need("--csv");
        else if (k == "--tag")         p.tag = need("--tag");
        else if (k == "--cuda-aware")  p.cuda_aware = true;
        else if (k == "--verbose")     p.verbose = true;
        else if (k == "--data")        p.data = need("--data");
        else if (k == "--out")         p.out = need("--out");
        else if (k == "--ref")         p.ref = need("--ref");
        else if (k == "--holdout")     p.holdout = need_d("--holdout");
        else if (k == "--tweight")     p.tweight = need_d("--tweight");
        else if (k == "--day")         p.day = (int)need_l("--day");
        else if (k == "--days")        p.days = (int)need_l("--days");
        else if (k == "--seed")        p.seed = (unsigned)need_l("--seed");
        else if (k == "--holdout-block") p.holdout_block = (int)need_l("--holdout-block");
        else if (k == "--eval-day")    p.eval_day = (int)need_l("--eval-day");
        else if (k == "-h" || k == "--help") { usage(argv[0]); std::exit(0); }
        else { std::fprintf(stderr, "unknown option %s\n", k.c_str()); usage(argv[0]); fail(2); }
    }
    auto bad = [](const char* msg) { std::fprintf(stderr, "%s\n", msg); fail(2); };
    if (p.dim != 2 && p.dim != 3) bad("--dim must be 2 or 3");
    if (!nstr.empty()) {
        const int given = parse_list(nstr, p.n, 3, 0, "--n");
        if (given != 1 && given != p.dim) bad("--n takes one value, or one value per dimension (NX,NY in 2D, NX,NY,NZ in 3D)");
    }
    if (p.dim == 2) p.n[2] = 1;
    for (int d = 0; d < p.dim; ++d) if (p.n[d] < 1) bad("--n must be at least 1 in every dimension");
    if (!dstr.empty()) {
        long d[3] = {0, 0, 0};
        const int given = parse_list(dstr, d, 3, 1, "--dims");
        for (int i = 0; i < 3; ++i) p.dims[i] = (int)d[i];
        if (given == 1) p.dims[1] = p.dims[2] = 0;
        for (int i = 0; i < 3; ++i) if (p.dims[i] < 0) bad("--dims values must be 0 (let MPI choose) or positive");
    }
    if (p.dim == 2) p.dims[2] = 1;
    if (p.solver != "jacobi" && p.solver != "cg") bad("--solver must be jacobi or cg");
    if (p.exchange != "sendrecv" && p.exchange != "isend" && p.exchange != "overlap" && p.exchange != "rma")
        bad("--exchange must be sendrecv|isend|overlap|rma");
    if (p.iters < 1) bad("--iters must be at least 1");
    if (p.tol < 0) bad("--tol must be 0 (fixed iterations) or positive");
    if (p.check_every < 0) bad("--check-every must be 0 (never) or positive");
    if (p.threads < 0) bad("--threads must be 0 (use OMP_NUM_THREADS) or positive");
    if (p.local_n < 0) bad("--local-n must be positive");
    if (p.holdout < 0 || p.holdout >= 1) bad("--holdout must be in [0,1)");
    if (p.holdout_block < 1) bad("--holdout-block must be at least 1");
    if (p.tweight < 0) bad("--tweight must not be negative");
    if (p.day < 0) bad("--day must not be negative");
    if (p.days < 0) bad("--days must be 0 (all remaining days) or positive");
    if (p.data_mode() && p.local_n > 0) bad("--local-n cannot be combined with --data");
    return p;
}

// Geometry of one rank's block (also used by the serial code with the whole grid as one block).
struct Local {
    int dim = 2;                      // problem dimension; a 2D block has nz = 1 and no z-neighbours
    int nx = 0, ny = 0, nz = 0;       // interior points owned by this rank
    long ox = 0, oy = 0, oz = 0;      // global index offset of local interior point (1,1,1)
    long sx = 1, sy = 0, sz = 0;      // strides
    size_t ntot = 0;                  // number of stored doubles including halos
    double hx = 0, hy = 0, hz = 0;    // grid spacings (hz unused in 2D)
    double ihx2 = 0, ihy2 = 0, ihz2 = 0; // 1/h^2 per direction (ihz2 = 0 in 2D)
    double diag = 0;                  // diagonal of the operator: 2*(ihx2+ihy2+ihz2)
    inline size_t idx(int i, int j, int k) const { return (size_t)k * sz + (size_t)j * sy + (size_t)i; }
};

inline Local make_local(const Params& p, const long gn[3], int nx, int ny, int nz, long ox, long oy, long oz) {
    Local L;
    L.dim = p.dim;
    L.nx = nx; L.ny = ny; L.nz = nz; L.ox = ox; L.oy = oy; L.oz = oz;
    L.sx = 1; L.sy = nx + 2; L.sz = (long)(nx + 2) * (ny + 2);
    L.ntot = (size_t)(nx + 2) * (ny + 2) * (nz + 2);
    if (p.data_mode()) {
        // pixels and days are the units: unit spacing in space, --tweight in time
        L.hx = L.hy = L.hz = 1.0;
        L.ihx2 = L.ihy2 = 1.0;
        L.ihz2 = p.dim == 3 ? p.tweight : 0.0;
    } else {
        L.hx = 1.0 / (gn[0] + 1); L.hy = 1.0 / (gn[1] + 1);
        L.ihx2 = 1.0 / (L.hx * L.hx); L.ihy2 = 1.0 / (L.hy * L.hy);
        if (p.dim == 3) { L.hz = 1.0 / (gn[2] + 1); L.ihz2 = 1.0 / (L.hz * L.hz); }
        else            { L.hz = 1.0; L.ihz2 = 0.0; }
    }
    L.diag = 2.0 * (L.ihx2 + L.ihy2 + L.ihz2);
    return L;
}

// Manufactured solution and right-hand side at a global interior index (gi,gj,gk), 1-based.
inline double exact_at(const Params& p, const Local& L, long gi, long gj, long gk) {
    const double x = gi * L.hx;
    double v = std::exp(x) * std::sin(PI * x) * std::sin(PI * gj * L.hy);
    if (p.dim == 3) v *= std::sin(PI * gk * L.hz);
    return v;
}
inline double rhs_at(const Params& p, const Local& L, long gi, long gj, long gk) {
    const double x = gi * L.hx;
    double v = std::exp(x) * std::sin(PI * gj * L.hy);
    if (p.dim == 3) v *= std::sin(PI * gk * L.hz);
    return v * ((p.dim * PI * PI - 1.0) * std::sin(PI * x) - 2.0 * PI * std::cos(PI * x));
}

// Block partition of n points over P ranks: ranks with coord < n%P get one extra point.
inline void partition(long n, int P, int coord, int& count, long& offset) {
    long base = n / P, rem = n % P;
    count = (int)(base + (coord < rem ? 1 : 0));
    offset = coord * base + (coord < rem ? coord : rem);
}

// Everything we record about one run.
struct Result {
    std::string impl;         // serial | omp | mpi | mpi_omp | mpi_cuda
    int ranks = 1, threads = 1;
    std::string dims_str = "1x1x1";
    long iters = 0; int converged = 0;
    double rel_resid = 0, l2_err = 0, linf_err = 0;   // data mode: l2_err = RMSE on hidden pixels, linf_err = max error
    double t_total = 0, t_setup = 0;
    double t_compute_max = 0, t_halo_max = 0, t_allreduce_max = 0;
    double t_compute_mean = 0, t_halo_mean = 0, t_allreduce_mean = 0;
    std::string hostname;
    // satellite gap filling only
    long n_obs = 0, n_unknown = 0, n_hidden = 0, n_land = 0;
    double mean_obs = 0;                 // mean of the observed pixels (the fill is computed as an anomaly around it)
    double mae_hidden = 0;               // mean absolute error on hidden pixels
    double rmse_baseline = 0;            // RMSE of the trivial fill "every hole = mean of the observed pixels"
    double rmse_ref_fill = 0;            // RMSE between filled pixels and the reference grid (if --ref)
    double rmse_ref_obs = 0;             // RMSE between observed pixels and the reference grid (sensor vs reference)
};

inline void write_csv(const Params& p, const Result& r) {
    if (p.csv.empty()) return;
    bool header = true;
    if (FILE* f = std::fopen(p.csv.c_str(), "r")) { std::fseek(f, 0, SEEK_END); header = std::ftell(f) == 0; std::fclose(f); }
    FILE* f = std::fopen(p.csv.c_str(), "a");
    if (!f) { std::perror("csv"); return; }
    if (header) std::fprintf(f,
        "impl,dim,nx,ny,nz,solver,exchange,ranks,threads,dims,iters,converged,rel_resid,l2_err,linf_err,"
        "t_total,t_setup,t_compute_max,t_halo_max,t_allreduce_max,t_compute_mean,t_halo_mean,t_allreduce_mean,"
        "check_every,tol,hostname,tag,"
        "data,holdout,holdout_block,tweight,n_obs,n_unknown,n_hidden,n_land,mean_obs,mae_hidden,rmse_baseline,rmse_ref_fill,rmse_ref_obs,eval_day\n");
    std::fprintf(f, "%s,%d,%ld,%ld,%ld,%s,%s,%d,%d,%s,%ld,%d,%.6e,%.17g,%.17g,"
        "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%g,%s,%s,"
        "%s,%g,%d,%g,%ld,%ld,%ld,%ld,%.6f,%.6f,%.6f,%.6f,%.6f,%d\n",
        r.impl.c_str(), p.dim, p.n[0], p.n[1], p.n[2], p.solver.c_str(), p.exchange.c_str(), r.ranks, r.threads,
        r.dims_str.c_str(), r.iters, r.converged, r.rel_resid, r.l2_err, r.linf_err,
        r.t_total, r.t_setup, r.t_compute_max, r.t_halo_max, r.t_allreduce_max,
        r.t_compute_mean, r.t_halo_mean, r.t_allreduce_mean, p.check_every, p.tol, r.hostname.c_str(), p.tag.c_str(),
        p.data.c_str(), p.holdout, p.holdout_block, p.tweight, r.n_obs, r.n_unknown, r.n_hidden, r.n_land, r.mean_obs,
        r.mae_hidden, r.rmse_baseline, r.rmse_ref_fill, r.rmse_ref_obs, p.eval_day);
    std::fclose(f);
}

inline void print_result(const Params& p, const Result& r) {
    std::printf("impl=%s dim=%d n=%ldx%ldx%ld solver=%s exchange=%s ranks=%d threads=%d dims=%s\n",
        r.impl.c_str(), p.dim, p.n[0], p.n[1], p.n[2], p.solver.c_str(), p.exchange.c_str(),
        r.ranks, r.threads, r.dims_str.c_str());
    if (p.data_mode()) {
        std::printf("data=%s pixels: observed=%ld missing=%ld hidden=%ld land=%ld  mean_obs=%.3f\n",
            p.data.c_str(), r.n_obs, r.n_unknown, r.n_hidden, r.n_land, r.mean_obs);
        std::printf("iters=%ld converged=%d rel_resid=%.3e", r.iters, r.converged, r.rel_resid);
        if (r.n_hidden > 0) std::printf("  hidden-pixel error: rmse=%.4f mae=%.4f max=%.4f  (baseline rmse=%.4f)",
            r.l2_err, r.mae_hidden, r.linf_err, r.rmse_baseline);
        if (!p.ref.empty()) std::printf("  vs reference: rmse_fill=%.4f rmse_obs=%.4f", r.rmse_ref_fill, r.rmse_ref_obs);
        std::printf("\n");
    } else {
        std::printf("iters=%ld converged=%d rel_resid=%.3e l2_err=%.6e linf_err=%.6e\n",
            r.iters, r.converged, r.rel_resid, r.l2_err, r.linf_err);
    }
    std::printf("t_total=%.4f s  (setup %.4f)  compute max/mean=%.4f/%.4f  halo max/mean=%.4f/%.4f  allreduce max/mean=%.4f/%.4f\n",
        r.t_total, r.t_setup, r.t_compute_max, r.t_compute_mean, r.t_halo_max, r.t_halo_mean,
        r.t_allreduce_max, r.t_allreduce_mean);
}

} // namespace poisson
