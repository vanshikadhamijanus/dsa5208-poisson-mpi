// summarize.cpp -- turn the CSV files written by the solvers into tables and plot data.
//
// Usage:  bin/summarize <results-dir> <out-dir> ["machine label"]
//         bin/summarize <results-dir> <out-dir> "label" <second-results-dir> "second label"   (adds the node-layout comparison)
//   reads   <results-dir>/{validation,serial_scaling,strong,weak,exchange,decomp,checkevery,hybrid,omp_threads,gpu,gpu_multi,gapfill}.csv
//   writes  <out-dir>/tables.md        every table in Markdown (same titles as the report uses)
//           <out-dir>/summary.txt      headline numbers
//           <out-dir>/*.dat            whitespace-separated data for analysis/plots.gp (gnuplot)
//
// Every configuration was run more than once; the MINIMUM time per configuration is kept,
// because any disturbance (another process, a page fault) can only add time.
// No external libraries: C++17 standard library only.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

struct Row {
    std::string impl, solver, exchange, dims, hostname, tag;
    int dim = 0; long nx = 0, ny = 0, nz = 0, iters = 0; int ranks = 1, threads = 1, converged = 0, check_every = 0;
    double rel_resid = 0, l2 = 0, linf = 0, t_total = 0, t_setup = 0, tc_max = 0, th_max = 0, ta_max = 0, tc_mean = 0, th_mean = 0, ta_mean = 0, tol = 0;
    // satellite gap filling columns (present in files written after the data extension; zero otherwise)
    std::string data; double holdout = 0, tweight = 0, mean_obs = 0, mae = 0, rmse_base = 0, rmse_ref_fill = 0, rmse_ref_obs = 0; int holdout_block = 1, eval_day = -1; long n_obs = 0, n_unknown = 0, n_hidden = 0, n_land = 0;
    long points() const { return nx * ny * nz; }
    int cores() const { return ranks * threads; }
    std::string key() const {
        std::ostringstream k; k << holdout_block << '|' << tweight << '|' << impl << '|' << dim << '|' << nx << '|' << ny << '|' << nz << '|' << solver << '|' << exchange << '|' << ranks << '|' << threads << '|' << dims << '|' << check_every << '|' << tag; return k.str();
    }
};

static std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> v; std::string cur; std::istringstream in(s);
    while (std::getline(in, cur, sep)) v.push_back(cur);
    if (!s.empty() && s.back() == sep) v.push_back("");
    return v;
}

static std::vector<Row> load(const std::string& path) {
    std::vector<Row> rows; std::ifstream f(path); std::string line;
    if (!f) return rows;
    std::getline(f, line); // header
    while (std::getline(f, line)) {
        auto c = split(line, ',');
        if (c.size() < 27) continue;
        Row r; r.impl = c[0]; r.dim = std::stoi(c[1]); r.nx = std::stol(c[2]); r.ny = std::stol(c[3]); r.nz = std::stol(c[4]);
        r.solver = c[5]; r.exchange = c[6]; r.ranks = std::stoi(c[7]); r.threads = std::stoi(c[8]); r.dims = c[9]; r.iters = std::stol(c[10]);
        r.converged = std::stoi(c[11]); r.rel_resid = std::stod(c[12]); r.l2 = std::stod(c[13]); r.linf = std::stod(c[14]);
        r.t_total = std::stod(c[15]); r.t_setup = std::stod(c[16]); r.tc_max = std::stod(c[17]); r.th_max = std::stod(c[18]); r.ta_max = std::stod(c[19]);
        r.tc_mean = std::stod(c[20]); r.th_mean = std::stod(c[21]); r.ta_mean = std::stod(c[22]); r.check_every = std::stoi(c[23]); r.tol = std::stod(c[24]);
        r.hostname = c[25]; r.tag = c[26];
        if (c.size() >= 40) {
            r.data = c[27]; r.holdout = std::stod(c[28]); r.holdout_block = std::stoi(c[29]); r.tweight = std::stod(c[30]);
            r.n_obs = std::stol(c[31]); r.n_unknown = std::stol(c[32]); r.n_hidden = std::stol(c[33]); r.n_land = std::stol(c[34]);
            r.mean_obs = std::stod(c[35]); r.mae = std::stod(c[36]); r.rmse_base = std::stod(c[37]); r.rmse_ref_fill = std::stod(c[38]); r.rmse_ref_obs = std::stod(c[39]);
            if (c.size() >= 41) r.eval_day = std::stoi(c[40]);
        }
        rows.push_back(r);
    }
    return rows;
}

// minimum t_total per configuration
static std::vector<Row> best(const std::vector<Row>& rows) {
    std::map<std::string, Row> m;
    for (const auto& r : rows) { auto it = m.find(r.key()); if (it == m.end() || r.t_total < it->second.t_total) m[r.key()] = r; }
    std::vector<Row> v; for (auto& kv : m) v.push_back(kv.second); return v;
}
template <class Pred> static std::vector<Row> where(const std::vector<Row>& v, Pred p) { std::vector<Row> o; for (auto& r : v) if (p(r)) o.push_back(r); return o; }

static const char* TAGS[4] = {"jac2d", "jac3d", "cg2d", "cg3d"};
static std::string title(const std::string& tag) {
    if (tag == "jac2d") return "Jacobi 2D"; if (tag == "jac3d") return "Jacobi 3D"; if (tag == "cg2d") return "CG 2D"; if (tag == "cg3d") return "CG 3D"; return tag;
}
static double slope(const std::vector<double>& x, const std::vector<double>& y) {   // least-squares slope of log y vs log x
    if (x.size() < 2) return NAN;
    double sx = 0, sy = 0, sxx = 0, sxy = 0; size_t n = x.size();
    for (size_t i = 0; i < n; ++i) { double lx = std::log(x[i]), ly = std::log(y[i]); sx += lx; sy += ly; sxx += lx * lx; sxy += lx * ly; }
    return (n * sxy - sx * sy) / (n * sxx - sx * sx);
}
template <class F> static std::vector<int> distinct(const std::vector<Row>& v, F f) { std::set<int> s; for (auto& r : v) s.insert(f(r)); return {s.begin(), s.end()}; }
static std::string fmt(double v, const char* f) { char b[64]; std::snprintf(b, sizeof b, f, v); return b; }
static std::string pct(double v) { return fmt(100 * v, "%.1f%%"); }

struct Out {
    std::string dir, label; std::ofstream tables, summary;
    Out(const std::string& d, const std::string& l) : dir(d), label(l), tables(d + "/tables.md"), summary(d + "/summary.txt") {
        tables << "# Tables generated by analysis/summarize.cpp\n\nMachine: " << label << "\n\n";
        summary << "machine: " << label << "\n";
    }
    std::ofstream dat(const std::string& name) { std::ofstream f(dir + "/" + name); f << "# generated by analysis/summarize.cpp\n"; return f; }
    void table(const std::string& t, const std::vector<std::string>& cols, const std::vector<std::vector<std::string>>& rows, const std::string& note = "") {
        tables << "### " << t << "\n\n|"; for (auto& c : cols) tables << " " << c << " |"; tables << "\n|"; for (size_t i = 0; i < cols.size(); ++i) tables << "---|"; tables << "\n";
        for (auto& r : rows) { tables << "|"; for (auto& c : r) tables << " " << c << " |"; tables << "\n"; }
        tables << "\n"; if (!note.empty()) tables << note << "\n\n"; tables << "\n";
    }
};

// ------------------------------------------------------------------ validation
static void do_validation(const std::vector<Row>& all, Out& O) {
    if (all.empty()) return;
    for (int dim : {2, 3}) {
        std::string tag = dim == 2 ? "order2d" : "order3d";
        auto d = where(all, [&](const Row& r) { return r.tag == tag; });
        std::sort(d.begin(), d.end(), [](const Row& a, const Row& b) { return a.nx < b.nx; });
        if (d.empty()) continue;
        std::vector<std::vector<std::string>> rows; double prev_err = 0; long prev_n = 0;
        for (auto& r : d) {
            double h = 1.0 / (r.nx + 1), ratio = prev_n ? prev_err / r.l2 : NAN, order = prev_n ? std::log(ratio) / std::log(double(r.nx + 1) / (prev_n + 1)) : NAN;
            rows.push_back({std::to_string(r.nx), fmt(h, "%.5f"), std::to_string(r.iters), fmt(r.rel_resid, "%.2e"), fmt(r.l2, "%.3e"), fmt(r.linf, "%.3e"), prev_n ? fmt(ratio, "%.2f") : "", prev_n ? fmt(order, "%.2f") : ""});
            if (prev_n) O.summary << "order_" << dim << "d_last: " << order << "\n";
            prev_err = r.l2; prev_n = r.nx;
        }
        O.table("Discretisation error against the exact solution, " + std::to_string(dim) + "D (serial CG, tol 1e-10)",
                {"nx", "h", "iters", "rel_resid", "l2_err", "linf_err", "ratio", "order"}, rows,
                "`ratio` is the error on the previous (coarser) grid divided by this one; 4.0 means the error quarters when h halves (second order). `order` is the measured convergence exponent.");
    }
    auto a = where(all, [](const Row& r) { return r.tag == "agree"; });
    if (!a.empty()) {
        auto cs = [](const Row& r) { return r.solver + " " + std::to_string(r.dim) + "D n=" + std::to_string(r.nx); };
        std::sort(a.begin(), a.end(), [&](const Row& x, const Row& y) { return std::make_tuple(cs(x), x.impl, x.ranks, x.threads, x.exchange) < std::make_tuple(cs(y), y.impl, y.ranks, y.threads, y.exchange); });
        std::vector<std::vector<std::string>> rows; std::map<std::string, double> ref; double maxdev = 0;
        for (auto& r : a) if (r.impl == "serial") ref[cs(r)] = r.l2;
        for (auto& r : a) {
            rows.push_back({cs(r), r.impl, std::to_string(r.ranks), std::to_string(r.threads), r.exchange, r.dims, std::to_string(r.iters), fmt(r.rel_resid, "%.3e"), fmt(r.l2, "%.10e"), fmt(r.linf, "%.10e")});
            if (ref.count(cs(r))) maxdev = std::max(maxdev, std::fabs(r.l2 - ref[cs(r)]));
        }
        O.table("Agreement of every implementation with the serial reference (same problem, same iterations)",
                {"case", "impl", "ranks", "threads", "exchange", "dims", "iters", "rel_resid", "l2_err", "linf_err"}, rows,
                "For Jacobi with a fixed sweep count every variant must agree to all printed digits (the arithmetic is identical; only the summation order of the diagnostics differs). For CG the dot products are summed in a different order on each rank count, so the residual differs in its last digits while the error agrees.");
        O.summary << "max_l2_deviation_from_serial: " << maxdev << "\n";
    }
    auto j = where(all, [](const Row& r) { return r.tag == "jacobi_conv"; });
    if (!j.empty()) {
        std::vector<std::vector<std::string>> rows;
        for (auto& r : j) rows.push_back({r.impl, std::to_string(r.dim), std::to_string(r.nx), std::to_string(r.ranks), std::to_string(r.iters), fmt(r.rel_resid, "%.3e"), fmt(r.l2, "%.4e"), fmt(r.t_total, "%.3f")});
        O.table("Jacobi run to convergence (tol 1e-8): serial and MPI agree on the iteration count", {"impl", "dim", "nx", "ranks", "iters", "rel_resid", "l2_err", "t_total"}, rows);
    }
}

// ------------------------------------------------------------------ serial scaling
static void do_serial(const std::vector<Row>& all, Out& O) {
    if (all.empty()) return;
    auto b = best(all); std::vector<std::vector<std::string>> rows;
    for (auto tag : TAGS) {
        auto d = where(b, [&](const Row& r) { return r.tag == tag; });
        std::sort(d.begin(), d.end(), [](const Row& x, const Row& y) { return x.points() < y.points(); });
        if (d.empty()) continue;
        auto f = O.dat(std::string("serial_") + tag + ".dat"); f << "# points t_total t_per_iter\n";
        std::vector<double> xs, yt, yi;
        for (auto& r : d) {
            double per = r.t_total / r.iters; xs.push_back(r.points()); yt.push_back(r.t_total); yi.push_back(per);
            f << r.points() << " " << r.t_total << " " << per << "\n";
            rows.push_back({title(tag), std::to_string(r.nx), std::to_string(r.points()), std::to_string(r.iters), fmt(r.t_total, "%.3f"), fmt(1e3 * per, "%.3f"), fmt(r.points() * r.iters / r.t_total / 1e6, "%.1f")});
        }
        O.summary << "serial_slope_total_" << tag << ": " << slope(xs, yt) << "\nserial_slope_per_iter_" << tag << ": " << slope(xs, yi) << "\n";
    }
    O.table("Serial runtime against problem size (minimum of repeats)", {"case", "n", "points", "iters", "t_total", "t_per_iter_ms", "Mupdates_per_s"}, rows,
            "Per-iteration time should grow linearly with N (slope 1 on log-log axes). Jacobi runs a fixed 200 sweeps, so total time is also linear; CG runs to tolerance 1e-8 and its iteration count grows like n, so total time grows like N^(1+1/dim).");
}

// ------------------------------------------------------------------ strong scaling
static void do_strong(const std::vector<Row>& all, Out& O) {
    if (all.empty()) return;
    auto b = best(all); std::vector<std::vector<std::string>> rows;
    for (auto tag : TAGS) {
        auto d = where(b, [&](const Row& r) { return r.tag == tag; });
        auto ser = where(d, [](const Row& r) { return r.impl == "serial"; });
        auto par = where(d, [](const Row& r) { return r.impl != "serial"; });
        std::sort(par.begin(), par.end(), [](const Row& x, const Row& y) { return x.cores() < y.cores(); });
        if (par.empty()) continue;
        double t1 = ser.empty() ? par[0].t_total : ser[0].t_total;
        auto f = O.dat(std::string("strong_") + tag + ".dat"); f << "# cores t_total speedup efficiency comm_frac\n";
        for (auto& r : par) {
            double sp = t1 / r.t_total, ef = sp / r.cores(), cf = (r.th_max + r.ta_max) / r.t_total, imb = r.tc_mean > 0 ? r.tc_max / r.tc_mean : 1;
            f << r.cores() << " " << r.t_total << " " << sp << " " << ef << " " << cf << "\n";
            rows.push_back({title(tag), std::to_string(r.nx), std::to_string(r.cores()), r.dims, std::to_string(r.iters), fmt(r.t_total, "%.3f"), fmt(sp, "%.2f"), fmt(ef, "%.2f"), fmt(r.tc_max, "%.3f"), fmt(imb, "%.2f"), fmt(r.th_max, "%.3f"), fmt(r.ta_max, "%.3f"), pct(cf)});
        }
        const Row& last = par.back();
        O.summary << "strong_" << tag << ": max_cores=" << last.cores() << " speedup=" << t1 / last.t_total << " efficiency=" << t1 / last.t_total / last.cores() << " serial_time=" << t1 << "\n";
    }
    O.table("Strong scaling (strong): fixed problem, increasing cores. Speedup is relative to the serial binary.",
            {"case", "n", "cores", "dims", "iters", "t_total", "speedup", "efficiency", "t_compute_max", "imbalance", "t_halo_max", "t_allreduce_max", "comm_frac"}, rows,
            "`imbalance` is the slowest rank's compute time divided by the mean over ranks; 1.0 means perfectly balanced work. A value well above 1 means some ranks finish late and the others wait for them inside the halo exchange or the all-reduce.");
}

// ------------------------------------------------------------------ weak scaling
static void do_weak(const std::vector<Row>& all, Out& O) {
    if (all.empty()) return;
    auto b = best(all); std::vector<std::vector<std::string>> rows;
    for (auto tag : TAGS) {
        auto d = where(b, [&](const Row& r) { return r.tag == tag; });
        std::sort(d.begin(), d.end(), [](const Row& x, const Row& y) { return x.cores() < y.cores(); });
        if (d.empty()) continue;
        double t1 = d[0].t_total;
        auto f = O.dat(std::string("weak_") + tag + ".dat"); f << "# cores t_total weak_efficiency comm_frac\n";
        for (auto& r : d) {
            double we = t1 / r.t_total, cf = (r.th_max + r.ta_max) / r.t_total, imb = r.tc_mean > 0 ? r.tc_max / r.tc_mean : 1;
            f << r.cores() << " " << r.t_total << " " << we << " " << cf << "\n";
            std::string grid = std::to_string(r.nx) + "x" + std::to_string(r.ny) + (r.dim == 3 ? "x" + std::to_string(r.nz) : "");
            rows.push_back({title(tag), std::to_string(r.cores()), grid, std::to_string(r.points()), r.dims, std::to_string(r.iters), fmt(r.t_total, "%.3f"), fmt(we, "%.2f"), fmt(imb, "%.2f"), pct(cf)});
        }
        O.summary << "weak_" << tag << ": max_cores=" << d.back().cores() << " efficiency=" << t1 / d.back().t_total << "\n";
    }
    O.table("Weak scaling: the grid grows with the rank count so each rank always owns the same number of points. Ideal is a flat time (efficiency 1).",
            {"case", "cores", "global_grid", "points", "dims", "iters", "t_total", "weak_efficiency", "imbalance", "comm_frac"}, rows);
}

// ------------------------------------------------------------------ exchange modes
static void do_exchange(const std::vector<Row>& all, Out& O) {
    if (all.empty()) return;
    auto b = best(all);
    std::set<std::pair<std::string, int>> groups; for (auto& r : b) groups.insert({r.tag, r.ranks});
    std::vector<std::pair<std::string, int>> g(groups.begin(), groups.end());
    std::sort(g.begin(), g.end(), [](auto& x, auto& y) { return std::tie(x.first, x.second) < std::tie(y.first, y.second); });
    const char* modes[4] = {"sendrecv", "isend", "overlap", "rma"};
    std::vector<std::vector<std::string>> rows;
    auto lab = O.dat("exchange_labels.dat"); lab << "# index label\n";
    for (size_t gi = 0; gi < g.size(); ++gi) lab << gi << " \"" << title(g[gi].first) << " p=" << g[gi].second << "\"\n";
    for (int mi = 0; mi < 4; ++mi) {
        auto f = O.dat(std::string("exchange_") + modes[mi] + ".dat"); f << "# group_index t_total t_halo_max label\n";
        for (size_t gi = 0; gi < g.size(); ++gi) {
            auto d = where(b, [&](const Row& r) { return r.tag == g[gi].first && r.ranks == g[gi].second && r.exchange == modes[mi]; });
            if (!d.empty()) f << gi << " " << d[0].t_total << " " << d[0].th_max << " \"" << title(g[gi].first) << " p=" << g[gi].second << "\"\n";
        }
    }
    for (auto& grp : g) {
        auto d = where(b, [&](const Row& r) { return r.tag == grp.first && r.ranks == grp.second; });
        auto ref = where(d, [](const Row& r) { return r.exchange == "isend"; });
        std::sort(d.begin(), d.end(), [](const Row& x, const Row& y) { return x.exchange < y.exchange; });
        std::string bestmode; double bt = 1e300;
        for (auto& r : d) {
            rows.push_back({title(grp.first), std::to_string(grp.second), r.exchange, fmt(r.t_total, "%.3f"), fmt(r.tc_max, "%.3f"), fmt(r.th_max, "%.3f"), fmt(r.ta_max, "%.3f"), ref.empty() ? "" : fmt(r.t_total / ref[0].t_total, "%.2f")});
            if (r.t_total < bt) { bt = r.t_total; bestmode = r.exchange; }
        }
        O.summary << "exchange_best " << title(grp.first) << " p=" << grp.second << ": " << bestmode << "\n";
    }
    O.table("Halo exchange strategies compared (same problem, same ranks). `vs_isend` is the time relative to the non-blocking baseline.",
            {"case", "cores", "exchange", "t_total", "t_compute_max", "t_halo_max", "t_allreduce_max", "vs_isend"}, rows);
}

// ------------------------------------------------------------------ process-grid shape
static double surface_cells(const Row& r) {
    auto p = split(r.dims, 'x'); double px = std::stod(p[0]), py = std::stod(p[1]), pz = std::stod(p[2]);
    double lx = r.nx / px, ly = r.ny / py, lz = r.nz / pz, s = 0;
    if (px > 1) s += 2 * ly * lz; if (py > 1) s += 2 * lx * lz; if (r.dim == 3 && pz > 1) s += 2 * lx * ly; return s;
}
static void do_decomp(const std::vector<Row>& all, Out& O) {
    if (all.empty()) return;
    auto b = best(all); std::vector<std::vector<std::string>> rows;
    // one group per rank count: shapes are only comparable at the same number of ranks.
    // decomp_<tag>.dat holds the largest rank count (the figure); decomp_<tag>_p<N>.dat every count.
    auto plist = distinct(b, [](const Row& r) { return r.ranks; });
    for (auto tag : {"jac2d", "jac3d"}) for (int p : plist) {
        auto d = where(b, [&](const Row& r) { return r.tag == tag && r.ranks == p; });
        std::sort(d.begin(), d.end(), [](const Row& x, const Row& y) { return surface_cells(x) < surface_cells(y); });
        if (d.empty()) continue;
        std::vector<std::string> names{std::string("decomp_") + tag + "_p" + std::to_string(p) + ".dat"};
        if (p == plist.back()) names.push_back(std::string("decomp_") + tag + ".dat");
        for (auto& nm : names) {
            auto f = O.dat(nm); f << "# index dims t_total t_halo_max halo_cells   (" << p << " ranks)\n";
            int i = 0; for (auto& r : d) f << i++ << " " << r.dims << " " << r.t_total << " " << r.th_max << " " << (long)surface_cells(r) << "\n";
        }
        for (auto& r : d) rows.push_back({title(tag), std::to_string(p), r.dims, std::to_string((long)surface_cells(r)), fmt(r.t_total, "%.3f"), fmt(r.tc_max, "%.3f"), fmt(r.th_max, "%.3f")});
    }
    O.table("Shape of the process grid at fixed rank count. `halo_cells_per_rank` is the number of boundary values each rank sends per sweep (its surface area); fewer means less communication volume, but contiguity of the faces also matters. Compare shapes only within the same `cores` value.",
            {"case", "cores", "dims", "halo_cells_per_rank", "t_total", "t_compute_max", "t_halo_max"}, rows);
}

// ------------------------------------------------------------------ residual-check interval
static void do_checkevery(const std::vector<Row>& all, Out& O) {
    if (all.empty()) return;
    auto b = best(all); std::vector<std::vector<std::string>> rows;
    auto plist = distinct(b, [](const Row& r) { return r.ranks; });
    for (auto tag : {"jac2d", "jac3d"}) for (int p : plist) {
        auto d = where(b, [&](const Row& r) { return r.tag == tag && r.ranks == p; });
        std::sort(d.begin(), d.end(), [](const Row& x, const Row& y) { int a = x.check_every ? x.check_every : 1000, c = y.check_every ? y.check_every : 1000; return a < c; });
        if (d.empty()) continue;
        std::vector<std::string> names{std::string("checkevery_") + tag + "_p" + std::to_string(p) + ".dat"};
        if (p == plist.back()) names.push_back(std::string("checkevery_") + tag + ".dat");
        for (auto& nm : names) {
            auto f = O.dat(nm); f << "# x(1000=never) t_total t_allreduce_max   (" << p << " ranks)\n";
            for (auto& r : d) f << (r.check_every ? r.check_every : 1000) << " " << r.t_total << " " << r.ta_max << "\n";
        }
        for (auto& r : d) {
            rows.push_back({title(tag), r.check_every ? std::to_string(r.check_every) : "never", std::to_string(r.ranks), fmt(r.t_total, "%.3f"), fmt(r.tc_max, "%.3f"), fmt(r.th_max, "%.3f"), fmt(r.ta_max, "%.3f")});
        }
    }
    O.table("Residual-check interval for Jacobi. Each check costs one extra halo exchange, one residual sweep and one MPI_Allreduce.",
            {"case", "check_every", "cores", "t_total", "t_compute_max", "t_halo_max", "t_allreduce_max"}, rows);
}

// ------------------------------------------------------------------ MPI+OpenMP split
static void do_hybrid(const std::vector<Row>& all, Out& O) {
    if (all.empty()) return;
    auto b = best(all); std::vector<std::vector<std::string>> rows;
    // one group per total core count (ranks x threads); hybrid_<tag>.dat is the largest, hybrid_<tag>_c<N>.dat every count.
    auto clist = distinct(b, [](const Row& r) { return r.cores(); });
    for (auto tag : TAGS) for (int c : clist) {
        auto d = where(b, [&](const Row& r) { return r.tag == tag && r.cores() == c; });
        std::sort(d.begin(), d.end(), [](const Row& x, const Row& y) { return x.threads < y.threads; });
        if (d.empty()) continue;
        std::vector<std::string> names{std::string("hybrid_") + tag + "_c" + std::to_string(c) + ".dat"};
        if (c == clist.back()) names.push_back(std::string("hybrid_") + tag + ".dat");
        for (auto& nm : names) {
            auto f = O.dat(nm); f << "# index label t_total comm_share_percent   (" << c << " cores)\n";
            int i = 0; for (auto& r : d) f << i++ << " " << r.ranks << "x" << r.threads << " " << r.t_total << " " << 100 * (r.th_max + r.ta_max) / r.t_total << "\n";
        }
        std::string bestsplit; double bt = 1e300;
        for (auto& r : d) {
            std::string lab = std::to_string(r.ranks) + "x" + std::to_string(r.threads);
            rows.push_back({title(tag), std::to_string(r.ranks), std::to_string(r.threads), std::to_string(r.cores()), r.dims, fmt(r.t_total, "%.3f"), fmt(r.tc_max, "%.3f"), fmt(r.th_max, "%.3f"), fmt(r.ta_max, "%.3f")});
            if (r.t_total < bt) { bt = r.t_total; bestsplit = lab; }
        }
        O.summary << "hybrid_best_split_" << tag << "_c" << c << ": " << bestsplit << "\n";
    }
    O.table("MPI+OpenMP: the total core count is fixed and divided between ranks and threads.",
            {"case", "ranks", "threads", "cores", "dims", "t_total", "t_compute_max", "t_halo_max", "t_allreduce_max"}, rows);
}

// ------------------------------------------------------------------ OpenMP threads
static void do_omp(const std::vector<Row>& all, Out& O) {
    if (all.empty()) return;
    auto b = best(all); std::vector<std::vector<std::string>> rows;
    for (auto tag : TAGS) {
        auto d = where(b, [&](const Row& r) { return r.tag == tag; });
        std::sort(d.begin(), d.end(), [](const Row& x, const Row& y) { return x.threads < y.threads; });
        if (d.empty()) continue;
        double t1 = d[0].t_total;
        auto f = O.dat(std::string("omp_") + tag + ".dat"); f << "# threads t_total speedup efficiency\n";
        for (auto& r : d) {
            double sp = t1 / r.t_total;
            f << r.threads << " " << r.t_total << " " << sp << " " << sp / r.threads << "\n";
            rows.push_back({title(tag), std::to_string(r.threads), fmt(r.t_total, "%.3f"), fmt(sp, "%.2f"), fmt(sp / r.threads, "%.2f")});
        }
    }
    O.table("OpenMP-only thread scaling (no MPI).", {"case", "threads", "t_total", "speedup", "efficiency"}, rows);
}

// ------------------------------------------------------------------ GPU
static void do_gpu(const std::vector<Row>& all, Out& O) {
    if (all.empty()) return;
    auto b = best(all); std::vector<std::vector<std::string>> rows;
    for (auto tag : TAGS) for (auto impl : {"mpi_cuda", "mpi_omp"}) {
        auto d = where(b, [&](const Row& r) { return r.tag == tag && r.impl == impl; });
        std::sort(d.begin(), d.end(), [](const Row& x, const Row& y) { return x.points() < y.points(); });
        if (d.empty()) continue;
        auto f = O.dat(std::string("gpu_") + tag + "_" + impl + ".dat"); f << "# points t_per_iter\n";
        for (auto& r : d) {
            f << r.points() << " " << r.t_total / r.iters << "\n";
            rows.push_back({title(tag), impl, std::to_string(r.nx), std::to_string(r.points()), std::to_string(r.threads), std::to_string(r.iters), fmt(r.t_total, "%.3f"), fmt(1e3 * r.t_total / r.iters, "%.3f"), fmt(r.points() * r.iters / r.t_total / 1e6, "%.1f"), fmt(r.l2, "%.3e")});
        }
    }
    O.table("GPU (MPI+CUDA, one rank, one GPU) against a full CPU node (MPI+OpenMP, one rank).",
            {"case", "impl", "n", "points", "threads", "iters", "t_total", "t_per_iter_ms", "Mupdates_per_s", "l2_err"}, rows);
}
// ------------------------------------------------------------------ satellite gap filling
static std::string gtitle(const std::string& tag) {
    if (tag == "sst2d_cg") return "SST 2D (one day), CG"; if (tag == "sst2d_jac") return "SST 2D (one day), Jacobi";
    if (tag == "sst3d_cg") return "SST 3D (year), CG"; if (tag == "sst3d_jac") return "SST 3D (year), Jacobi"; return tag;
}
static void do_gapfill(const std::vector<Row>& all, Out& O) {
    if (all.empty()) return;
    auto b = best(all);
    // 1. strong scaling on the real data
    std::vector<std::vector<std::string>> rows;
    for (auto tag : {"sst2d_cg", "sst2d_jac", "sst3d_cg", "sst3d_jac"}) {
        auto d = where(b, [&](const Row& r) { return r.tag == tag; });
        auto ser = where(d, [](const Row& r) { return r.impl == "serial"; });
        auto par = where(d, [](const Row& r) { return r.impl != "serial"; });
        std::sort(par.begin(), par.end(), [](const Row& x, const Row& y) { return std::make_tuple(x.cores(), x.impl) < std::make_tuple(y.cores(), y.impl); });
        if (par.empty()) continue;
        double t1 = ser.empty() ? par[0].t_total : ser[0].t_total;
        auto f = O.dat(std::string("gapfill_") + tag + ".dat"); f << "# cores t_total speedup efficiency comm_frac   (pure MPI rows only; the table also lists OpenMP and hybrid)\n";
        for (auto& r : par) {
            double sp = t1 / r.t_total, ef = sp / r.cores(), cf = (r.th_max + r.ta_max) / r.t_total, imb = r.tc_mean > 0 ? r.tc_max / r.tc_mean : 1;
            if (r.impl == "mpi") f << r.cores() << " " << r.t_total << " " << sp << " " << ef << " " << cf << "\n";
            std::string grid = std::to_string(r.nx) + "x" + std::to_string(r.ny) + (r.dim == 3 ? "x" + std::to_string(r.nz) : "");
            rows.push_back({gtitle(tag), r.impl, grid, std::to_string(r.cores()), r.dims, std::to_string(r.iters), fmt(r.t_total, "%.3f"), fmt(sp, "%.2f"), fmt(ef, "%.2f"), fmt(imb, "%.2f"), pct(cf), fmt(r.l2, "%.4f")});
        }
        const Row& last = par.back();
        O.summary << "gapfill_" << tag << ": max_cores=" << last.cores() << " speedup=" << t1 / last.t_total << " efficiency=" << t1 / last.t_total / last.cores() << " serial_time=" << t1 << " iters=" << last.iters << " rmse_hidden=" << last.l2 << "\n";
    }
    if (!rows.empty()) O.table("Strong scaling on the satellite data (speedup relative to the serial binary). `rmse_hidden` is the error on the hidden pixels and must not depend on the rank count.",
        {"case", "impl", "grid", "cores", "dims", "iters", "t_total", "speedup", "efficiency", "imbalance", "comm_frac", "rmse_hidden"}, rows);
    // 2. accuracy of the fill: hidden patch size and time weight
    auto acc = where(all, [](const Row& r) { return r.tag == "acc2d" || r.tag == "acc3d" || r.tag == "acc3d_day"; });
    if (!acc.empty()) {
        std::sort(acc.begin(), acc.end(), [](const Row& x, const Row& y) { return std::make_tuple(x.tag, x.tweight, x.holdout_block, x.solver) < std::make_tuple(y.tag, y.tweight, y.holdout_block, y.solver); });
        rows.clear();
        auto f2 = O.dat("gapfill_acc2d.dat"); f2 << "# patch_side rmse mae max baseline\n";
        auto f3 = O.dat("gapfill_acc3d.dat"); f3 << "# patch_side rmse mae max baseline   (tweight 1, all days)\n";
        auto f3d = O.dat("gapfill_acc3d_day.dat"); f3d << "# patch_side rmse mae max baseline   (tweight 1, the 2D day only)\n";
        auto ft = O.dat("gapfill_tweight.dat"); ft << "# tweight rmse mae iters t_total   (3D, patch side 16, all days)\n";
        auto ftd = O.dat("gapfill_tweight_day.dat"); ftd << "# tweight rmse mae iters t_total rmse_2d_fill   (3D, patch side 16, the 2D day only; last column: the 2D fill of that day)\n";
        double rmse2d = NAN; for (auto& r : acc) if (r.tag == "acc2d" && r.holdout_block == 16) rmse2d = r.l2;
        for (auto& r : acc) {
            rows.push_back({r.tag == "acc2d" ? "2D (one day)" : r.tag == "acc3d_day" ? "3D (year), error on the 2D day" : "3D (year), all days", r.solver, std::to_string(r.holdout_block), fmt(r.tweight, "%g"), std::to_string(r.n_hidden), fmt(r.l2, "%.4f"), fmt(r.mae, "%.4f"), fmt(r.linf, "%.3f"), fmt(r.rmse_base, "%.4f"), r.rmse_ref_fill > 0 ? fmt(r.rmse_ref_fill, "%.4f") : "", r.rmse_ref_obs > 0 ? fmt(r.rmse_ref_obs, "%.4f") : "", std::to_string(r.iters), fmt(r.t_total, "%.2f")});
            if (r.tag == "acc2d") f2 << r.holdout_block << " " << r.l2 << " " << r.mae << " " << r.linf << " " << r.rmse_base << "\n";
            if (r.tag == "acc3d" && r.tweight == 1.0) f3 << r.holdout_block << " " << r.l2 << " " << r.mae << " " << r.linf << " " << r.rmse_base << "\n";
            if (r.tag == "acc3d_day" && r.tweight == 1.0) f3d << r.holdout_block << " " << r.l2 << " " << r.mae << " " << r.linf << " " << r.rmse_base << "\n";
            if (r.tag == "acc3d" && r.holdout_block == 16) ft << r.tweight << " " << r.l2 << " " << r.mae << " " << r.iters << " " << r.t_total << "\n";
            if (r.tag == "acc3d_day" && r.holdout_block == 16) ftd << r.tweight << " " << r.l2 << " " << r.mae << " " << r.iters << " " << r.t_total << " " << rmse2d << "\n";
        }
        O.table("Accuracy of the fill on hidden pixels. `patch` is the side of the square patches of observed pixels that were hidden (1 = single pixels); `baseline` is the error of filling every hole with the mean observed temperature; `rmse_ref_*` compare the filled (resp. observed) pixels with the gap-free blended analysis.",
            {"case", "solver", "patch", "tweight", "n_hidden", "rmse", "mae", "max", "baseline", "rmse_ref_fill", "rmse_ref_obs", "iters", "t_total"}, rows);
    }
}

static void do_gpu_multi(const std::vector<Row>& all, Out& O) {
    if (all.empty()) return;
    auto b = best(all); std::vector<std::vector<std::string>> rows;
    std::sort(b.begin(), b.end(), [](const Row& x, const Row& y) { return std::tie(x.tag, x.ranks, x.exchange) < std::tie(y.tag, y.ranks, y.exchange); });
    for (auto& r : b) rows.push_back({r.tag, std::to_string(r.ranks), r.exchange, std::to_string(r.nx), std::to_string(r.ny), std::to_string(r.nz), std::to_string(r.iters), fmt(r.t_total, "%.3f"), fmt(r.tc_max, "%.3f"), fmt(r.th_max, "%.3f"), fmt(r.ta_max, "%.3f"), fmt(r.l2, "%.3e")});
    O.table("Multi-GPU runs (one MPI rank per GPU).", {"tag", "ranks", "exchange", "nx", "ny", "nz", "iters", "t_total", "t_compute_max", "t_halo_max", "t_allreduce_max", "l2_err"}, rows);
}

// ------------------------------------------------------------------ satellite data on the GPU node
static void do_gapfill_gpu(const std::vector<Row>& all, Out& O) {
    if (all.empty()) return;
    auto b = best(all); std::vector<std::vector<std::string>> rows;
    std::sort(b.begin(), b.end(), [](const Row& x, const Row& y) { return std::make_tuple(x.tag, x.ranks, x.impl) < std::make_tuple(y.tag, y.ranks, y.impl); });
    for (auto& r : b) {
        std::string grid = std::to_string(r.nx) + "x" + std::to_string(r.ny) + (r.dim == 3 ? "x" + std::to_string(r.nz) : "");
        std::string dev = r.impl == "mpi_cuda" ? std::to_string(r.ranks) + " GPU" + (r.ranks > 1 ? "s" : "") : std::to_string(r.cores()) + " CPU cores";
        rows.push_back({r.tag, r.impl, dev, grid, std::to_string(r.iters), fmt(r.t_total, "%.3f"), fmt(1e3 * r.t_total / r.iters, "%.2f"), fmt(r.l2, "%.4f")});
        O.summary << "gapfill_gpu " << r.tag << " " << dev << ": t_total=" << r.t_total << " rmse_hidden=" << r.l2 << "\n";
    }
    O.table("Satellite gap filling on the GPU node: one or two GPUs against the 36 CPU cores of the same node (OpenMP). The hidden-pixel error must equal the CPU value.",
            {"case", "impl", "device", "grid", "iters", "t_total", "ms_per_iter", "rmse_hidden"}, rows);
}

// ------------------------------------------------------------------ node layout: same runs in a second results folder
// Matches every configuration that exists in both folders (same key) and compares the times.
// Used for 256 cores as 4 nodes x 64 cores (main folder) against 8 nodes x 32 cores (second folder).
static void do_layout(const std::string& in, const std::string& alt, const std::string& altlab, Out& O) {
    const char* files[] = {"strong", "weak", "exchange", "checkevery", "decomp", "hybrid", "gapfill"};
    std::vector<std::vector<std::string>> rows;
    auto f = O.dat("layout_compare.dat"); f << "# index label t_main t_alt   (strong and weak scaling only)\n";
    int idx = 0;
    for (auto file : files) {
        auto A = best(load(in + "/" + file + ".csv")), B = best(load(alt + "/" + file + ".csv"));
        std::map<std::string, Row> mb; for (auto& r : B) mb[r.key()] = r;
        std::vector<std::pair<Row, Row>> pairs;
        for (auto& r : A) { auto it = mb.find(r.key()); if (it != mb.end()) pairs.push_back({r, it->second}); }
        if (pairs.empty()) continue;
        std::sort(pairs.begin(), pairs.end(), [](auto& x, auto& y) { return std::make_tuple(x.first.tag, x.first.threads, x.first.exchange, x.first.dims, x.first.check_every, x.first.holdout_block, x.first.tweight) < std::make_tuple(y.first.tag, y.first.threads, y.first.exchange, y.first.dims, y.first.check_every, y.first.holdout_block, y.first.tweight); });
        double logsum = 0;
        for (auto& pr : pairs) {
            const Row &a = pr.first, &b = pr.second;
            std::string variant = std::string(file) == "exchange" ? a.exchange : std::string(file) == "decomp" ? a.dims
                : std::string(file) == "checkevery" ? (a.check_every ? std::to_string(a.check_every) : "never")
                : (a.tag.rfind("acc", 0) == 0 ? "patch " + std::to_string(a.holdout_block) + ", tweight " + fmt(a.tweight, "%g") : "");
            double ca = (a.th_max + a.ta_max) / a.t_total, cb = (b.th_max + b.ta_max) / b.t_total;
            rows.push_back({file, title(a.tag), std::to_string(a.ranks) + "x" + std::to_string(a.threads), variant, fmt(a.t_total, "%.3f"), fmt(b.t_total, "%.3f"), fmt(b.t_total / a.t_total, "%.2f"), pct(ca), pct(cb)});
            logsum += std::log(b.t_total / a.t_total);
            if (std::string(file) == "strong" || std::string(file) == "weak")
                f << idx++ << " \"" << title(a.tag) << " " << file << "\" " << a.t_total << " " << b.t_total << "\n";
        }
        O.summary << "layout_" << file << ": pairs=" << pairs.size() << " geomean_ratio_alt_over_main=" << std::exp(logsum / pairs.size()) << "\n";
    }
    O.table("Node layout: the same runs in " + O.label + " (main) and " + altlab + " (alt). `ratio` is alt time / main time; above 1 means the alternative layout is slower.",
            {"file", "case", "ranks x threads", "variant", "t_main", "t_alt", "ratio", "comm_main", "comm_alt"}, rows);
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s <results-dir> <out-dir> [label] [second-results-dir second-label]\n", argv[0]); return 2; }
    std::string in = argv[1], out = argv[2], label = argc > 3 ? argv[3] : "local machine";
    Out O(out, label);
    if (!O.tables) { std::fprintf(stderr, "cannot write to %s (does the folder exist?)\n", out.c_str()); return 1; }
    do_validation(load(in + "/validation.csv"), O);
    do_serial(load(in + "/serial_scaling.csv"), O);
    do_strong(load(in + "/strong.csv"), O);
    do_weak(load(in + "/weak.csv"), O);
    do_exchange(load(in + "/exchange.csv"), O);
    do_decomp(load(in + "/decomp.csv"), O);
    do_checkevery(load(in + "/checkevery.csv"), O);
    do_hybrid(load(in + "/hybrid.csv"), O);
    do_omp(load(in + "/omp_threads.csv"), O);
    do_gpu(load(in + "/gpu.csv"), O);
    do_gpu_multi(load(in + "/gpu_multi.csv"), O);
    do_gapfill(load(in + "/gapfill.csv"), O);
    do_gapfill_gpu(load(in + "/gapfill_gpu.csv"), O);
    if (argc > 5) do_layout(in, argv[4], argv[5], O);
    std::printf("wrote %s/tables.md, %s/summary.txt and the .dat files for analysis/plots.gp\n", out.c_str(), out.c_str());
    return 0;
}
