// gridtool.cpp -- prepare and inspect the satellite grids (no Python anywhere).
//
//   bin/gridtool convert  OUT.grid [--mask MASK.asc] [--stride S] [--var NAME] DAY1.asc DAY2.asc ...
//       Stack the daily ERDDAP ASCII grids (one per day, sorted by file name, which
//       ends in _YYYY-MM-DD.asc) into one grid file. Rows are reordered so that
//       latitude ascends. Empty values become NaN (missing). With --mask, the land
//       mask of the blended product (a byte grid on its own lat/lon axes) is
//       resampled by nearest neighbour and land pixels become LAND_VALUE.
//       --stride S keeps every S-th pixel in both directions (smaller test grids).
//   bin/gridtool regrid   TARGET.grid OUT.grid [--offset X] [--stride S] REF1.asc REF2.asc ...
//       Resample the daily reference grids (e.g. the blended analysis in kelvin, so
//       --offset -273.15) onto the pixel centres of TARGET.grid by nearest neighbour,
//       producing a grid file with the same nx, ny, nt. Land pixels of TARGET are
//       copied as LAND_VALUE.
//   bin/gridtool info     FILE.grid [--days]
//       Print the header and the pixel statistics (observed / missing / land); with
//       --days, one line per day.
//   bin/gridtool slice    FILE.grid K OUT.txt
//       Write day K as a text matrix (one row per latitude, "NaN" for missing/land)
//       for gnuplot's `matrix with image`.
//   bin/gridtool diff     A.grid KA B.grid KB OUT.txt
//       Write day KA of A minus day KB of B as a text matrix where both are valid.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
#include "gridio.hpp"

using namespace poisson;
static const float NaN = std::numeric_limits<float>::quiet_NaN();

// One ERDDAP .asc grid: a 2D field plus its latitude and longitude axes.
struct Asc { int nlat = 0, nlon = 0; std::vector<double> lat, lon; std::vector<float> v; /* [lat][lon] as stored */ };

static std::vector<double> parse_numbers(const std::string& line) {
    std::vector<double> out; std::string tok; std::istringstream in(line);
    while (std::getline(in, tok, ',')) {
        size_t a = tok.find_first_not_of(" \t\r"); if (a == std::string::npos) { out.push_back(NAN); continue; }
        size_t b = tok.find_last_not_of(" \t\r"); std::string s = tok.substr(a, b - a + 1);
        if (s.empty() || s == "NaN" || s == "nan") out.push_back(NAN); else out.push_back(std::atof(s.c_str()));
    }
    return out;
}

static bool read_asc(const std::string& path, Asc& g) {
    std::ifstream f(path); if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }
    std::string line; std::vector<std::vector<double>> rows; std::string pending_axis;
    while (std::getline(f, line)) {
        if (!line.empty() && line[0] == '[') {                       // a data row: "[t][a][j], v, v, ..."
            size_t p = line.find("], "); if (p == std::string::npos) continue;
            rows.push_back(parse_numbers(line.substr(p + 3)));
        } else if (!pending_axis.empty()) {                          // the values of an axis announced on the previous line
            auto vals = parse_numbers(line);
            if (pending_axis == "latitude") g.lat = vals; else if (pending_axis == "longitude") g.lon = vals;
            pending_axis.clear();
        } else {
            size_t p = line.rfind('['), d = line.rfind('.');
            if (p != std::string::npos && d != std::string::npos && d < p) {
                std::string name = line.substr(d + 1, p - d - 1);
                if (name == "latitude" || name == "longitude") pending_axis = name;
            }
        }
    }
    if (rows.empty() || g.lat.empty() || g.lon.empty()) { std::fprintf(stderr, "%s: no grid found\n", path.c_str()); return false; }
    g.nlat = (int)g.lat.size(); g.nlon = (int)g.lon.size();
    if ((int)rows.size() != g.nlat) { std::fprintf(stderr, "%s: %zu rows but %d latitudes\n", path.c_str(), rows.size(), g.nlat); return false; }
    g.v.assign((size_t)g.nlat * g.nlon, NaN);
    for (int j = 0; j < g.nlat; ++j) {
        if ((int)rows[j].size() != g.nlon) { std::fprintf(stderr, "%s: row %d has %zu values, expected %d\n", path.c_str(), j, rows[j].size(), g.nlon); return false; }
        for (int i = 0; i < g.nlon; ++i) g.v[(size_t)j * g.nlon + i] = (float)rows[j][i];
    }
    return true;
}

// Index of the axis entry nearest to x (axis may ascend or descend).
static int nearest(const std::vector<double>& ax, double x) {
    int best = 0; double bd = std::fabs(ax[0] - x);
    for (size_t k = 1; k < ax.size(); ++k) { double d = std::fabs(ax[k] - x); if (d < bd) { bd = d; best = (int)k; } }
    return best;
}
static std::string date_from_name(const std::string& path) {
    size_t e = path.rfind(".asc"); if (e == std::string::npos || e < 10) return "0000-00-00";
    return path.substr(e - 10, 10);
}

// Build the target axes (ascending latitude, ascending longitude, optional stride) from the first day.
struct Target { int nx, ny; std::vector<double> lon, lat; std::vector<int> ilon, ilat; /* source indices */ };
static Target make_target(const Asc& a, int stride) {
    Target T;
    std::vector<int> jl(a.nlat), il(a.nlon);
    for (int j = 0; j < a.nlat; ++j) jl[j] = j;
    for (int i = 0; i < a.nlon; ++i) il[i] = i;
    std::sort(jl.begin(), jl.end(), [&](int x, int y) { return a.lat[x] < a.lat[y]; });
    std::sort(il.begin(), il.end(), [&](int x, int y) { return a.lon[x] < a.lon[y]; });
    for (int j = 0; j < a.nlat; j += stride) { T.ilat.push_back(jl[j]); T.lat.push_back(a.lat[jl[j]]); }
    for (int i = 0; i < a.nlon; i += stride) { T.ilon.push_back(il[i]); T.lon.push_back(a.lon[il[i]]); }
    T.nx = (int)T.lon.size(); T.ny = (int)T.lat.size();
    return T;
}

static int cmd_convert(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: gridtool convert OUT.grid [--mask MASK.asc] [--stride S] [--var NAME] DAY.asc ...\n"); return 2; }
    std::string out = argv[2], maskfile, var = "sst_degC"; int stride = 1; std::vector<std::string> days;
    for (int a = 3; a < argc; ++a) {
        std::string k = argv[a];
        if (k == "--mask" && a + 1 < argc) maskfile = argv[++a];
        else if (k == "--stride" && a + 1 < argc) stride = std::max(1, std::atoi(argv[++a]));
        else if (k == "--var" && a + 1 < argc) var = argv[++a];
        else days.push_back(k);
    }
    std::sort(days.begin(), days.end());
    Asc first; if (!read_asc(days[0], first)) return 1;
    Target T = make_target(first, stride);
    std::vector<uint8_t> land((size_t)T.nx * T.ny, 0);
    if (!maskfile.empty()) {
        Asc m; if (!read_asc(maskfile, m)) return 1;
        long nl = 0;
        for (int j = 0; j < T.ny; ++j) for (int i = 0; i < T.nx; ++i) {
            int mj = nearest(m.lat, T.lat[j]), mi = nearest(m.lon, T.lon[i]);
            float mv = m.v[(size_t)mj * m.nlon + mi];
            int bits = std::isnan(mv) ? 0 : (int)mv;
            if (bits == 0 || (bits & 2)) { land[(size_t)j * T.nx + i] = 1; ++nl; }     // bit 2 = land in the blended mask; 0 = fill value
        }
        std::printf("mask: %ld of %d land pixels (%.1f%%)\n", nl, T.nx * T.ny, 100.0 * nl / (T.nx * T.ny));
    }
    GridHeader h = make_header(T.nx, T.ny, (int)days.size(), (float)T.lon[0], (float)(T.nx > 1 ? T.lon[1] - T.lon[0] : 0),
                               (float)T.lat[0], (float)(T.ny > 1 ? T.lat[1] - T.lat[0] : 0), date_from_name(days[0]), var);
    FILE* f = std::fopen(out.c_str(), "wb"); if (!f) { std::perror(out.c_str()); return 1; }
    std::fwrite(&h, sizeof h, 1, f);
    std::vector<float> slab((size_t)T.nx * T.ny);
    long tot_obs = 0; const long sea = (long)T.nx * T.ny - std::count(land.begin(), land.end(), 1);
    std::string expect = h.date0;
    for (size_t t = 0; t < days.size(); ++t) {
        Asc a; if (t == 0) a = first; else if (!read_asc(days[t], a)) return 1;
        if (a.nlat != first.nlat || a.nlon != first.nlon) { std::fprintf(stderr, "%s: grid size differs from the first day\n", days[t].c_str()); return 1; }
        std::string d = date_from_name(days[t]);
        if (d != expect) std::fprintf(stderr, "warning: %s is not the expected date %s (days must be consecutive)\n", days[t].c_str(), expect.c_str());
        expect = add_days(d, 1);
        long obs = 0;
        for (int j = 0; j < T.ny; ++j) for (int i = 0; i < T.nx; ++i) {
            size_t q = (size_t)j * T.nx + i;
            float v = a.v[(size_t)T.ilat[j] * a.nlon + T.ilon[i]];
            if (land[q]) v = LAND_VALUE; else if (!std::isnan(v)) ++obs;
            slab[q] = v;
        }
        tot_obs += obs;
        std::fwrite(slab.data(), sizeof(float), slab.size(), f);
        std::printf("%s  observed %6.1f%% of sea pixels\n", d.c_str(), sea ? 100.0 * obs / sea : 0.0);
    }
    std::fclose(f);
    std::printf("wrote %s: %d x %d pixels x %zu days, lon %.3f..%.3f lat %.3f..%.3f, observed %.1f%% of sea pixels overall\n",
                out.c_str(), T.nx, T.ny, days.size(), T.lon.front(), T.lon.back(), T.lat.front(), T.lat.back(),
                sea ? 100.0 * tot_obs / (sea * (double)days.size()) : 0.0);
    return 0;
}

static int cmd_regrid(int argc, char** argv) {
    if (argc < 5) { std::fprintf(stderr, "usage: gridtool regrid TARGET.grid OUT.grid [--offset X] REF.asc ...\n"); return 2; }
    std::string target = argv[2], out = argv[3]; double offset = 0; std::vector<std::string> days;
    for (int a = 4; a < argc; ++a) { std::string k = argv[a]; if (k == "--offset" && a + 1 < argc) offset = std::atof(argv[++a]); else days.push_back(k); }
    std::sort(days.begin(), days.end());
    GridHeader h; if (!read_header(target, h)) { std::fprintf(stderr, "cannot read %s\n", target.c_str()); return 1; }
    if ((int)days.size() != h.nt) { std::fprintf(stderr, "target has %d days but %zu reference files were given\n", h.nt, days.size()); return 1; }
    FILE* ft = std::fopen(target.c_str(), "rb"); FILE* fo = std::fopen(out.c_str(), "wb");
    if (!ft || !fo) { std::perror("regrid"); return 1; }
    GridHeader ho = h; std::memset(ho.varname, 0, sizeof ho.varname); std::strncpy(ho.varname, "reference", 15);
    std::fwrite(&ho, sizeof ho, 1, fo);
    std::vector<float> tslab((size_t)h.nx * h.ny), oslab(tslab.size());
    std::vector<int> jmap, imap;
    for (int t = 0; t < h.nt; ++t) {
        Asc r; if (!read_asc(days[t], r)) return 1;
        if (t == 0) {
            jmap.resize(h.ny); imap.resize(h.nx);
            for (int j = 0; j < h.ny; ++j) jmap[j] = nearest(r.lat, h.lat0 + j * (double)h.dlat);
            for (int i = 0; i < h.nx; ++i) imap[i] = nearest(r.lon, h.lon0 + i * (double)h.dlon);
        }
        std::fseek(ft, (long)grid_offset(h, t, 0, 0), SEEK_SET);
        if (std::fread(tslab.data(), sizeof(float), tslab.size(), ft) != tslab.size()) { std::fprintf(stderr, "short read\n"); return 1; }
        for (int j = 0; j < h.ny; ++j) for (int i = 0; i < h.nx; ++i) {
            size_t q = (size_t)j * h.nx + i;
            float v = r.v[(size_t)jmap[j] * r.nlon + imap[i]];
            oslab[q] = tslab[q] == LAND_VALUE ? LAND_VALUE : std::isnan(v) ? NaN : (float)(v + offset);
        }
        std::fwrite(oslab.data(), sizeof(float), oslab.size(), fo);
    }
    std::fclose(ft); std::fclose(fo);
    std::printf("wrote %s (%d x %d x %d, nearest-neighbour resampled, offset %g)\n", out.c_str(), h.nx, h.ny, h.nt, offset);
    return 0;
}

static bool read_slice(FILE* f, const GridHeader& h, int t, std::vector<float>& s) {
    s.resize((size_t)h.nx * h.ny);
    std::fseek(f, (long)grid_offset(h, t, 0, 0), SEEK_SET);
    return std::fread(s.data(), sizeof(float), s.size(), f) == s.size();
}

static int cmd_info(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: gridtool info FILE.grid [--days]\n"); return 2; }
    bool perday = argc > 3 && std::string(argv[3]) == "--days";
    GridHeader h; if (!read_header(argv[2], h)) { std::fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }
    std::printf("%s: %d x %d pixels x %d days, variable %s, first day %s\n", argv[2], h.nx, h.ny, h.nt, h.varname, h.date0);
    std::printf("lon %.4f .. %.4f step %.4f   lat %.4f .. %.4f step %.4f\n", h.lon0, h.lon0 + (h.nx - 1) * (double)h.dlon, h.dlon, h.lat0, h.lat0 + (h.ny - 1) * (double)h.dlat, h.dlat);
    FILE* f = std::fopen(argv[2], "rb"); std::vector<float> s;
    long obs = 0, miss = 0, land = 0; double sum = 0, mn = 1e30, mx = -1e30;
    for (int t = 0; t < h.nt; ++t) {
        if (!read_slice(f, h, t, s)) { std::fprintf(stderr, "short read\n"); return 1; }
        long o = 0, m = 0, l = 0;
        for (float v : s) { if (v == LAND_VALUE) ++l; else if (std::isnan(v)) ++m; else { ++o; sum += v; mn = std::min(mn, (double)v); mx = std::max(mx, (double)v); } }
        obs += o; miss += m; land += l;
        if (perday) std::printf("%s  observed %6.1f%%  missing %6.1f%% of sea pixels\n", add_days(h.date0, t).c_str(), 100.0 * o / std::max(1L, o + m), 100.0 * m / std::max(1L, o + m));
    }
    std::fclose(f);
    std::printf("pixels: observed %ld (%.1f%% of sea), missing %ld (%.1f%% of sea), land %ld (%.1f%% of box)\n", obs, 100.0 * obs / std::max(1L, obs + miss), miss, 100.0 * miss / std::max(1L, obs + miss), land, 100.0 * land / ((double)h.nx * h.ny * h.nt));
    if (obs) std::printf("observed values: mean %.3f  min %.3f  max %.3f\n", sum / obs, mn, mx);
    return 0;
}

static int write_matrix(const std::string& out, const GridHeader& h, const std::vector<float>& s) {
    std::ofstream o(out); if (!o) { std::perror(out.c_str()); return 1; }
    o << "# lon0 " << h.lon0 << " dlon " << h.dlon << " lat0 " << h.lat0 << " dlat " << h.dlat << "\n";
    for (int j = 0; j < h.ny; ++j) {
        for (int i = 0; i < h.nx; ++i) { float v = s[(size_t)j * h.nx + i]; if (v == LAND_VALUE || std::isnan(v)) o << "NaN"; else o << v; o << (i + 1 < h.nx ? ' ' : '\n'); }
    }
    return 0;
}
static int cmd_slice(int argc, char** argv) {
    if (argc < 5) { std::fprintf(stderr, "usage: gridtool slice FILE.grid K OUT.txt\n"); return 2; }
    GridHeader h; if (!read_header(argv[2], h)) { std::fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }
    int k = std::atoi(argv[3]); if (k < 0 || k >= h.nt) { std::fprintf(stderr, "day %d out of range\n", k); return 2; }
    FILE* f = std::fopen(argv[2], "rb"); std::vector<float> s; bool ok = read_slice(f, h, k, s); std::fclose(f);
    if (!ok) return 1;
    return write_matrix(argv[4], h, s);
}
static int cmd_diff(int argc, char** argv) {
    if (argc < 7) { std::fprintf(stderr, "usage: gridtool diff A.grid KA B.grid KB OUT.txt\n"); return 2; }
    GridHeader ha, hb; if (!read_header(argv[2], ha) || !read_header(argv[4], hb)) { std::fprintf(stderr, "cannot read inputs\n"); return 1; }
    if (ha.nx != hb.nx || ha.ny != hb.ny) { std::fprintf(stderr, "grids differ in size\n"); return 1; }
    int k = std::atoi(argv[3]), kb = std::atoi(argv[5]);
    if (k < 0 || k >= ha.nt || kb < 0 || kb >= hb.nt) { std::fprintf(stderr, "day index out of range\n"); return 2; }
    FILE* fa = std::fopen(argv[2], "rb"); FILE* fb = std::fopen(argv[4], "rb"); std::vector<float> a, b;
    bool ok = read_slice(fa, ha, k, a) && read_slice(fb, hb, kb, b); std::fclose(fa); std::fclose(fb);
    if (!ok) return 1;
    double se = 0; long n = 0;
    for (size_t q = 0; q < a.size(); ++q) {
        bool va = a[q] != LAND_VALUE && !std::isnan(a[q]), vb = b[q] != LAND_VALUE && !std::isnan(b[q]);
        if (va && vb) { a[q] -= b[q]; se += (double)a[q] * a[q]; ++n; } else a[q] = NaN;
    }
    std::printf("day %d vs day %d: %ld common pixels, rmse %.4f\n", k, kb, n, n ? std::sqrt(se / n) : 0.0);
    return write_matrix(argv[6], ha, a);
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: gridtool convert|regrid|info|slice|diff ...  (see the comment at the top of tools/gridtool.cpp)\n"); return 2; }
    std::string cmd = argv[1];
    if (cmd == "convert") return cmd_convert(argc, argv);
    if (cmd == "regrid")  return cmd_regrid(argc, argv);
    if (cmd == "info")    return cmd_info(argc, argv);
    if (cmd == "slice")   return cmd_slice(argc, argv);
    if (cmd == "diff")    return cmd_diff(argc, argv);
    std::fprintf(stderr, "unknown command %s\n", cmd.c_str());
    return 2;
}
