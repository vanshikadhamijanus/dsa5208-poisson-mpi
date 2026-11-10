// gridio.hpp -- the satellite data: grid file format, per-rank block loading,
// the pixel mask, the right-hand side of the gap-filling problem, and the
// error measures. Shared by the serial, MPI, MPI+OpenMP and MPI+CUDA drivers.
//
// Grid file (written by bin/gridtool from the downloaded ASCII grids):
//   GridHeader (84 bytes, little-endian, packed) followed by
//   float32 value[nt][ny][nx]   (time slowest, longitude fastest; latitude ascending)
//   value = temperature in degrees C, NaN = missing (cloud), LAND_VALUE = land.
//
// Mask codes per cell:
//   M_UNKNOWN  missing pixel, to be filled (an unknown of the linear system)
//   M_OBS      observed pixel, held fixed (Dirichlet data)
//   M_LAND     land or outside the box: excluded; coasts act as zero-flux walls
//   M_HIDDEN   observed pixel deliberately hidden (--holdout): solved as unknown,
//              its true value kept to measure the error of the fill
//
// The linear system is written for the anomaly a = u - mean(observed) on the
// unknown cells only. For an unknown cell c with active (non-land) neighbours
// n and weights w_n (1 in space, --tweight in time):
//     (sum_n w_n) a_c - sum_{n unknown} w_n a_n = sum_{n observed} w_n (obs_n - mean)
// The right-hand side f carries the observed neighbours; observed and land
// cells keep the value 0 in every solver array so the same halo exchange and
// dot products work unchanged. An unknown cell without any active neighbour
// (a one-pixel pond) gets the identity row a_c = 0, i.e. the mean.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <string>
#include <vector>
#include "common.hpp"

namespace poisson {

constexpr float LAND_VALUE = -999.0f;
enum : uint8_t { M_UNKNOWN = 0, M_OBS = 1, M_LAND = 2, M_HIDDEN = 3 };

#pragma pack(push, 1)
struct GridHeader {
    char magic[8];                  // "SSTGRD01"
    int32_t nx, ny, nt;             // longitudes, latitudes, days
    float lon0, dlon, lat0, dlat;   // centre of pixel (0,0) and spacing in degrees; latitude ascending
    char date0[16];                 // first day, "YYYY-MM-DD"
    char varname[16];               // e.g. "sst_degC"
    int32_t reserved[4];
};
#pragma pack(pop)
static_assert(sizeof(GridHeader) == 84, "GridHeader must be 84 bytes");

inline GridHeader make_header(int nx, int ny, int nt, float lon0, float dlon, float lat0, float dlat,
                              const std::string& date0, const std::string& var) {
    GridHeader h; std::memset(&h, 0, sizeof h);
    std::memcpy(h.magic, "SSTGRD01", 8);
    h.nx = nx; h.ny = ny; h.nt = nt; h.lon0 = lon0; h.dlon = dlon; h.lat0 = lat0; h.dlat = dlat;
    std::strncpy(h.date0, date0.c_str(), 15); std::strncpy(h.varname, var.c_str(), 15);
    return h;
}
inline bool read_header(const std::string& path, GridHeader& h) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    bool ok = std::fread(&h, sizeof h, 1, f) == 1 && std::memcmp(h.magic, "SSTGRD01", 8) == 0;
    std::fclose(f);
    return ok;
}
inline size_t grid_offset(const GridHeader& h, long t, long j, long i) {
    return sizeof(GridHeader) + (((size_t)t * h.ny + j) * h.nx + i) * sizeof(float);
}

// "YYYY-MM-DD" plus k days (proleptic Gregorian, Howard Hinnant's algorithms).
inline long days_from_civil(int y, int m, int d) {
    y -= m <= 2; const long era = (y >= 0 ? y : y - 399) / 400; const unsigned yoe = (unsigned)(y - era * 400);
    const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1; const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + (long)doe - 719468;
}
inline std::string civil_from_days(long z) {
    z += 719468; const long era = (z >= 0 ? z : z - 146096) / 146097; const unsigned doe = (unsigned)(z - era * 146097);
    const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; const long y = (long)yoe + era * 400;
    const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100); const unsigned mp = (5 * doy + 2) / 153;
    const unsigned d = doy - (153 * mp + 2) / 5 + 1; const unsigned m = mp + (mp < 10 ? 3 : -9);
    char b[32]; std::snprintf(b, sizeof b, "%04ld-%02u-%02u", y + (m <= 2), m, d); return b;
}
inline std::string add_days(const std::string& date, long k) {
    int y = 0, m = 0, d = 0; if (std::sscanf(date.c_str(), "%d-%d-%d", &y, &m, &d) != 3) return date;
    return civil_from_days(days_from_civil(y, m, d) + k);
}

// Deterministic pseudo-random number in [0,1) for a global pixel index: the same
// pixels are hidden whatever the rank count (splitmix64).
inline double hash01(long gi, long gj, long gk, unsigned seed) {
    uint64_t x = (uint64_t)gk * 0x9E3779B97F4A7C15ULL ^ (uint64_t)gj * 0xBF58476D1CE4E5B9ULL ^ (uint64_t)gi * 0x94D049BB133111EBULL ^ ((uint64_t)seed << 32);
    x += 0x9E3779B97F4A7C15ULL; x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL; x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL; x ^= x >> 31;
    return (double)(x >> 11) * (1.0 / 9007199254740992.0);
}

// Set the global grid size from the file header. Call before building the decomposition.
inline void data_configure(Params& P) {
    GridHeader h;
    if (!read_header(P.data, h)) { std::fprintf(stderr, "cannot read grid file %s\n", P.data.c_str()); fail(2); }
    if (P.day < 0 || P.day >= h.nt) { std::fprintf(stderr, "--day %d outside the file's %d days\n", P.day, h.nt); fail(2); }
    P.n[0] = h.nx; P.n[1] = h.ny;
    if (P.dim == 3) { int avail = h.nt - P.day; P.n[2] = (P.days > 0 && P.days < avail) ? P.days : avail; }
    else P.n[2] = 1;
    if (P.eval_day >= 0 && (P.eval_day < P.day || P.eval_day >= P.day + P.n[2])) {
        std::fprintf(stderr, "--eval-day %d is outside the selected days %d..%ld\n", P.eval_day, P.day, P.day + P.n[2] - 1); fail(2);
    }
    if (!P.ref.empty()) {
        GridHeader r;
        if (!read_header(P.ref, r) || r.nx != h.nx || r.ny != h.ny || r.nt != h.nt) {
            std::fprintf(stderr, "--ref must be a grid file with the same nx, ny, nt as --data\n"); fail(2);
        }
    }
}

// Read this rank's block, halo included, of the time slices [day0, day0+nsel). Halo
// cells outside the global box get LAND_VALUE. "Outside" in time means outside the
// selected window, not outside the file: with --day 90 --days 30 the day after the
// window (day 120) must not leak into the last slice's time halo as observed data.
// `dst` has L.ntot entries.
inline bool read_block(const std::string& path, const GridHeader& h, const Local& L, int day0, long nsel, float* dst) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    for (int k = 0; k <= L.nz + 1; ++k) {
        const long gk = L.oz + k - 1;                       // slice within the selected range
        for (int j = 0; j <= L.ny + 1; ++j) {
            const long gj = L.oy + j - 1;
            float* out = dst + L.idx(0, j, k);
            if (gk < 0 || gk >= nsel || gk >= (long)h.nt - day0 || gj < 0 || gj >= h.ny) { std::fill(out, out + L.nx + 2, LAND_VALUE); continue; }
            // columns ox-1 .. ox+nx, clamped to [0, nx)
            long i0 = L.ox - 1, i1 = L.ox + L.nx;           // inclusive global column range of the stored row
            long c0 = std::max(i0, 0L), c1 = std::min(i1, (long)h.nx - 1);
            std::fill(out, out + L.nx + 2, LAND_VALUE);
            if (c1 < c0) continue;
            if (std::fseek(f, (long)grid_offset(h, day0 + gk, gj, c0), SEEK_SET) != 0) { std::fclose(f); return false; }
            if (std::fread(out + (c0 - i0), sizeof(float), (size_t)(c1 - c0 + 1), f) != (size_t)(c1 - c0 + 1)) { std::fclose(f); return false; }
        }
    }
    std::fclose(f);
    return true;
}

struct DataBlock {
    GridHeader hdr;
    int day0 = 0;
    std::vector<float> obs;          // L.ntot: value, NaN (missing) or LAND_VALUE; halo included
    std::vector<uint8_t> mask;       // L.ntot
    std::vector<float> ref;          // L.ntot or empty
    std::vector<double> solution;    // L.ntot: the solved anomaly, filled in by solve()
    double mean = 0;                 // global mean of the observed pixels (set by finish_data)
    long n_obs = 0, n_unknown = 0, n_hidden = 0, n_land = 0;   // local interior counts
    double sum_obs = 0;              // local sum of observed values
};

// Read the block and classify every cell (halo included; the global mean is not known yet).
inline DataBlock load_data(const Params& P, const Local& L) {
    DataBlock D;
    if (!read_header(P.data, D.hdr)) { std::fprintf(stderr, "cannot read %s\n", P.data.c_str()); fail(2); }
    D.day0 = P.day;
    D.obs.assign(L.ntot, LAND_VALUE);
    if (!read_block(P.data, D.hdr, L, D.day0, P.n[2], D.obs.data())) { std::fprintf(stderr, "error reading %s\n", P.data.c_str()); fail(2); }
    if (!P.ref.empty()) {
        GridHeader rh;
        if (!read_header(P.ref, rh)) { std::fprintf(stderr, "cannot read %s\n", P.ref.c_str()); fail(2); }
        D.ref.assign(L.ntot, LAND_VALUE);
        if (!read_block(P.ref, rh, L, D.day0, P.n[2], D.ref.data())) { std::fprintf(stderr, "error reading %s\n", P.ref.c_str()); fail(2); }
    }
    D.mask.assign(L.ntot, M_LAND);
    for (int k = 0; k <= L.nz + 1; ++k)
        for (int j = 0; j <= L.ny + 1; ++j)
            for (int i = 0; i <= L.nx + 1; ++i) {
                const size_t c = L.idx(i, j, k);
                const float v = D.obs[c];
                uint8_t m;
                if (v == LAND_VALUE) m = M_LAND;
                else if (std::isnan(v)) m = M_UNKNOWN;
                else {
                    m = M_OBS;
                    if (P.holdout > 0 && hash01((L.ox + i) / P.holdout_block, (L.oy + j) / P.holdout_block, P.day + L.oz + k - 1, P.seed) < P.holdout) m = M_HIDDEN;   // absolute day index: the same pixels are hidden in a 2D and a 3D run
                }
                D.mask[c] = m;
                const bool interior = i >= 1 && i <= L.nx && j >= 1 && j <= L.ny && k >= 1 && k <= L.nz;
                if (interior) {
                    if (m == M_OBS) { ++D.n_obs; D.sum_obs += v; }
                    else if (m == M_UNKNOWN) ++D.n_unknown;
                    else if (m == M_HIDDEN) ++D.n_hidden;
                    else ++D.n_land;
                }
            }
    return D;
}

// Global mean of the observed pixels, then the right-hand side f. `sum` is a
// global-sum functor (MPI_Allreduce in the parallel drivers).
template <class SumFn>
inline void finish_data(DataBlock& D, const Local& L, SumFn sum, double* f) {
    const double s = sum(D.sum_obs), c = sum((double)D.n_obs);
    D.mean = c > 0 ? s / c : 0.0;
    const long sy = L.sy, sz = L.sz;
    const double w[3] = {L.ihx2, L.ihy2, L.ihz2};
    const long step[3] = {1, sy, sz};
    const int nd = L.ihz2 > 0 ? 3 : 2;
    const double mean = D.mean;
    const uint8_t* m = D.mask.data();
    const float* o = D.obs.data();
#pragma omp parallel for collapse(2) schedule(static)
    for (int k = 1; k <= L.nz; ++k)
        for (int j = 1; j <= L.ny; ++j)
            for (int i = 1; i <= L.nx; ++i) {
                const size_t cc = L.idx(i, j, k);
                double rhs = 0.0;
                if (m[cc] == M_UNKNOWN || m[cc] == M_HIDDEN)
                    for (int d = 0; d < nd; ++d) {
                        if (m[cc - step[d]] == M_OBS) rhs += w[d] * (o[cc - step[d]] - mean);
                        if (m[cc + step[d]] == M_OBS) rhs += w[d] * (o[cc + step[d]] - mean);
                    }
                f[cc] = rhs;
            }
}

// Local error sums after the solve. u holds the anomaly (0 at observed/land cells).
struct DataSums {
    double se_h = 0, ae_h = 0, mx_h = 0, se_base = 0;      // hidden pixels: squared error, abs error, max, baseline (mean) error
    double se_ref_fill = 0, se_ref_obs = 0;                 // against the reference grid
    double n_h = 0, n_fill = 0, n_refobs = 0;               // counts (as doubles so one reduction type serves)
};
inline DataSums data_local_errors(const DataBlock& D, const Local& L, const double* u, int eval_slice = -1) {
    // eval_slice: local index (0-based within the selected range) of the one day to evaluate, or -1 for all
    DataSums S;
    const bool have_ref = !D.ref.empty();
    for (int k = 1; k <= L.nz; ++k) {
        if (eval_slice >= 0 && L.oz + k - 1 != eval_slice) continue;
        for (int j = 1; j <= L.ny; ++j)
            for (int i = 1; i <= L.nx; ++i) {
                const size_t c = L.idx(i, j, k);
                const uint8_t m = D.mask[c];
                if (m == M_LAND) continue;
                const double fill = u[c] + D.mean;
                if (m == M_HIDDEN) {
                    const double e = fill - D.obs[c], eb = D.mean - D.obs[c];
                    S.se_h += e * e; S.ae_h += std::fabs(e); S.mx_h = std::max(S.mx_h, std::fabs(e)); S.se_base += eb * eb; S.n_h += 1;
                }
                if (have_ref) {
                    const float r = D.ref[c];
                    if (r == LAND_VALUE || std::isnan(r)) continue;
                    if (m == M_OBS) { const double e = D.obs[c] - r; S.se_ref_obs += e * e; S.n_refobs += 1; }
                    else            { const double e = fill - r;     S.se_ref_fill += e * e; S.n_fill += 1; }
                }
            }
    }
    return S;
}

// The filled field of this rank's interior as float32, k-major, ready to be written.
inline void compose_output(const DataBlock& D, const Local& L, const double* u, std::vector<float>& out) {
    out.resize((size_t)L.nx * L.ny * L.nz);
    size_t q = 0;
    for (int k = 1; k <= L.nz; ++k)
        for (int j = 1; j <= L.ny; ++j)
            for (int i = 1; i <= L.nx; ++i) {
                const size_t c = L.idx(i, j, k);
                const uint8_t m = D.mask[c];
                out[q++] = m == M_LAND ? LAND_VALUE : m == M_OBS ? D.obs[c] : (float)(u[c] + D.mean);
            }
}

// Header of an output file covering the selected slices.
inline GridHeader output_header(const DataBlock& D, const Params& P) {
    GridHeader h = D.hdr;
    h.nt = (int32_t)P.n[2];
    std::string d0 = add_days(D.hdr.date0, D.day0);
    std::memset(h.date0, 0, sizeof h.date0); std::strncpy(h.date0, d0.c_str(), 15);
    std::memset(h.varname, 0, sizeof h.varname); std::strncpy(h.varname, "sst_filled", 15);
    return h;
}

// Serial writer (whole grid is one block).
inline bool write_grid_serial(const std::string& path, const GridHeader& h, const std::vector<float>& data) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    bool ok = std::fwrite(&h, sizeof h, 1, f) == 1 && std::fwrite(data.data(), sizeof(float), data.size(), f) == data.size();
    std::fclose(f);
    return ok;
}

} // namespace poisson
