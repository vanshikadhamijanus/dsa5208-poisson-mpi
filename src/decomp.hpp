// decomp.hpp -- MPI domain decomposition and halo exchange.
//
// The global grid is split into a Cartesian grid of blocks, one per MPI rank
// (MPI_Cart_create). Each block stores one layer of halo cells that mirror the
// neighbouring rank's outermost interior layer. Before each stencil sweep the
// halos are refreshed. Four exchange strategies are implemented so their cost
// can be compared:
//   sendrecv : blocking MPI_Sendrecv, one direction at a time
//   isend    : all MPI_Irecv/MPI_Isend posted at once, then MPI_Waitall
//   overlap  : as isend, but the deep interior is computed while messages fly
//   rma      : one-sided MPI_Put into the neighbour's halo inside a fence epoch
// Non-contiguous faces are described with MPI_Type_create_subarray so that no
// manual packing is needed on the CPU.
#pragma once
#include <mpi.h>
#include "common.hpp"
#include "kernels.hpp"

namespace poisson {

struct Decomp {
    MPI_Comm cart = MPI_COMM_NULL;
    int rank = 0, size = 1;
    int dims[3] = {1, 1, 1}, coords[3] = {0, 0, 0};
    int lo[3] = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};  // neighbour towards smaller index
    int hi[3] = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};  // neighbour towards larger index
    Local L;
    // datatypes describing the faces of *this* rank's block, per dimension d:
    MPI_Datatype send_lo[3], recv_lo[3], send_hi[3], recv_hi[3];
    // datatypes describing the halo layer of the lo / hi *neighbour's* block (for RMA puts)
    MPI_Datatype nbr_lo_halo[3], nbr_hi_halo[3];
    MPI_Win win = MPI_WIN_NULL;
    int ndims_active = 3;   // 2 for a 2D problem (no z exchange)
};

// Build a subarray type that selects the face of a (nx+2)x(ny+2)x(nz+2) block
// at position `pos` along dimension d, covering only the interior in the other dims.
inline MPI_Datatype face_type(int nx, int ny, int nz, int d, int pos) {
    int sizes[3]    = {nz + 2, ny + 2, nx + 2};  // C order: slowest first (k, j, i)
    int subsizes[3] = {nz, ny, nx};
    int starts[3]   = {1, 1, 1};
    int cd = 2 - d;                               // dimension d=0 (x) is the last C index
    subsizes[cd] = 1; starts[cd] = pos;
    MPI_Datatype t;
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &t);
    MPI_Type_commit(&t);
    return t;
}

inline Decomp make_decomp(Params& P) {
    Decomp D;
    MPI_Comm_rank(MPI_COMM_WORLD, &D.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &D.size);
    D.ndims_active = P.dim;
    for (int d = 0; d < 3; ++d) D.dims[d] = P.dims[d];
    if (P.dim == 2) D.dims[2] = 1;
    MPI_Dims_create(D.size, 3, D.dims);            // fills in the zeros as evenly as possible
    int periods[3] = {0, 0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 3, D.dims, periods, 1, &D.cart);
    MPI_Comm_rank(D.cart, &D.rank);
    MPI_Cart_coords(D.cart, D.rank, 3, D.coords);
    for (int d = 0; d < 3; ++d) MPI_Cart_shift(D.cart, d, 1, &D.lo[d], &D.hi[d]);

    // weak scaling: global size follows the process grid
    if (P.local_n > 0) for (int d = 0; d < 3; ++d) P.n[d] = P.local_n * D.dims[d];
    if (P.dim == 2) P.n[2] = 1;
    for (int d = 0; d < P.dim; ++d)
        if (P.n[d] < D.dims[d]) {
            if (D.rank == 0) std::fprintf(stderr, "grid too small for process grid in dimension %d\n", d);
            MPI_Abort(MPI_COMM_WORLD, 3);
        }

    int cnt[3]; long off[3];
    for (int d = 0; d < 3; ++d) partition(P.n[d], D.dims[d], D.coords[d], cnt[d], off[d]);
    D.L = make_local(P, P.n, cnt[0], cnt[1], cnt[2], off[0], off[1], off[2]);

    const int nx = D.L.nx, ny = D.L.ny, nz = D.L.nz;
    const int n[3] = {nx, ny, nz};
    for (int d = 0; d < 3; ++d) {
        D.send_lo[d] = face_type(nx, ny, nz, d, 1);          // first interior layer
        D.recv_lo[d] = face_type(nx, ny, nz, d, 0);          // lo halo
        D.send_hi[d] = face_type(nx, ny, nz, d, n[d]);       // last interior layer
        D.recv_hi[d] = face_type(nx, ny, nz, d, n[d] + 1);   // hi halo
        // The neighbour's block may differ in size by one point along the split
        // dimension; recompute its extents so the RMA target type is exact.
        int ncnt[3] = {nx, ny, nz}; long noff;
        if (D.lo[d] != MPI_PROC_NULL) {
            partition(P.n[d], D.dims[d], D.coords[d] - 1, ncnt[d], noff);
            D.nbr_lo_halo[d] = face_type(ncnt[0], ncnt[1], ncnt[2], d, ncnt[d] + 1); // their hi halo
        } else D.nbr_lo_halo[d] = MPI_DATATYPE_NULL;
        ncnt[0] = nx; ncnt[1] = ny; ncnt[2] = nz;
        if (D.hi[d] != MPI_PROC_NULL) {
            partition(P.n[d], D.dims[d], D.coords[d] + 1, ncnt[d], noff);
            D.nbr_hi_halo[d] = face_type(ncnt[0], ncnt[1], ncnt[2], d, 0);          // their lo halo
        } else D.nbr_hi_halo[d] = MPI_DATATYPE_NULL;
    }
    return D;
}

inline void free_decomp(Decomp& D) {
    for (int d = 0; d < 3; ++d) {
        MPI_Type_free(&D.send_lo[d]); MPI_Type_free(&D.recv_lo[d]);
        MPI_Type_free(&D.send_hi[d]); MPI_Type_free(&D.recv_hi[d]);
        if (D.nbr_lo_halo[d] != MPI_DATATYPE_NULL) MPI_Type_free(&D.nbr_lo_halo[d]);
        if (D.nbr_hi_halo[d] != MPI_DATATYPE_NULL) MPI_Type_free(&D.nbr_hi_halo[d]);
    }
    if (D.win != MPI_WIN_NULL) MPI_Win_free(&D.win);
    MPI_Comm_free(&D.cart);
}

inline std::string dims_string(const Decomp& D) {
    char b[64]; std::snprintf(b, sizeof b, "%dx%dx%d", D.dims[0], D.dims[1], D.dims[2]); return b;
}

// ---- exchange strategies -------------------------------------------------

inline void exchange_sendrecv(const Decomp& D, double* u) {
    for (int d = 0; d < D.ndims_active; ++d) {
        // shift "up": send my hi face to hi neighbour, receive lo neighbour's hi face into my lo halo
        MPI_Sendrecv(u, 1, D.send_hi[d], D.hi[d], 10 + d,
                     u, 1, D.recv_lo[d], D.lo[d], 10 + d, D.cart, MPI_STATUS_IGNORE);
        // shift "down"
        MPI_Sendrecv(u, 1, D.send_lo[d], D.lo[d], 20 + d,
                     u, 1, D.recv_hi[d], D.hi[d], 20 + d, D.cart, MPI_STATUS_IGNORE);
    }
}

inline int exchange_start(const Decomp& D, double* u, MPI_Request* req) {
    int n = 0;
    for (int d = 0; d < D.ndims_active; ++d) {
        MPI_Irecv(u, 1, D.recv_lo[d], D.lo[d], 10 + d, D.cart, &req[n++]);
        MPI_Irecv(u, 1, D.recv_hi[d], D.hi[d], 20 + d, D.cart, &req[n++]);
    }
    for (int d = 0; d < D.ndims_active; ++d) {
        MPI_Isend(u, 1, D.send_hi[d], D.hi[d], 10 + d, D.cart, &req[n++]);
        MPI_Isend(u, 1, D.send_lo[d], D.lo[d], 20 + d, D.cart, &req[n++]);
    }
    return n;
}
inline void exchange_wait(int n, MPI_Request* req) { MPI_Waitall(n, req, MPI_STATUSES_IGNORE); }

// One-sided: every rank exposes its whole block through a window; neighbours put
// their faces straight into my halo. The fence pair delimits the access epoch.
inline void exchange_rma(Decomp& D, double* u) {
    MPI_Win_fence(MPI_MODE_NOPRECEDE, D.win);
    for (int d = 0; d < D.ndims_active; ++d) {
        if (D.hi[d] != MPI_PROC_NULL) MPI_Put(u, 1, D.send_hi[d], D.hi[d], 0, 1, D.nbr_hi_halo[d], D.win);
        if (D.lo[d] != MPI_PROC_NULL) MPI_Put(u, 1, D.send_lo[d], D.lo[d], 0, 1, D.nbr_lo_halo[d], D.win);
    }
    MPI_Win_fence(MPI_MODE_NOSUCCEED, D.win);
}

// Windows are tied to a buffer, so the solver (re)creates them when the exchanged array changes.
inline void rma_attach(Decomp& D, double* u) {
    if (D.win != MPI_WIN_NULL) MPI_Win_free(&D.win);
    MPI_Win_create(u, (MPI_Aint)(D.L.ntot * sizeof(double)), sizeof(double), MPI_INFO_NULL, D.cart, &D.win);
}

} // namespace poisson

// ---- parallel output of the filled satellite field -------------------------
// Every rank writes its interior block into the global [nt][ny][nx] float array
// with one collective MPI-IO call through a subarray view; rank 0 writes the header.
#include "gridio.hpp"
namespace poisson {
inline bool write_grid_mpi(const Decomp& D, const Params& P, const GridHeader& h, const std::vector<float>& block) {
    MPI_File fh;
    if (MPI_File_open(D.cart, P.out.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh) != MPI_SUCCESS) return false;
    MPI_File_set_size(fh, 0);
    if (D.rank == 0) MPI_File_write_at(fh, 0, &h, (int)sizeof h, MPI_BYTE, MPI_STATUS_IGNORE);
    int sizes[3]    = {(int)P.n[2], (int)P.n[1], (int)P.n[0]};
    int subsizes[3] = {D.L.nz, D.L.ny, D.L.nx};
    int starts[3]   = {(int)D.L.oz, (int)D.L.oy, (int)D.L.ox};
    MPI_Datatype view;
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &view);
    MPI_Type_commit(&view);
    MPI_File_set_view(fh, (MPI_Offset)sizeof h, MPI_FLOAT, view, "native", MPI_INFO_NULL);
    int rc = MPI_File_write_all(fh, block.data(), (int)block.size(), MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_Type_free(&view);
    MPI_File_close(&fh);
    return rc == MPI_SUCCESS;
}
} // namespace poisson
