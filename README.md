# Filling cloud gaps in satellite sea-surface temperature with a parallel Poisson solver (MPI + OpenMP, MPI + CUDA)

DSA5208 Project 3, National University of Singapore, AY2026/27 Semester 1.
Team: Aditya Vijindra Pandhari (A0328872R), Apurwa Kumari (A0333260R) and
Vanshika Dhamija (A0333989H).

Repository: https://github.com/vanshikadhamijanus/dsa5208-poisson-mpi
Report: `report/report.pdf`, edited in `report/report.docx` (see "Editing the report").

Satellite infrared sensors cannot see the sea surface through clouds, so every
daily temperature map has holes. We fill them by solving the Laplace equation
(the Poisson equation with zero source) on the missing pixels, with the observed
pixels held fixed and the coast acting as an insulated wall. Each missing pixel
becomes the average of its neighbours. In 2D one day is filled from its spatial
neighbours. In 3D a whole year is stacked with time as the third axis, so a hole
is also filled from the days before and after.

The solver (Jacobi and conjugate gradient on a 2D or 3D finite-difference grid)
is written once in C++ and compiled as a serial program and as three parallel
programs: MPI (domain decomposition), MPI+OpenMP (threads inside each rank) and
MPI+CUDA (one GPU per rank). A manufactured Poisson problem with a known exact
solution validates the solvers and provides the scaling benchmarks. The
satellite data provide the application.

## Results in brief

All timings are from the NUS Vanda cluster. The report gives the details and the caveats.

- Every build (serial, OpenMP, MPI, MPI+OpenMP, MPI+CUDA) gives the serial answer: the same iteration count on every rank count, thread split and node layout and on the GPU, and the same error apart from rounding in the last digits.
- Speedup at 256 cores: 74 (Jacobi 2D), 77 (Jacobi 3D), 137 (CG 2D) and 103 (CG 3D). Inside one node memory bandwidth is the limit (efficiency about 0.3 by 32 cores). From one node to four, scaling is linear or better.
- Weak scaling is flat from one node to four (efficiency 0.94 to 1.00 for three of the four problems).
- At 256 ranks CG is limited by the all-reduce (about 0.1 ms each over TCP), and Jacobi by the halo exchange, where the memory layout of the faces matters more than the exchange strategy.
- Pure MPI was as fast as or faster than every MPI+OpenMP split.
- One A40 GPU is 1.9 to 2.7 times faster than 36 CPU cores on large grids and 1.2 to 1.5 times faster than a full 64-core node on Jacobi. On CG it ties with the full node.
- Gap filling: about 0.15 °C error on 64 km holes, against 0.92 °C for filling with the mean. The whole year fills in 2.5 s on 256 cores.

## Layout

```
dsa5208-poisson-mpi/
  Makefile                    builds the programs into bin/ and runs the tests (make check)
  src/common.hpp              problem definition, command-line options (checked), CSV output
  src/kernels.hpp             stencil, dot product and axpy kernels, plain and masked (gap filling)
  src/gridio.hpp              satellite grid file format, per-rank block loading, pixel mask, error measures
  src/solver.hpp              Jacobi and CG loops, written once for every backend and both problems
  src/decomp.hpp              MPI Cartesian decomposition, the four halo-exchange modes, MPI-IO output
  src/serial_main.cpp         serial (and OpenMP-only) driver
  src/mpi_main.cpp            MPI and MPI+OpenMP driver
  src/cuda/                   MPI+CUDA driver, CUDA kernels, and a header that runs the CUDA code without a GPU
  tools/gridtool.cpp          data tool: downloaded ASCII grids -> grid file, plus info, slices and differences
  scripts/data/download.sh    downloads the daily satellite maps for a region and year (curl, no account)
  scripts/local/check.sh      pass/fail tests (make check)
  scripts/local/run_all.sh    every experiment on a laptop or workstation -> results/local/*.csv
  scripts/local/maps.sh       draws the observed and filled maps for one day -> results/figures/sst_maps.png
  scripts/atlas/              cluster settings (env.sh) and the PBS job generator (gen_jobs.sh)
  scripts/vanda/              Vanda jobs: build, GPU (one and two GPUs), hardware check, optional repeats
  vanda_jobs/                 the exact job files that produced results/cluster and results/cluster_8x32
  analysis/summarize.cpp      CSV -> tables.md, summary.txt and plot data (bin/summarize)
  analysis/plots.gp, maps.gp  gnuplot scripts for the figures and the satellite maps
  results/local/              raw measurements from the laptop
  results/cluster/            raw measurements from Vanda (4 nodes x 64 cores) and the GPU node
  results/cluster_8x32/       the same 256-rank runs on 8 nodes x 32 cores
  results/figures/            laptop figures and tables (the OpenMP-only figure and the maps are used in the report)
  results/figures_cluster/    the Vanda figures and tables used in the report
  report/report.docx          the report, for editing
  report/report.pdf           the report, as submitted
```

Not in the repository: the satellite data (`data/`, about 2.5 GB) and the
filled fields (`results/fields/`, about 600 MB) are too large, and the Data
section rebuilds them. Compiled programs (`bin/`) are rebuilt with `make all`.
Job logs and terminal output are kept out, as the lecturer asked; the
measurements themselves are in the CSV files.

Everything is C++ (and CUDA for the GPU build), with bash for the scripts and
gnuplot for the figures.

## Build and test

Requirements: a C++17 compiler, an MPI implementation (Open MPI or MPICH),
OpenMP (on macOS: `brew install libomp`), `curl` and `gnuplot`, and `nvcc` for
the GPU build. On macOS: `xcode-select --install`, then
`brew install open-mpi libomp gnuplot`. On Ubuntu:
`sudo apt install build-essential openmpi-bin libopenmpi-dev gnuplot curl`.

```
make all            # bin/poisson_serial  bin/poisson_omp  bin/poisson_mpi  bin/poisson_mpi_omp  bin/gridtool
make cuda           # bin/poisson_mpi_cuda       (needs nvcc; set CUDA_ARCH=sm_80 etc.)
make cuda_emul      # bin/poisson_mpi_cuda_emul  (the CUDA source run on the CPU, for testing)
make check          # builds everything except cuda, then runs the tests
```

On macOS use `make all CXX=clang++`. On a cluster, load the compiler and MPI
modules first and compile on a compute node (see "On Vanda" below).

`make check` runs `scripts/local/check.sh`: 37 pass/fail tests in about 15
seconds, with at most 8 MPI ranks. Each test runs one configuration and
compares its iteration count and error with the serial run. They cover every
exchange mode on uneven blocks, blocks one cell thick, CG with overlap, Jacobi
stopping at a tolerance, the CUDA driver through its emulation layer, ten
invalid command lines that must be rejected and, if `data/india_2025.grid`
exists, the satellite data, including a byte-for-byte check of the MPI-IO
output. It must end with `37 passed, 0 failed` (fewer tests run without the
data). Set `MPIRUN=...` to change the launcher; the default is
`mpirun --oversubscribe` (Open MPI).

A ten-second smoke test: both lines must print `iters=183` and `l2_err=3.445134e-05`.

```
bin/poisson_serial --dim 2 --n 128 --solver cg --tol 1e-8 --iters 100000
mpirun -np 4 bin/poisson_mpi --dim 2 --n 128 --solver cg --tol 1e-8 --iters 100000
```

If the build fails on macOS with missing C++ headers (`<cmath>` or `<vector>`
not found), the Command Line Tools' libc++ is incomplete. Build against the SDK
instead, for example
`make all CXXFLAGS="-O2 -std=c++17 -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX15.sdk -nostdinc++ -isystem /Library/Developer/CommandLineTools/SDKs/MacOSX15.sdk/usr/include/c++/v1"`.

## Data

The source is the NOAA CoastWatch ERDDAP server (https://coastwatch.noaa.gov/erddap),
which serves subsets of gridded satellite products as plain text without an account.

| file | ERDDAP dataset | what it is |
|---|---|---|
| `sst_YYYY-MM-DD.asc` | `noaacwN20ACSPOSSTL3GCDaily` | NOAA-20 VIIRS sea-surface temperature, daily, 0.0375 degree (about 4 km), with cloud gaps |
| `mask.asc` | `noaacwBLENDEDsstDaily` (variable `mask`) | land/water mask of the blended analysis, 0.05 degree |
| `ref_YYYY-MM-DD.asc` (optional) | `noaacwBLENDEDsstDaily` (variable `analysed_sst`) | NOAA geo-polar blended analysis, daily, gap-free, used as an independent reference |

The region is the seas around India, latitude 5 N to 25 N and longitude 65 E to 95 E,
801 x 534 pixels per day, one third land, for the year 2025. The script also
knows a `singapore` box (8 S to 8 N, 96 E to 120 E) and accepts any custom box.

```
scripts/data/download.sh india 2025 ref        # 365 days + reference, about 1.1 GB, 20-40 minutes
make gridtool
bin/gridtool convert data/india_2025.grid --mask data/raw/india/mask.asc data/raw/india/sst_2025-*.asc
bin/gridtool regrid  data/india_2025.grid data/india_2025_ref.grid --offset -273.15 data/raw/india/ref_2025-*.asc
bin/gridtool info    data/india_2025.grid --days   # observed / missing share per day
```

`bin/gridtool convert` stacks the daily text grids into one binary file
(`src/gridio.hpp` documents the format: an 84-byte header, then float32
`[day][lat][lon]`, NaN for missing and -999 for land). Each MPI rank reads its
own block of that file, so no rank holds the whole year. To try the gap filler
on one day, `scripts/data/download.sh india 2025-04-10` fetches that day; convert
it the same way into `data/test.grid` and run
`mpirun -np 4 bin/poisson_mpi --data data/test.grid --dim 2 --solver cg --tol 1e-6 --iters 100000 --holdout 0.1`.

## Run

All programs share one command line. Without `--data` they solve the
manufactured Poisson problem, whose exact solution is
`u = exp(x) sin(pi x) sin(pi y) [sin(pi z)]`. With `--data` they fill the
satellite grid.

```
bin/poisson_serial --dim 2 --n 1024 --solver cg --tol 1e-8 --iters 100000
mpirun -np 16 bin/poisson_mpi --dim 3 --n 192 --solver jacobi --iters 200 --exchange overlap
OMP_NUM_THREADS=4 mpirun -np 4 bin/poisson_mpi_omp --threads 4 --dim 2 --n 2048 --iters 200
mpirun -np 2 bin/poisson_mpi_cuda --dim 3 --n 256 --iters 200 --cuda-aware

# satellite gap filling: one day in 2D, the whole year in 3D
mpirun -np 8 bin/poisson_mpi --data data/india_2025.grid --dim 2 --day 99 --solver cg --tol 1e-6 --iters 100000 \
       --holdout 0.1 --holdout-block 16 --out results/fields/fill2d_day99.grid
mpirun -np 16 bin/poisson_mpi --data data/india_2025.grid --dim 3 --tweight 1 --solver cg --tol 1e-6 --iters 100000 \
       --holdout 0.1 --holdout-block 16 --ref data/india_2025_ref.grid --out results/fields/fill3d.grid
scripts/local/maps.sh data/india_2025.grid 99 results/fields/fill2d_day99.grid results/fields/fill3d.grid results/figures/sst_maps.png
```

| option | meaning |
|---|---|
| `--dim 2\|3` | square or cube (manufactured problem); one day or the year stack (data) |
| `--n N` or `--n NX,NY[,NZ]` | global interior grid points: one value, or one per dimension |
| `--local-n M` | weak scaling: each rank owns M points per dimension, and the global grid follows the process grid |
| `--solver jacobi\|cg` | iteration |
| `--iters K` | maximum iterations (default 1000) |
| `--tol T` | stop when the relative residual drops below T (default 0: run exactly `--iters` sweeps) |
| `--check-every C` | Jacobi only: compute the global residual every C sweeps (0 = never) |
| `--exchange sendrecv\|isend\|overlap\|rma` | halo exchange strategy |
| `--dims PX,PY[,PZ]` | force the process grid shape (the product must equal the number of ranks) |
| `--threads T` | OpenMP threads per rank |
| `--csv FILE --tag TEXT` | append one result row to FILE |
| `--data FILE` | satellite grid file (sets the grid size) |
| `--day K`, `--days N` | 2D: the day to fill (0-based). 3D: first day and number of days (0 = all) |
| `--tweight W` | 3D: weight of the time neighbours relative to the space neighbours (default 1) |
| `--holdout F`, `--holdout-block B` | hide a fraction F of the observed pixels, in BxB patches, to measure the fill error |
| `--eval-day K` | 3D: report the hidden-pixel error for day K only, to compare with the 2D fill of that day |
| `--ref FILE` | gap-free reference grid on the same grid (made by `gridtool regrid`) |
| `--out FILE` | write the filled field as a grid file (MPI-IO in parallel) |

A value that is not a number or is out of range (for example `--n 0` or
`--iters -3`) stops the program with a message. Each run prints the iteration
count, the relative residual, the error (against the exact solution, or on the
hidden pixels with the trivial "fill with the mean" baseline) and the time split
into compute, halo exchange and all-reduce.

## Reproduce the results

### On a laptop or workstation

```
scripts/local/run_all.sh 16 2                         # cores, repeats; about 20 minutes plus the data stage
make figures LABEL="your machine"                     # bin/summarize + gnuplot -> results/figures/
```

The first block of `run_all.sh` is the validation (convergence order, Jacobi
stopping at the same sweep in serial and in parallel, every implementation on
the same problem). A third argument picks stages (for example `data`). The data
stage runs only if `data/india_2025.grid` exists (`GRID=`, `DAY=` and `DAYS3D=`
override the file, the 2D day and the number of days in 3D). `OUT=dir` writes
somewhere other than `results/local`.

`make figures` compiles `analysis/summarize.cpp`, which keeps the faster of the
repeated runs and derives speedup, efficiency, communication share,
convergence order, log-log slopes and the gap-filling tables, and then runs
`analysis/plots.gp`. The outputs are the PNG figures, `tables.md` and
`summary.txt`. Axes for time, problem size and core count are logarithmic.

### On Vanda

The cluster runs were made on NUS Vanda in November 2026. Vanda CPU nodes have
72 cores; the GPU nodes have two A40 GPUs. Work in `/scratch/$USER` and compile
on a compute node, because the login node has a different CPU. Copy the
repository there (without `data/raw`, `bin` and `results/fields`) together with
`data/india_2025.grid` and `data/india_2025_ref.grid`, then, from the
repository folder:

```
BUILD=$(qsub scripts/vanda/build.pbs)                 # compiles everything and runs the smoke test
PROJECT=personal-<NUSNET id> PPN=64 RANKS="1 2 4 8 16 32 64 128 256" MEM=200gb \
MODULES="module load gcc/13.1.0 openmpi4/gcc/4.1.8" \
MPIRUN='mpirun --prefix /cm/shared/apps/openmpi4/gcc/4.1.8 -hostfile $PBS_NODEFILE --bind-to core --mca mca_base_component_show_load_errors 0 -x LD_LIBRARY_PATH -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES' \
scripts/atlas/gen_jobs.sh                             # writes jobs/node_1.pbs, mpi_128.pbs, mpi_256.pbs, hybrid_256.pbs
sed -i '/--map-by ppr/! s/--bind-to core/--map-by core --bind-to core/' jobs/*.pbs
sed -i 's/^  mpirun \(.*--exchange \$mode\)/  timeout 900 mpirun \1/' jobs/mpi_256.pbs
for j in jobs/*.pbs; do qsub -W depend=afterok:$BUILD $j; done
qsub -W depend=afterok:$BUILD scripts/vanda/gpu.pbs   # one GPU against 36 CPU cores (gpu_multi.pbs: two GPUs)
qstat -u $USER                                        # H = held for the build, Q = waiting, R = running
```

What these settings fix, and what else the runs used:

- The free queue stops at 160 cores, so the 256-core jobs ran under a project (queue `batch_cpu`).
- Open MPI's launcher is not on the remote nodes' path (`--prefix`), and the site default binds each rank to a whole socket (`--map-by core --bind-to core`).
- MPI runs over TCP; UCX does not start on Vanda (`PML ucx cannot be selected`).
- The 8-node layout of the report is the same generator with `PPN=32 RANKS=256 MEM=100gb`, writing to `results/cluster_8x32`.
- `scripts/vanda/hwinfo.pbs` records a CPU node's processor and where each of 64 ranks runs. `scripts/vanda/repeat_256.pbs` adds repeats at 256 ranks (optional, not used in the report). Change their `-P` line to your project.

Back on the laptop, copy `results/cluster` and `results/cluster_8x32` into
`results/` and make the figures:

```
make figures RESULTS=results/cluster FIGDIR=results/figures_cluster \
  LABEL="NUS Vanda, 4 nodes x 64 cores" \
  ALT=results/cluster_8x32 ALTLABEL="NUS Vanda, 8 nodes x 32 cores"
```

On another cluster (for example Atlas, 96 cores per node), edit
`scripts/atlas/env.sh` (queue, ranks per node, modules) and run
`scripts/atlas/gen_jobs.sh --submit`.

## Editing the report

`report/report.docx` is the editable report. Open it in Word, make the changes
and export with File > Save As (or Export) > PDF, saving over `report/report.pdf`.
When Word asks whether to update the fields, choose Yes; this refreshes the
table of contents. The figures are images inside the document: if a figure is
regenerated with `make figures`, replace it in Word (right-click > Change
Picture) with the new PNG from `results/figures_cluster/` or `results/figures/`.

## Known limitations

- A 2D grid is stored as a 3D block one layer thick, and the 2D kernels still read the two (zero) halo planes in z. A 2D update therefore moves about 40 bytes instead of 24. Skipping those terms made 2D Jacobi 1.6 times faster on a laptop with the same answers; the report's timings use the kernel as it is.
- On Vanda all messages between nodes went over TCP, because the UCX (RDMA) layer did not start.
- CG has no preconditioner, and the GPU dot product uses one `atomicAdd` per thread, which costs about 1 ms per CG iteration (report Section 7.3).
- The arrays are first written by the master thread, so ranks whose threads span both sockets read half their memory from the other socket (report Section 7.2).
