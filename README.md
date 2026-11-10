# Filling cloud gaps in satellite sea-surface temperature with a parallel Poisson solver (MPI + OpenMP, MPI + CUDA)

DSA5208 Project 3, National University of Singapore, AY2026/27 Semester 1.
Team: Aditya Vijindra Pandhari (A0328872R), Apurwa Kumari (A0333260R) and
Vanshika Dhamija (A0333989H).

Repository: https://github.com/vanshikadhamijanus/dsa5208-poisson-mpi
Report: `report/report.pdf` (source `report/report.md`).

Satellite infrared sensors cannot see the sea surface through clouds, so every
daily temperature map has holes. We fill them by solving the Laplace equation
(the Poisson equation with zero source) on the missing pixels, with the observed
pixels held fixed and the coast acting as an insulated wall, so each missing pixel
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

All timings are from the NUS Vanda cluster unless marked otherwise. The report
gives the details and the caveats.

- Every build (serial, OpenMP, MPI, MPI+OpenMP, MPI+CUDA) gives the same iteration count and error on every rank count, thread split and node layout, and on the GPU.
- Speedup at 256 cores: 74 (Jacobi 2D), 77 (Jacobi 3D), 137 (CG 2D) and 103 (CG 3D). Inside one node memory bandwidth is the limit (efficiency about 0.3 by 32 cores). From one node to four, scaling is linear or better.
- Weak scaling is flat from one node to four (efficiency 0.94 to 1.00 for three of the four problems).
- At 256 ranks CG is limited by the all-reduce (about 0.1 ms each over TCP), and Jacobi by the halo exchange, where the memory layout of the faces matters more than the exchange strategy.
- Pure MPI was as fast as or faster than every MPI+OpenMP split.
- One A40 GPU is 1.9 to 2.7 times faster than 36 CPU cores on large grids, and 1.2 to 1.5 times faster than a full 64-core node on Jacobi. On CG it ties with the full node.
- Gap filling: about 0.15 °C error on 64 km holes (against 0.92 °C for filling with the mean). The whole year fills in 2.5 s on 256 cores.

## Layout

```
dsa5208-poisson-mpi/
  Makefile                    builds the executables into bin/ and runs the tests (make check)
  README.md                   this file
  src/common.hpp              problem definition, command-line options (checked), CSV output
  src/kernels.hpp             stencil, dot product and axpy kernels, plain and masked (gap filling)
  src/gridio.hpp              satellite grid file format, per-rank block loading, pixel mask, error measures
  src/solver.hpp              Jacobi and CG loops, written once for every backend and both problems
  src/decomp.hpp              MPI Cartesian decomposition, the four halo-exchange modes, MPI-IO output
  src/serial_main.cpp         serial (and OpenMP-only) driver
  src/mpi_main.cpp            MPI and MPI+OpenMP driver
  src/cuda/gpu_kernels.cuh    CUDA kernels (plain and masked)
  src/cuda/mpi_cuda_main.cu   MPI+CUDA driver
  src/cuda/cuda_compat.hpp    lets the CUDA code be compiled and tested without a GPU
  tools/gridtool.cpp          data tool: downloaded ASCII grids -> grid file, plus info, slices and differences
  scripts/data/download.sh    downloads the daily satellite maps for a region and year (curl, no account)
  scripts/local/check.sh      pass/fail tests (make check)
  scripts/local/run_all.sh    every experiment on a laptop or workstation -> results/local/*.csv
  scripts/local/maps.sh       draws the observed and filled maps for one day -> results/figures/sst_maps.png
  scripts/atlas/env.sh        cluster settings to edit (queue, ranks per node, modules)
  scripts/atlas/gen_jobs.sh   writes the PBS jobs for the CPU cluster (1 to 256 cores) into jobs/
  scripts/vanda/build.pbs     builds every program on a Vanda compute node
  scripts/vanda/gpu.pbs       GPU job: one GPU against 36 CPU cores of the same node (gpu_multi.pbs: two GPUs)
  scripts/vanda/hwinfo.pbs    records a CPU node's processor and where each of 64 ranks is placed
  scripts/vanda/repeat_256.pbs  optional: three more repeats of the 256-rank strong-scaling runs
  analysis/summarize.cpp      CSV -> tables.md, summary.txt and plot data (bin/summarize)
  analysis/plots.gp           gnuplot script: plot data -> the PNG figures
  analysis/maps.gp            gnuplot script: the satellite maps
  results/local/*.csv         raw measurements from the laptop
  results/figures/            laptop figures and tables (the OpenMP-only figure and the maps are used in the report)
  results/cluster/*.csv       raw measurements from Vanda, 4 nodes x 64 cores, and the GPU node
  results/cluster_8x32/*.csv  the same 256-rank runs on 8 nodes x 32 cores (node-layout comparison)
  results/cluster_round1/     the failed first round of Vanda runs (not used in the report)
  results/figures_cluster/    the Vanda figures and tables used in the report
  vanda_jobs/                 the exact PBS job files that produced results/cluster and results/cluster_8x32
  report/report.md, report.pdf  the project report (build with report/build.sh)
  report/report_laptop_version.md  an earlier, laptop-only version of the report, kept for comparison
  handbook/handbook.pdf       a long beginner's guide to the code, written during development
  tools/                      Markdown -> PDF build (pandoc + headless Chrome), stylesheets, handbook source
```

Everything is C++ (and CUDA for the GPU build), with bash for the scripts and
gnuplot for the figures. No Python is used.

**Not in this repository.** The satellite data (`data/`, about 2.5 GB) and the
filled fields (`results/fields/`, about 600 MB) are too large for GitHub. The
commands in the Data section rebuild them. Compiled programs (`bin/`) are
rebuilt with `make all`. Job logs and terminal output (`logs_vanda/`, `*.log`,
PBS `*.o<jobid>` files) are kept out of the repository, as the lecturer asked;
the measurements themselves are in the CSV files under `results/`.

## Build

Requirements: a C++17 compiler, an MPI implementation (Open MPI or MPICH),
OpenMP (on macOS: `brew install libomp`), `curl` and `gnuplot`, and `nvcc` for
the GPU build.

```
make all            # bin/poisson_serial  bin/poisson_omp  bin/poisson_mpi  bin/poisson_mpi_omp  bin/gridtool
make cuda           # bin/poisson_mpi_cuda       (needs nvcc; set CUDA_ARCH=sm_80 etc.)
make cuda_emul      # bin/poisson_mpi_cuda_emul  (the CUDA source run on the CPU, for testing)
make check          # builds all of the above except cuda, then runs the tests
```

On macOS use `make all CXX=clang++`. On the clusters, load the compiler and MPI
modules first (Atlas: `source /app1/ebenv`; Vanda: compile on a compute node,
see `scripts/vanda/build.pbs`).

## Test

`make check` runs `scripts/local/check.sh`: 37 pass/fail tests in about 15
seconds on a laptop, with at most 8 MPI ranks. Each test runs one configuration
and compares its iteration count and error with the serial run. The tests cover
every exchange mode on uneven blocks, blocks one cell thick, CG with overlap,
Jacobi stopping at a tolerance, the CUDA driver through its emulation layer,
ten invalid command lines that must be rejected and, if `data/india_2025.grid`
exists, the satellite data (including a byte-for-byte check of the MPI-IO
output). Set `MPIRUN=...` to change the launcher. The default is
`mpirun --oversubscribe`, for Open MPI.

## Data

Source: the NOAA CoastWatch ERDDAP server (https://coastwatch.noaa.gov/erddap),
which serves subsets of gridded satellite products as plain text without an account.

| file | ERDDAP dataset | what it is |
|---|---|---|
| `sst_YYYY-MM-DD.asc` | `noaacwN20ACSPOSSTL3GCDaily` | NOAA-20 VIIRS sea-surface temperature, daily, 0.0375 degree (about 4 km), with cloud gaps |
| `mask.asc` | `noaacwBLENDEDsstDaily` (variable `mask`) | land/water mask of the blended analysis, 0.05 degree |
| `ref_YYYY-MM-DD.asc` (optional) | `noaacwBLENDEDsstDaily` (variable `analysed_sst`) | NOAA geo-polar blended analysis, daily, gap-free, used as an independent reference |

Region: the seas around India, latitude 5 N to 25 N, longitude 65 E to 95 E
(Arabian Sea, Bay of Bengal, Sri Lanka), 801 x 534 pixels per day, one third
land. Year: 2025. The script also knows a `singapore` box (8 S to 8 N, 96 E to
120 E) and accepts any custom box.

```
scripts/data/download.sh india 2025 ref        # 365 days + reference, about 1.1 GB, 20-40 minutes
make gridtool
bin/gridtool convert data/india_2025.grid --mask data/raw/india/mask.asc data/raw/india/sst_2025-*.asc
bin/gridtool regrid  data/india_2025.grid data/india_2025_ref.grid --offset -273.15 data/raw/india/ref_2025-*.asc
bin/gridtool info    data/india_2025.grid --days   # observed / missing share per day
```

`bin/gridtool convert` stacks the daily text grids into one binary file
(`src/gridio.hpp` documents the format: an 84-byte header, then float32
`[day][lat][lon]`, NaN for missing and -999 for land). The MPI ranks read their
own block of that file directly, so no rank holds the whole year. For a quick
test, `scripts/data/download.sh india 2025-02-15` fetches one day.

## Run

All binaries share one command line. Without `--data` they solve the
manufactured Poisson problem with a known exact solution. With `--data` they
fill the satellite grid.

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
| `--n N` or `--n NX,NY[,NZ]` | global interior grid points: one value, or one per dimension (manufactured problem) |
| `--local-n M` | weak scaling: each rank owns M points per dimension; the global grid follows the process grid |
| `--solver jacobi\|cg` | iteration |
| `--iters K` | maximum iterations (default 1000) |
| `--tol T` | stop when the relative residual drops below T (default 0: run exactly `--iters` sweeps) |
| `--check-every C` | Jacobi only: compute the global residual every C sweeps (0 = never) |
| `--exchange sendrecv\|isend\|overlap\|rma` | halo exchange strategy |
| `--dims PX,PY[,PZ]` | force the process grid shape (the product must equal the number of ranks) |
| `--threads T` | OpenMP threads per rank |
| `--csv FILE --tag TEXT` | append one result row to FILE |
| `--data FILE` | satellite grid file; the grid size comes from its header |
| `--day K`, `--days N` | 2D: the day to fill (0-based); 3D: first day and number of days (0 = all) |
| `--tweight W` | 3D: weight of the time neighbours relative to the space neighbours (default 1) |
| `--holdout F`, `--holdout-block B` | hide a fraction F of the observed pixels, in BxB patches, to measure the fill error |
| `--eval-day K` | 3D: report the hidden-pixel error for day K only, so a 3D fill can be compared with the 2D fill of that day |
| `--ref FILE` | gap-free reference grid for comparison (same grid; made by `gridtool regrid`) |
| `--out FILE` | write the filled field as a grid file (MPI-IO in parallel) |

Every numeric option is checked: a value that is not a number, or is out of
range (for example `--n 0` or `--iters -3`), stops the program with a message.

Each manufactured-problem run prints the iteration count, the relative residual
and the L2 and maximum error against the exact solution
`u = exp(x) sin(pi x) sin(pi y) [sin(pi z)]`. Each data run prints the pixel
counts (observed, missing, hidden, land), the RMSE, MAE and maximum error of the
fill on the hidden pixels, the RMSE of the trivial fill (every hole set to the
mean temperature) as a baseline, and, with `--ref`, the RMSE against the
reference analysis. Both print the time split into compute, halo exchange and
all-reduce.

## Reproduce the figures

Local (laptop or workstation):

```
scripts/local/run_all.sh 16 2                         # about 20 minutes on 16 cores, plus the data stage
make figures LABEL="your machine"                     # bin/summarize + gnuplot -> results/figures/
```

The last stage of `run_all.sh` (`data`) runs only if `data/india_2025.grid`
exists (`GRID=... DAY=... DAYS3D=...` override the file, the 2D day and the
number of days in 3D). It writes `results/local/gapfill.csv`, the filled fields
in `results/fields/` and the map figure. Set `OUT=dir` to write somewhere other
than `results/local`.

Cluster (Atlas or Vanda CPU nodes, up to 256 cores):

```
# edit scripts/atlas/env.sh: queue name, ranks per node, module lines
# copy data/india_2025.grid (and _ref.grid) into data/ on the cluster first
scripts/atlas/gen_jobs.sh --submit          # node_1.pbs (serial, every rank count up to PPN, design, hybrid, data) + mpi_NP.pbs per multi-node count
qsub scripts/vanda/gpu.pbs                   # GPU against CPU cores, manufactured problem and the satellite data
qsub scripts/vanda/gpu_multi.pbs             # two GPUs (if the queue allows ngpus=2)
# when they finish:
make figures RESULTS=results/cluster FIGDIR=results/figures_cluster LABEL="NUS Vanda"
```

`make figures` compiles `analysis/summarize.cpp` and runs it on the CSV files.
It keeps the faster of the repeated runs and derives speedup, efficiency,
communication share, convergence order, log-log slopes and the gap-filling
accuracy tables. It then runs `analysis/plots.gp` with gnuplot. The outputs are
the PNG figures, `tables.md` (every table in Markdown) and `summary.txt`
(headline numbers). Axes for time, problem size and core count are logarithmic.

## Step by step

### A. On your own computer

1. Install the tools once. macOS: `xcode-select --install`, then `brew install open-mpi libomp gnuplot pandoc`.
   Ubuntu/Debian: `sudo apt install build-essential openmpi-bin libopenmpi-dev gnuplot pandoc curl`.
   Check with `which mpicxx mpirun gnuplot curl`.
2. Build and test: `make all CXX=clang++` on macOS (or `make all` on Linux), then `make check`.
   `make check` must end with `37 passed, 0 failed` (fewer tests run if the data are missing).
3. Smoke test (ten seconds):
   ```
   bin/poisson_serial --dim 2 --n 128 --solver cg --tol 1e-8 --iters 100000
   mpirun -np 4 bin/poisson_mpi --dim 2 --n 128 --solver cg --tol 1e-8 --iters 100000
   ```
   Both lines must print the same `iters=183` and `l2_err=3.445134e-05`.
4. Get the data (once, needs internet): `scripts/data/download.sh india 2025 ref`, then the `bin/gridtool` commands of the Data section.
   One day is enough to try the gap filler: `scripts/data/download.sh india 2025-04-10`,
   `bin/gridtool convert data/test.grid --mask data/raw/india/mask.asc data/raw/india/sst_2025-04-10.asc`, then
   `mpirun -np 4 bin/poisson_mpi --data data/test.grid --dim 2 --solver cg --tol 1e-6 --iters 100000 --holdout 0.1`.
5. Run every local experiment: `scripts/local/run_all.sh 16 2`
   (first argument: cores to use; second: repeats per configuration; optional third: a comma list of stages,
   for example `data` to rerun only the satellite part).
6. Make the tables and figures: `make figures LABEL="my laptop"` writes `results/figures/`.
7. Rebuild the report PDF after editing `report/report.md`: `report/build.sh` (needs pandoc and Google Chrome).

### B. On the NUS HPC clusters (Atlas for CPUs, Vanda for 256 cores and the GPUs)

The two clusters have different limits, checked on 3 November 2026:

| | Atlas (`atlas8.nus.edu.sg`) | Vanda (`vanda.nus.edu.sg`) |
|---|---|---|
| node size | 96 cores | 72 cores (some 64); GPU nodes with 2 GPUs |
| queue | `parallel`: at most 96 cores per job, 4 running jobs | `-q auto_free` (free tier) or `-P <project>` (see `hpc project`) |
| toolchain | `source /app1/ebenv; module load GCC/13.2.0 OpenMPI/4.1.6-GCC-13.2.0` | `module load gcc/13.1.0 openmpi4/gcc/4.1.8 cuda12.4/toolkit/12.4.1` |
| where to work | home directory | `/scratch/$USER` (compile on a compute node: the login node has a different CPU) |

Atlas:

1. Copy the folder up without the raw downloads and laptop binaries (compute nodes have no internet):
   `rsync -avz --exclude data/raw --exclude results/fields --exclude bin dsa5208-poisson-mpi USERNAME@atlas8.nus.edu.sg:~/`
2. `ssh USERNAME@atlas8.nus.edu.sg`, `cd dsa5208-poisson-mpi`, load the toolchain from the table, `make all`, then the smoke test of step A3.
3. `scripts/atlas/gen_jobs.sh` writes `jobs/node_1.pbs` (one 96-core job holding everything) using the Atlas values in `scripts/atlas/env.sh`. Submit it with `qsub jobs/node_1.pbs`.
4. Watch with `qstat -u $USER`. The log is `node_1.o<jobid>`, and the measurements are appended to `results/cluster/*.csv`.

Vanda:

1. Log in (`ssh USERNAME@vanda.nus.edu.sg`). Your Atlas home is visible read-only at `/atlas/home/$USER`, so copy from there:
   `mkdir -p /scratch/$USER && rsync -a --exclude bin --exclude jobs --exclude results/cluster --exclude 'node_1.o*' /atlas/home/$USER/dsa5208-poisson-mpi /scratch/$USER/`
   (or rsync from the laptop to `vanda.nus.edu.sg:/scratch/USERNAME/` with the excludes of Atlas step 1).
2. Build on a compute node and queue every run behind the build, from the login node in `/scratch/$USER/dsa5208-poisson-mpi`:
   ```
   BUILD=$(qsub scripts/vanda/build.pbs)
   QSUB_OPTS="-W depend=afterok:$BUILD" \
   QUEUE=auto_free QUEUE_MULTI=auto_free PPN=64 RANKS="1 2 4 8 16 32 64 128 256" MEM=200gb \
   MODULES="module load gcc/13.1.0 openmpi4/gcc/4.1.8" MPIRUN='mpirun -hostfile $PBS_NODEFILE' \
   scripts/atlas/gen_jobs.sh --submit
   qsub -W depend=afterok:$BUILD scripts/vanda/gpu.pbs
   qsub -W depend=afterok:$BUILD scripts/vanda/gpu_multi.pbs
   ```
   `build.pbs` compiles the CPU programs and the CUDA program and runs a smoke test (log: `poisson_build.o<jobid>`). The generator writes and submits `node_1.pbs`, `mpi_128.pbs`, `mpi_256.pbs` and `hybrid_256.pbs` with 64 ranks per node, so 256 ranks are 4 nodes; with a project add `PROJECT=<name>`. Every job waits until the build has succeeded.
3. Watch with `qstat -u $USER` (`H` = held until the build finishes, `Q` = waiting, `R` = running).

**What the report's Vanda runs actually used (November 2026).** The settings above did not work unchanged. These did:

- **Project queue for 256 cores.** The free `cpu_parallel` queue stops at 160 cores, so the CPU jobs ran under the project (`PROJECT=personal-<NUSNET id>`, queue `batch_cpu`).
- **Full `mpirun` line.** Open MPI's launcher is not on the remote nodes' path, and the site default binds each rank to a whole socket. The line was:
  ```
  MPIRUN='mpirun --prefix /cm/shared/apps/openmpi4/gcc/4.1.8 -hostfile $PBS_NODEFILE --bind-to core --mca mca_base_component_show_load_errors 0 -x LD_LIBRARY_PATH -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES'
  ```
- **Two edits to the generated jobs before `qsub`.** One places pure-MPI ranks core by core, and the other caps each one-sided exchange run at 15 minutes:
  ```
  sed -i '/--map-by ppr/! s/--bind-to core/--map-by core --bind-to core/' jobs/*.pbs
  sed -i 's/^  mpirun \(.*--exchange \$mode\)/  timeout 900 mpirun \1/' jobs/mpi_256.pbs
  ```
- **TCP only.** MPI runs over TCP, the site default. UCX does not start on Vanda (`PML ucx cannot be selected`).
- **The 8-node layout.** The second layout in the report (8 nodes × 32 cores) is the same generator with `PPN=32 RANKS=256 MEM=100gb`, writing to `results/cluster_8x32` (`sed -i 's#results/cluster#results/cluster_8x32#g'` on the generated jobs).

Back on the laptop, keep the machines and layouts apart:

```
rsync -av USERNAME@vanda.nus.edu.sg:/scratch/USERNAME/dsa5208-poisson-mpi/results/cluster results/
rsync -av USERNAME@vanda.nus.edu.sg:/scratch/USERNAME/dsa5208-poisson-mpi/results/cluster_8x32 results/
make figures RESULTS=results/cluster FIGDIR=results/figures_cluster \
  LABEL="NUS Vanda, 4 nodes x 64 cores" \
  ALT=results/cluster_8x32 ALTLABEL="NUS Vanda, 8 nodes x 32 cores"
```

`ALT` is optional. When given, `bin/summarize` adds a node-layout table and
gnuplot draws `layout_compare.png`. Then run `report/build.sh`.

If the build fails on macOS with missing C++ headers (`<cmath>` or `<vector>`
not found), the Command Line Tools' libc++ is incomplete. Build the summariser
against the SDK instead:

```
SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX15.sdk
make figures CXXFLAGS="-O2 -std=c++17 -isysroot $SDK -nostdinc++ -isystem $SDK/usr/include/c++/v1" ...
```

Queue names, module names, node sizes and the scratch location differ between
sites, which is why they sit in `scripts/atlas/env.sh` and at the top of the
job files rather than in the code.

## Validation

`scripts/local/run_all.sh` starts with a validation block. It checks that the
discretisation error falls by 4 times when the grid spacing halves, that Jacobi
run to a tolerance stops at the same iteration in serial and in parallel, and
that the same fixed-iteration problem gives the same errors in every
implementation (serial, OpenMP, MPI with each exchange mode, MPI+OpenMP,
emulated MPI+CUDA). `make check` turns the agreement checks into pass/fail
tests and adds edge cases. For the satellite data the check is the same in
spirit: every implementation and rank count must print the same hidden-pixel
error, and the hidden pixels are chosen by a hash of their global position so
that the same pixels are hidden whatever the decomposition.

## Known limitations

- A 2D grid is stored as a 3D block one layer thick, and the 2D kernels still read the two (zero) halo planes in z. That costs memory traffic: a 2D update moves about 40 bytes instead of 24. Skipping those terms made 2D Jacobi 1.6 times faster on a laptop with the same answers. The report's timings use the kernel as it is.
- On Vanda all messages between nodes went over TCP, because the UCX (RDMA) layer did not start.
- CG has no preconditioner, and the GPU dot product uses one `atomicAdd` per thread, which costs about 1 ms per CG iteration (report Section 7.3).
- The arrays are first written by the master thread, so ranks whose threads span both sockets read half their memory from the other socket (report Section 7.2).
