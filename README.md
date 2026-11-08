# Filling cloud gaps in satellite sea-surface temperature with a parallel Poisson solver (MPI + OpenMP, MPI + CUDA)

DSA5208 Project 3 (NUS, AY2026/27 Semester 1). Team: Aditya Vijindra Pandhari
(A0328872R), Apurwa Kumari (A0333260R), Vanshika Dhamija (A0333989H).

Satellite infrared sensors cannot see the sea surface through clouds, so every
daily temperature map has holes. We fill them by solving the Laplace equation
(the Poisson equation with zero source) on the missing pixels, with the observed
pixels held fixed and the coast acting as an insulated wall: each missing pixel
becomes the average of its neighbours. In 2D one day is filled from its spatial
neighbours; in 3D a whole year is stacked with time as the third axis, so a hole
is also filled from the days before and after. The solver (Jacobi and conjugate
gradient on a 2D/3D finite-difference grid) is parallelised by domain
decomposition with MPI, then with OpenMP threads inside each rank (MPI+OpenMP)
and with one GPU per rank (MPI+CUDA). A manufactured Poisson problem with a
known exact solution validates the solvers and provides the scaling benchmarks;
the satellite data provides the application and the data conclusions.

The written report is `report/report.pdf`. A beginner's handbook that explains
every file and every number is `handbook/handbook.pdf`.

## Project status (for team review, 8 November 2026)

The deadline is **13 November 2026**. The code, the cluster runs and the report
are done. A few small items are still open (listed below). When reviewing,
start with `report/report.pdf`, then this section.

### What we did, and why

| Step | What | Why | Where to look |
|---|---|---|---|
| 1. Solver | Jacobi and conjugate gradient (CG) for the 2D and 3D Poisson equation, written once and compiled as serial, OpenMP, MPI, MPI+OpenMP and MPI+CUDA programs | The brief asks for MPI plus one intra-node layer (OpenMP) and a GPU test; one source tree guarantees every version does the same arithmetic | `src/`, report Section 3 |
| 2. Validation | A problem with a known exact answer; error falls 4× each time the grid spacing halves; every version and rank count gives the same digits | Shows the code is correct before timing it | report Section 4 |
| 3. Application | One year (2025) of daily satellite sea-surface temperature around India from NOAA; cloud holes filled by the Poisson (Laplace) solver, in 2D (one day) and 3D (the year, time as 3rd axis); accuracy checked by hiding observed pixels | Gives the project a real dataset and a data conclusion, as the brief asks | `scripts/data/`, report Sections 2.6 and 8 |
| 4. Laptop study | Development, validation and a first scaling study on a 16-core laptop | Fast to iterate; only machine available before cluster access | `results/local/`, `results/figures/` |
| 5. Cluster runs (NUS Vanda) | Serial and 1 to 256 MPI ranks (1, 2 and 4 nodes); halo-exchange strategies, process-grid shapes and residual-check interval at 64 and 256 ranks; MPI+OpenMP splits at 64 and 256 cores; satellite runs to 256 ranks | The brief requires strong and weak scaling to at least 256 cores | `results/cluster/`, `logs_vanda/`, `vanda_jobs/jobs_4x64/` |
| 6. Second node layout | The same 256-rank jobs on 8 nodes × 32 cores, submitted as a fallback while the 4-node jobs waited in the queue; both ran, so the report compares them | Backup against queue delays; turned out to be a useful result (shared half-nodes are 1.4 to 8.3× slower) | `results/cluster_8x32/`, `vanda_jobs/jobs_8x32/`, report Section 6.7 |
| 7. GPU | MPI+CUDA on one NVIDIA A40 against 36 CPU cores of the same node, manufactured problem and satellite data | The brief requires at least one GPU test with MPI+OpenMP | `results/cluster/gpu.csv`, report Section 7.3 |
| 8. Analysis and report | `bin/summarize` (C++) turns the CSV files into tables and plot data, gnuplot draws the figures, the report was rewritten around the Vanda results | Every number in the report is regenerated from raw CSVs with one command | `analysis/`, `results/figures_cluster/`, `report/` |

How the cluster runs were made to work (details in Section B below and
report Section 5.1): the first round failed because remote nodes could not find
Open MPI (`--prefix` fixed it) and Vanda binds ranks to whole sockets by default
(`--bind-to core` / `--map-by core` fixed it). MPI runs over TCP because UCX
does not start on Vanda. The 256-core jobs ran under the project queue
(`batch_cpu`), because the free queue stops at 160 cores.

### Main results (one line each)

- Correct everywhere: same iteration counts and errors on every rank count, thread split, node layout and the GPU.
- Speedup at 256 cores: 74 (Jacobi 2D), 77 (Jacobi 3D), 137 (CG 2D), 103 (CG 3D). Inside one node memory bandwidth is the limit (efficiency about 0.3 by 32 cores); from 1 to 4 nodes scaling is linear or better.
- Weak scaling is flat from 1 node to 4 (efficiency 0.94 to 1.00 across nodes); all the loss is inside a node.
- At 256 ranks CG is limited by the all-reduce (about 0.1 ms each over TCP); Jacobi by the halo exchange, where face layout matters more than exchange strategy.
- Pure MPI beat every MPI+OpenMP split on Vanda (memory placement and single-threaded halo packing; both fixable).
- GPU (A40) is 1.9 to 2.7× faster than 36 CPU cores on large grids, slower on small ones.
- Gap filling: about 0.15 °C error for 64 km holes; the whole year fills in 2.5 s on 256 cores.

### What is remaining

| Item | Owner | Status |
|---|---|---|
| Team review of `report/report.pdf` (content, length, anything unclear) | Aditya, Apurwa | **to do** |
| CPU model of Vanda's CPU nodes: one sentence in report Sections 5.1 and 9.4 (job `hwinfo`, 1532679, queued) | Vanshika | waiting on queue |
| Two-GPU job (`poisson_gpu_multi`, 1527245) is still queued. Optional; if it has not run by about 11 Nov the report stays as it is (it says the job did not run) | Vanshika | optional |
| Check the report against the module's page/word limit (currently about 23 000 words, 34 pages) | team | **to do** |
| Final PDF build (`report/build.sh`) and submission | Vanshika | after review |

`report/report_laptop_version.md` is the earlier, laptop-only version of the
report, kept for comparison. It is not part of the submission.

### Not in this repository

The satellite data (`data/`, about 2.5 GB) and the filled fields
(`results/fields/`, about 600 MB) are too large for GitHub. They are rebuilt by
the commands in the Data section (`scripts/data/download.sh` then
`bin/gridtool`). Compiled programs (`bin/`) are rebuilt with `make all`.

## Layout

```
poisson-mpi/
  Makefile                builds the executables into bin/
  README.md               this file
  src/common.hpp          problem definition, command-line options, CSV output
  src/kernels.hpp         stencil, dot product, axpy kernels; plain and masked (gap filling) versions
  src/gridio.hpp          satellite grid file format, per-rank block loading, pixel mask, error measures
  src/solver.hpp          Jacobi and CG loops, written once for every backend and both problems
  src/decomp.hpp          MPI Cartesian decomposition, the four halo-exchange modes, MPI-IO output
  src/serial_main.cpp     serial (and OpenMP-only) driver
  src/mpi_main.cpp        MPI and MPI+OpenMP driver
  src/cuda/gpu_kernels.cuh    CUDA kernels (plain and masked)
  src/cuda/mpi_cuda_main.cu   MPI+CUDA driver
  src/cuda/cuda_compat.hpp    lets the CUDA code be compiled and tested without a GPU
  tools/gridtool.cpp          C++ data tool: downloaded ASCII grids -> grid file; info, slices, differences
  scripts/data/download.sh    downloads the daily satellite maps for a region and year (curl, no account)
  scripts/local/run_all.sh    every experiment on a laptop/workstation -> results/local/*.csv
  scripts/local/maps.sh       draws the observed / filled maps for one day -> results/figures/sst_maps.png
  scripts/atlas/env.sh        cluster settings to edit (queue, ranks per node, modules)
  scripts/atlas/gen_jobs.sh   writes the PBS jobs for the CPU cluster (1 .. 256 cores) into jobs/
  scripts/vanda/gpu.pbs       GPU job (Vanda): one GPU against the 36 CPU cores of the node; gpu_multi.pbs for two GPUs
  analysis/summarize.cpp      C++ tool: CSV -> tables.md, summary.txt and plot data (bin/summarize)
  analysis/plots.gp           gnuplot script: plot data -> the PNG figures
  analysis/maps.gp            gnuplot script: the satellite maps
  data/                       the satellite grids (not in the submission; rebuilt by the two commands below)
  results/local/*.csv         raw measurements from the laptop
  results/fields/*.grid       the filled temperature fields (the 3D one is 625 MB; both are regenerated by the data stage, not submitted)
  results/figures/            the laptop figures and tables (OpenMP-only figure and satellite maps are used in the report)
  results/cluster/*.csv       raw measurements from NUS Vanda, 4 nodes x 64 cores, and the GPU node
  results/cluster_8x32/*.csv  the same 256-rank runs on 8 nodes x 32 cores (node-layout comparison)
  results/figures_cluster/    the Vanda figures and tables used in the report (tables.md, summary.txt)
  logs_vanda/                 the Vanda job logs
  vanda_jobs/                 the exact PBS job files that produced results/cluster and results/cluster_8x32
  report/report.md, report.pdf   the project report (build with report/build.sh)
  handbook/handbook.pdf       a beginner's handbook explaining every file and every number
  tools/                      Markdown -> PDF build (pandoc + headless Chrome), stylesheets, handbook_source.md
```

Everything is C++ (and CUDA for the GPU build), with bash for the scripts and
gnuplot for the figures. No Python is used anywhere.

## Build

Requirements: a C++17 compiler, an MPI implementation (Open MPI or MPICH),
OpenMP (libomp on macOS: `brew install libomp`), `curl` and `gnuplot`, and
`nvcc` for the GPU build.

```
make all            # bin/poisson_serial  bin/poisson_omp  bin/poisson_mpi  bin/poisson_mpi_omp  bin/gridtool
make cuda           # bin/poisson_mpi_cuda       (needs nvcc; set CUDA_ARCH=sm_80 etc.)
make cuda_emul      # bin/poisson_mpi_cuda_emul  (the CUDA source run on the CPU, for testing)
```

On macOS use `make all CXX=clang++`. On the clusters load the compiler and MPI
modules first (Atlas: `source /app1/ebenv`; Vanda: compile inside an
interactive job, see `scripts/vanda/gpu.pbs`).

## Data

Source: NOAA CoastWatch ERDDAP server (https://coastwatch.noaa.gov/erddap), which
serves subsets of gridded satellite products as plain text without an account.

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
own block of that file directly, so no rank ever holds the whole year. For a
quick test, `scripts/data/download.sh india 2025-02-15` fetches one day.

## Run

All binaries share one command line. Without `--data` they solve the manufactured
Poisson problem with a known exact solution; with `--data` they fill the
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
| `--n N` or `--n NX,NY,NZ` | global interior grid points per dimension (manufactured problem) |
| `--local-n M` | weak scaling: each rank owns M points per dimension; the global grid follows the process grid |
| `--solver jacobi\|cg` | iteration |
| `--iters K` | maximum iterations (default 1000) |
| `--tol T` | stop when the relative residual drops below T (default 0: run exactly `--iters` sweeps) |
| `--check-every C` | Jacobi only: compute the global residual every C sweeps (0 = never) |
| `--exchange sendrecv\|isend\|overlap\|rma` | halo exchange strategy |
| `--dims PX,PY[,PZ]` | force the process grid shape |
| `--threads T` | OpenMP threads per rank |
| `--csv FILE --tag TEXT` | append one result row to FILE |
| `--data FILE` | satellite grid file; the grid size comes from its header |
| `--day K`, `--days N` | 2D: the day to fill (0-based); 3D: first day and number of days (0 = all) |
| `--tweight W` | 3D: weight of the time neighbours relative to the space neighbours (default 1) |
| `--holdout F`, `--holdout-block B` | hide a fraction F of the observed pixels, in BxB patches, to measure the fill error |
| `--eval-day K` | 3D: report the hidden-pixel error for day K only, so a 3D fill can be compared with the 2D fill of that day |
| `--ref FILE` | gap-free reference grid for comparison (same grid; made by `gridtool regrid`) |
| `--out FILE` | write the filled field as a grid file (MPI-IO in parallel) |

Each manufactured-problem run prints the iteration count, the relative residual
and the L2 and max error against the exact solution
`u = exp(x) sin(pi x) sin(pi y) [sin(pi z)]`. Each data run prints the pixel
counts (observed, missing, hidden, land), the RMSE, MAE and maximum error of
the fill on the hidden pixels, the RMSE of the trivial fill (every hole set to
the mean temperature) as a baseline, and, with `--ref`, the RMSE against the
reference analysis. Both print the time split into compute, halo exchange and
all-reduce.

## Reproduce the figures

Local (laptop/workstation):

```
scripts/local/run_all.sh 16 2                         # about 20 minutes on 16 cores plus the data stage
make figures LABEL="your machine"                     # bin/summarize + gnuplot -> results/figures/
```

The last stage of `run_all.sh` (`data`) runs only if `data/india_2025.grid`
exists (`GRID=... DAY=... DAYS3D=...` override the file, the 2D day and the
number of days in 3D). It writes `results/local/gapfill.csv`, the filled fields
in `results/fields/` and the map figure.

Cluster (Atlas or Vanda CPU nodes, up to 256 cores):

```
# edit scripts/atlas/env.sh: queue name, ranks per node, module lines
# copy data/india_2025.grid (and _ref.grid) into poisson-mpi/data/ on the cluster first
scripts/atlas/gen_jobs.sh --submit          # one job per node-size: node_1.pbs (serial + every rank count up to PPN, design, hybrid, data) + mpi_NP.pbs per multi-node count
qsub scripts/vanda/gpu.pbs                   # GPU vs CPU node, manufactured problem and the satellite data
qsub scripts/vanda/gpu_multi.pbs             # two GPUs (if the queue allows ngpus=2)
# when they finish:
make figures RESULTS=results/cluster FIGDIR=results/figures_cluster LABEL="NUS Vanda"
```

`make figures` compiles `analysis/summarize.cpp`, runs it on the CSV files (it
keeps the minimum of the repeated runs and derives speedup, efficiency,
communication share, convergence order, log-log slopes and the gap-filling
accuracy tables), and then runs `analysis/plots.gp` with gnuplot. Outputs: the
PNG figures, `tables.md` (every table in Markdown) and `summary.txt` (headline
numbers). gnuplot is the only tool needed besides a C++ compiler (`brew install
gnuplot` on macOS, `module load gnuplot` or `apt install gnuplot` elsewhere).

## Step by step: what to run, in what order

### A. On your own computer (laptop or workstation)

1. Install the tools once. macOS: `xcode-select --install`, then `brew install open-mpi libomp gnuplot pandoc`.
   Ubuntu/Debian: `sudo apt install build-essential openmpi-bin libopenmpi-dev gnuplot pandoc curl`.
   Check with `which mpicxx mpirun gnuplot curl`.
2. Build everything: `cd poisson-mpi && make all CXX=clang++` on macOS, or `make all` on Linux.
   This is the first thing to run; it produces `bin/poisson_serial`, `bin/poisson_omp`, `bin/poisson_mpi`, `bin/poisson_mpi_omp` and `bin/gridtool`.
   Optional: `make cuda_emul` builds the CUDA code as plain C++ for a logic check without a GPU.
3. Smoke test (ten seconds):
   ```
   bin/poisson_serial --dim 2 --n 128 --solver cg --tol 1e-8 --iters 100000
   mpirun -np 4 bin/poisson_mpi --dim 2 --n 128 --solver cg --tol 1e-8 --iters 100000
   ```
   Both lines must print the same `iters=`, `l2_err=` and `linf_err=`.
4. Get the data (once, needs internet): `scripts/data/download.sh india 2025 ref`, then the three `bin/gridtool` commands from the Data section.
   One day is enough to try the gap filler: `scripts/data/download.sh india 2025-04-10` and
   `bin/gridtool convert data/test.grid --mask data/raw/india/mask.asc data/raw/india/sst_2025-04-10.asc`, then
   `mpirun -np 4 bin/poisson_mpi --data data/test.grid --dim 2 --solver cg --tol 1e-6 --iters 100000 --holdout 0.1`.
5. Run every local experiment (writes `results/local/*.csv`): `scripts/local/run_all.sh 16 2`
   (first argument: cores to use; second: repeats per configuration; optional third: a comma list of stages,
   e.g. `data` to rerun only the satellite part).
6. Make the tables and figures: `make figures LABEL="my laptop"` writes `results/figures/`.
7. Rebuild the report PDF after editing `report/report.md`: `report/build.sh` (needs pandoc and Google Chrome).

### B. On the NUS HPC clusters (Atlas for CPUs, Vanda for 256 cores and the GPUs)

The two clusters have different limits, checked on 3 November 2026:

| | Atlas (`atlas8.nus.edu.sg`) | Vanda (`vanda.nus.edu.sg`) |
|---|---|---|
| node size | 96 cores | 72 cores (some 64); GPU nodes with 2 GPUs |
| queue | `parallel`: at most 96 cores per job, 4 running jobs | `-q auto_free` (free tier) or `-P <project>` (see `hpc project`) |
| toolchain | `source /app1/ebenv; module load GCC/13.2.0 OpenMPI/4.1.6-GCC-13.2.0` | `module load gcc/13.1.0 openmpi4/gcc/4.1.8 cuda12.4/toolkit/12.4.1` |
| where to work | home directory | `/scratch/$USER` (compile inside an interactive job: the login node has a different CPU) |
| what it gives | serial, 1 to 96 ranks, design study, hybrid splits of one node, satellite runs | the same up to 64 ranks per node, plus 128 and 256 ranks on 2 and 4 nodes, and the GPU jobs |

Atlas:

1. Copy the folder up without the raw downloads and laptop binaries (compute nodes have no internet):
   `rsync -avz --exclude data/raw --exclude results/fields --exclude bin poisson-mpi USERNAME@atlas8.nus.edu.sg:~/`
2. `ssh USERNAME@atlas8.nus.edu.sg`, `cd poisson-mpi`, load the toolchain from the table, `make all`.
   Smoke test: `bin/poisson_serial --dim 2 --n 128 --solver cg --tol 1e-8 --iters 100000` and the same with `mpirun -np 4 bin/poisson_mpi` must print the same `iters=` and `l2_err=`.
3. `scripts/atlas/gen_jobs.sh` writes `jobs/node_1.pbs` (one 96-core job holding everything) using the Atlas values in `scripts/atlas/env.sh`. Submit it with `qsub jobs/node_1.pbs`.
4. Watch with `qstat -u $USER`; the log is `node_1.o<jobid>` and the measurements append to `results/cluster/*.csv`.

Vanda:

1. Log in (`ssh USERNAME@vanda.nus.edu.sg`). Your Atlas home is visible read-only at `/atlas/home/$USER`, so copy from there:
   `mkdir -p /scratch/$USER && rsync -a --exclude bin --exclude jobs --exclude results/cluster --exclude 'node_1.o*' /atlas/home/$USER/poisson-mpi /scratch/$USER/`
   (or rsync from the laptop to `vanda.nus.edu.sg:/scratch/USERNAME/` with the excludes of Atlas step 1).
2. Build on a compute node (the login node has a different CPU) and queue every run behind the build, all from the login node in `/scratch/$USER/poisson-mpi`:
   ```
   BUILD=$(qsub scripts/vanda/build.pbs)
   QSUB_OPTS="-W depend=afterok:$BUILD" \
   QUEUE=auto_free QUEUE_MULTI=auto_free PPN=64 RANKS="1 2 4 8 16 32 64 128 256" MEM=200gb \
   MODULES="module load gcc/13.1.0 openmpi4/gcc/4.1.8" MPIRUN='mpirun -hostfile $PBS_NODEFILE' \
   scripts/atlas/gen_jobs.sh --submit
   qsub -W depend=afterok:$BUILD scripts/vanda/gpu.pbs
   qsub -W depend=afterok:$BUILD scripts/vanda/gpu_multi.pbs
   ```
   `build.pbs` compiles the CPU programs and the CUDA program and runs a smoke test (log: `poisson_build.o<jobid>`). The generator writes and submits `node_1.pbs`, `mpi_128.pbs`, `mpi_256.pbs` and `hybrid_256.pbs` with 64 ranks per node, so 256 ranks are 4 nodes; with a project add `PROJECT=<name>`. Every job waits until the build has succeeded. To build interactively instead: `qsub -I -q auto_free -l select=1:ncpus=8:mem=32gb -l walltime=00:30:00`, then load the toolchain and `make all cuda`.
3. Watch with `qstat -u $USER` (`H` = held until the build finishes, `Q` = waiting, `R` = running).

**What the report's Vanda runs actually used (November 2026).** The settings above did not work unchanged. These are the ones that did:

- **Project queue for 256 cores.** The free `cpu_parallel` queue stops at 160 cores, so the CPU jobs ran under the project (`PROJECT=personal-<NUSNET id>`, queue `batch_cpu`).
- **Full `mpirun` line.** Open MPI's launcher is not on the remote nodes' path, and the site default binds each rank to a whole socket. The line was:
  ```
  MPIRUN='mpirun --prefix /cm/shared/apps/openmpi4/gcc/4.1.8 -hostfile $PBS_NODEFILE --bind-to core --mca mca_base_component_show_load_errors 0 -x LD_LIBRARY_PATH -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES'
  ```
- **Two edits to the generated jobs before `qsub`.** One places pure-MPI ranks core by core; the other caps each one-sided exchange run at 15 minutes:
  ```
  sed -i '/--map-by ppr/! s/--bind-to core/--map-by core --bind-to core/' jobs/*.pbs
  sed -i 's/^  mpirun \(.*--exchange \$mode\)/  timeout 900 mpirun \1/' jobs/mpi_256.pbs
  ```
- **TCP only.** MPI runs over TCP, the site default. UCX does not start on Vanda (`PML ucx cannot be selected`).
- **The 8-node layout.** The second layout in the report (8 nodes × 32 cores) is the same generator with `PPN=32 RANKS=256 MEM=100gb`, writing to `results/cluster_8x32` (`sed -i 's#results/cluster#results/cluster_8x32#g'` on the generated jobs).
- **Job logs.** The logs are `<jobname>.o<jobid>` in the project folder.

Back on the laptop, keep the machines and layouts apart:

```
rsync -av USERNAME@vanda.nus.edu.sg:/scratch/USERNAME/poisson-mpi/results/cluster poisson-mpi/results/
rsync -av USERNAME@vanda.nus.edu.sg:/scratch/USERNAME/poisson-mpi/results/cluster_8x32 poisson-mpi/results/
make figures RESULTS=results/cluster FIGDIR=results/figures_cluster \
  LABEL="NUS Vanda, 4 nodes x 64 cores" \
  ALT=results/cluster_8x32 ALTLABEL="NUS Vanda, 8 nodes x 32 cores"
```

`ALT` is optional. When given, `bin/summarize` adds a node-layout table (`layout_compare.dat` and the "Node layout" section of `tables.md`), and gnuplot draws `layout_compare.png`. Then run `report/build.sh`.

If the build fails on macOS with missing C++ headers (`<cmath>` or `<vector>` not found), the Command Line Tools' libc++ is incomplete. Build the summariser against the SDK instead:

```
SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX15.sdk
make figures CXXFLAGS="-O2 -std=c++17 -isysroot $SDK -nostdinc++ -isystem $SDK/usr/include/c++/v1" ...
```

Queue names, module names, node sizes and the scratch location are what differ between sites, which is why they sit in `scripts/atlas/env.sh` and at the top of the two GPU job files rather than in the code.

## Validation

`scripts/local/run_all.sh` starts with a validation block that (1) checks the
discretisation error falls by 4x when the grid spacing halves, (2) checks that
Jacobi run to a tolerance stops at the same iteration in serial and in
parallel, and (3) runs the same fixed-iteration problem through every
implementation (serial, OpenMP, MPI with each exchange mode, MPI+OpenMP,
emulated MPI+CUDA) and tabulates the errors, which agree to all printed digits.
For the satellite data the check is the same in spirit: the hidden-pixel error
printed by every implementation and every rank count must be identical (the
`rmse_hidden` column of the gap-filling table), and the hidden pixels are chosen
by a hash of their global position so that the same pixels are hidden whatever
the decomposition.
