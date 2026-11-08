#!/bin/bash
# Run every local experiment on a laptop/workstation and write CSVs to results/local/.
# Usage:  scripts/local/run_all.sh [MAXCORES] [REPEATS] [STAGES]
# Each timing configuration is run REPEATS times; the analysis takes the minimum.
# STAGES is "all" (default) or a comma list of: validation,serial,strong,weak,design,hybrid,omp,data
# The data stage needs the satellite grid file (default data/india_2025.grid; override with GRID=...),
# the day index for the 2D case (DAY, default 99 = 2025-04-10) and optionally DAYS3D to use fewer days in 3D.
set -u
cd "$(dirname "$0")/../.."
MAXCORES=${1:-16}
REPEATS=${2:-2}
STAGES=${3:-all}
want() { [ "$STAGES" = all ] || [[ ",$STAGES," == *",$1,"* ]]; }
OUT=${OUT:-results/local}
mkdir -p "$OUT"
LOG="$OUT/run.log"
echo "=== run_all.sh started $(date)  maxcores=$MAXCORES repeats=$REPEATS stages=$STAGES" | tee -a "$LOG"

S=bin/poisson_serial; O=bin/poisson_omp; M=bin/poisson_mpi; MO=bin/poisson_mpi_omp
for b in $S $O $M $MO; do [ -x $b ] || { echo "missing $b: run make all first"; exit 1; }; done

# Rank counts to test: powers of two up to MAXCORES, plus any EXTRA_RANKS (e.g. EXTRA_RANKS=12
# on a 12 performance + 4 efficiency core laptop, to see the effect of the slow cores)
RANKS=(); r=1; while [ $r -le $MAXCORES ]; do RANKS+=($r); r=$((r*2)); done
for e in ${EXTRA_RANKS:-}; do RANKS+=($e); done
RANKS=($(printf '%s\n' "${RANKS[@]}" | sort -n | uniq))

run_serial() { # csv tag args...
  local csv=$1 tag=$2; shift 2
  for ((k=0;k<REPEATS;k++)); do $S --csv "$csv" --tag "$tag" "$@" | grep -E '^iters=|^t_total' | tr '\n' ' ' | tee -a "$LOG"; echo | tee -a "$LOG"; done
}
run_omp() { # csv tag threads args...
  local csv=$1 tag=$2 th=$3; shift 3
  for ((k=0;k<REPEATS;k++)); do OMP_NUM_THREADS=$th $O --threads $th --csv "$csv" --tag "$tag" "$@" | grep -E '^t_total' | tee -a "$LOG"; done
}
run_mpi() { # csv tag np args...
  local csv=$1 tag=$2 np=$3; shift 3
  for ((k=0;k<REPEATS;k++)); do mpirun -np $np $M --csv "$csv" --tag "$tag" "$@" | grep -E '^t_total' | tee -a "$LOG"; done
}
run_hybrid() { # csv tag np threads args...
  local csv=$1 tag=$2 np=$3 th=$4; shift 4
  for ((k=0;k<REPEATS;k++)); do OMP_NUM_THREADS=$th mpirun -np $np $MO --threads $th --csv "$csv" --tag "$tag" "$@" | grep -E '^t_total' | tee -a "$LOG"; done
}
say() { echo "--- $* ($(date +%H:%M:%S))" | tee -a "$LOG"; }

# ---------------------------------------------------------------- 1. validation
if want validation; then
CSV=$OUT/validation.csv; rm -f "$CSV"
say "validation: convergence order (serial CG to tight tolerance)"
for n in 32 64 128 256 512 1024; do $S --dim 2 --n $n --solver cg --tol 1e-10 --iters 1000000 --csv "$CSV" --tag order2d >/dev/null; done
for n in 8 16 32 64 128;         do $S --dim 3 --n $n --solver cg --tol 1e-10 --iters 1000000 --csv "$CSV" --tag order3d >/dev/null; done
say "validation: jacobi to convergence on a small grid (serial, 4 ranks)"
$S --dim 2 --n 64 --solver jacobi --tol 1e-8 --iters 1000000 --check-every 50 --csv "$CSV" --tag jacobi_conv >/dev/null
mpirun -np 4 $M --dim 2 --n 64 --solver jacobi --tol 1e-8 --iters 1000000 --check-every 50 --csv "$CSV" --tag jacobi_conv >/dev/null
$S --dim 3 --n 24 --solver jacobi --tol 1e-8 --iters 1000000 --check-every 50 --csv "$CSV" --tag jacobi_conv >/dev/null
mpirun -np 8 $M --dim 3 --n 24 --solver jacobi --tol 1e-8 --iters 1000000 --check-every 50 --csv "$CSV" --tag jacobi_conv >/dev/null
say "validation: serial vs every parallel variant, fixed 300 Jacobi sweeps and CG to 1e-9"
$S --dim 2 --n 300 --iters 300 --csv "$CSV" --tag agree >/dev/null
$S --dim 3 --n 45  --iters 300 --csv "$CSV" --tag agree >/dev/null
$S --dim 2 --n 300 --solver cg --tol 1e-9 --iters 100000 --csv "$CSV" --tag agree >/dev/null
$S --dim 3 --n 45  --solver cg --tol 1e-9 --iters 100000 --csv "$CSV" --tag agree >/dev/null
OMP_NUM_THREADS=4 $O --threads 4 --dim 2 --n 300 --iters 300 --csv "$CSV" --tag agree >/dev/null
OMP_NUM_THREADS=4 $O --threads 4 --dim 3 --n 45 --solver cg --tol 1e-9 --iters 100000 --csv "$CSV" --tag agree >/dev/null
for mode in sendrecv isend overlap rma; do
  mpirun -np 6 $M --dim 2 --n 300 --iters 300 --exchange $mode --csv "$CSV" --tag agree >/dev/null
  mpirun -np 12 $M --dim 3 --n 45 --iters 300 --exchange $mode --csv "$CSV" --tag agree >/dev/null
  OMP_NUM_THREADS=2 mpirun -np 6 $MO --threads 2 --dim 2 --n 300 --solver cg --tol 1e-9 --iters 100000 --exchange $mode --csv "$CSV" --tag agree >/dev/null
  OMP_NUM_THREADS=2 mpirun -np 7 $MO --threads 2 --dim 3 --n 45 --solver cg --tol 1e-9 --iters 100000 --exchange $mode --csv "$CSV" --tag agree >/dev/null
done
if [ -x bin/poisson_mpi_cuda_emul ]; then
  say "validation: CUDA driver through the CPU emulation layer"
  mpirun -np 4 bin/poisson_mpi_cuda_emul --dim 2 --n 300 --iters 300 --csv "$CSV" --tag agree >/dev/null
  mpirun -np 4 bin/poisson_mpi_cuda_emul --dim 2 --n 300 --iters 300 --exchange overlap --csv "$CSV" --tag agree >/dev/null
  mpirun -np 8 bin/poisson_mpi_cuda_emul --dim 3 --n 45 --iters 300 --csv "$CSV" --tag agree >/dev/null
  mpirun -np 6 bin/poisson_mpi_cuda_emul --dim 2 --n 300 --solver cg --tol 1e-9 --iters 100000 --csv "$CSV" --tag agree >/dev/null
  mpirun -np 7 bin/poisson_mpi_cuda_emul --dim 3 --n 45 --solver cg --tol 1e-9 --iters 100000 --exchange overlap --csv "$CSV" --tag agree >/dev/null
fi

fi
# ---------------------------------------------------------------- 2. serial scaling with problem size
if want serial; then
CSV=$OUT/serial_scaling.csv; rm -f "$CSV"
say "serial scaling: jacobi 2D"; for n in 128 256 512 1024 2048 4096; do run_serial "$CSV" jac2d --dim 2 --n $n --iters 200; done
say "serial scaling: jacobi 3D"; for n in 16 32 64 128 256;         do run_serial "$CSV" jac3d --dim 3 --n $n --iters 200; done
say "serial scaling: cg 2D";     for n in 128 256 512 1024 2048;    do run_serial "$CSV" cg2d --dim 2 --n $n --solver cg --tol 1e-8 --iters 1000000; done
say "serial scaling: cg 3D";     for n in 32 64 128 192 256;        do run_serial "$CSV" cg3d --dim 3 --n $n --solver cg --tol 1e-8 --iters 1000000; done

fi
# ---------------------------------------------------------------- 3. strong scaling (fixed problem, more ranks)
if want strong; then
CSV=$OUT/strong.csv; rm -f "$CSV"
run_serial "$CSV" jac2d --dim 2 --n 2048 --iters 200
run_serial "$CSV" jac3d --dim 3 --n 192  --iters 200
run_serial "$CSV" cg2d  --dim 2 --n 1024 --solver cg --tol 1e-8 --iters 1000000
run_serial "$CSV" cg3d  --dim 3 --n 128  --solver cg --tol 1e-8 --iters 1000000
for np in "${RANKS[@]}"; do
  say "strong scaling np=$np"
  run_mpi "$CSV" jac2d $np --dim 2 --n 2048 --iters 200
  run_mpi "$CSV" jac3d $np --dim 3 --n 192  --iters 200
  run_mpi "$CSV" cg2d  $np --dim 2 --n 1024 --solver cg --tol 1e-8 --iters 1000000
  run_mpi "$CSV" cg3d  $np --dim 3 --n 128  --solver cg --tol 1e-8 --iters 1000000
done

fi
# ---------------------------------------------------------------- 4. weak scaling (fixed work per rank)
if want weak; then
CSV=$OUT/weak.csv; rm -f "$CSV"
for np in "${RANKS[@]}"; do
  say "weak scaling np=$np"
  run_mpi "$CSV" jac2d $np --dim 2 --local-n 1024 --iters 200
  run_mpi "$CSV" jac3d $np --dim 3 --local-n 64   --iters 200
  run_mpi "$CSV" cg2d  $np --dim 2 --local-n 512  --solver cg --iters 300
  run_mpi "$CSV" cg3d  $np --dim 3 --local-n 48   --solver cg --iters 300
done

fi
# ---------------------------------------------------------------- 5. design choices at the largest rank count
NP=${RANKS[$((${#RANKS[@]}-1))]}     # last element (Bash 3 has no negative indices)
if want design; then
CSV=$OUT/exchange.csv; rm -f "$CSV"
for np in $((NP/2)) $NP; do for mode in sendrecv isend overlap rma; do
  say "exchange mode $mode np=$np"
  run_mpi "$CSV" jac2d $np --dim 2 --n 2048 --iters 200 --exchange $mode
  run_mpi "$CSV" jac3d $np --dim 3 --n 192  --iters 200 --exchange $mode
  run_mpi "$CSV" cg2d  $np --dim 2 --n 1024 --solver cg --iters 300 --exchange $mode
done; done

CSV=$OUT/decomp.csv; rm -f "$CSV"
say "decomposition shape, np=$NP"
for d in "$NP,1" "$((NP/2)),2" "$((NP/4)),4" "2,$((NP/2))" "1,$NP"; do run_mpi "$CSV" jac2d $NP --dim 2 --n 2048 --iters 200 --dims $d; done
for d in "$NP,1,1" "1,1,$NP" "$((NP/4)),4,1" "$((NP/4)),2,2" "1,$((NP/4)),4" "2,2,$((NP/4))"; do run_mpi "$CSV" jac3d $NP --dim 3 --n 192 --iters 200 --dims $d; done

CSV=$OUT/checkevery.csv; rm -f "$CSV"
say "residual check interval, np=$NP"
for c in 0 1 10 100; do
  run_mpi "$CSV" jac2d $NP --dim 2 --n 2048 --iters 200 --check-every $c
  run_mpi "$CSV" jac3d $NP --dim 3 --n 192  --iters 200 --check-every $c
done

fi
# ---------------------------------------------------------------- 6. MPI+OpenMP: ranks x threads = MAXCORES
if want hybrid; then
CSV=$OUT/hybrid.csv; rm -f "$CSV"
np=$NP; th=1
while [ $np -ge 1 ]; do
  say "hybrid ranks=$np threads=$th"
  run_hybrid "$CSV" jac2d $np $th --dim 2 --n 2048 --iters 200
  run_hybrid "$CSV" jac3d $np $th --dim 3 --n 192  --iters 200
  run_hybrid "$CSV" cg2d  $np $th --dim 2 --n 1024 --solver cg --tol 1e-8 --iters 1000000
  run_hybrid "$CSV" cg3d  $np $th --dim 3 --n 128  --solver cg --tol 1e-8 --iters 1000000
  np=$((np/2)); th=$((th*2))
done

fi
# ---------------------------------------------------------------- 7. OpenMP only: thread scaling
if want omp; then
CSV=$OUT/omp_threads.csv; rm -f "$CSV"
for th in "${RANKS[@]}"; do
  say "openmp threads=$th"
  run_omp "$CSV" jac2d $th --dim 2 --n 2048 --iters 200
  run_omp "$CSV" jac3d $th --dim 3 --n 192  --iters 200
  run_omp "$CSV" cg2d  $th --dim 2 --n 1024 --solver cg --tol 1e-8 --iters 1000000
  run_omp "$CSV" cg3d  $th --dim 3 --n 128  --solver cg --tol 1e-8 --iters 1000000
done
fi
# ---------------------------------------------------------------- 8. satellite gap filling on the real data
GRID=${GRID:-data/india_2025.grid}; DAY=${DAY:-99}; DAYS3D=${DAYS3D:-0}
if want data && [ -f "$GRID" ]; then
CSV=$OUT/gapfill.csv; rm -f "$CSV"
mkdir -p results/fields
D2="--data $GRID --dim 2 --day $DAY --holdout 0.1 --holdout-block 16"
D3="--data $GRID --dim 3 --days $DAYS3D --holdout 0.1 --holdout-block 16 --tweight 1"
say "gap filling: serial baselines (CG to 1e-6, Jacobi 1000 sweeps)"
run_serial "$CSV" sst2d_cg  $D2 --solver cg --tol 1e-6 --iters 100000 --out results/fields/fill2d_day$DAY.grid
run_serial "$CSV" sst2d_jac $D2 --solver jacobi --iters 1000
run_serial "$CSV" sst3d_cg  $D3 --solver cg --tol 1e-6 --iters 100000 --out results/fields/fill3d.grid
run_serial "$CSV" sst3d_jac $D3 --solver jacobi --iters 200
for np in "${RANKS[@]}"; do
  say "gap filling: strong scaling np=$np"
  run_mpi "$CSV" sst2d_cg  $np $D2 --solver cg --tol 1e-6 --iters 100000
  run_mpi "$CSV" sst2d_jac $np $D2 --solver jacobi --iters 1000
  run_mpi "$CSV" sst3d_cg  $np $D3 --solver cg --tol 1e-6 --iters 100000
  run_mpi "$CSV" sst3d_jac $np $D3 --solver jacobi --iters 200
done
say "gap filling: OpenMP only and MPI+OpenMP at $MAXCORES cores"
run_omp "$CSV" sst2d_cg $NP $D2 --solver cg --tol 1e-6 --iters 100000
run_omp "$CSV" sst3d_cg $NP $D3 --solver cg --tol 1e-6 --iters 100000
run_hybrid "$CSV" sst3d_cg $((NP/4)) 4 $D3 --solver cg --tol 1e-6 --iters 100000
say "gap filling: accuracy against the size of the hidden patches and the time weight (np=$NP)"
REF=${REF:-${GRID%.grid}_ref.grid}; [ -f "$REF" ] && REFOPT="--ref $REF" || REFOPT=""
for b in 1 4 16 64; do
  mpirun -np $NP $M --data $GRID --dim 2 --day $DAY --holdout 0.1 --holdout-block $b --solver cg --tol 1e-6 --iters 100000 $REFOPT --csv "$CSV" --tag acc2d | grep -E "^iters" | tee -a "$LOG"
  mpirun -np $NP $M --data $GRID --dim 3 --days $DAYS3D --holdout 0.1 --holdout-block $b --tweight 1 --solver cg --tol 1e-6 --iters 100000 $REFOPT --csv "$CSV" --tag acc3d | grep -E "^iters" | tee -a "$LOG"
  mpirun -np $NP $M --data $GRID --dim 3 --days $DAYS3D --holdout 0.1 --holdout-block $b --tweight 1 --solver cg --tol 1e-6 --iters 100000 $REFOPT --eval-day $DAY --csv "$CSV" --tag acc3d_day | grep -E "^iters" | tee -a "$LOG"
done
for w in 0.01 0.03 0.1 0.3 3; do
  mpirun -np $NP $M --data $GRID --dim 3 --days $DAYS3D --holdout 0.1 --holdout-block 16 --tweight $w --solver cg --tol 1e-6 --iters 100000 $REFOPT --csv "$CSV" --tag acc3d | grep -E "^iters" | tee -a "$LOG"
  mpirun -np $NP $M --data $GRID --dim 3 --days $DAYS3D --holdout 0.1 --holdout-block 16 --tweight $w --solver cg --tol 1e-6 --iters 100000 $REFOPT --eval-day $DAY --csv "$CSV" --tag acc3d_day | grep -E "^iters" | tee -a "$LOG"
done
say "gap filling: maps"
scripts/local/maps.sh "$GRID" $DAY results/fields/fill2d_day$DAY.grid results/fields/fill3d.grid results/figures/sst_maps.png | tee -a "$LOG"
elif want data; then echo "data stage skipped: $GRID not found (see README, section Data)" | tee -a "$LOG"
fi
echo "=== run_all.sh finished $(date)" | tee -a "$LOG"
