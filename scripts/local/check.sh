#!/bin/bash
# Pass/fail tests: every parallel build must give the serial answer, edge cases
# included, and invalid options must be rejected.
#
#   make check            (builds first)   or   scripts/local/check.sh
#
# Needs at most 8 MPI ranks and takes about a minute on a laptop. Set MPIRUN to
# change the launcher (default "mpirun --oversubscribe", for Open MPI).
# The data tests run only if data/india_2025.grid exists (override with GRID=...);
# the --days window test also needs the raw daily files in data/raw/india/.
set -u
cd "$(dirname "$0")/../.."
MPIRUN=${MPIRUN:-mpirun --oversubscribe}
S=bin/poisson_serial; O=bin/poisson_omp; M=bin/poisson_mpi; MO=bin/poisson_mpi_omp; C=bin/poisson_mpi_cuda_emul
for b in $S $O $M $MO; do [ -x $b ] || { echo "missing $b: run make all first"; exit 1; }; done
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
npass=0; nfail=0
ok()  { echo "PASS  $*"; npass=$((npass+1)); }
bad() { echo "FAIL  $*"; nfail=$((nfail+1)); }

# The answer of a run: iteration count and error (manufactured problem), or
# iteration count and hidden-pixel errors (satellite data).
answer() {
  "$@" 2>&1 | awk '/^iters=/{ for (i = 1; i <= NF; i++) { if ($i ~ /^\(baseline/) break; n = split($i, a, "=");
        if (n == 2 && (a[1] == "iters" || a[1] == "l2_err" || a[1] == "rmse" || a[1] == "mae" || a[1] == "max")) printf "%s=%s ", a[1], a[2] } }'
}
same() { # name reference command...
  local name=$1 ref=$2; shift 2
  local got; got=$(answer "$@")
  if [ -n "$ref" ] && [ "$got" = "$ref" ]; then ok "$name"; else bad "$name: expected [$ref] got [$got]"; fi
}
rejects() { # name command...
  local name=$1; shift
  if "$@" >/dev/null 2>&1; then bad "accepted invalid input: $name"; else ok "rejects $name"; fi
}

echo "--- Jacobi, fixed sweeps: the iterates must be identical in every build"
ref=$(answer $S --dim 2 --n 61 --iters 300)
same "OpenMP 3 threads, 2D" "$ref" env OMP_NUM_THREADS=3 $O --threads 3 --dim 2 --n 61 --iters 300
for mode in sendrecv isend overlap rma; do
  same "MPI 6 ranks, $mode, 2D (uneven blocks)" "$ref" $MPIRUN -np 6 $M --dim 2 --n 61 --iters 300 --exchange $mode
done
same "MPI+OpenMP 3x2, overlap, 2D" "$ref" env OMP_NUM_THREADS=2 $MPIRUN -np 3 $MO --threads 2 --dim 2 --n 61 --iters 300 --exchange overlap
ref=$(answer $S --dim 3 --n 37 --iters 200)
for mode in sendrecv isend overlap rma; do
  same "MPI 6 ranks, $mode, 3D n=37 (uneven blocks)" "$ref" $MPIRUN -np 6 $M --dim 3 --n 37 --iters 200 --exchange $mode
done

echo "--- blocks one cell thick (every cell touches a halo)"
ref=$(answer $S --dim 3 --n 8 --iters 100)
for mode in sendrecv isend overlap rma; do
  same "3D, 1x1x8 ranks (one z-layer each), $mode" "$ref" $MPIRUN -np 8 $M --dim 3 --n 8 --iters 100 --dims 1,1,8 --exchange $mode
done
same "3D, 8x1x1 ranks (one x-layer each), overlap" "$ref" $MPIRUN -np 8 $M --dim 3 --n 8 --iters 100 --dims 8,1,1 --exchange overlap
same "3D, 2x2x2 ranks of 4^3, overlap" "$ref" $MPIRUN -np 8 $M --dim 3 --n 8 --iters 100 --dims 2,2,2 --exchange overlap
ref=$(answer $S --dim 2 --n 8 --iters 100)
same "2D, 1x8 ranks (one row each), overlap" "$ref" $MPIRUN -np 8 $M --dim 2 --n 8 --iters 100 --dims 1,8 --exchange overlap

echo "--- conjugate gradient and the stopping rule"
ref=$(answer $S --dim 3 --n 33 --solver cg --tol 1e-8 --iters 10000)
same "CG 3D, MPI+OpenMP 4x2, overlap" "$ref" env OMP_NUM_THREADS=2 $MPIRUN -np 4 $MO --threads 2 --dim 3 --n 33 --solver cg --tol 1e-8 --iters 10000 --exchange overlap
ref=$(answer $S --dim 3 --n 8 --solver cg --tol 1e-10 --iters 1000)
same "CG 3D, one z-layer per rank, overlap" "$ref" $MPIRUN -np 8 $M --dim 3 --n 8 --solver cg --tol 1e-10 --iters 1000 --dims 1,1,8 --exchange overlap
ref=$(answer $S --dim 2 --n 32 --tol 1e-6 --check-every 10 --iters 100000)
same "Jacobi to tolerance, 4 ranks stop at the same sweep" "$ref" $MPIRUN -np 4 $M --dim 2 --n 32 --tol 1e-6 --check-every 10 --iters 100000

if [ -x $C ]; then
  echo "--- the CUDA driver through its CPU emulation layer"
  ref=$(answer $S --dim 2 --n 61 --iters 300)
  same "CUDA (emulated) 4 ranks, overlap, 2D" "$ref" $MPIRUN -np 4 $C --dim 2 --n 61 --iters 300 --exchange overlap
  ref=$(answer $S --dim 3 --n 8 --iters 100)
  same "CUDA (emulated) one z-layer per rank, overlap" "$ref" $MPIRUN -np 8 $C --dim 3 --n 8 --iters 100 --dims 1,1,8 --exchange overlap
  ref=$(answer $S --dim 3 --n 33 --solver cg --tol 1e-8 --iters 10000)
  same "CUDA (emulated) CG 3D, 3 ranks" "$ref" $MPIRUN -np 3 $C --dim 3 --n 33 --solver cg --tol 1e-8 --iters 10000
else
  echo "SKIP  CUDA emulation tests (make cuda_emul builds $C)"
fi

echo "--- invalid options"
rejects "--n 0" $S --n 0
rejects "--n -5" $S --n -5
rejects "--n abc" $S --n abc
rejects "--iters -3" $S --iters -3
rejects "--tol -1" $S --tol -1
rejects "--dim 3 --n 64,64 (two values in 3D)" $S --dim 3 --n 64,64
rejects "--dim 4" $S --dim 4
rejects "--holdout 1.5" $S --holdout 1.5
rejects "an unknown option" $S --bogus
rejects "--dims 3,3 on 4 ranks" $MPIRUN -np 4 $M --dims 3,3

GRID=${GRID:-data/india_2025.grid}
if [ -f "$GRID" ]; then
  echo "--- satellite data ($GRID)"
  ref=$(answer $S --data "$GRID" --dim 2 --day 99 --holdout 0.1 --holdout-block 16 --solver cg --tol 1e-6 --iters 100000)
  same "2D gap fill, 7 ranks, rma" "$ref" $MPIRUN -np 7 $M --data "$GRID" --dim 2 --day 99 --holdout 0.1 --holdout-block 16 --solver cg --tol 1e-6 --iters 100000 --exchange rma
  ref=$(answer $S --data "$GRID" --dim 3 --day 90 --days 10 --holdout 0.1 --holdout-block 16 --solver cg --tol 1e-6 --iters 100000)
  same "3D gap fill, 10 days, 8 ranks, overlap" "$ref" $MPIRUN -np 8 $M --data "$GRID" --dim 3 --day 90 --days 10 --holdout 0.1 --holdout-block 16 --solver cg --tol 1e-6 --iters 100000 --exchange overlap
  $S --data "$GRID" --dim 3 --day 90 --days 5 --solver jacobi --iters 50 --out "$TMP/serial.grid" >/dev/null 2>&1
  $MPIRUN -np 6 $M --data "$GRID" --dim 3 --day 90 --days 5 --solver jacobi --iters 50 --dims 3,2,1 --out "$TMP/mpi.grid" >/dev/null 2>&1
  if [ -s "$TMP/serial.grid" ] && cmp -s "$TMP/serial.grid" "$TMP/mpi.grid"; then ok "MPI-IO output of 6 ranks is byte-identical to the serial file"
  else bad "MPI-IO output of 6 ranks differs from the serial file"; fi
  RAW=data/raw/india
  if [ -x bin/gridtool ] && [ -f $RAW/mask.asc ] && ls $RAW/sst_2025-04-0[1-5].asc >/dev/null 2>&1; then
    # A window cut from the year (--day 90 --days 5 = 1-5 April) must be solved exactly
    # like a file that holds only those five days: the day after the window is not data.
    bin/gridtool convert "$TMP/five.grid" --mask $RAW/mask.asc $RAW/sst_2025-04-0[1-5].asc >/dev/null 2>&1
    $S --data "$TMP/five.grid" --dim 3 --solver cg --tol 1e-8 --iters 100000 --out "$TMP/a.grid" >/dev/null 2>&1
    $S --data "$GRID" --dim 3 --day 90 --days 5 --solver cg --tol 1e-8 --iters 100000 --out "$TMP/b.grid" >/dev/null 2>&1
    if [ -s "$TMP/a.grid" ] && cmp -s "$TMP/a.grid" "$TMP/b.grid"; then ok "a --days window gives the same field as a file holding only those days"
    else bad "a --days window differs from a file holding only those days"; fi
  else
    echo "SKIP  --days window test (needs bin/gridtool and $RAW/sst_2025-04-0[1-5].asc)"
  fi
else
  echo "SKIP  data tests ($GRID not found)"
fi

echo "=== $npass passed, $nfail failed"
[ $nfail -eq 0 ]
