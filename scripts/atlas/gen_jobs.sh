#!/bin/bash
# Generate (and optionally submit) the PBS jobs for the CPU scaling study.
#   scripts/atlas/gen_jobs.sh            -> writes jobs/*.pbs
#   scripts/atlas/gen_jobs.sh --submit   -> also runs qsub on each
#
# Site settings come from scripts/atlas/env.sh (queue, ranks per node, modules).
# The rank counts to run come from RANKS (default "1 2 4 8 16 32 64 96 128 256").
# Every rank count that fits on one node (<= PPN) is run inside ONE job,
# jobs/node_1.pbs, together with the serial baselines, the design study, the
# hybrid splits of one node and the satellite-data runs: queues often cap the
# number of running jobs per user, and a single job avoids that. Every rank
# count above PPN gets its own multi-node job (mpi_NP.pbs plus data_NP.pbs),
# submitted to QUEUE_MULTI (default: QUEUE), and the largest of them also gets
# the design study, the hybrid splits and the data accuracy study.
#
# Hybrid jobs use Open MPI's --map-by ppr:N:node:pe=T; with MPICH or Intel MPI
# replace it with that implementation's binding flags (-ppn N, I_MPI_PIN_DOMAIN).
# Results are appended to results/cluster/*.csv; copy that folder back and run
#   make figures RESULTS=results/cluster FIGDIR=results/figures_cluster LABEL="NUS Atlas"
set -u
cd "$(dirname "$0")/../.."
source scripts/atlas/env.sh
QUEUE_MULTI=${QUEUE_MULTI:-$QUEUE}
RANKS=${RANKS:-"1 2 4 8 16 32 64 96 128 256"}
GRID=${GRID:-data/india_2025.grid}; DAY=${DAY:-99}
REF=${GRID%.grid}_ref.grid
mkdir -p jobs results/cluster results/fields
SUBMIT=${1:-}

S=bin/poisson_serial; M=bin/poisson_mpi; MO=bin/poisson_mpi_omp; OUT=results/cluster
[ -f "$REF" ] && REFOPT="--ref $REF" || REFOPT=""
MAXNP=0; for n in $RANKS; do [ $n -gt $MAXNP ] && MAXNP=$n; done

header() { # name queue nodes ppn threads walltime
  local name=$1 queue=$2 nodes=$3 ppn=$4 th=$5 wall=$6 memsel=""
  [ -n "${MEM:-}" ] && memsel=":mem=$MEM"
  echo "#!/bin/bash"
  echo "#PBS -N $name"
  if [ -n "${PROJECT:-}" ]; then echo "#PBS -P $PROJECT"; else echo "#PBS -q $queue"; fi
  cat <<H
#PBS -l select=$nodes:ncpus=$((ppn*th)):mpiprocs=$ppn:ompthreads=$th$memsel
#PBS -l walltime=$wall
#PBS -j oe
#PBS -k oed
cd \$PBS_O_WORKDIR
$MODULES
export OMP_NUM_THREADS=$th
export OMP_PROC_BIND=close OMP_PLACES=cores
mkdir -p $OUT results/fields
echo "=== \$PBS_JOBNAME started \$(date) on \$(hostname)"
H
}
footer() { echo 'echo "=== $PBS_JOBNAME finished $(date)"'; }

serial_runs() {
  cat <<B
# --- serial size scaling and the serial baselines (1 core) ---
for r in 1 2; do
for n in 256 512 1024 2048 4096 8192;   do $S --dim 2 --n \$n --iters 200 --csv $OUT/serial_scaling.csv --tag jac2d; done
for n in 32 64 128 256 512;             do $S --dim 3 --n \$n --iters 200 --csv $OUT/serial_scaling.csv --tag jac3d; done
for n in 256 512 1024 2048 4096;        do $S --dim 2 --n \$n --solver cg --tol 1e-8 --iters 1000000 --csv $OUT/serial_scaling.csv --tag cg2d; done
for n in 32 64 128 256;                 do $S --dim 3 --n \$n --solver cg --tol 1e-8 --iters 1000000 --csv $OUT/serial_scaling.csv --tag cg3d; done
$S --dim 2 --n 8192 --iters 200 --csv $OUT/strong.csv --tag jac2d
$S --dim 3 --n 512  --iters 200 --csv $OUT/strong.csv --tag jac3d
$S --dim 2 --n 4096 --solver cg --tol 1e-8 --iters 1000000 --csv $OUT/strong.csv --tag cg2d
$S --dim 3 --n 256  --solver cg --tol 1e-8 --iters 1000000 --csv $OUT/strong.csv --tag cg3d
done
B
  if [ -f "$GRID" ]; then cat <<B
# --- serial baselines on the satellite data ---
for r in 1 2; do
$S --data $GRID --dim 2 --day $DAY --holdout 0.1 --holdout-block 16 --solver cg --tol 1e-6 --iters 100000 --csv $OUT/gapfill.csv --tag sst2d_cg --out results/fields/fill2d_day$DAY.grid
$S --data $GRID --dim 2 --day $DAY --holdout 0.1 --holdout-block 16 --solver jacobi --iters 1000 --csv $OUT/gapfill.csv --tag sst2d_jac
$S --data $GRID --dim 3 --holdout 0.1 --holdout-block 16 --tweight 1 --solver cg --tol 1e-6 --iters 100000 --csv $OUT/gapfill.csv --tag sst3d_cg --out results/fields/fill3d.grid
$S --data $GRID --dim 3 --holdout 0.1 --holdout-block 16 --tweight 1 --solver jacobi --iters 200 --csv $OUT/gapfill.csv --tag sst3d_jac
done
B
  fi
}

scaling_runs() { # NP
  local NP=$1
  cat <<B
# --- $NP ranks: strong scaling (fixed problem) and weak scaling (fixed work per rank) ---
for r in 1 2; do
$MPIRUN -np $NP $M --dim 2 --n 8192 --iters 200 --csv $OUT/strong.csv --tag jac2d
$MPIRUN -np $NP $M --dim 3 --n 512  --iters 200 --csv $OUT/strong.csv --tag jac3d
$MPIRUN -np $NP $M --dim 2 --n 4096 --solver cg --tol 1e-8 --iters 1000000 --csv $OUT/strong.csv --tag cg2d
$MPIRUN -np $NP $M --dim 3 --n 256  --solver cg --tol 1e-8 --iters 1000000 --csv $OUT/strong.csv --tag cg3d
$MPIRUN -np $NP $M --dim 2 --local-n 1024 --iters 200 --csv $OUT/weak.csv --tag jac2d
$MPIRUN -np $NP $M --dim 3 --local-n 64   --iters 200 --csv $OUT/weak.csv --tag jac3d
$MPIRUN -np $NP $M --dim 2 --local-n 512  --solver cg --iters 300 --csv $OUT/weak.csv --tag cg2d
$MPIRUN -np $NP $M --dim 3 --local-n 48   --solver cg --iters 300 --csv $OUT/weak.csv --tag cg3d
done
B
  if [ -f "$GRID" ]; then cat <<B
# --- $NP ranks: satellite data ---
for r in 1 2; do
$MPIRUN -np $NP $M --data $GRID --dim 2 --day $DAY --holdout 0.1 --holdout-block 16 --solver cg --tol 1e-6 --iters 100000 --csv $OUT/gapfill.csv --tag sst2d_cg
$MPIRUN -np $NP $M --data $GRID --dim 2 --day $DAY --holdout 0.1 --holdout-block 16 --solver jacobi --iters 1000 --csv $OUT/gapfill.csv --tag sst2d_jac
$MPIRUN -np $NP $M --data $GRID --dim 3 --holdout 0.1 --holdout-block 16 --tweight 1 --solver cg --tol 1e-6 --iters 100000 --csv $OUT/gapfill.csv --tag sst3d_cg
$MPIRUN -np $NP $M --data $GRID --dim 3 --holdout 0.1 --holdout-block 16 --tweight 1 --solver jacobi --iters 200 --csv $OUT/gapfill.csv --tag sst3d_jac
done
B
  fi
}

design_runs() { # NP
  local NP=$1
  cat <<B
# --- $NP ranks: exchange strategy and residual-check interval ---
for r in 1 2; do
for mode in sendrecv isend overlap rma; do
  $MPIRUN -np $NP $M --dim 2 --n 8192 --iters 200 --exchange \$mode --csv $OUT/exchange.csv --tag jac2d
  $MPIRUN -np $NP $M --dim 3 --n 512  --iters 200 --exchange \$mode --csv $OUT/exchange.csv --tag jac3d
  $MPIRUN -np $NP $M --dim 2 --n 4096 --solver cg --iters 300 --exchange \$mode --csv $OUT/exchange.csv --tag cg2d
done
for c in 0 1 10 100; do
  $MPIRUN -np $NP $M --dim 2 --n 8192 --iters 200 --check-every \$c --csv $OUT/checkevery.csv --tag jac2d
  $MPIRUN -np $NP $M --dim 3 --n 512  --iters 200 --check-every \$c --csv $OUT/checkevery.csv --tag jac3d
done
done
B
}
decomp_runs() { # NP: process-grid shapes (only for rank counts with the needed factorisations)
  local NP=$1 shapes2 shapes3
  case $NP in
    256) shapes2="256,1 64,4 16,16 4,64 1,256"; shapes3="256,1,1 1,1,256 16,16,1 8,8,4 4,8,8 1,16,16" ;;
    192) shapes2="192,1 48,4 16,12 4,48 1,192"; shapes3="192,1,1 1,1,192 16,12,1 8,6,4 4,6,8 1,12,16" ;;
    96)  shapes2="96,1 24,4 12,8 4,24 1,96";    shapes3="96,1,1 1,1,96 12,8,1 6,4,4 4,4,6 1,8,12" ;;
    64)  shapes2="64,1 16,4 8,8 4,16 1,64";     shapes3="64,1,1 1,1,64 8,8,1 4,4,4 1,8,8" ;;
    *)   return ;;
  esac
  cat <<B
# --- $NP ranks: shape of the process grid ---
for r in 1 2; do
for d in $shapes2; do $MPIRUN -np $NP $M --dim 2 --n 8192 --iters 200 --dims \$d --csv $OUT/decomp.csv --tag jac2d; done
for d in $shapes3; do $MPIRUN -np $NP $M --dim 3 --n 512  --iters 200 --dims \$d --csv $OUT/decomp.csv --tag jac3d; done
done
B
}
hybrid_runs() { # NODES PPN: ranks x threads = NODES*PPN, split from PPN ranks per node down to 1
  local NODES=$1 ppn=$2 th=1
  echo "# --- MPI+OpenMP on $NODES node(s): ranks x threads = $((NODES*ppn)) ---"
  while [ $ppn -ge 1 ]; do
    local NP=$((ppn*NODES))
    cat <<B
export OMP_NUM_THREADS=$th
for r in 1 2; do
$MPIRUN -np $NP --map-by ppr:$ppn:node:pe=$th $MO --threads $th --dim 2 --n 8192 --iters 200 --csv $OUT/hybrid.csv --tag jac2d
$MPIRUN -np $NP --map-by ppr:$ppn:node:pe=$th $MO --threads $th --dim 3 --n 512  --iters 200 --csv $OUT/hybrid.csv --tag jac3d
$MPIRUN -np $NP --map-by ppr:$ppn:node:pe=$th $MO --threads $th --dim 2 --n 4096 --solver cg --tol 1e-8 --iters 1000000 --csv $OUT/hybrid.csv --tag cg2d
$MPIRUN -np $NP --map-by ppr:$ppn:node:pe=$th $MO --threads $th --dim 3 --n 256  --solver cg --tol 1e-8 --iters 1000000 --csv $OUT/hybrid.csv --tag cg3d
done
B
    [ -f "$GRID" ] && echo "$MPIRUN -np $NP --map-by ppr:$ppn:node:pe=$th $MO --threads $th --data $GRID --dim 3 --holdout 0.1 --holdout-block 16 --tweight 1 --solver cg --tol 1e-6 --iters 100000 --csv $OUT/gapfill.csv --tag sst3d_cg"
    ppn=$((ppn/2)); th=$((th*2))
  done
  echo "export OMP_NUM_THREADS=1"
}
accuracy_runs() { # NP
  local NP=$1
  [ -f "$GRID" ] || return
  cat <<B
# --- $NP ranks: accuracy of the fill (hidden patch size, time weight) ---
for b in 1 4 16 64; do
  $MPIRUN -np $NP $M --data $GRID --dim 2 --day $DAY --holdout 0.1 --holdout-block \$b --solver cg --tol 1e-6 --iters 100000 $REFOPT --csv $OUT/gapfill.csv --tag acc2d
  $MPIRUN -np $NP $M --data $GRID --dim 3 --holdout 0.1 --holdout-block \$b --tweight 1 --solver cg --tol 1e-6 --iters 100000 $REFOPT --csv $OUT/gapfill.csv --tag acc3d
  $MPIRUN -np $NP $M --data $GRID --dim 3 --holdout 0.1 --holdout-block \$b --tweight 1 --solver cg --tol 1e-6 --iters 100000 $REFOPT --eval-day $DAY --csv $OUT/gapfill.csv --tag acc3d_day
done
for w in 0.01 0.03 0.1 0.3 3; do
  $MPIRUN -np $NP $M --data $GRID --dim 3 --holdout 0.1 --holdout-block 16 --tweight \$w --solver cg --tol 1e-6 --iters 100000 $REFOPT --csv $OUT/gapfill.csv --tag acc3d
  $MPIRUN -np $NP $M --data $GRID --dim 3 --holdout 0.1 --holdout-block 16 --tweight \$w --solver cg --tol 1e-6 --iters 100000 $REFOPT --eval-day $DAY --csv $OUT/gapfill.csv --tag acc3d_day
done
B
}

# ---------------------------------------------------------------- one job for everything that fits on one node
ONE=""; MULTI=""
for n in $RANKS; do if [ $n -le $PPN ]; then ONE="$ONE $n"; else MULTI="$MULTI $n"; fi; done
BIGONE=0; for n in $ONE; do BIGONE=$n; done           # largest single-node rank count
{
  header node_1 "$QUEUE" 1 "$PPN" 1 "${WALL_NODE:-08:00:00}"
  serial_runs
  for n in $ONE; do scaling_runs $n; done
  [ $BIGONE -ge 64 ] && design_runs 64
  [ $BIGONE -gt 64 ] && design_runs $BIGONE
  decomp_runs $BIGONE
  hybrid_runs 1 "$PPN"
  [ -z "$MULTI" ] && accuracy_runs $BIGONE
  footer
} > jobs/node_1.pbs

# ---------------------------------------------------------------- one job per multi-node rank count
for NP in $MULTI; do
  nodes=$(( (NP + PPN - 1) / PPN )); ppn=$(( NP / nodes ))
  {
    header mpi_$NP "$QUEUE_MULTI" $nodes $ppn 1 "${WALL:-04:00:00}"
    scaling_runs $NP
    if [ $NP -eq $MAXNP ]; then design_runs $NP; decomp_runs $NP; accuracy_runs $NP; fi
    footer
  } > jobs/mpi_$NP.pbs
  if [ $NP -eq $MAXNP ]; then
    { header hybrid_$NP "$QUEUE_MULTI" $nodes $ppn 1 "${WALL:-04:00:00}"; hybrid_runs $nodes $ppn; footer; } > jobs/hybrid_$NP.pbs
  fi
done

echo "generated in jobs/:"; ls jobs/
echo "node job: ranks$ONE on one $PPN-core node (queue $QUEUE); multi-node jobs: ranks$MULTI (queue $QUEUE_MULTI)"
[ -f "$GRID" ] || echo "note: $GRID not found, the satellite runs were left out"
# QSUB_OPTS is passed to every qsub, e.g. QSUB_OPTS="-W depend=afterok:<build job id>"
if [ "$SUBMIT" = "--submit" ]; then for j in jobs/*.pbs; do qsub ${QSUB_OPTS:-} "$j"; done; fi
