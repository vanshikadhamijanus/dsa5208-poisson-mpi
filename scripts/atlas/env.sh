# Site configuration for the NUS HPC clusters. Edit these before submitting.
# Check queue names and limits with:  gstat   and   qstat -Qf <queue>
# Node sizes:  pbsnodes -a | grep resources_available.ncpus | sort | uniq -c
# Compile on Atlas after:   source /app1/ebenv; module load GCC/13.2.0 OpenMPI/4.1.6-GCC-13.2.0
# Compile on Vanda inside:  qsub -I -l select=1:ncpus=1 -l walltime=00:10:00
#
# Atlas (checked 3 Nov 2026): queue "parallel" allows at most 96 cores per job and
# per user, 4 running jobs; the nodes have 96 cores. Queue "hpc6a" allows one job
# of up to 192 cores on 2 nodes but refused our account ("Unauthorized Request"). 256 cores therefore need Vanda (72-core nodes:
# PPN=64 and RANKS="... 256" give 4 nodes).

QUEUE=${QUEUE:-parallel}            # queue for the single-node job (all rank counts up to PPN)
QUEUE_MULTI=${QUEUE_MULTI:-$QUEUE}   # queue for the multi-node jobs (rank counts above PPN); Atlas hpc6a is not open to student accounts
PPN=${PPN:-96}                      # MPI ranks per node = cores per node (Atlas 96, Vanda 72 -> use 64)
RANKS=${RANKS:-"1 2 4 8 16 32 64 96"}   # Atlas: one 96-core node is the limit; on Vanda: "1 2 4 8 16 32 64 128 256"
WALL=${WALL:-04:00:00}              # wall time of each multi-node job
WALL_NODE=${WALL_NODE:-08:00:00}    # wall time of the single-node job (it runs everything up to PPN ranks)
PROJECT_DIR=${PROJECT_DIR:-$HOME/poisson-mpi}
MODULES=${MODULES:-"source /app1/ebenv; module load GCC/13.2.0 OpenMPI/4.1.6-GCC-13.2.0"}   # Atlas, verified; on Vanda use the names from 'module avail'
GPU_MODULES=${GPU_MODULES:-"module load cuda; module load openmpi"}
MPIRUN=${MPIRUN:-"mpirun"}           # some sites want 'mpirun -hostfile $PBS_NODEFILE'
MEM=${MEM:-}                        # per-node memory request, e.g. MEM=200gb (Vanda wants one; leave empty on Atlas)
PROJECT=${PROJECT:-}                # Vanda project name (hpc project); when set, "#PBS -P" replaces "#PBS -q"
#
# Vanda (checked 3 Nov 2026): free tier is "-q auto_free" (no project), nodes have 72 cores
# (22 have 64), GPU nodes have 2 GPUs, work directory must be /scratch/$USER, and compiling
# must be done inside an interactive job on a compute node. Settings to use there
# (the hostfile makes mpirun span the nodes even if Open MPI lacks PBS support):
#   QUEUE=auto_free QUEUE_MULTI=auto_free PPN=64 RANKS="1 2 4 8 16 32 64 128 256" MEM=200gb \
#   MODULES="module load gcc/13.1.0 openmpi4/gcc/4.1.8" MPIRUN='mpirun -hostfile $PBS_NODEFILE' scripts/atlas/gen_jobs.sh
