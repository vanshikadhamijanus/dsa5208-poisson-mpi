# Makefile for the Poisson MPI+X project.
#
# Targets:
#   make serial     -> bin/poisson_serial   (no parallelism; the baseline)
#   make omp        -> bin/poisson_omp      (OpenMP threads only)
#   make mpi        -> bin/poisson_mpi      (MPI only)
#   make mpi_omp    -> bin/poisson_mpi_omp  (MPI + OpenMP)
#   make cuda       -> bin/poisson_mpi_cuda (MPI + CUDA; needs nvcc)
#   make gridtool   -> bin/gridtool        (satellite data: ASCII grids -> grid file, info, slices)
#   make all        -> the four CPU binaries and gridtool
#   make check      -> pass/fail tests: every build against the serial answer (scripts/local/check.sh)
#   make figures    -> bin/summarize + gnuplot: results/<RESULTS>/*.csv -> results/figures/*.png, tables.md
#                      (make figures RESULTS=results/cluster FIGDIR=results/figures_cluster LABEL="NUS Vanda")
#                      ALT=results/cluster_8x32 ALTLABEL="8 nodes x 32 cores" adds the node-layout comparison
#
# Override the compilers on the command line if needed, e.g.
#   make all CXX=g++ MPICXX=mpicxx OMPFLAGS=-fopenmp
# On macOS with Homebrew libomp the defaults below work with Apple clang.

CXX      ?= g++
MPICXX   ?= mpicxx
NVCC     ?= nvcc
# Instruction set. On x86-64 Linux the default is AVX2 (x86-64-v3) rather than
# -march=native: cluster login nodes often have a different CPU from the compute
# nodes (Vanda says so explicitly), and a binary built for the login node can stop
# with "Illegal instruction" on a compute node. On Apple silicon: -mcpu=native.
UNAME   := $(shell uname)
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),x86_64)
  ARCHFLAG ?= -march=x86-64-v3
else
  ARCHFLAG ?= -mcpu=native
endif
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra $(ARCHFLAG)

# GPU code: one binary for V100, A100, A40/A10, L40/4090 and H100 unless CUDA_ARCH
# names one architecture (e.g. make cuda CUDA_ARCH=sm_86). nvcc uses mpicxx as its
# host compiler (-ccbin), so the MPI include and link flags come from the wrapper.
CUDA_ARCH ?=
ifeq ($(strip $(CUDA_ARCH)),)
  CUDA_GENCODE ?= -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80 \
                  -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 \
                  -gencode arch=compute_90,code=sm_90
else
  CUDA_GENCODE ?= -arch=$(CUDA_ARCH)
endif

ifeq ($(UNAME),Darwin)
  OMPFLAGS ?= -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include
  OMPLIBS  ?= -L/opt/homebrew/opt/libomp/lib -lomp
else
  OMPFLAGS ?= -fopenmp
  OMPLIBS  ?= -fopenmp
endif

HDRS := src/common.hpp src/kernels.hpp src/solver.hpp src/gridio.hpp
BIN  := bin

.PHONY: all serial omp mpi mpi_omp cuda cuda_emul summarize gridtool figures check clean

RESULTS ?= results/local
FIGDIR  ?= results/figures
LABEL   ?= local machine
ALT     ?=
ALTLABEL ?= second layout

all: serial omp mpi mpi_omp gridtool

$(BIN):
	mkdir -p $(BIN)

serial: $(BIN)/poisson_serial
omp:    $(BIN)/poisson_omp
mpi:    $(BIN)/poisson_mpi
mpi_omp:$(BIN)/poisson_mpi_omp
cuda:   $(BIN)/poisson_mpi_cuda
cuda_emul: $(BIN)/poisson_mpi_cuda_emul
summarize: $(BIN)/summarize
gridtool:  $(BIN)/gridtool

$(BIN)/summarize: analysis/summarize.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BIN)/gridtool: tools/gridtool.cpp src/gridio.hpp src/common.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -Isrc $< -o $@

check: all cuda_emul
	scripts/local/check.sh

figures: $(BIN)/summarize
	mkdir -p $(FIGDIR)
	$(BIN)/summarize $(RESULTS) $(FIGDIR) "$(LABEL)" $(if $(ALT),"$(ALT)" "$(ALTLABEL)")
	gnuplot -e "dir='$(FIGDIR)'" analysis/plots.gp
	@echo "figures written to $(FIGDIR)"

$(BIN)/poisson_serial: src/serial_main.cpp $(HDRS) | $(BIN)
	$(CXX) $(CXXFLAGS) -Isrc $< -o $@

$(BIN)/poisson_omp: src/serial_main.cpp $(HDRS) | $(BIN)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -Isrc $< -o $@ $(OMPLIBS)

$(BIN)/poisson_mpi: src/mpi_main.cpp src/decomp.hpp $(HDRS) | $(BIN)
	$(MPICXX) $(CXXFLAGS) -Isrc $< -o $@

$(BIN)/poisson_mpi_omp: src/mpi_main.cpp src/decomp.hpp $(HDRS) | $(BIN)
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) -Isrc $< -o $@ $(OMPLIBS)

$(BIN)/poisson_mpi_cuda: src/cuda/mpi_cuda_main.cu src/cuda/gpu_kernels.cuh src/cuda/cuda_compat.hpp src/decomp.hpp $(HDRS) | $(BIN)
	$(NVCC) -O3 -std=c++17 $(CUDA_GENCODE) -ccbin $(MPICXX) -Isrc $< -o $@

# The CUDA source compiled as plain C++ through src/cuda/cuda_compat.hpp: no GPU,
# kernels run sequentially on the CPU. For logic testing only.
$(BIN)/poisson_mpi_cuda_emul: src/cuda/mpi_cuda_main.cu src/cuda/gpu_kernels.cuh src/cuda/cuda_compat.hpp src/decomp.hpp $(HDRS) | $(BIN)
	$(MPICXX) $(CXXFLAGS) -x c++ -Isrc $< -o $@

clean:
	rm -rf $(BIN)
