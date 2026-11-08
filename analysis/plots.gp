# plots.gp -- draw every figure from the .dat files written by bin/summarize.
# Usage:  gnuplot -e "dir='results/figures'" analysis/plots.gp
# Axes for time, problem size and core count are logarithmic, as the lecturer asked.
if (!exists("dir")) dir = "results/figures"
exists_file(f) = system("[ -f '".f."' ] && echo 1 || echo 0") + 0
set terminal pngcairo size 1500,430 enhanced font "Helvetica,11"
set grid lc rgb "#dddddd"
set key top left box opaque
set style line 1 lc rgb "#1f77b4" pt 7 ps 1.1 lw 1.6
set style line 2 lc rgb "#ff7f0e" pt 5 ps 1.1 lw 1.6
set style line 3 lc rgb "#2ca02c" pt 9 ps 1.1 lw 1.6
set style line 4 lc rgb "#d62728" pt 13 ps 1.1 lw 1.6
set style line 9 lc rgb "black" dt 2 lw 1.2
tags  = "jac2d jac3d cg2d cg3d"
names = "Jacobi-2D Jacobi-3D CG-2D CG-3D"
name(i) = word(names, i)
D(prefix, i) = dir."/".prefix."_".word(tags, i).".dat"

# ---------------------------------------------------------------- serial scaling
set output dir."/serial_scaling.png"
set multiplot layout 1,2
set logscale xy; set xlabel "grid points N (total)"; set ylabel "seconds"
set title "Serial: total time vs problem size"
plot for [i=1:4] D("serial",i) u 1:2 w lp ls i t name(i)
set title "Serial: time per iteration vs problem size"
plot for [i=1:4] D("serial",i) u 1:3 w lp ls i t name(i)
unset multiplot

# ---------------------------------------------------------------- strong scaling
set output dir."/strong_scaling.png"
set multiplot layout 1,3
set logscale xy; set xlabel "cores p"; set ylabel "time (s)"; set title "Strong scaling: time"
plot for [i=1:4] D("strong",i) u 1:2 w lp ls i t name(i)
set ylabel "speedup T1/Tp"; set title "Strong scaling: speedup"
plot for [i=1:4] D("strong",i) u 1:3 w lp ls i t name(i), x ls 9 t "ideal"
unset logscale y; set yrange [0:1.3]; set ylabel "efficiency T1/(p Tp)"; set title "Strong scaling: efficiency"
plot for [i=1:4] D("strong",i) u 1:4 w lp ls i t name(i), 1 ls 9 notitle
unset multiplot
set autoscale y; unset logscale

set terminal pngcairo size 800,430 enhanced font "Helvetica,11"
set output dir."/strong_comm_fraction.png"
set logscale x; set xlabel "cores p"; set ylabel "communication share of runtime (%)"
set title "Share of time in halo exchange + all-reduce (max over ranks)"
plot for [i=1:4] D("strong",i) u 1:($5*100) w lp ls i t name(i)
unset logscale

# ---------------------------------------------------------------- weak scaling
set terminal pngcairo size 1500,430 enhanced font "Helvetica,11"
set output dir."/weak_scaling.png"
set multiplot layout 1,2
set logscale x; set xlabel "cores p"; set ylabel "time (s)"; set title "Weak scaling: time (constant work per rank)"
plot for [i=1:4] D("weak",i) u 1:2 w lp ls i t name(i)
set yrange [0:1.3]; set ylabel "weak efficiency T1/Tp"; set title "Weak scaling: efficiency"
plot for [i=1:4] D("weak",i) u 1:3 w lp ls i t name(i), 1 ls 9 notitle
unset multiplot
set autoscale y; unset logscale

# ---------------------------------------------------------------- exchange modes (grouped bars, log y)
set output dir."/exchange_modes.png"
set multiplot layout 1,2
set style data histogram; set style histogram clustered gap 1; set style fill solid 0.85 border -1
set boxwidth 0.9; set logscale y; set ylabel "seconds (log scale)"; set xlabel ""
set xtics rotate by -20 font ",9"
set title "Halo exchange strategy: total time"
plot dir."/exchange_sendrecv.dat" u 2:xtic(4) t "sendrecv" lc rgb "#1f77b4", \
     dir."/exchange_isend.dat"    u 2 t "isend"    lc rgb "#ff7f0e", \
     dir."/exchange_overlap.dat"  u 2 t "overlap"  lc rgb "#2ca02c", \
     dir."/exchange_rma.dat"      u 2 t "rma"      lc rgb "#d62728"
set title "Halo exchange strategy: halo time (max over ranks)"
plot dir."/exchange_sendrecv.dat" u 3:xtic(4) t "sendrecv" lc rgb "#1f77b4", \
     dir."/exchange_isend.dat"    u 3 t "isend"    lc rgb "#ff7f0e", \
     dir."/exchange_overlap.dat"  u 3 t "overlap"  lc rgb "#2ca02c", \
     dir."/exchange_rma.dat"      u 3 t "rma"      lc rgb "#d62728"
unset multiplot
unset logscale; set xtics rotate by 0 font ",11"; set style data lines

# ---------------------------------------------------------------- process-grid shape
set output dir."/decomposition_shape.png"
set multiplot layout 1,2
set style fill solid 0.8 border -1; set boxwidth 0.7; set ylabel "seconds"; set yrange [0:*]; set xtics rotate by -25 font ",9"
set title "Process-grid shape, Jacobi 2D (sorted by halo cells per rank)"
plot D("decomp",1) u 1:3:xtic(2) w boxes lc rgb "#1f77b4" t "total", D("decomp",1) u 1:4 w boxes lc rgb "#444444" t "halo (max)"
set title "Process-grid shape, Jacobi 3D (sorted by halo cells per rank)"
plot D("decomp",2) u 1:3:xtic(2) w boxes lc rgb "#ff7f0e" t "total", D("decomp",2) u 1:4 w boxes lc rgb "#444444" t "halo (max)"
unset multiplot
set autoscale y; set xtics rotate by 0 font ",11"

# ---------------------------------------------------------------- residual-check interval
set terminal pngcairo size 800,430 enhanced font "Helvetica,11"
set output dir."/check_interval.png"
set logscale x; set xtics ("1" 1, "10" 10, "100" 100, "never" 1000); set xlabel "sweeps between residual checks"; set ylabel "seconds"
set title "Cost of the global residual check (Jacobi, 200 sweeps)"
plot D("checkevery",1) u 1:2 w lp ls 1 t "Jacobi 2D total", D("checkevery",1) u 1:3 w lp ls 1 dt 3 t "Jacobi 2D all-reduce", \
     D("checkevery",2) u 1:2 w lp ls 2 t "Jacobi 3D total", D("checkevery",2) u 1:3 w lp ls 2 dt 3 t "Jacobi 3D all-reduce"
unset logscale; set xtics autofreq

# ---------------------------------------------------------------- MPI+OpenMP split
set terminal pngcairo size 1500,430 enhanced font "Helvetica,11"
set output dir."/hybrid_split.png"
set multiplot layout 1,2
set xlabel "MPI ranks x OpenMP threads"; set ylabel "time (s)"; set title "MPI+OpenMP: same core count, different split (largest core count)"
plot for [i=1:4] D("hybrid",i) u 1:3:xtic(2) w lp ls i t name(i)
set ylabel "communication share (%)"; set title "Communication share vs split"
plot for [i=1:4] D("hybrid",i) u 1:4:xtic(2) w lp ls i t name(i)
unset multiplot
# the same study on one 64-core node, when the results also hold it (hybrid_<tag>_c64.dat)
if (exists_file(dir."/hybrid_jac2d_c64.dat")) {
  set output dir."/hybrid_split_64.png"
  set multiplot layout 1,2
  set ylabel "time (s)"; set title "MPI+OpenMP on one node: 64 cores, different split"
  plot for [i=1:4] dir."/hybrid_".word(tags,i)."_c64.dat" u 1:3:xtic(2) w lp ls i t name(i)
  set ylabel "communication share (%)"; set title "Communication share vs split (one node)"
  plot for [i=1:4] dir."/hybrid_".word(tags,i)."_c64.dat" u 1:4:xtic(2) w lp ls i t name(i)
  unset multiplot
}

# ---------------------------------------------------------------- OpenMP threads vs MPI
if (exists_file(dir."/omp_jac2d.dat")) {   # OpenMP-only runs exist only in the laptop results
set output dir."/openmp_threads.png"
set multiplot layout 1,2
set logscale xy; set xlabel "threads / cores"; set ylabel "speedup"; set title "Shared memory (OpenMP) vs MPI on one machine"
set key top left font ",8"
plot for [i=1:4] D("omp",i) u 1:3 w lp ls i t name(i)." OpenMP", \
     for [i=1:4] D("strong",i) u 1:3 w lp ls i dt 2 lw 1 t name(i)." MPI", x ls 9 t "ideal"
unset logscale y; set yrange [0:1.3]; set ylabel "efficiency"; set title "OpenMP efficiency"; set key top right font ",10"
plot for [i=1:4] D("omp",i) u 1:4 w lp ls i t name(i), 1 ls 9 notitle
unset multiplot
set autoscale y; unset logscale
}

# ---------------------------------------------------------------- GPU vs CPU node (only if the cluster files exist)
if (exists_file(dir."/gpu_jac2d_mpi_cuda.dat")) {
  set terminal pngcairo size 800,430 enhanced font "Helvetica,11"
  set output dir."/gpu_vs_cpu.png"
  set logscale xy; set xlabel "grid points N"; set ylabel "seconds per iteration"; set title "One GPU vs one CPU node (OpenMP)"
  plot for [i=1:4] dir."/gpu_".word(tags,i)."_mpi_cuda.dat" u 1:2 w lp ls i t name(i)." GPU", \
       for [i=1:4] dir."/gpu_".word(tags,i)."_mpi_omp.dat"  u 1:2 w lp ls i dt 2 t name(i)." CPU node"
}

# ---------------------------------------------------------------- satellite gap filling (only if the data runs exist)
gtags  = "sst2d_cg sst2d_jac sst3d_cg sst3d_jac"
gnames = "SST-2D-CG SST-2D-Jacobi SST-3D-CG SST-3D-Jacobi"
G(i) = dir."/gapfill_".word(gtags, i).".dat"
if (exists_file(G(1)) || exists_file(G(3))) {
  set terminal pngcairo size 1500,430 enhanced font "Helvetica,11"
  set output dir."/gapfill_scaling.png"
  set multiplot layout 1,3
  set logscale xy; set xlabel "cores p"; set ylabel "time (s)"; set title "Satellite gap filling: time"
  plot for [i=1:4] G(i) u 1:2 w lp ls i t word(gnames, i)
  set ylabel "speedup T1/Tp"; set title "Satellite gap filling: speedup"
  plot for [i=1:4] G(i) u 1:3 w lp ls i t word(gnames, i), x ls 9 t "ideal"
  unset logscale y; set yrange [0:1.3]; set ylabel "efficiency T1/(p Tp)"; set title "Satellite gap filling: efficiency"
  plot for [i=1:4] G(i) u 1:4 w lp ls i t word(gnames, i), 1 ls 9 notitle
  unset multiplot
  set autoscale y; unset logscale
}
if (exists_file(dir."/gapfill_acc2d.dat")) {
  set terminal pngcairo size 1500,430 enhanced font "Helvetica,11"
  set output dir."/gapfill_accuracy.png"
  set multiplot layout 1,2
  set logscale x; set xlabel "side of the hidden patch (pixels of 4 km)"; set ylabel "RMSE on hidden pixels (deg C)"
  set title "Fill error against the size of the hole"
  set key top left
  plot dir."/gapfill_acc2d.dat" u 1:2 w lp ls 1 t "2D: the day alone, space only", \
       dir."/gapfill_acc3d_day.dat" u 1:2 w lp ls 3 t "3D: the year, error on the same day", \
       dir."/gapfill_acc3d.dat" u 1:2 w lp ls 2 t "3D: the year, error on all days", \
       dir."/gapfill_acc2d.dat" u 1:5 w lp ls 9 t "baseline: mean temperature (the day)"
  unset logscale x; set logscale x
  set xlabel "time weight (--tweight)"; set ylabel "RMSE on hidden pixels (deg C)"; set title "3D: weight of the time neighbours (16-pixel patches)"
  plot dir."/gapfill_tweight_day.dat" u 1:2 w lp ls 3 t "3D, error on the 2D day", dir."/gapfill_tweight.dat" u 1:2 w lp ls 2 t "3D, error on all days", \
       dir."/gapfill_tweight_day.dat" u 1:6 w l ls 1 dt 2 t "2D fill of the day (16-pixel patches)"
  unset multiplot
  unset logscale
}

# ---------------------------------------------------------------- node layout (only if a second results folder was given)
if (exists_file(dir."/layout_compare.dat")) {
  set terminal pngcairo size 1100,460 enhanced font "Helvetica,11"
  set output dir."/layout_compare.png"
  set style data histogram; set style histogram clustered gap 1; set style fill solid 0.85 border -1
  set boxwidth 0.9; set logscale y; set ylabel "seconds (log scale)"; set xlabel ""
  set xtics rotate by -25 font ",9"; set key top left
  set title "256 cores: 4 nodes x 64 cores against 8 nodes x 32 cores"
  plot dir."/layout_compare.dat" u 3:xtic(2) t "4 x 64 (whole nodes)" lc rgb "#1f77b4", \
       ''                         u 4 t "8 x 32 (half nodes, shared)" lc rgb "#ff7f0e"
  unset logscale; set xtics rotate by 0 font ",11"; set style data lines
}
