# Run from the repository root: gnuplot example/plots.gnuplot
set encoding utf8
colors='#d97706 #2e8b57 #3465a4'
markers='9 5 7'
shapeIndex(s) = s eq '512x512x512' ? 0 : s eq '1024x1024x1024' ? 1 : s eq '2048x2048x2048' ? 2 : s eq '4096x4096x4096' ? 3 : s eq '256x2048x1024' ? 4 : s eq '2048x256x1024' ? 5 : 6

# Runtime above, speedup against the original baseline below.
do for [case=1:4] {
 reset
 set encoding utf8
 set terminal svg size 1600,1000 enhanced font 'DejaVu Sans,16'
 heat=(case>2)
 gpu=(case==1 || case==3)
 data=case==1 ? 'example/matmul/results/runtimes.csv' : case==2 ? 'example/matmul/results/cpu-runtimes.csv' : case==3 ? 'example/heatEquation/figures/a30/runtime.csv' : 'example/heatEquation/figures/genoa/runtime.csv'
 outputFile=case==1 ? 'example/matmul/results/runtime-comparison.svg' : case==2 ? 'example/matmul/results/cpu-runtime-comparison.svg' : case==3 ? 'example/heatEquation/figures/a30/runtime.svg' : 'example/heatEquation/figures/genoa/runtime.svg'
 title=case==1 ? 'Matmul — NVIDIA A30, FP32' : case==2 ? 'Matmul — 2 × AMD EPYC 7713, FP32, 128 physical cores' : case==3 ? 'Heat equation — NVIDIA A30, FP64' : 'Heat equation — 2 × AMD EPYC 9654, FP64, 192 physical cores'
 dcol=heat ? 20 : gpu ? 2 : 3
 tcol=heat ? 21 : gpu ? 5 : 6
 scale=heat ? 1000.0 : 1.0
 paths=(!heat && gpu) ? 3 : 2
 labels='Default Tuned Strict-FP32-cuBLAS'
 footer=heat ? 'Complete timestep includes boundary kernel. 31 paired samples × 64 steps; speedup bars: paired-bootstrap 95% intervals.' : gpu ? 'Five run medians × 31 samples/path; runtime bars span run medians. Speedup: ratio of plotted medians.' : 'Five run medians × 7 samples/path; runtime bars span run medians. Both paths use a fixed 128-thread OpenMP team.'
 set output outputFile
 set datafile separator ','
 set border 3
 set tics out nomirror
 set grid ytics lc rgb '#dddddd'
 set key top left opaque
 set xrange [-0.5:(heat ? 7.5 : 6.5)]
 set multiplot title title font ',22'
 set lmargin at screen 0.10
 set rmargin at screen 0.97
 set bmargin at screen 0.43
 set tmargin at screen 0.90
 set logscale y
 set ylabel (heat ? 'Runtime (ms / timestep)' : 'Runtime (ms)')
 set format x ''
 set label 1 footer at screen 0.10,0.03 font ',12'
 if (heat) {
  plot for [impl=1:2] data every ::1 using (log(column(5)/128)/log(2)):(column(impl==1 ? dcol : tcol)*scale) with linespoints pt int(word(markers,impl)) ps 1.4 lw 2 lc rgb word(colors,impl) title word(labels,impl)
 } else {
  plot for [impl=1:paths] data every ::1 using (shapeIndex(strcol(gpu ? 1 : 2))+(impl-2)*0.08):(column(impl==1 ? dcol : impl==2 ? tcol : 8)):(column(impl==1 ? dcol+1 : impl==2 ? tcol+1 : 9)):(column(impl==1 ? dcol+2 : impl==2 ? tcol+2 : 10)) with yerrorbars pt int(word(markers,impl)) ps 1.4 lw 2 lc rgb word(colors,impl) title word(labels,impl)
 }
 unset label 1
 unset logscale y
 unset key
 set bmargin at screen 0.19
 set tmargin at screen 0.37
 set yrange [0.85:*]
 set ylabel 'Speedup vs default (×)'
 set xlabel (heat ? 'Interior grid side length N (N × N)' : 'Matrix shape M × N × K')
 set format x '%g'
 if (heat) {
  set xtics font ',13'
  set xtics ('128' 0, '256' 1, '512' 2, '1024' 3, '2048' 4, '4096' 5, '8192' 6, '16384' 7)
  plot 1 with lines dt 2 lc rgb '#777777' notitle, data every ::1 using (log(column(5)/128)/log(2)):24:25:26 with yerrorbars pt 5 ps 1.4 lw 2 lc rgb word(colors,2) notitle
 } else {
  set xtics font ',13'
  set xtics ('512×512×512' 0, '1024×1024×1024' 1, '2048×2048×2048' 2, '4096×4096×4096' 3, '256×2048×1024' 4, '2048×256×1024' 5, '1023×1009×997' 6) rotate by -20
  plot 1 with lines dt 2 lc rgb '#777777' notitle, for [impl=2:paths] data every ::1 using (shapeIndex(strcol(gpu ? 1 : 2))+(impl-2)*0.08):(column(dcol)/column(impl==2 ? tcol : 8)) with linespoints pt int(word(markers,impl)) ps 1.4 lw 2 lc rgb word(colors,impl) notitle
 }
 unset multiplot
 unset output
}

# Overlay the hierarchy ceilings and executed-operation points for both kernels.
do for [case=1:2] {
 reset
 set encoding utf8
 set terminal svg size 1600,1000 enhanced font 'DejaVu Sans,16'
 heat=(case==2)
 data=heat ? 'example/heatEquation/figures/a30/hierarchical-roofline.csv' : 'example/matmul/results/roofline-points.csv'
 set output (heat ? 'example/heatEquation/figures/a30/hierarchical-roofline.svg' : 'example/matmul/results/hierarchical-roofline.svg')
 set datafile separator ','
 set border 3
 set tics out nomirror
 set grid xtics ytics lc rgb '#dddddd'
 set logscale xy
 set xrange [0.01:10000]
 set yrange [100:10000]
 set lmargin at screen 0.10
 set rmargin at screen 0.77
 set bmargin at screen 0.20
 set tmargin at screen 0.90
 set key at screen 0.79,0.89 left top opaque font ',14'
 set xlabel 'Arithmetic intensity (executed FLOP/byte)'
 set ylabel 'Executed throughput (GFLOP/s)'
 set title (heat ? 'Heat equation — NVIDIA A30, FP64 hierarchical rooflines, 8192²' : 'Matmul — NVIDIA A30, FP32 hierarchical rooflines') font ',22'
 footer=heat ? 'Three stencil launches per implementation; application replay, cache/clock control disabled. Boundary kernel excluded.' : 'Seven shapes, one launch per implementation; kernel replay, cache flushing and base clocks. Separate from runtime measurements.'
 set label 1 footer at screen 0.10,0.04 font ',12'
 if (!heat) { set label 2 'DRAM point labels: 1=512³  2=1024³  3=2048³  4=4096³  5=256×2048×1024  6=2048×256×1024  7=1023×1009×997' at screen 0.10,0.08 font ',12' }
 stats data every ::1 using (column(heat ? 14 : 6)) nooutput
 compute=STATS_median
 levels='DRAM L2 L1'
 levelMarkers='7 5 8'
 implementations=heat ? 'default winner' : 'default tuned cublas'
 labels=heat ? 'Default Tuned' : 'Default Tuned cuBLAS'
 paths=heat ? 2 : 3
 array bandwidth[3]
 do for [level=1:3] {
  matchLevel=heat ? (level==1 ? 'DRAM' : level==2 ? 'L2 return' : 'L1 global/local') : word(levels,level)
  stats data every ::1 using (strcol(heat ? 4 : 3) eq matchLevel ? column(heat ? 8 : 7) : 1/0) nooutput
  bandwidth[level]=STATS_median
 }
 plot for [level=1:3] (x*bandwidth[level]<compute ? x*bandwidth[level] : compute) with lines lw 2 dt level lc rgb '#777777' title (word(levels,level).' ceiling'), \
      for [impl=1:paths] for [level=1:3] data every ::1 using (strcol(heat ? 4 : 3) eq (heat ? (level==1 ? 'DRAM' : level==2 ? 'L2 return' : 'L1 global/local') : word(levels,level)) && strcol(heat ? 1 : 2) eq word(implementations,impl) ? column(heat ? 13 : 4) : 1/0):(column(heat ? 11 : 5)) with points pt int(word(levelMarkers,level)) ps 1.5 lc rgb word(colors,impl) title (word(labels,impl).' / '.word(levels,level)), \
      for [impl=1:paths] data every ::1 using (!heat && strcol(3) eq 'DRAM' && strcol(2) eq word(implementations,impl) ? $4 : 1/0):5:(sprintf('%d',shapeIndex(strcol(1))+1)) with labels offset char 0.7,0.5 font ',12' tc rgb word(colors,impl) notitle
 unset output
}
