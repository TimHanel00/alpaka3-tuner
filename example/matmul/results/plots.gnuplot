# Run from example/matmul/results: gnuplot plots.gnuplot
reset
set terminal svg size 1600,800 enhanced font 'DejaVu Sans,14'
set output 'runtime-comparison.svg'
set datafile separator ','
set title "Measured compiled matmul winners — NVIDIA A30, strict FP32\nMedian of five run medians; 31 samples per implementation per run"
set xlabel 'Matrix shape M × N × K'
set ylabel 'Runtime (ms, logarithmic scale)'
set logscale y
set yrange [0.03:30]
set xrange [-0.6:6.6]
set grid ytics mytics lc rgb '#dddddd'
set border 3
set tics out nomirror
set xtics rotate by -20
set key top left opaque
set bars 1.5
set label 1 'Error bars: minimum–maximum run medians. Queue-event intervals; cuBLAS handle setup excluded.' at screen 0.1,0.02 font ',11'
set bmargin 6
plot 'runtimes.csv' every ::1 using ($0-0.18):8:9:10:xticlabels(1) with yerrorbars pt 7 ps 1.4 lw 2 lc rgb '#3465a4' title 'Strict FP32 cuBLAS', \
     '' every ::1 using ($0):5:6:7 with yerrorbars pt 5 ps 1.4 lw 2 lc rgb '#2e8b57' title 'Tuned matmul', \
     '' every ::1 using ($0+0.18):2:3:4 with yerrorbars pt 9 ps 1.5 lw 2 lc rgb '#d97706' title 'Default matmul (original baseline)'

reset
set terminal svg size 1600,800 enhanced font 'DejaVu Sans,14'
set output 'cpu-runtime-comparison.svg'
set datafile separator ','
set title "Compiled matmul on 2 × AMD EPYC 7713 (128 physical cores total; SMT disabled)\nFixed 128-thread OpenMP team; tile layout and numBlocks tuned jointly" font ',17'
set xlabel 'Matrix shape M × N × K'
set ylabel 'Runtime (ms, logarithmic scale)'
set logscale y
set grid ytics mytics lc rgb '#dddddd'
set border 3
set tics out nomirror
set xtics rotate by -15
set key top left opaque
set bars 1.5
set bmargin 7
set label 1 'Median of five independent run medians, seven samples per path; error bars: minimum–maximum run medians.' at screen 0.09,0.04 font ',11'
set label 2 'Same exclusive node, CPU affinity and NUMA policy for both paths. Queue-event intervals include lookup.' at screen 0.09,0.02 font ',11'
set xrange [-0.6:6.6]
set xtics ('512x512x512' 0, '1024x1024x1024' 1, '2048x2048x2048' 2, '4096x4096x4096' 3, '256x2048x1024' 4, '2048x256x1024' 5, '1023x1009x997' 6)
plot 'cpu-runtimes.csv' every ::1 using ($0-0.10):6:7:8 with yerrorbars pt 5 ps 1.4 lw 2 lc rgb '#2e8b57' title 'CPU-tuned matmul', \
     '' every ::1 using ($0+0.10):3:4:5 with yerrorbars pt 9 ps 1.5 lw 2 lc rgb '#d97706' title 'Default matmul (original baseline)'

reset
set terminal svg size 1800,780 enhanced font 'DejaVu Sans,13'
set output 'hierarchical-roofline.svg'
set datafile separator ','
set logscale xy
set xrange [0.1:10000]
set yrange [100:10000]
set grid xtics ytics lc rgb '#dddddd'
set border 3
set tics out nomirror
set xlabel 'Arithmetic intensity (executed FP32 FLOP/byte)'
set ylabel 'Executed FP32 throughput (GFLOP/s)'
set key bottom right opaque font ',11'
set multiplot layout 1,3 title "Hierarchical FP32 rooflines — NVIDIA A30\nSeven shapes, one profiled launch per implementation; cache flush and base clocks" margins 0.075,0.98,0.20,0.82 spacing 0.07,0.04
set label 1 '1=512³  2=1024³  3=2048³  4=4096³  5=256×2048×1024  6=2048×256×1024  7=1023×1009×997' at screen 0.06,0.07 font ',11'
set label 2 'ncu 2025.2.1 hierarchical single-precision section; L1 covers global/local traffic. Profiling is separate from runtime plots.' at screen 0.06,0.035 font ',11'
# Ceilings are medians of the 21 counter-derived per-launch ceilings.
stats 'roofline-points.csv' every ::1 using 6 nooutput
compute=STATS_median
levels='DRAM L2 L1'
colors='#d97706 #2e8b57 #3465a4'
implementations='default tuned cublas'
labels='Default Tuned Strict-FP32-cuBLAS'
markers='9 5 7'
do for [panel=1:3] {
 level=word(levels,panel)
 stats 'roofline-points.csv' every ::1 using (strcol(3) eq level ? $7 : 1/0) nooutput
 bandwidth=STATS_median
 set title (panel==3 ? 'L1 (global/local)' : level)
 plot (x*bandwidth<compute ? x*bandwidth : compute) with lines lw 2 lc rgb '#777777' title 'Counter-derived ceiling', \
      for [impl=1:3] 'roofline-points.csv' every ::1 using (strcol(3) eq level && strcol(2) eq word(implementations,impl) ? $4 : 1/0):5 with points pt int(word(markers,impl)) ps 1.35 lc rgb word(colors,impl) title word(labels,impl), \
      for [impl=1:3] 'roofline-points.csv' every ::1 using (strcol(3) eq level && strcol(2) eq word(implementations,impl) ? $4 : 1/0):5:(sprintf('%d',int(($0-1)/9)+1)) with labels offset char 0.6,0.5 font ',10' tc rgb word(colors,impl) notitle
}
unset multiplot

