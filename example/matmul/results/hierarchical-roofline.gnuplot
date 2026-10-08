# Run from example/matmul/results: gnuplot hierarchical-roofline.gnuplot
set terminal pngcairo size 1800,780 enhanced font 'DejaVu Sans,13'
set output 'hierarchical-roofline.png'
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
stats 'roofline-ceilings.csv' every ::1 using 2 nooutput
compute=STATS_median
levels='DRAM L2 L1'
colors='#d97706 #2e8b57 #3465a4'
implementations='default tuned cublas'
labels='Default Tuned Strict-FP32-cuBLAS'
markers='9 5 7'
do for [panel=1:3] {
 level=word(levels,panel)
 stats 'roofline-ceilings.csv' every ::(panel)::(panel) using 3 nooutput
 bandwidth=STATS_mean
 set title (panel==3 ? 'L1 (global/local)' : level)
 plot (x*bandwidth<compute ? x*bandwidth : compute) with lines lw 2 lc rgb '#777777' title 'Counter-derived ceiling', \
      for [impl=1:3] 'roofline-points.csv' every ::1 using (strcol(3) eq level && strcol(2) eq word(implementations,impl) ? $4 : 1/0):5 with points pt int(word(markers,impl)) ps 1.35 lc rgb word(colors,impl) title word(labels,impl), \
      for [impl=1:3] 'roofline-points.csv' every ::1 using (strcol(3) eq level && strcol(2) eq word(implementations,impl) ? $4 : 1/0):5:(sprintf('%d',int(($0-1)/9)+1)) with labels offset char 0.6,0.5 font ',10' tc rgb word(colors,impl) notitle
}
unset multiplot
