# Run from example/matmul/results: gnuplot runtime-comparison.gnuplot
set terminal pngcairo size 1600,800 enhanced font 'DejaVu Sans,14'
set output 'runtime-comparison.png'
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
