# Run from example/matmul/results: gnuplot cpu-runtime-comparison.gnuplot
set terminal pngcairo size 1600,800 enhanced font 'DejaVu Sans,14'
set output 'cpu-runtime-comparison.png'
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
