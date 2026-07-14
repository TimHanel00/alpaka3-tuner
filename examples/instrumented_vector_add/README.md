# Instrumented vector add

This example combines a runtime `runtimeOffset` (`RVals{0, 1}`) and a
compile-time `simdWidth` (`CVals<1, 2, 4, 8>`), and the reserved Alpaka
`numBlocks` launch parameter (`{1, 2, 4, 8, 16}`). It runs with
`CpuOmpBlocks`, where host thread blocks are meaningful and `numThreads` must
remain one. Its generated YAML selects `bayesian_optimization` and assigns 250
measured launches to each of the 40 Cartesian configurations, for exactly
10,000 `context.tune` calls.
Mann-Whitney early retirement is deliberately disabled in this controlled
benchmark so every configuration has an equally sized history; production
sessions enable it by default.

```sh
cmake --build build --target alpakaTune_instrumented_vector_add
build/examples/instrumented_vector_add/alpakaTune_instrumented_vector_add \
  /tmp/alpakaTune-vector-add-10000
python3 examples/instrumented_vector_add/plot.py \
  /tmp/alpakaTune-vector-add-10000/samples.csv \
  /tmp/alpakaTune-vector-add-10000/runtime.png
```

`samples.csv` records every raw launch together with the MAD-filtered median
for that configuration. The plot overlays the raw series with this less-peaky
per-record history. A starred label is emitted only when a configuration
completes its measurement budget and its final robust estimate improves on
every previously-completed configuration; it is never a provisional moving
median. `summary.txt` reports the final winner and both raw and robust timing
summaries.
