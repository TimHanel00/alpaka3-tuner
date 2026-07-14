Instrumentation example
=======================

``examples/instrumented_vector_add`` is an end-to-end benchmark of the new
interface. It combines ``RVals{0, 1}`` for a runtime offset with
``CVals<1, 2, 4, 8>`` for a compile-time SIMD width and the reserved
``numBlocks`` launch parameter. It uses
``CpuOmpBlocks``, where block count is physical launch parallelism and host
``numThreads`` is constrained to one. Its generated YAML selects
``bayesian_optimization`` and gives every candidate the same fixed
measurement budget.

Mann-Whitney early retirement is disabled only for this controlled benchmark
so that every candidate receives the same number of measurements. The normal
YAML default enables early retirement. Each CSV row contains the raw
launch time and the configuration's MAD-filtered median; the plot overlays the
two histories. A labelled best is emitted only when a configuration completes
its measurement budget and its final robust estimate improves on every
previously-completed configuration; provisional, moving estimates are not
labelled as best.

The executable writes ``samples.csv`` and ``summary.txt``. ``plot.py`` draws
time against runtime and labels each confirmed improvement in completion
order.

.. code-block:: sh

   build/examples/instrumented_vector_add/alpakaTune_instrumented_vector_add /tmp/vector-add
   python3 examples/instrumented_vector_add/plot.py /tmp/vector-add/samples.csv /tmp/vector-add/runtime.png
