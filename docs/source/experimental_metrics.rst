Experimental metrics objectives
===============================

This integration is currently available on the ``tunerEnhancedMetrics`` feature
branch. The general scoring-function API in :doc:`custom_metrics` is also
available on ``dev``. Metrics remain optional on the feature branch; ordinary
runtime tuning works without them.

Build and dependency ownership
------------------------------

.. code-block:: sh

   git switch tunerEnhancedMetrics
   cmake -S . -B build-metrics -G Ninja \
       -DCMAKE_BUILD_TYPE=Release \
       -DalpakaTune_BUILD_EXAMPLES=ON \
       -DalpakaTune_BUILD_TESTING=ON \
       -DalpakaTune_DEP_METRICS=ON \
       -DalpakaMetrics_DEP_PAPI=OFF
   cmake --build build-metrics

``alpakaTune_DEP_METRICS`` defaults to ``OFF``. With it enabled, alpakaTune
fetches `alpakaMetrics
<https://github.com/TimHanel00/alpakaMetrics/tree/b2f3fbf5086f875a1fea41dabe198df8bf76f7f1>`_
first and uses its Alpaka dependency. The metrics project's pinned source owns
that revision; the tuner does not populate its standalone Alpaka dependency.
Both libraries and all consumers share one ``alpaka::alpaka`` target.

With metrics disabled, the fallback remains Alpaka revision
``b7d339d07056a9a2a6c4051cc3927157bc5f0d51``. Metrics and PAPI are not fetched or
discovered in this configuration.

An existing build directory can be switched without removing its cache:

.. code-block:: sh

   cmake -S . -B build-metrics -DalpakaTune_DEP_METRICS=OFF
   cmake --build build-metrics
   cmake -S . -B build-metrics -DalpakaTune_DEP_METRICS=ON
   cmake --build build-metrics

Reconfiguration regenerates the active target graph. Dependencies owned by
metrics occupy a separate FetchContent directory from the standalone fallback;
inactive directories remain cached. A standalone Alpaka source override and
stale metrics-revision cache entries do not override metrics ownership when
metrics is enabled.

PAPI is independent of the integration option. Set
``alpakaMetrics_DEP_PAPI=ON`` for counters. Upstream defaults to bundled PAPI;
``alpakaMetrics_USE_SYSTEM_PAPI=ON`` selects an installed PAPI. Its component
options control additional native events. Disabling PAPI retains queue elapsed
time and the adapter. Counter availability also depends on hardware and access
permissions; successful configuration does not guarantee collection.

Using configured queues
-----------------------

Link the optional ``alpakaTune::metrics`` CMake target and include
``<alpakaTune/metrics.hpp>``. An installed consumer requests
``find_package(alpakaTune CONFIG REQUIRED COMPONENTS metrics)``; the installation
must have been built with metrics enabled.

.. code-block:: cpp

   auto queue = alpakaMetrics::makeQueue(
       device.makeQueue(alpaka::queueKind::nonBlocking, alpaka::timing::enabled),
       alpakaMetrics::Config{.metrics = {
           alpakaMetrics::metric::elapsedTime,
           alpakaMetrics::metric::instructions}});
   auto objective = alpakaTune::customMetric(
       "instructions_v1", [](alpakaMetrics::Result const &result) {
           auto const &count = result.getMetric("instructions");
           if (!count.isAvailable())
               throw std::runtime_error{count.diagnostic};
           if (count.descriptor.unit != alpakaMetrics::MetricUnit::count ||
               count.descriptor.scope != alpakaMetrics::MetricScope::queueWorkerThread)
               throw std::runtime_error{"Incompatible instruction count"};
           return count.asDouble();
       });
   auto tuner = alpakaTune::makeTuner(config, tunables, device, objective);
   auto launch = alpakaTune::metrics::enqueue(tuner, queue, frameSpec, bundle);
   alpakaTune::metrics::provideMetrics(tuner, launch);
   queue.clearMeasurements();

The helper launches the tuner's selected kernel once and retains that exact
measurement. Explicit submission waits for completion and calls the configured
scoring function. It validates the tuner and execution index, so an older handle
cannot score a newer launch. Queue copies and unrelated submissions cannot
change the retained measurement; clearing queue history leaves it valid.
Consume it before requesting the next tuner launch, as required by the
last-enqueue contract.

Use ``queue.getUnderlyingQueue()`` for Alpaka memory-copy operations that require
an ordinary Alpaka queue. Explicitly wait for producing work before transferring
or validating buffers. Default timing also accepts the metrics queue, but the
examples use the adapter's own elapsed result when it is the objective, avoiding
a second timing layer.

Runnable examples
-----------------

All three examples retain runtime as their default. Additional objectives are
``elapsed-time``, ``instructions``, ``l2-misses``, and ``weighted``:

* ``vectorAdd`` introduces the configured queue and defaults to instructions as
  its weighted counter.
* ``heatEquation2D`` tunes only the stencil; its boundary launch is fixed and
  uses the same underlying queue. L2 misses are its default weighted counter.
* ``matrixMultiplication`` compares tiling choices under instructions or cache
  misses, with L2 misses as its default weighted counter.

The small test configuration bounds the tuning horizon and disables persistence:

.. code-block:: sh

   export ALPAKA_TUNE_CONFIG="$PWD/tests/adaptive-smoke.yaml"
   build-metrics/example/vectorAdd/alpakaTune_vectorAdd \
       -n 16384 -r 2 --objective elapsed-time --backend host:cpu --executor CpuSerial
   build-metrics/example/heatEquation2D/alpakaTune_heatEquation2D \
       -n 64 -t 20 -d 0.0001 --objective elapsed-time --backend host:cpu --executor CpuSerial
   build-metrics/example/matrixMultiplication/alpakaTune_matrixMultiplication \
       -m 32 -n 32 -k 32 -r 2 --objective elapsed-time --backend host:cpu --executor CpuSerial

Enable PAPI before requesting a counter objective. For example:

.. code-block:: sh

   build-metrics/example/vectorAdd/alpakaTune_vectorAdd \
       -n 16384 -r 2 --objective instructions --backend host:cpu --executor CpuSerial
   build-metrics/example/matrixMultiplication/alpakaTune_matrixMultiplication \
       -m 32 -n 32 -k 32 -r 2 --objective weighted --counter l2-misses \
       --time-weight 0.5 --counter-weight 0.5 \
       --time-scale-seconds 0.001 --counter-scale 100000 \
       --backend host:cpu --executor CpuSerial

Weighted scoring is ``timeWeight * seconds / timeScale + counterWeight * count /
counterScale``. Equal weights are the default; both positive normalization scales
must be supplied. Weights must be finite and non-negative, with at least one
positive weight. The illustrated scales are user-selected reference values,
not measured calibrations. Keep them fixed across candidates and comparable runs.

The examples print their objective identity and latest available component
values, units, scopes, provider, and score. Formula version, counter selection,
weights, and scales are included in persistence identity. Raw readings remain
outside tuner persistence; the scalar score uses existing history formats.

Collection boundaries
---------------------

Queue counters are supported for explicit CPU serial kernels and describe
the queue worker thread. OpenMP/TBB and GPU kernel counters report unsupported
scope. Counter examples require ``--backend host:cpu --executor CpuSerial``.
Adapter elapsed time describes the queue interval, including applicable marker
bookkeeping. Include that overhead when interpreting the result.

Requested unavailable counters stop the example with their status, scope, and
diagnostic. There is no automatic runtime fallback or zero substitution.
L2 miss presets retain their hardware-dependent event meaning and are not
transferred-byte counts. Queue elapsed time and counts must have compatible
attribution before combining them.

The queue collector does not provide per-kernel energy, occupancy, or bandwidth
measurements, GPU counter attribution, or thread-team aggregation.
See `alpakaMetrics collection semantics
<https://github.com/TimHanel00/alpakaMetrics#implemented-measurement-semantics>`_.
Instrumentation and synchronization alter execution cost; evaluate performance
separately with sufficient repetitions.

Run the integration tests
-------------------------

.. code-block:: sh

   ctest --test-dir build-metrics -R 'alpakaTune_(metrics_tests|metric_objective_tests|dependency_switching|.*_metrics)$' \
       --output-on-failure

Elapsed-time smoke tests validate all three algorithms. Deterministic result
fixtures check scoring, units, scope, and unavailable-counter behavior. The real
instruction-counter case skips when collection is unavailable. Dependency tests
check reconfiguration with metrics enabled and disabled.
