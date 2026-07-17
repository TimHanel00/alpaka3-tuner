Instrumenting a kernel
======================

A ``Tuner`` represents one particular tuning context. It owns the candidate
scheduler, measurements, and winner for one device, kernel-bundle type, and
launch prototype. Instrument a normal Alpaka launch by replacing
``queue.enqueue(spec, bundle)`` with ``tuner.enqueue(queue, spec, bundle)``.

Minimal example
---------------

.. code-block:: cpp

   #include <tuning.hpp>

   inline constexpr auto chunkSize =
       ALPAKA_TUNE_TUNABLE("chunkSize");

   auto config = alpakaTune::TunerConfig::fromYaml("tuning.yaml");
   auto tunables = alpakaTune::TunableBundle{
       chunkSize(alpakaTune::RVals{64u, 128u, 256u})};
   auto tuner = alpakaTune::makeTuner(
       config, tunables, device, executor, "vector-add");

   auto bundle = alpaka::KernelBundle{
       VectorAddKernel{}, inputA, inputB, output,
       alpakaTune::markTunable(chunkSize)};

   while (!tuner.isTuningComplete())
       tuner.enqueue(queue, frameSpec, bundle);

The marker is replaced by the selected value before Alpaka receives the
bundle. Once tuning completes, later ``tuner.enqueue`` calls replay the winner.
Reuse one ``TunerConfig`` for several tuners when they should share settings
and a persistence file; every tuner still owns independent runtime state.

Launch-shape convenience
------------------------

``makeTuner(config, device, frameSpec, identity)`` creates correlated
``numFrames`` and ``frameExtent`` candidates. Omitting ``config`` uses
``tunerConfig()``. The corresponding ``ThreadSpec`` overload creates paired
``numBlocks`` and ``numThreads`` candidates. Explicit
``FrameExtentTuning`` and ``NumFramesTuning`` arguments replace the generated
candidate sets.

The mirrored Alpaka examples under ``example/`` use the same instrumentation
pattern. For a complete runtime example, see
``example/vectorAdd/src/vectorAdd.cpp``; nested configuration tuning is shown
in ``example/tuneTheTuner/src/tuneTheTuner.cpp``.
