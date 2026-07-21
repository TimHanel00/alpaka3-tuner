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

Launch-shape tuning
-------------------

``makeTuner`` always receives a completed ``TunableBundle``. Its only two
forms are ``makeTuner(config, tunables, device, identity...)`` and
``makeTuner(tunables, device, identity...)``; the latter snapshots
``tunerConfig()``. Entries after the device contribute only to the persistent
identity. Strings use their value, and supported Alpaka objects such as an
executor use their Alpaka name.

``tuneFrameExtent`` and ``tuneNumFrames`` create independent named entries.
Putting both directly in a bundle exposes their Cartesian product:

.. code-block:: cpp

   auto tunables = alpakaTune::TunableBundle{
       alpakaTune::tuneFrameExtent(frameSpec, frameExtentCandidates),
       alpakaTune::tuneNumFrames(frameSpec, numFramesCandidates),
       chunkSize(alpakaTune::RVals{64u, 128u, 256u})};

Use ``makeFrameSpecTuning(frameSpec)`` for the generated defaults plus a lazy
coverage-preserving relation. A tuning fragment flattens into the enclosing
bundle alongside ordinary kernel parameters:

.. code-block:: cpp

   auto tunables = alpakaTune::TunableBundle{
       alpakaTune::makeFrameSpecTuning(frameSpec),
       chunkSize(alpakaTune::RVals{64u, 128u, 256u})};

Explicit candidates can be correlated with the same factory:

.. code-block:: cpp

   auto frameTuning = alpakaTune::makeFrameSpecTuning(
       alpakaTune::tuneFrameExtent(frameSpec, frameExtentCandidates),
       alpakaTune::tuneNumFrames(frameSpec, numFramesCandidates),
       alpakaTune::preserveCoverage(frameSpec));

``tuneNumBlocks``, ``tuneNumThreads``, and ``makeThreadSpecTuning`` provide the
corresponding ``ThreadSpec`` interface.

The mirrored Alpaka examples under ``example/`` use the same instrumentation
pattern. For a complete runtime example, see
``example/vectorAdd/src/vectorAdd.cpp``; nested configuration tuning is shown
in ``example/tuneTheTuner/src/tuneTheTuner.cpp``.
