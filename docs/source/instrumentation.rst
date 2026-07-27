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

   constexpr std::size_t applicationMinimum = 50'000u;
   std::size_t launches = 0u;
   while (launches < applicationMinimum || !tuner.completed()) {
       tuner.enqueue(queue, frameSpec, bundle);
       ++launches;
   }

The marker is replaced by the selected value before Alpaka receives the
bundle. The application owns this loop and decides how often the kernel is
needed. The example above deliberately combines two application choices: run
at least 50,000 launches, and keep running until the configured tuning-policy
goal has been reached. A real application may use only its own loop bound and
ignore ``completed()`` entirely.

In ``online_fixed``, ``completed()`` reports a terminal tuning state and later
launches replay the winner. In ``online_adaptive`` with a horizon, it reports
only that the configured admission and cooling horizon has been reached. The
tuner remains active: later calls still recommend, measure, revisit
configurations, and update the residual adapter. Therefore ``completed()`` is
policy information, not an instruction from the library to stop the
application.
``isTuningComplete()`` reports the same adaptive horizon completion while the
internal scheduler remains active. ``completionReason()`` stays unavailable
because no terminal replay state was entered.
Horizon-less adaptive mode never reports completion, so an application using
the combined loop condition above must configure a horizon.

Reuse one ``TunerConfig`` for several tuners when they should share settings
and optional history access; every tuner still owns independent runtime state.

Instrumentation overhead
------------------------

Online measurement is not free. A measured ``Tuner::enqueue`` call performs
strategy recommendation and admission, rebuilds the selected launch, submits
the kernel between reusable Alpaka events, waits for those event boundaries,
updates the rolling statistics, and stages enabled history state. The
synchronized launch path commonly adds
approximately 20--40 microseconds per call on GPU workloads once tuner and
model state are initialized. This is a representative engineering estimate,
not a backend-independent guarantee; device, driver, queue state, host load,
strategy, and enabled application-side tracing can change it.

Before timing, the tuner enqueues and waits for a reusable start event. It then
starts the host clock, submits exactly the selected kernel, enqueues a reusable
completion event, and waits for that event. Consequently,
``LaunchObservation::runtimeSeconds`` excludes older work already resident in
the queue, but still contains host launch submission and event-wait latency in
addition to device execution. Alpaka currently exposes portable event ordering
and completion, but not a generic elapsed-device-time query, so this is an
event-delimited host observation rather than a CUDA- or HIP-specific device
timestamp. Recommendation time is reported separately, while application-side
logging outside ``Tuner::enqueue`` is not included.

For a 200-microsecond kernel, a 20--40-microsecond measurement cost is already
about 10--20 percent of that runtime. Online tuning pays off only when the
accumulated time saved by better configurations exceeds measurement,
recommendation, exploration, and integration costs. Short or infrequently
called kernels are consequently often better served by an offline workflow:
collect a history in a dedicated run, then use ``offline`` mode to replay the
recorded best configuration without synchronized timing or history updates.

Short-kernel diagnostic
^^^^^^^^^^^^^^^^^^^^^^^

Once a tuner first observes a measured runtime below 200 microseconds, it
emits one warning for that tuner context through ``std::clog``. The warning is
not repeated on later launches.

``TunerInfo::instrumentationOverheadWarning`` remains populated afterward and
reports the triggering runtime, the 200-microsecond threshold, and the
representative 20--40-microsecond overhead range. Applications can use this
diagnostic to exclude short kernels from online tuning. It is runtime
information and does not alter the persistence schema or candidate-selection
policy.

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
