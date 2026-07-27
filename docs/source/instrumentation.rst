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

alpakaTune selects its runtime clock from the Alpaka API and device-kind tags
at compile time. Applications should construct the queue passed to
``Tuner::enqueue`` with
``alpakaTune::makeQueue(device, alpaka::queueKind::nonBlocking,
alpakaTune::timing::enabled)``. Timing is an explicit tag independent of the
queue's blocking behavior; the temporary implementation currently requires a
non-blocking timed queue so markers can be submitted adjacent to the kernel.
CUDA and HIP use timing-enabled native events behind an internal backend
adapter. Before measuring, the adapter enqueues and waits for an Alpaka event
on the application queue. It then submits a start event, exactly the selected
kernel, and an end event consecutively on that non-blocking timed queue.
Waiting for the end event preserves the synchronous tuner contract, while
``LaunchObservation::runtimeSeconds`` comes from the device-event timestamps.
This avoids folding host submission and wake-up latency into the candidate
runtime.

The host API uses a synchronized ``std::chrono::steady_clock`` interval. SYCL
CPU devices select that host implementation directly. For SYCL GPU devices,
alpakaTune temporarily returns a minimal Alpaka-compatible measurement queue
created with ``sycl::property::queue::enable_profiling``. The kernel event's
``command_start`` and ``command_end`` timestamps provide the device runtime.
This compatibility layer is intended to disappear once Alpaka exposes the same
capability through its public queue and event interfaces.
``LaunchObservation::runtimeMeasurementSource`` and
``TunerInfo::runtimeMeasurementSource`` report either ``device_event`` or
``host_clock``; fallback therefore never masquerades as device timing.
Recommendation time is reported separately, while application-side logging
outside ``Tuner::enqueue`` is not included.

Device-event timing improves the runtime sample used for candidate ranking; it
does not remove the synchronization cost paid by the application. The external
queue boundary, end-event wait, strategy, statistics, and persistence work
remain part of the overall measured-enqueue overhead.

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

Use ``makeFrameSpecTuning(frameSpec)`` for the generated defaults. Default
``frameExtent`` values cover 32, 64, 128, 256, 512, and 1024 logical elements.
For an N-dimensional extent, every power-of-two factorization is generated
with nondecreasing components. Alpaka's final, fastest-varying index therefore
contains the largest component. The original extent is retained as a safe
fallback for small or nonstandard launch prototypes.

Default ``numFrames`` values start at one and end at the input FrameSpec value.
They contain repeated halves plus integer midpoints between adjacent splits.
For example, an upper limit of 1024 includes 256, 512, 768, and 1024. In
multiple dimensions, Alpaka ``mapToND`` ordering forms their Cartesian product,
with the final component varying fastest.

``defaultNumFramesCandidates`` accepts an optional refinement-level argument.
The default value of one inserts the midpoints above. Each additional level
inserts another midpoint between every adjacent value; for example,
``defaultNumFramesCandidates(upperNumFrames, 3u)`` approximates eighth-interval
spacing between successive halvings. This provides a denser runtime space
without changing the FrameSpec tunable's dimensionality.

The complete default factory applies two lazy relations: only the documented
extent factorizations are legal after vector-component recombination, and each
candidate's logical coverage must be less than or equal to the original
FrameSpec coverage. A tuning fragment flattens into the enclosing bundle
alongside ordinary kernel parameters:

.. code-block:: cpp

   auto tunables = alpakaTune::TunableBundle{
       alpakaTune::makeFrameSpecTuning(frameSpec),
       chunkSize(alpakaTune::RVals{64u, 128u, 256u})};

The generators can also be composed independently. This example keeps the
default extent space but supplies a larger application-specific upper limit
for ``numFrames``:

.. code-block:: cpp

   auto upperNumFrames = frameSpec.getNumFrames() * 4u;
   auto tunables = alpakaTune::TunableBundle{
       alpakaTune::makeDefaultFrameExtentTuning(frameSpec),
       alpakaTune::tuneNumFrames(
           frameSpec,
           alpakaTune::defaultNumFramesCandidates(upperNumFrames))};

There is deliberately no implicit coverage relation when an application
supplies either tuning entry. The application must add
``doesNotExceedCoverage(frameSpec)``, ``preserveCoverage(frameSpec)``, or its
own semantic restriction when one is required. ``doesNotExceedCoverage``
accepts less-than-or-equal coverage independently in every dimension;
``preserveCoverage`` requires exact equality. Explicit candidates can be
correlated with the same factory:

.. code-block:: cpp

   auto frameTuning = alpakaTune::makeFrameSpecTuning(
       alpakaTune::tuneFrameExtent(frameSpec, frameExtentCandidates),
       alpakaTune::tuneNumFrames(frameSpec, numFramesCandidates),
       alpakaTune::preserveCoverage(frameSpec));

An ``RVals<Vec<T, N>>`` remains one named N-dimensional launch tunable. The
strategy sees N scalar dimensions and the selected components are reconstructed
into one ``alpaka::Vec<T, N>`` before launch. Component-wise recombination is
why multidimensional vector spaces may need an explicit relation.

``tuneNumBlocks``, ``tuneNumThreads``, and ``makeThreadSpecTuning`` provide the
corresponding ``ThreadSpec`` interface.

The mirrored Alpaka examples under ``example/`` use the same instrumentation
pattern. For a complete runtime example, see
``example/vectorAdd/src/vectorAdd.cpp``; nested configuration tuning is shown
in ``example/tuneTheTuner/src/tuneTheTuner.cpp``.
