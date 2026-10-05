Automatic candidate spaces
===========================

Declare the allowed values once and let online exploration register a bounded
set of configurations from them. Measurements guide later generation toward
promising neighborhoods while global probes keep other parts of the domain
represented. This avoids constructing the entire Cartesian product up front.
The domain declaration stays fixed; the registered catalog grows.

Start with launch geometry
--------------------------

Use the launch prototype to declare an automatic space::

   auto launchTuning = alpakaTune::makeAutomaticLaunchTuning(frameSpec);
   auto tuner = alpakaTune::makeTuner(
       config, alpakaTune::TunableBundle{launchTuning}, device);
   tuner.enqueue(queue, frameSpec, alpaka::KernelBundle{kernel, arguments...});

The same helper accepts a ``ThreadSpec``. A ``FrameSpec`` exposes
``frameExtent`` and ``numFrames``; a ``ThreadSpec`` exposes ``numThreads`` and
``numBlocks``. Their automatic defaults preserve the original coverage on
**each axis**, include the original geometry, and propose power-of-two
alternatives. The tuner checks the actual derived thread geometry against the
selected device's total and per-axis limits, the executor's thread support,
and Alpaka's dynamic shared-memory query before registering candidates.
This uses the pinned Alpaka API and requires no Alpaka changes.

Override one parameter when you know more about the kernel::

   auto launchTuning = alpakaTune::makeAutomaticLaunchTuning(
       frameSpec,
       alpakaTune::tuneFrameExtent(frameSpec,
           alpakaTune::autoCandidates(alpakaTune::RVals{
               alpaka::Vec{64u}, alpaka::Vec{128u}})));

The other launch domain remains automatic. An override must belong to the
same launch family; do not mix physical ThreadSpec overrides into a FrameSpec
automatic helper. Vector components remain independent tuning dimensions;
the coverage relation couples the generated launch parameters explicitly.
For unusual override extents, also supply compatible counts if the defaults
do not contain them.

A kernel that traverses its entire problem independently of launch coverage
can opt into wider geometry with
``makeAutomaticLaunchTuning(frameSpec, alpakaTune::fullTraversal)``. Make this
promise only when the kernel actually handles that geometry. Existing manual
``tuneFrameExtent``, ``tuneNumFrames``, ``tuneNumThreads``, ``tuneNumBlocks`` and
explicit restrictions continue to work.

Runtime and compiled parameters
-------------------------------

Automatic spaces also apply to ordinary kernel parameters::

   inline constexpr auto tile = ALPAKA_TUNE_TUNABLE("tile");
   inline constexpr auto unroll = ALPAKA_TUNE_TUNABLE("unroll");
   auto tunables = alpakaTune::TunableBundle{
       tile(alpakaTune::autoCandidates(
           alpakaTune::domain::interval(1u, 4096u),
           alpakaTune::hint::logarithmic,
           alpakaTune::hint::alignment(32u),
           alpakaTune::hint::preferred(128u))),
       unroll(alpakaTune::autoCandidates(
           alpakaTune::domain::compileInterval<1u, 8u>(
               alpakaTune::hint::logarithmic)))};

An integral runtime interval is stored as bounds and a step, without building
a vector for the entire range. Floating-point intervals use a bounded sampling
resolution; their declared cardinality and exhaustion proof remain unknown.
Hints control numeric sampling, alignment, preferred initial values, and
categorical neighborhood sampling. Bounds and alternatives remain immutable;
feedback adds configurations to the registered catalog.

``compileInterval<minimum, maximum>()`` generates at most eight linearly
spaced compiled alternatives. The logarithmic overload doubles the lower
bound and includes both bounds: ``compileInterval<1u, 8u>(hint::logarithmic)``
produces the compiled alternatives ``1, 2, 4, 8``. The concise spelling is
``autoCandidates<1u, 8u>(hint::logarithmic)``; bounds alone are sufficient.
Those alternatives are compiled ahead of time. Feedback decides which combinations to register and
explore; it cannot create new compiled values while the application runs.
You can still supply exact ``CVals``, ``CTypes``, or ``RVals``. Wrap them in
``autoCandidates`` to generate combinations progressively, or use them
directly for the existing fixed Cartesian space.

Declare a dependent scalar runtime domain with named parents::

   inline constexpr auto workers = ALPAKA_TUNE_TUNABLE("workers");
   auto workersDomain = alpakaTune::autoCandidates(
       alpakaTune::domain::dependent(
           alpakaTune::domain::interval(1u, 64u), tile,
           [](unsigned chosenTile) {
               return alpakaTune::domain::interval(1u, std::min(64u, chosenTile / 32u));
           }));

The generator returns allowed values inside the declared child domain; it
must accept every parent value that may be proposed. Multiple parents can be
passed as ``std::tuple{parent1, parent2}``. Missing parents are compile errors;
dependency cycles are rejected when constructing the tuner. Add explicit
identity/context arguments to ``makeTuner`` when captured restrictions or
dependency generators change meaning between workloads.

Generation, selection, and stopping
-----------------------------------

All tuners use one internal candidate-space provider. Manual spaces retain
the existing Cartesian IDs. Automatic spaces keep an append-only catalog of
exact domain indices. Adding configurations never changes older IDs or
measurement associations. Built-in strategies select registered IDs; numeric
coordinates are model features rather than candidate identities. A custom
strategy can implement ``supportsCandidateCatalog`` and ``recommendCandidate``
plus the exact-ID feedback callbacks; its existing normalized API continues
to serve manual spaces.

The defaults register up to 32 legal candidates. After 16 newly finished
**distinct** candidates, generation adds up to 16 candidates: approximately
three quarters local proposals around the four best measured configurations
and one quarter global probes. Local proposals perturb individual parameters
and parameter pairs. Sampling is bounded by 2,048 attempts per generation
batch and a 4,096-candidate catalog. Small enumerable domains have a separate
exhaustion cursor. Adaptive revisits update timing histories without counting
as new exploration coverage.

Finishing the current pool triggers another generation attempt. When generation
stops, online exploration completes once all eligible registered configurations
have measurements; fixed selection also requires their retirement. Execution
and retirement limits can end exploration earlier. Fixed selection can also
finish at its strategy-retry guard. Completion reasons distinguish:

* ``all_configurations``: full domain exhaustion is proven;
* ``candidate_budget``: the generated catalog reached its size limit;
* ``plateau``: distinct observations stopped improving after global probes;
* ``generation_stalled``: bounded attempts could not find a new legal point.

Plateau stopping is optional and disabled by default. ``info().space`` reports
registered count, optional declared combination count, distinct measured
count, revision, generation state, and domain exhaustion. Benchmark history
inspection reports registered-pool coverage separately from full domain
coverage.

Generation and adaptation have different limits. ``space.maximum_candidates``
bounds the registered catalog; ``tuning.maximum_executions`` bounds exploration
launches, including warm-ups and repeated measurements. Neither limits the
application's launch loop. ``candidate_budget`` does not mean that every value
in the declared domain was tested.

Online exploration can grow automatic spaces with either selection policy.
Offline exploration loads the saved catalog and performs no generation.
After exploration ends, fixed selection replays its winner; adaptive selection
remeasures only valid configurations that already have measurements, probing
an alternative at ``adaptive_probe_interval``. Registered but unmeasured
configurations cannot enter this reuse path. See :doc:`execution_modes`.

Configure generation separately from execution budgets::

   schema_version: 4
   tuning:
     exploration: online
     selection: fixed
     strategy: bayesian_optimization
     runs_per_candidate: 3
     maximum_executions: 10000
   space:
     initial_candidates: 32
     refinement_batch_size: 16
     refinement_interval: 16
     maximum_candidates: 4096
     maximum_generation_attempts: 2048
     plateau_patience: null
     minimum_relative_improvement: 0.01

This is a complete policy file; it uses process-local history. Add the
``history`` settings from :doc:`history_workflows` to save the catalog and
measurements. Generation settings belong to ``config.space`` in C++, for
example ``config.space.maximumCandidates = 4096u``. They affect automatic
spaces; explicit Cartesian candidate lists retain their existing enumeration.

Persistence and learned models
------------------------------

Compact and complete histories store a versioned catalog, its enumeration
cursor, revision, and random generator state. A context fingerprint describes
the domains and hints; it does not change when the catalog grows. Online
reload retains candidate IDs and measurements while execution budgets and
admission counters start a new run. Offline adaptive selection reuses the
measured subset of the restored catalog, so a changed workload need not trigger
another generation phase. Compact sampling can reduce that measured subset;
use enough retained configurations to provide useful alternatives. Manual
histories keep their existing format.

Automatic spaces use learned feature schema 2 and score registered candidates
in bounded batches, including candidates added later. Schema 1 artifacts
continue to work for manual spaces. Using a schema 1 artifact with an
automatic space reports an explicit incompatible-artifact fallback. Schema 2
support enables inference with a compatible artifact; it does not claim that
an existing model was retrained or that automatic tuning beats an offline
baseline.
