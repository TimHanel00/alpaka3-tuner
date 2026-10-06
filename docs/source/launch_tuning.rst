Choose candidates and launch shapes
===================================

Begin with a small set of choices whose correctness you can explain. The
:doc:`tutorial <getting_started>` tunes ``batchSize`` using a named marker:

.. code-block:: cpp

   inline constexpr auto batchSize = ALPAKA_TUNE_TUNABLE("batchSize");
   auto tunables = alpakaTune::TunableBundle{
       batchSize(alpakaTune::RVals{32u, 64u, 128u})};
   auto bundle = alpaka::KernelBundle{
       AddOne{}, input, output, extent, batchSize};

Only the marker is substituted. Buffers, extent, and other ordinary arguments
are passed through. Every non-launch tunable must appear in the prototype.
Build the candidate list before ``makeTuner``; subsequent changes to the list
or configuration do not change an existing tuner.

``RVals{1, 2, 3}`` means exactly those values, not a range specification. Use
``generate::linSpace(first, last, step)`` or
``generate::logSpace(first, last, factor)`` for generated runtime candidates.
Candidate values can come from application configuration or device discovery.
These generators build explicit lists before tuning. To declare a bounded
domain whose combinations are generated progressively from feedback, use
``autoCandidates`` as described in :doc:`automatic_spaces`.
``markTunable`` and ``Tunable`` are compatibility forms; new code can use the
marker directly as above.

Multidimensional tunables
-------------------------

Vector components are independent by default. Make that choice explicit with
``mdPolicy::independent``, or use ``mdPolicy::listed`` to keep each listed
vector intact:

.. code-block:: cpp

   using Vec2 = alpaka::Vec<std::size_t, 2u>;
   inline constexpr auto tile = ALPAKA_TUNE_TUNABLE("tile");
   auto choices = alpakaTune::RVals{Vec2{8u, 2u}, Vec2{16u, 4u}};
   auto independent = tile(choices, alpakaTune::mdPolicy::independent);
   auto listed = tile(choices, alpakaTune::mdPolicy::listed);
   auto tunables = alpakaTune::TunableBundle{listed};

``independent`` produces four shapes: ``{8,2}``, ``{8,4}``, ``{16,2}``,
and ``{16,4}``. ``listed`` produces only ``{8,2}`` and ``{16,4}``, with
no predicate needed. Other tunables still combine with these choices.

The same policies work for compile-time vectors:

.. code-block:: cpp

   using Small = alpaka::CVec<std::size_t, 8u, 2u>;
   using Large = std::integer_sequence<std::size_t, 16u, 4u>;
   auto tunables = alpakaTune::TunableBundle{
       tile(alpakaTune::CTypes<Small, Large>{}, alpakaTune::mdPolicy::listed)};

Only the two listed variants are compiled. ``mdPolicy::independent`` compiles
all four component combinations. See :doc:`compile_time_tuning` for kernel
arguments. Policies apply to explicit ``RVals`` or ``CTypes`` vector lists;
every vector must have the same number of components.

Launch helpers accept the policy as their third argument. For a two-dimensional
``frame``, use:

.. code-block:: cpp

   auto tunables = alpakaTune::TunableBundle{
       alpakaTune::tuneFrameExtent(
           frame, alpakaTune::CTypes<Small, Large>{}, alpakaTune::mdPolicy::listed)};

``tuneNumFrames``, ``tuneNumBlocks``, and ``tuneNumThreads`` accept the same
policy. ``listed`` limits choices within one vector; use a relation when
different tunables must satisfy a shared condition such as coverage.

Tune a FrameSpec
----------------

``FrameSpec`` describes logical work with ``numFrames`` and ``frameExtent``.
The executor maps that work onto physical workers. For the tutorial's ``Index``
and ``frame``, replace its tunable bundle with:

.. code-block:: cpp

   auto tunables = alpakaTune::TunableBundle{
       candidates,
       alpakaTune::tuneNumFrames(
           frame, alpakaTune::RVals<Index>{Index{1u}, Index{2u}, Index{4u}}),
       alpakaTune::tuneFrameExtent(
           frame, alpakaTune::RVals<Index>{Index{32u}, Index{64u}, Index{128u}})};

The helpers use reserved launch names; **do not** add their markers to the
kernel arguments. They replace the relevant fields in the launch specification.
This space has ``3 × 3 × 3 = 27`` candidates. Increase the tutorial's training
launch count or guard if you want to measure them all repeatedly.

Changing geometry can change which elements a kernel visits. The tutorial's
``makeIdxMap`` traverses the whole problem range with any of these shapes.
A kernel that handles only one element per physical worker may instead need
exact coverage. Attach that requirement explicitly:

.. code-block:: cpp

   auto launchChoices = alpakaTune::makeFrameSpecTuning(
       alpakaTune::tuneFrameExtent(frame, extentCandidates),
       alpakaTune::tuneNumFrames(frame, frameCountCandidates),
       alpakaTune::preserveCoverage(frame));

Here the two candidate lists are application-defined. ``preserveCoverage``
requires exact logical coverage; ``doesNotExceedCoverage`` permits less than
or equal coverage in every dimension. Neither relation is added by the
individual ``tune...`` helpers.

For generated defaults, ``makeFrameSpecTuning(frame)`` bundles default frame
extents, frame counts, and a less-than-or-equal coverage restriction. Default
extents use power-of-two factorizations of 32 through 1024, plus the original
extent. Frame counts use halvings of the original count and their midpoints.
Use these only when reduced coverage still lets your kernel traverse all data.

Tune physical blocks and threads
--------------------------------

``ThreadSpec`` describes exact physical ``numBlocks`` and ``numThreads``.
``tuneNumBlocks``, ``tuneNumThreads``, and ``makeThreadSpecTuning`` provide
its corresponding interface. A tuner must use one launch-specification family;
mixing FrameSpec and ThreadSpec names is rejected.

Choose values supported by the executor. ``CpuSerial`` and ``CpuOmpBlocks``
require a thread-block extent of one, so varying physical thread counts is not
useful there. GPU thread counts must fit the device/kernel limits; use a small
valid set suited to the algorithm. Logical ``frameExtent`` is not a direct
GPU thread-count setting.

Generate launch choices automatically
-------------------------------------

To let the tuner generate both launch dimensions, use:

.. code-block:: cpp

   auto launchChoices = alpakaTune::makeAutomaticLaunchTuning(frame);
   auto tunables = alpakaTune::TunableBundle{candidates, launchChoices};

This preserves the prototype's coverage on every axis and includes its original
geometry. The generated catalog grows during online exploration; loaded measured
configurations can be reused with either offline selection policy. Device and
kernel resource checks apply before a generated candidate is registered.
The same helper accepts a ``ThreadSpec``. See :doc:`automatic_spaces` for
overrides, dependent domains, generation budgets, and the explicit
``fullTraversal`` option for kernels that handle variable coverage.

Restrict combinations
---------------------

Explicit candidate dimensions form a Cartesian product. Use ``constrain`` and
``restrict`` to reject incompatible combinations before they run:

.. code-block:: cpp

   auto tunables = alpakaTune::constrain(
       alpakaTune::TunableBundle{
           tile(alpakaTune::RVals{32u, 64u}),
           workers(alpakaTune::RVals{32u, 64u})},
       alpakaTune::restrict(workers, tile,
           [](std::uint32_t workerCount, std::uint32_t tileExtent) {
               return workerCount <= tileExtent;
           }));

``tile`` and ``workers`` are named markers declared like ``batchSize``.
Relations are evaluated lazily during admission. ``info().candidateCount``
counts the original Cartesian space, including combinations later rejected.
Inspect ``restrictionRejectedCount`` for rejections; use
:doc:`post_evaluation_validation` when validity depends on the result.
