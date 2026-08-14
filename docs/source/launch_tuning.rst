Launch tuning
=============

A tuner owns the tuning state for one device and one kernel bundle type.
Construct it from a ``TunerConfig`` and named ``TunableBundle``, then pass the
device plus optional identity-only Alpaka or application entries. Alpaka
``deviceKind``, ``api``,
``Device``, and executor objects use their Alpaka name; strings and values
accepted by ``std::to_string`` are also accepted.

.. code-block:: cpp

   inline constexpr auto scale = ALPAKA_TUNE_TUNABLE("scale");

   auto tunableBundle = alpakaTune::TunableBundle{
       scale(alpakaTune::RVals{0, 256, 16})};
   auto tuner = alpakaTune::makeTuner(
       alpakaTune::tunerConfig(), tunableBundle, device,
       alpaka::deviceKind::cpu, alpaka::api::host, executor, "vector-add");

   auto prototype = alpaka::KernelBundle{
       Kernel{}, input, output, scale};
   tuner.enqueue(queue, frameSpec, prototype);

The parameter object in the prototype bundle is a placeholder only: it is
never sent to the device. The tuner rebuilds the Alpaka ``KernelBundle``
with the selected value in every parameter slot before calling
``queue.enqueue``. An ordinary argument remains fixed. Every non-launch
tunable must appear in at least one parameter slot. ``Tunable{scale, values}``,
``named``, and ``markTunable`` remain compatibility spellings for this direct
frontend.

Braced ``RVals`` arguments are explicit candidates, so ``RVals{1, 2, 3}``
selects those three values. For a non-integral candidate such as an Alpaka
vector, spell the element type and provide a vector of values:
``RVals<Index>{std::vector<Index>{Index{1}, Index{2}}}``.

``generate::linSpace(first, last, step)`` and
``generate::logSpace(first, last, factor)`` provide the corresponding scalar
or Alpaka-vector runtime candidates.

Runtime candidate lists may be assembled from configuration files, command
line arguments, device discovery, or other runtime state. ``makeTuner`` moves
or copies the completed bundle into the tuner. Candidate dimensions, indices,
scheduling state, and the persistence fingerprint are fixed from that
snapshot; candidate lists do not change during tuning.

Multidimensional parameters
---------------------------

Every component of an Alpaka vector is an independent backend dimension. The
candidate vectors provide the component value sets; duplicates are removed
per component before the Cartesian product is formed:

.. code-block:: cpp

   inline constexpr auto tile = ALPAKA_TUNE_TUNABLE("tile");

   auto tunables = alpakaTune::TunableBundle{
       tile(alpakaTune::RVals<Vec2>{
           Vec2{8, 2},
           Vec2{16, 4}})};

This produces backend dimension sizes ``{2, 2}`` and reconstructs ``Vec2``
values ``{8,2}``, ``{8,4}``, ``{16,2}``, and ``{16,4}`` for the kernel. A
strategy still sees only one flat normalized coordinate per component.
For example, one three-dimensional integral vector tunable and one integral
scalar tunable occupy exactly four entries in the strategy's
``std::vector<float>`` configuration.

Dependent component combinations do not change the backend representation.
Express them as a lazy unary restriction over the reconstructed vector:

.. code-block:: cpp

   alpakaTune::restrict(tile, [](auto const& value) {
       return value == Vec2{8, 2} || value == Vec2{16, 4};
   })

Constraint relations
--------------------

Build the normal Cartesian product first, then attach relationships between
named parameter values with ``restrict``. For example, this space exposes all
tile and worker choices while accepting only combinations in which the worker
count fits into the tile:

.. code-block:: cpp

   inline constexpr auto tile = ALPAKA_TUNE_TUNABLE("tile");
   inline constexpr auto workers = ALPAKA_TUNE_TUNABLE("workers");

   auto tunables = alpakaTune::constrain(
       alpakaTune::TunableBundle{
           tile(alpakaTune::RVals{1, 2, 3}),
           workers(alpakaTune::RVals{1, 2, 3})},
       alpakaTune::restrict(
           workers,
           tile,
           [](alpaka::concepts::VectorOrScalar auto const& workerCount,
              alpaka::concepts::VectorOrScalar auto const& tileExtent) {
               return workerCount <= tileExtent;
           }));

The relation is evaluated only when the scheduler reaches a candidate. It is
not expanded into a second list and needs no separate relation identifier.
The same workflow accepts scalar values and Alpaka vectors. ``CTypes``
candidates such as ``alpaka::CVec`` and ``std::integer_sequence`` are
materialized as their vector values before the predicate is called.

``tuner.info()`` returns a read-only snapshot including the Cartesian
candidate count, the number rejected by relations, the number measured, the
execution count, and the selected configuration when tuning is complete.

Reserved launch names
---------------------

The names ``frameExtent``, ``numFrames``, ``numThreads``, and ``numBlocks``
need no marker. ``numFrames`` and ``frameExtent`` replace the corresponding
fields of a ``FrameSpec``; ``numBlocks`` and ``numThreads`` do the same for a
``ThreadSpec``. A tuner targets one of these launch specifications. Combining
FrameSpec and ThreadSpec parameters in one tuner is rejected because it
duplicates the logical and physical decomposition in the same search space.

Choose launch candidates that are valid for the selected Alpaka executor. For
example, ``CpuSerial`` and ``CpuOmpBlocks`` require a thread-block extent of
one, so ``numBlocks`` is their meaningful physical launch parameter. A CUDA or
HIP tuner can additionally tune ``numThreads`` (usually warp-aligned values)
and may tune ``numBlocks`` when the kernel is valid for every Cartesian pair.
``numFrames`` is the portable logical alternative; it commonly maps to host
blocks. ``frameExtent`` changes only the logical decomposition, not a direct
thread count.

Noise-cancelling scheduling
---------------------------

Each ``tuner.enqueue`` performs one launch. The internal queue keeps up to 50
active candidates and measures a candidate no more than three consecutive
times while alternatives exist when a ``queue`` section enables it. These YAML
defaults reduce thermal, frequency, and operating-system noise. Warm-up
launches are scheduled but excluded from the measured result. Without a queue,
accepted strategy recommendations are measured directly after mandatory
constraints and optional horizon rejection.

Measured tuning calls require a timing-enabled non-blocking queue. An
``offline`` winner or a completed ``online_fixed`` winner may instead use a
timing-disabled queue when ``replay_fast_path`` is explicitly enabled; that
production replay bypasses the timer and runtime-history instrumentation.
