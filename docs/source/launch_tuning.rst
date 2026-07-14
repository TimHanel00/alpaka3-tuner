Launch tuning
=============

A context owns the tuning state for one device and one kernel bundle type.
Construct it from named ``Tunables`` and pass the device plus optional,
identity-only Alpaka or application entries. Alpaka ``deviceKind``, ``api``,
``Device``, and executor objects use their Alpaka name; strings and values
accepted by ``std::to_string`` are also accepted.

.. code-block:: cpp

   inline constexpr auto scale = ALPAKA_TUNE_NAME("scale");

   auto tunables = alpakaTune::Tunables{
       alpakaTune::named(scale, alpakaTune::RVals{0, 256, 16})};
   auto context = alpakaTune::contextBuilder().createContextWith(
       tunables, device, alpaka::deviceKind::cpu, alpaka::api::host,
       executor, "vector-add");

   auto prototype = alpaka::KernelBundle{
       Kernel{}, input, output, alpakaTune::markTunable(scale)};
   context.tune(queue, frameSpec, prototype);

``markTunable`` is a placeholder only: it is never sent to the device. The
context rebuilds the Alpaka ``KernelBundle`` with the selected value in every
marked slot before calling ``queue.enqueue``. An unmarked argument remains
fixed. Every non-launch tunable must appear in at least one marker.

``RVals{maximum}``, ``RVals{minimum, maximum}``, and
``RVals{minimum, maximum, step}`` create inclusive integral ranges. For a
non-integral candidate such as an Alpaka vector, spell the element type and
provide a vector of values: ``RVals<Index>{std::vector<Index>{Index{1},
Index{2}}}``.

Reserved launch names
---------------------

The names ``frameExtent``, ``numFrames``, ``numThreads``, and ``numBlocks``
need no marker. Their candidates replace the corresponding field of the
``FrameSpec`` passed to ``tune``. Frame and thread candidates participate in
the same Cartesian tuning space. A thread specification is derived from the
resolved frame specification using Alpaka3's internal adjustment operation;
explicit block/thread candidates then override the derived fields.

Choose launch candidates that are valid for the selected Alpaka executor. For
example, ``CpuSerial`` and ``CpuOmpBlocks`` require a thread-block extent of
one, so ``numBlocks`` is their meaningful physical launch parameter. A CUDA or
HIP context can additionally tune ``numThreads`` (usually warp-aligned values)
and may tune ``numBlocks`` when the kernel is valid for every Cartesian pair.
``numFrames`` is the portable logical alternative; it commonly maps to host
blocks. ``frameExtent`` changes only the logical decomposition, not a direct
thread count.

Noise-cancelling scheduling
---------------------------

Each ``context.tune`` performs one launch. The internal queue keeps up to 50
active candidates and measures a candidate no more than three consecutive
times while alternatives exist. These YAML defaults reduce thermal, frequency,
and operating-system noise. Warm-up launches are scheduled but excluded from
the measured result.
