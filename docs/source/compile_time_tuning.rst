Compile-time tuning
===================

Use ``CVals`` for values that must be part of the kernel argument type. Each
``CVals`` combination is compiled into a launch functor when the tuner first
sees the prototype bundle. The tuner keeps those functors in a table keyed
by the selected compile-time configuration, while the normal scheduler chooses
the key together with runtime and launch candidates.

.. code-block:: cpp

   inline constexpr auto width = ALPAKA_TUNE_TUNABLE("simdWidth");
   auto tunables = alpakaTune::TunableBundle{
       width(alpakaTune::CVals<1u, 2u, 4u, 8u>{})};
   auto prototype = alpaka::KernelBundle{
       Kernel{}, buffers..., width};

Inside a generic kernel operator, ``decltype(widthArgument)::value`` is a
compile-time constant. ``RVals`` does not change the argument type, whereas
``CVals`` does.

Use ``CTypes`` when a candidate is itself a compile-time type. This applies to
ordinary kernel parameters and to the reserved ``frameExtent``, ``numFrames``,
``numBlocks``, and ``numThreads`` launch parameters. Both Alpaka ``CVec`` types
and ``std::integer_sequence`` types are accepted; an integer sequence is
materialized as the corresponding ``CVec`` when it is used by a launch or a
constraint predicate.

Every component is an independent backend dimension. Candidate component
sets are deduplicated before their Cartesian product is compiled:

.. code-block:: cpp

   inline constexpr auto tile = ALPAKA_TUNE_TUNABLE("tile");

   auto tunables = alpakaTune::TunableBundle{
       tile(alpakaTune::CTypes<
           alpaka::CVec<std::size_t, 8u, 2u>,
           std::integer_sequence<std::size_t, 16u, 4u>>{})};

This compiles the four reconstructed vectors ``{8,2}``, ``{8,4}``,
``{16,2}``, and ``{16,4}``. They occupy two normalized strategy coordinates
while the kernel continues to receive one compile-time vector.

.. code-block:: cpp

   using Small = alpaka::CVec<std::size_t, 8u>;
   using Large = std::integer_sequence<std::size_t, 16u>;

   auto tunables = alpakaTune::TunableBundle{
       alpakaTune::Tunable{
           alpakaTune::frameExtent,
           alpakaTune::CTypes<Small, Large>{}}};

``FrameExtentTuning{...}`` is the convenience frontend for exposing these
compile-time frame extents through ``makeTuner`` while deriving the
matching number of frames.

The mirrored Alpaka examples deliberately retain their upstream kernel source
shape. Add compile-time candidates only where an existing kernel argument is
already a semantic compile-time choice; do not create a renamed tuner-only
copy of an example to demonstrate ``CVals``.
