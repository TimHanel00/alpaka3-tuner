Tune compiled alternatives
==========================

Use ``RVals`` when a kernel accepts a runtime value. Use ``CVals`` when you
want separately compiled alternatives, such as an unroll factor or a SIMD
width. Every alternative must still implement the required computation.
More compile-time combinations increase compilation time and binary size;
start with a few meaningful choices.

Try the same tutorial with CVals
--------------------------------

The tutorial kernel accepts both an integer and an integral-constant argument.
Its CMake option changes only the candidate declaration:

.. code-block:: cpp

   auto candidates = batchSize(alpakaTune::CVals<32u, 64u, 128u>{});
   auto tunables = alpakaTune::TunableBundle{candidates};

Build in a separate directory:

.. code-block:: sh

   cmake -S docs/examples/first_tuner -B build-tutorial-ct \
     -DCMAKE_BUILD_TYPE=Release -DalpakaTune_TUTORIAL_COMPILE_TIME=ON
   cmake --build build-tutorial-ct --target first_tuner --parallel 4
   ./build-tutorial-ct/first_tuner

It still checks the same 20 launches and exposes three candidates. Compilation
produces the alternatives; execution selects among them. The tuner does not
invoke a compiler during tuning.

Inside a generic kernel, ``decltype(argument)::value`` provides the selected
``CVals`` constant for ``if constexpr`` or template arguments. Ordinary
arithmetic can use its conversion to the underlying value, as the tutorial
does. That type-dependent expression would not work with an ``RVals`` integer.

Runtime, compile-time, and launch candidates can coexist in one bundle. Their
Cartesian product is searched by the same tuner; only the compile-time
combinations need compiled launch variants. Changing candidate kinds also
changes the context identity, so recollect history before offline replay.

Type candidates and vectors
---------------------------

Use ``CTypes`` when a candidate is itself a type. For compile-time launch
vectors, both Alpaka ``CVec`` and ``std::integer_sequence`` are accepted:

.. code-block:: cpp

   using Small = alpaka::CVec<std::size_t, 32u>;
   using Large = std::integer_sequence<std::size_t, 64u>;
   auto tunables = alpakaTune::TunableBundle{
       alpakaTune::tuneFrameExtent(frame, alpakaTune::CTypes<Small, Large>{})};

The sequence is materialized as a ``CVec`` for launch and restriction
predicates. This tunes the frame extent and leaves frame count fixed. Add
``tuneNumFrames`` for an independent second dimension, or a relation when
coverage must be preserved.

Multidimensional vectors tune components independently by default.
Pass ``mdPolicy::listed`` when constructing the tunable to compile only the
listed vectors, or ``mdPolicy::independent`` to make the default explicit.
See the multidimensional examples in :doc:`launch_tuning`.
