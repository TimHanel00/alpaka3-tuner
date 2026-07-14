Compile-time tuning
===================

Use ``CVals`` for values that must be part of the kernel argument type. Each
``CVals`` combination is compiled into a launch functor when the context first
sees the prototype bundle. The context keeps those functors in a table keyed
by the selected compile-time configuration, while the normal scheduler chooses
the key together with runtime and launch candidates.

.. code-block:: cpp

   inline constexpr auto width = ALPAKA_TUNE_NAME("simdWidth");
   auto tunables = alpakaTune::Tunables{
       alpakaTune::named(width, alpakaTune::CVals<1u, 2u, 4u, 8u>{})};
   auto prototype = alpaka::KernelBundle{
       Kernel{}, buffers..., alpakaTune::markTunable(width)};

Inside a generic kernel operator, ``decltype(widthArgument)::value`` is a
compile-time constant. ``RVals`` does not change the argument type, whereas
``CVals`` does.

The complete runnable example uses Alpaka3 SIMD width:

.. literalinclude:: ../../examples/compile_time_tuning/src/main.cpp
   :language: cpp
   :caption: examples/compile_time_tuning/src/main.cpp
