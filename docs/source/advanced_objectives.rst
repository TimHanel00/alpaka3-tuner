Advanced objective examples
===========================

The examples below are complete starting points for application objectives
and post-evaluation constraints. Both use the existing interfaces described in
:doc:`custom_metrics` and :doc:`post_evaluation_validation`.

Adaptive Pi: swap accuracy and runtime
--------------------------------------

Build with ``alpakaTune_BUILD_EXAMPLES=ON`` and run the same executable with
either objective:

.. code-block:: sh

   cmake --build build --target alpakaTune_adaptivePi
   build/example/adaptivePi/alpakaTune_adaptivePi --objective accuracy --runtime-budget-ms 10
   build/example/adaptivePi/alpakaTune_adaptivePi --objective runtime --max-error 1e-3

Accuracy mode minimizes absolute Pi error and accepts only calculations taking
at most 10 ms. Runtime mode uses the default timing metric and accepts only
results with absolute Pi error at most ``1e-3``. Both limits are command-line
arguments with these defaults. The runtime budget applies to each complete
calculation, not the whole search, and is checked after execution.

The same calculator and search loop cover 320 configurations with two tunable
host-kernel arguments:

* maximum split depth: 2, 4, 6, 8;
* samples per final boundary tile: 32 through 2560, in steps of 32.

A parallel tile kernel retains squares with four inside corners, discards
squares whose nearest point is outside, and subdivides boundary squares.
Monte Carlo sampling occurs only in unresolved tiles at the maximum depth.
Retired tiles never re-enter the frontier. Each final boundary tile is a block
work item; its threads sample points and reduce their hits in shared memory.
Each retained leaf writes one area-weighted value into a 2D contribution buffer,
which is summed with ``onHost::reduce``. Two alternating active masks keep refinement on the device, with no append
counters or explicit global atomics. Geometric bounds and area
conservation independently check the adaptive calculation.

A single host invocation orchestrates the complete multi-launch workflow; its
outer FrameSpec always has one worker. Inner launches use fixed FrameSpecs.
In runtime mode, synchronized timing of that host invocation includes
all refinement phases, transfers, and reduction. Buffer allocation is outside
the calculation. The example runs enabled device/executor pairings with
available hardware.
Use ``--backend api:deviceKind`` and ``--executor name`` to select a pairing;
for example ``--backend cuda:nvidiaGpu`` in a CUDA build.

See ``example/adaptivePi/README.md``, ``src/adaptivePi.cpp``, and
``src/PiKernel.hpp``, ``src/PiCalculator.hpp``, and ``src/AdaptivePi.hpp``
for the commented implementation. It loosely follows the
quarter-circle sampling idea from `Alpaka's Monte Carlo integration example
<https://github.com/alpaka-group/alpaka/blob/develop/example/monteCarloIntegration/src/monteCarloIntegration.cpp>`_.
Its adaptive refinement and tuning scheme are independent.

Integration tests run both objectives, check feasible winner replay, and verify
that zero runtime or zero error tolerance leaves no valid candidate. One
measurement per candidate bounds test duration. Tests inspect real device leaves
and retain the geometry and accuracy checks, but supply deterministic 5 ms and
20 ms durations through the inspection hook when testing the 10 ms constraint.
This tests acceptance and rejection independently of sanitizer overhead, SYCL
startup, and runner speed. The executable always uses measured durations;
production timing comparisons should use enough repetitions for their noise
level.

Run the integration cases
-------------------------

.. code-block:: sh

   ctest --test-dir build -R alpakaTune_adaptive_pi_tests --output-on-failure

Enable ``alpakaTune_BUILD_TESTING=ON``. The Pi integration case exercises all
enabled backends with available devices.
The algorithm, objectives, and assertions are shared between the executable
examples and their tests.
