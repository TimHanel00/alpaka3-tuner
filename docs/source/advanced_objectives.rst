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

The same calculator and search loop cover 320 independent configurations:

* maximum split depth: 2, 4, 6, 8;
* points per final boundary tile: 32, 64, 128, 256, 512;
* inner FrameSpec frame count: 1, 2, 4, 8;
* inner FrameSpec frame extent: 32, 64, 128, 256.

A parallel tile kernel retains squares with four inside corners, discards
squares whose nearest point is outside, and subdivides boundary squares.
Monte Carlo sampling occurs only in unresolved tiles at the maximum depth.
Retired tiles never re-enter the frontier. Geometric bounds and area
conservation independently check the adaptive calculation.

A single host invocation orchestrates the complete multi-launch workflow; its
outer FrameSpec always has one worker. The geometry parameters control inner
launches. In runtime mode, synchronized timing of that host invocation includes
all refinement phases, transfers, and reduction. Buffer allocation is outside
the calculation. Add ``--cuda`` in a CUDA build to run tile evaluation on a GPU.

See ``example/adaptivePi/README.md``, ``src/adaptivePi.cpp``, and
``src/AdaptivePi.hpp`` for the commented implementation. It loosely follows the
quarter-circle sampling idea from `Alpaka's Monte Carlo integration example
<https://github.com/alpaka-group/alpaka/blob/develop/example/monteCarloIntegration/src/monteCarloIntegration.cpp>`_.
Its adaptive refinement and tuning scheme are independent.

Integration tests run both objectives, check feasible winner replay, and verify
that zero runtime or zero error tolerance leaves no valid candidate. One
measurement per candidate bounds test duration; production timing comparisons
should use enough repetitions for their noise level.

CUDA vector-add: occupancy as the objective
-------------------------------------------

With ``alpaka_DEP_CUDA=ON`` and ``alpakaTune_BUILD_EXAMPLES=ON``:

.. code-block:: sh

   cmake --build build-cuda --target alpakaTune_cudaOccupancy
   build-cuda/example/cudaOccupancy/alpakaTune_cudaOccupancy

This independent example launches a native vector-add function through
Alpaka's public ``enqueueNativeFn`` hook. It measures five block sizes with
CUDA's `occupancy API
<https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__OCCUPANCY.html>`_
and provides ``1 - predictedOccupancy`` through a custom-metric tuner. This
maximizes predicted occupancy using the tuner's minimization contract.

The function passed to the occupancy query is the actual launched function.
The example uses the selected device's maximum threads per multiprocessor and
the kernel's actual dynamic shared-memory size. It avoids depending on Alpaka's
internal wrapper types. Source guards require CUDA compilation and the enabled
Alpaka CUDA target. Runtime additionally requires a GPU; the CTest case reports
an explicit skip if hardware is unavailable.

Predicted occupancy describes potential residency. It is neither measured
achieved occupancy nor evidence that the chosen block size is fastest.
``example/cudaOccupancy/README.md`` and ``src/cudaOccupancy.cpp`` document the complete
pattern. The same source is tested, including full vector output checks, changing
inputs, actual candidate propagation, score storage, maximum-occupancy winner
selection, and terminal replay.

Achieved occupancy can also be supplied by a profiler-backed collector: use
``1 - achievedPercent / 100`` as the custom objective after collection finishes.
Nsight Compute reports ``sm__warps_active.avg.pct_of_peak_sustained_active``;
collecting it requires hardware-counter access through Nsight Compute or
CUPTI/NVPerf and may replay kernels. See the `NVIDIA profiling guide
<https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html>`_. The executable
above queries predicted occupancy and does not implement that counter collector.

Run the integration cases
-------------------------

.. code-block:: sh

   ctest --test-dir build -R 'alpakaTune_(adaptive_pi|cuda_occupancy)_tests' --output-on-failure

Enable ``alpakaTune_BUILD_TESTING=ON``. CUDA builds also include a Pi GPU case.
The algorithm, objectives, and assertions are shared between the executable
examples and their tests.
