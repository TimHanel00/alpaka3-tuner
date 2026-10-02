Integrate into a real application
=================================

Keep the tuner outside the repeated launch loop. Reconstructing it each step
restarts scheduling and run counters. Use separate tuners for different
kernels, devices, or workload identities; sharing a ``TunerConfig`` shares
settings, not their runtime state.

Supply current arguments
------------------------

You can rebuild the prototype with current buffers or other ordinary arguments
at each call while keeping its type, launch prototype, and workload identity
consistent:

.. code-block:: cpp

   for (std::size_t step = 0u; step < numberOfSteps; ++step) {
       auto bundle = alpaka::KernelBundle{
           Kernel{}, currentInput, currentOutput, extent, batchSize};
       tuner.enqueue(queue, frameSpec, bundle);
       std::swap(currentInput, currentOutput);
   }

All tunable markers must still be present. A material change to the problem
size, algorithm, or validity conditions should use another tuning context.
See :doc:`history_workflows` for explicit identity entries.

Queue ordering and measurement
------------------------------

For the default runtime objective, create a non-blocking timed queue:

.. code-block:: cpp

   auto queue = device.makeQueue(
       alpaka::queueKind::nonBlocking, alpaka::timing::enabled);

Measured calls submit start/kernel/end events on that queue and wait for the
measurement. CUDA/HIP use native events, SYCL uses profiling timestamps, and
the host backend measures its queue timeline with a host clock. Input copies,
logging, and checks outside the interval are not part of the kernel sample.

Use the same queue for dependent transfers and kernels where practical. If an
input was produced on another queue, wait for that producer or establish an
explicit dependency **before** the tuned launch. Before copying results on a
separate queue, wait for the queue that produced them. A wait on an unrelated
queue does not order either operation.

Do not rely on measured-call synchronization for production correctness:
terminal fast-path replay and custom-metric launches may return asynchronously.
Order dependent work on the producing queue, or wait before reading its result.

Inspect progress without parsing logs
-------------------------------------

.. code-block:: cpp

   auto observation = tuner.enqueueObserved(queue, frameSpec, bundle);
   auto info = tuner.info();
   if (observation.runtimeSeconds)
       reportRuntime(*observation.runtimeSeconds);

``enqueueObserved`` still performs exactly one launch. Its optional timing
field is empty for unmeasured launches and custom objectives. ``info`` is a
read-only snapshot: candidate count, tracked execution count, rejection counts,
and selection/completion diagnostics. ``lastConfig()`` identifies the launch
that actually ran; ``history()`` contains successful executions in order.
Adaptive horizon completion does not imply a terminal winner.

When is online tuning worthwhile?
---------------------------------

Synchronization, strategy work, exploration, and statistics add application
cost beyond the measured kernel runtime. The existing short-kernel diagnostic
uses a 200-microsecond threshold and a representative 20--40-microsecond
instrumentation estimate; actual overhead depends on the backend and workload.
Treat that estimate as guidance, not a measured guarantee for your application.
The warning is emitted once per context and retained in
``info().instrumentationOverheadWarning``.

Compare total application time, including training and replay, with a sensible
fixed configuration. Tiny or infrequently called kernels often benefit more
from :doc:`saved-history replay <history_workflows>` than continuous tuning.
