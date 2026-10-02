Custom tuning metrics
=====================

By default, alpakaTune minimizes synchronized kernel runtime. Applications can
instead provide a non-negative metric that is only available after the launch,
such as energy consumption, an iteration count, or a larger workflow cost.

Select this interface at tuner construction with ``customMetric()``:

.. code-block:: cpp

   auto tuner = alpakaTune::makeTuner(
       config, tunables, device,
       alpakaTune::customMetric("energy_joules"), "workload-identity");

   auto const before = readEnergyCounter();
   tuner.enqueue(queue, frameSpec, bundle);
   alpaka::onHost::wait(queue);
   auto const energy = readEnergyCounter() - before;
   tuner.provideMetric(energy);

``readEnergyCounter`` is an application-provided collector, not an alpakaTune
function. Wait for the producing queue before reading a result or a counter.
A timing-disabled non-blocking queue is sufficient for this objective.

The ``customMetric()`` token is a compile-time policy even though its label is
a runtime ``std::string``. It is supplied among the identity arguments after
the device and cannot be selected through YAML. Custom-metric tuners compile
out ``KernelTimer`` use and accept timing-disabled queues. Timing remains the
default when no token is present.

alpakaTune minimizes the metric. Values passed to ``provideMetric()`` must be
finite and non-negative. The label is exposed through ``metricName()`` and
``TunerInfo`` and is included in the persistent fingerprint and history
metadata. Histories produced for different metric kinds or labels are not
interchangeable.

Last-enqueue contract
---------------------

``provideMetric(value)`` always targets the most recent successful enqueue on
that tuner. It updates ``lastConfig().metricValue`` and the corresponding
execution-history entry. It cannot provide a value for an older launch, and a
second call for the same launch fails.

During online tuning, call ``provideMetric()`` before the next enqueue. If the
next enqueue begins while the preceding launch still has no metric, alpakaTune
permanently rejects that preceding candidate, removes it from active scheduler
and strategy state, and continues with another candidate. The launch remains
in ``history()`` with no metric as an audit record. ``TunerInfo`` reports this
path separately through ``missingMetricCandidateCount`` and
``missingMetricRejectedCount``.

An explicit ``lastConfig().valid = false`` takes precedence over a missing
metric, so one launch is not classified twice. Offline and terminal winner
replays do not require a metric; ``provideMetric()`` may still attach one to
their execution record, but it is not added to tuning statistics.

``enqueueObserved()`` necessarily returns before an application metric is
provided. Its ``metricValue`` is therefore empty in custom mode. Inspect
``lastConfig()`` after ``provideMetric()`` when the attached value is needed.

Persistence
-----------

Compact histories store the metric kind and name together with
``median_metric_value``. Complete histories additionally retain raw candidate
samples and the ``missing_metric_candidates`` bitmap. Candidates rejected for
a missing metric are excluded from winner selection and compact history while
their prior samples remain available in complete history for diagnostics.

Runnable feature examples
-------------------------

See :doc:`advanced_objectives` for the adaptive Pi example that swaps accuracy
and runtime objectives with post-evaluation constraints.

Scoring multiple inputs
-----------------------

Register a scoring function to combine application-provided measurements:

.. code-block:: cpp

   constexpr double timeWeight = 0.7, energyWeight = 0.3;
   constexpr double timeScale = 0.001, energyScale = 0.01;
   auto objective = alpakaTune::customMetric(
       "weighted_cost_v1", [=](double seconds, double joules) {
           return timeWeight * seconds / timeScale
                + energyWeight * joules / energyScale;
       });
   auto tuner = alpakaTune::makeTuner(
       config, tunables, device, objective,
       "weighted-v1:t=0.7:e=0.3:seconds=0.001:joules=0.01");
   tuner.enqueue(queue, frameSpec, bundle);
   alpaka::onHost::wait(queue);
   tuner.provideMetrics(elapsedSeconds, consumedJoules);

``elapsedSeconds`` and ``consumedJoules`` must be collected by your application
for this launch. The example scales are fixed reference values in seconds and
joules, not automatically estimated from the candidates.

The callable is stored by value without type erasure. Evaluation is explicit;
the tuner neither collects these inputs nor invokes the callable during enqueue.
Inputs can be scalars or an application-defined result object. A single finite,
non-negative score is minimized using the same statistics as scalar metrics.
Normalize different units using fixed positive reference scales. Users choose
weights, minimization transforms, and nonlinear formulas; this is scalar scoring,
not Pareto optimization.

Include the formula version, weights, normalization scales, and input meanings
in the metric name or stable identity entries. Captured state cannot be inferred
automatically. Changing it without changing identity can reuse incompatible
histories. Persistence stores the final score, not the inputs or callable.

The last-enqueue contract also applies to ``provideMetrics()``. Eligibility is
checked before the callable runs, so duplicate submissions do not invoke it.
Exceptions and invalid scores leave the launch awaiting a metric; retry before
the next enqueue, or explicitly invalidate the candidate. Terminal replays may
attach a score without adding it to tuning statistics. Existing
``provideMetric(double)`` remains available on callable-objective tuners.
