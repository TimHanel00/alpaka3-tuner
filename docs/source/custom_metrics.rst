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

   tuner.enqueue(queue, frameSpec, bundle);
   auto const energy = readEnergyCounter();
   tuner.provideMetric(energy);

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
