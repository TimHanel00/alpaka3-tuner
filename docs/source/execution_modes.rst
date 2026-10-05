Choose exploration and selection independently
==============================================

``exploration`` controls whether new configurations may be searched.
``selection`` controls whether the selected configuration keeps adapting to
measured performance. Offline adaptive selection can measure known configurations
without starting another search.

.. list-table::
   :header-rows: 1

   * - Exploration
     - Selection
     - Behaviour
   * - ``online``
     - ``fixed``
     - Search until completion, then lock the best valid measured configuration.
   * - ``online``
     - ``adaptive``
     - Search while permitted, then continue adapting among known configurations.
   * - ``offline``
     - ``fixed``
     - Load compatible measured history and replay its best valid configuration.
   * - ``offline``
     - ``adaptive``
     - Load compatible measured history, remeasure it, and switch as scores change.

.. code-block:: cpp

   config.exploration = alpakaTune::ExplorationPolicy::offline;
   config.selection = alpakaTune::SelectionPolicy::adaptive;
   config.adaptiveProbeInterval = 10u;

Online fixed selection requires ``maximumExecutions`` or
``maximumRetiredConfigurations``. Online adaptive selection accepts either limit
but can also explore indefinitely. Limits end exploration without stopping
adaptive selection. A configured ``horizon`` only controls admission and cooling;
reaching it does not end exploration.

For a bounded search followed by ongoing adaptation, use a complete policy file
such as:

.. code-block:: yaml

   schema_version: 4
   tuning:
     exploration: online
     selection: adaptive
     strategy: exhaustive
     runs_per_candidate: 3
     maximum_executions: 1000
     adaptive_probe_interval: 10
     history_window_size: 10

Up to 1,000 exploration launches can search and measure configurations.
Later launches reuse the measured set, without new strategy recommendations.
Automatic spaces may finish exploration sooner when generation stops and the
registered pool is measured. Their generation budgets are configured separately
under ``space``; see :doc:`automatic_spaces`. To begin directly with reuse of
saved measurements, choose offline/adaptive as in :doc:`history_workflows`.

``completed()`` and ``isTuningComplete()`` mean exploration has ended.
``completionReason()`` explains why. ``info().explorationComplete`` reports the
same status; ``info().selectionLocked`` identifies a valid fixed replay winner.
Adaptive selection keeps running after exploration completes.
``bestCandidateIndex()`` and winner accessors then report the current best;
their value can change with adaptive selection.

Reuse known configurations
--------------------------

After exploration ends, adaptive selection measures the current best on ordinary
launches and probes a known alternative on every tenth successful reuse launch.
``adaptiveProbeInterval`` changes this interval. Alternatives rotate by candidate
index, excluding the current best, rejected candidates, and configurations without
measurements. A single eligible configuration is measured continuously.

Each measurement refreshes the existing rolling history. The best robust estimate
selects subsequent ordinary launches. Probes bypass search strategies, horizon
admission gates, candidate generation, and queued warm-ups. Queue scheduling still
applies during online exploration. Invalidating a known configuration excludes it
from both selection and probes. No eligible measured configuration is an error.

Offline exploration requires compatible measured history. History read/write
permissions are independent: set ``write: false`` to adapt without changing the
input file. Workload changes can arrive through ordinary kernel inputs or custom
metrics within the tuner's existing kernel and launch context.

Keep your application loop
--------------------------

.. code-block:: cpp

   for (std::size_t step = 0u; step < numberOfSteps; ++step)
       tuner.enqueue(queue, frameSpec, bundle);

An application owns its launch count. Use ``while (!tuner.completed())`` only for
a dedicated finite exploration run. Online adaptive exploration without a limit
may never complete.

Replay without measurement
--------------------------

``config.replayFastPath = true`` requires fixed selection. Once the winner is
locked, launches bypass timing events, execution-budget updates, strategy work,
and sample/persistence updates. A timing-disabled queue is then allowed; calls
may return before the kernel completes. Keep queue dependencies ordered or wait
before using results. Successful launches remain visible in ``history()``.

Adaptive timing measurements require a timed queue even with offline exploration.
Custom objectives retain the ``provideMetric()`` contract. See
:doc:`history_workflows` and :doc:`execution_reference` for reuse and restart details.
