Choose how long to tune
=======================

Use ``online_fixed`` for a finite learning phase, ``online_adaptive`` for a
changing workload, and ``offline`` to reuse a saved result. These names are
YAML spellings; C++ uses ``TuningMode::onlineFixed``, ``onlineAdaptive``, and
``offline``.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Mode
     - Behaviour
     - ``completed()`` means
   * - ``online_fixed``
     - Measure candidates until a terminal guard, then replay the best.
     - The finite learning phase ended. Check that a valid winner exists.
   * - ``online_adaptive``
     - Keep proposing and measuring with rolling histories.
     - The optional horizon was reached; adaptation continues. Without a
       horizon it stays false.
   * - ``offline``
     - Load compatible measured history and replay its best candidate.
     - A terminal replay state is ready after initialization.

The tutorial uses fixed mode. It sets ``maximumExecutions`` as a guard;
alternatively set ``maximumRetiredConfigurations``. Fixed mode requires at
least one. It may finish earlier when all legal candidates have retired.
``runsPerCandidate`` is its per-candidate measurement cap, and must fit in
``historyWindowSize``. Confidence checks may retire candidates earlier.

Keep your application loop
--------------------------

A simulation should still run its required time steps:

.. code-block:: cpp

   for (std::size_t step = 0u; step < numberOfSteps; ++step)
       tuner.enqueue(queue, frameSpec, bundle);

Use ``while (!tuner.completed())`` only when a dedicated **fixed-mode training
run** should end with tuning. A horizon-less adaptive tuner never completes,
so that loop would run indefinitely. A horizon is an admission/cooling schedule,
not a launch budget and not a transition to winner-only replay.

For continuous adaptation, start from a fresh ``TunerConfig{}``, whose mode is
``onlineAdaptive`` with neither fixed-mode guards nor a horizon. If you change
a fixed policy to adaptive, reset ``maximumExecutions`` and
``maximumRetiredConfigurations`` first. The strategy then controls revisits
directly. Add a horizon
only when you want the admission schedule described in
:doc:`execution_reference`.

Replay without measurement
--------------------------

Set ``config.replayFastPath = true`` for offline or fixed-mode production
replay. It removes timing events, tracked execution-budget updates, strategy
work, and sample/persistence updates from terminal launches. These calls may
use a timing-disabled queue and can return before the kernel completes.
Keep dependencies ordered on that queue or wait explicitly before using results.
Successful launches still appear in ``history()``.

Online timing calls need a timed non-blocking queue until a terminal winner
exists. Leave the tutorial's queue timed throughout if you want one queue for
both phases. ``replayFastPath`` is invalid in adaptive mode. Without that option,
a timing-metric tuner still requires a timed queue for replay.

Saved history seeds a new online learning phase; it does not resume an old
execution budget or switch an online tuner immediately into offline replay.
See :doc:`history_workflows` for runnable two-process examples and
:doc:`execution_reference` for the exact admission and restart contract.
