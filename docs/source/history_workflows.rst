History workflows
=================

Persistence access, execution mode, and proposal strategy are independent
choices. This page gives complete configurations for the most common two-run
workflows and explains which state crosses the process boundary.

Parameter ownership
-------------------

The mode-specific fields are deliberately disjoint:

.. list-table::
   :header-rows: 1

   * - Mode
     - Mode-specific fields
     - Meaning
   * - ``online_fixed``
     - ``maximum_executions`` and
       ``maximum_retired_configurations``
     - Terminal guards. At least one must be present.
   * - ``online_adaptive``
     - ``horizon`` and
       ``horizon_offset_with_active_history``
     - Schedule length and history-aware starting point. Neither is a
       terminal application budget.
   * - ``offline``
     - none
     - Replays the best configuration from compatible measured history.

Mixing these fields is a configuration error. In particular,
``maximum_executions`` no longer doubles as an adaptive schedule parameter.
The surrounding application still owns its total number of kernel launches in
every mode.

The sampling fields also have different roles. In ``online_fixed``,
``runs_per_candidate`` is a hard per-candidate measurement cap. In
``online_adaptive``, one admitted residency records
``max_consecutive_runs - warmup_runs`` timings. For example, a warm-up count of
one and a consecutive-run count of four record exactly three timings during
each residency. A later revisit starts another residency and may replace old
samples in the rolling ``history_window_size``.

Collect a fresh learned history
-------------------------------

The first run can ignore a pre-existing file while still publishing its result
once at normal process shutdown:

.. code-block:: yaml

   schema_version: 3
   tuning:
     mode: online_adaptive
     strategy: learned_hybrid
     random_seed: 17
     warmup_runs: 1
     max_consecutive_runs: 4
     history_window_size: 12
     horizon: 40000
     horizon_offset_with_active_history: 0.8
   history:
     file: learned-history.json
     read: false
     write: true
     sample_count: 100
   complete_history:
     file: learned-complete-history.json
     read: false
     write: true
   learning:
     model: /path/to/model.atml
     fallback: random
     candidate_pool_size: 4096
     candidate_batch_size: 256

``read: false`` means that an old file cannot influence this run. ``write:
true`` does not cause per-launch file writes: both in-memory stores flush once
during normal shutdown. Existing files are replaced because their contents
were explicitly excluded from this run.

Continue learned adaptation without modifying history
------------------------------------------------------

The second run uses the same tuner fingerprint, model artifact, and history
path, but changes only the access policy:

.. code-block:: yaml

   schema_version: 3
   tuning:
     mode: online_adaptive
     strategy: learned_hybrid
     warmup_runs: 1
     max_consecutive_runs: 4
     history_window_size: 12
     horizon: 40000
     horizon_offset_with_active_history: 0.8
   history:
     file: learned-history.json
     read: true
     write: false
   complete_history:
     file: learned-complete-history.json
     read: true
     write: false
   learning:
     model: /path/to/model.atml
     fallback: random

Compatible measured history activates the history-aware schedule. If ``q`` is
the new run's progress from zero to one, the unchanged sigmoid and Boltzmann
functions receive

.. math::

   x = h + (1-h)q,

where ``h`` is ``horizon_offset_with_active_history``. With the default 0.8,
the second run begins at schedule position 0.8 and stretches the remaining
0.2 across all new ``horizon`` launches. Candidate run counts and execution
counters restart at zero, while the retained timings remain in their original
rolling histories. Those old samples inform the strategy but do not consume
the new run's measurement or horizon counts.

The learned residual adapter resumes from compact history first. Both files
store its
coefficients, retained residual observations, partial update-batch progress,
and update count. Restore occurs only when the tuner fingerprint and exact
model-artifact digest are compatible. Timing history remains usable if this
optional adapter state is absent or invalid. A valid compact adapter wins over
the duplicated complete-history state. ``write: false`` guarantees that both
input files remain unchanged even though the in-memory adapter continues
learning during the second run.

Retune online-fixed with history
--------------------------------

A completed fixed run may also seed another fixed run. Both runs use the same
mode and limits; only the history access policy needs to change:

.. code-block:: yaml

   schema_version: 3
   tuning:
     mode: online_fixed
     strategy: learned_hybrid
     warmup_runs: 1
     runs_per_candidate: 4
     minimum_runs_per_candidate: 4
     noise_cancellation_window: 20
     max_consecutive_runs: 5
     maximum_executions: 20000
     maximum_retired_configurations: null
     history_window_size: 12
   history:
     file: fixed-history.json
     read: true
     write: false
   complete_history:
     file: fixed-complete-history.json
     read: true
     write: false
   learning:
     model: /path/to/model.atml

For a fresh first run, set both ``read`` flags to ``false`` and the desired
``write`` flags to ``true``. In the second run above, each candidate keeps its
loaded timing window and the learned strategy restores its compatible adapter.
Nevertheless, every candidate's run-local sample count, the global execution
count, scheduler residency, and terminal state start again from zero. The
retained samples therefore inform estimates, but each candidate can still
contribute four new measurements before its current-run cap is reached. The
earlier terminal winner is not replayed unless the second run independently
reaches a terminal condition.

Collect random, then replay offline
-----------------------------------

A strategy-independent history can first be collected with random proposals:

.. code-block:: yaml

   schema_version: 3
   tuning:
     mode: online_adaptive
     strategy: random
     random_seed: 17
     warmup_runs: 1
     max_consecutive_runs: 4
     history_window_size: 12
     horizon: 40000
     horizon_offset_with_active_history: 0.8
   history:
     file: random-history.json
     read: false
     write: true

The subsequent replay needs no schedule or fixed-mode limit:

.. code-block:: yaml

   schema_version: 3
   tuning:
     mode: offline
     strategy: random
   history:
     file: random-history.json
     read: true
     write: false

``offline`` selects the best robust estimate already present in the history.
It performs no warm-ups, measurements, strategy recommendations, residual
updates, or persistence writes. Strategy and mode are intentionally excluded
from the tuner fingerprint, so measured history may cross this boundary while
the kernel, device, launch prototype, tuning dimensions, and restrictions must
still match.

Access-policy summary
---------------------

.. list-table::
   :header-rows: 1

   * - Goal
     - ``read``
     - ``write``
     - Filesystem result
   * - Fresh collection
     - ``false``
     - ``true``
     - Ignore old contents; publish this run once at shutdown.
   * - Read-only continuation or replay
     - ``true``
     - ``false``
     - Load compatible state; leave the file unchanged.
   * - Read-modify-write continuation
     - ``true``
     - ``true``
     - Load and merge; publish the newest snapshot once at shutdown.
   * - In-process-only tuning
     - either
     - either
     - Omit ``file``; no filesystem access occurs.

Normal shutdown is required for a requested write. Abrupt process termination
cannot guarantee that the final JSON snapshot is emitted.
