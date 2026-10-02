Common integration problems
===========================

Start by checking the mode, queue, and workload identity. ``tuner.info()`` and
``tuner.lastConfig()`` usually distinguish scheduling from kernel failures.

A training loop never ends
--------------------------

``online_adaptive`` without a horizon never reports ``completed()``. With a
horizon it reports schedule completion but keeps adapting. Use your application's
own loop bound, or choose fixed mode for a dedicated finite training run.
See :doc:`execution_modes`.

YAML is rejected
----------------

Use schema 3 and include ``tuning.runs_per_candidate`` even for offline mode.
Fixed mode requires a completion guard; adaptive mode accepts a horizon instead
of fixed guards. ``runs_per_candidate`` must fit in ``history_window_size`` in
fixed mode. ``queue.max_consecutive_runs`` must exceed ``queue.warmup_runs``.
The files in :doc:`history_workflows` are complete runnable policies.

No offline winner or cache not loaded
-------------------------------------

Finish a collection process normally before starting replay. Check its working
directory and ``history.file`` path. Collection and replay must use the same
kernel-bundle type, device, launch prototype, tunables, restrictions, metric,
and identity entries. Different problem-size labels or changed candidate
kinds produce different contexts. Unsupported old file schemas need a fresh
collection. ``loadedFromCache()`` reports compatible restored state, not the
mere existence of a JSON file.

A timed launch rejects the queue
--------------------------------

Use a timing-enabled **non-blocking** queue on the tuner's device for online
runtime measurement. A timing-disabled queue requires an initialized terminal
winner and ``replayFastPath`` for a timing-metric tuner. A custom objective can
use a timing-disabled queue, but the application must wait for its measurements
and submit a score. See :doc:`custom_metrics`.

Results change when tuning geometry
-----------------------------------

Verify coverage and bounds for every candidate. Frame extents are logical
shapes, not physical thread counts. Vector components recombine independently;
restrict dependent combinations. If correctness is known only after execution,
set ``lastConfig().valid = false`` before the next enqueue. If all candidates
are invalid, there is no safe winner to replay.

A custom candidate disappears
-----------------------------

Submit ``provideMetric`` or ``provideMetrics`` after the producing work is
ready and before the next enqueue. Omission rejects the previous candidate;
an invalid score is not replaced with zero. See the last-enqueue contract in
:doc:`custom_metrics`.

Performance is worse with tuning
--------------------------------

Check total runtime, space size, warm-ups, and measurement overhead. Reduce the
search to meaningful candidates; consider finite training and fast-path replay.
A runtime sample excludes much of the host-side tuning cost, so it alone does
not establish an application speedup. See :doc:`instrumentation`.
