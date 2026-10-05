Common integration problems
===========================

Start by checking both policies, the queue, and workload identity. ``tuner.info()`` and
``tuner.lastConfig()`` usually distinguish scheduling from kernel failures.

A training loop never ends
--------------------------

Online adaptive exploration without a completion limit can keep searching
indefinitely. A cooling horizon does not report ``completed()``. Use your
application's own loop bound for ongoing adaptation, or add an exploration
completion limit for a finite search. Choose fixed selection if later launches
should replay a locked winner.
See :doc:`execution_modes`.

YAML is rejected
----------------

Use schema 4 and include ``tuning.runs_per_candidate`` even for offline exploration.
Replace the removed ``tuning.mode`` with ``exploration`` and ``selection``;
see the migration table in :doc:`configuration`.
Online fixed selection requires a completion guard; online adaptive selection
also accepts completion guards. Offline exploration rejects these guards.
A horizon requires online/adaptive. ``runs_per_candidate`` must fit in
``history_window_size`` with fixed selection. ``adaptive_probe_interval``
must be positive. ``replay_fast_path`` requires fixed selection.
``queue.max_consecutive_runs`` must exceed ``queue.warmup_runs``.
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
Offline/adaptive needs valid measured configurations as well; a saved catalog
of unmeasured points is insufficient. It cannot generate replacements when all
known configurations have been invalidated or rejected for missing metrics.

Completed, but the selected configuration changes
-------------------------------------------------

``completed()`` reports that exploration ended. Adaptive selection continues
remeasuring known configurations and may change its winner. Check
``info().selectionLocked`` for fixed replay. Both exploration and selection
are shown in ``info()``; see :doc:`execution_modes`.

An automatic space stops before every domain value is tested
------------------------------------------------------------

Check ``info().space.state`` and ``completionReason()``. Catalog budgets,
plateau detection, and bounded generation attempts can end generation before
full domain exhaustion. ``registeredCandidateCount`` describes the current
pool; ``domainExhausted`` reports whether exhaustion is proven. Increase
generation or execution budgets when more exploration is needed, or refine
the domain and restrictions. See :doc:`automatic_spaces`.

A timed launch rejects the queue
--------------------------------

Use a timing-enabled **non-blocking** queue on the tuner's device for runtime
measurement during exploration and adaptive reuse, including offline/adaptive.
A timing-disabled queue requires an initialized fixed winner and
``replayFastPath`` for a timing-metric tuner. A custom objective can
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
