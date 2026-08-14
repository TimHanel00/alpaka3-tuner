Post-evaluation validation
==========================

Tuning-space restrictions decide whether a candidate is legal before it runs.
Some requirements can only be checked from the produced result: numerical
error, convergence, memory consumption reported by the algorithm, or another
application-level quality metric. alpakaTune supports this second stage through
the ordered execution history.

After each launch, ``lastConfig()`` returns the configuration that was actually
submitted. It is equivalent to ``history().back()``; it is not the last raw
strategy proposal. Set its shared ``valid`` flag from the application result:

.. code-block:: cpp

   tuner.enqueue(queue, frameSpec, bundle);
   copyResultToHost();

   auto& executed = tuner.lastConfig();
   executed.valid = relativeError(result, reference) <= maximumError;

``history()`` is an in-process, execution-ordered span of all successfully
submitted launches. Every entry contains ``executionIndex``, ``candidateIndex``,
the normalized ``configuration``, an optional measured runtime, and ``valid``.
Entries for the same candidate share one validity state, so invalidating any of
them invalidates that candidate.

The application may revise the flag until the next enqueue. Before selecting
the next launch, the tuner consumes false flags and makes the invalidation
permanent for that tuner:

* an active queued candidate is removed;
* the candidate cannot be admitted again;
* its runtime is excluded from strategy observations and winner selection;
* compact persistence omits it, while complete persistence retains its samples
  with the user-invalidation bitmap for diagnostics.

The execution record remains in ``history()`` as an audit of work that really
ran. A failed launch does not create an entry. Terminal fast-path replays do
create entries, although they still do not add timing samples or update the
tracked execution budget.

Pre-launch and post-launch rejection
------------------------------------

A tuning-space restriction is evaluated during admission. A rejected proposal
never runs and therefore never appears in ``history()``. Application
invalidation happens only after a successful launch and remains visible in the
history. ``TunerInfo`` reports these paths separately through
``rejectedCandidateCount`` and ``restrictionRejectedCount`` versus
``userInvalidatedCandidateCount`` and ``userInvalidatedRejectedCount``.

In ``online_fixed`` mode, invalid candidates count as resolved. If no measured
candidate remains valid, the tuner completes with
``TunerCompletionReason::noValidConfiguration``. ``bestCandidateIndex()`` and a
later enqueue then fail instead of replaying a known-invalid configuration.

Accuracy-versus-runtime example
-------------------------------

The ``postEvaluation`` example tunes the number of terms used by a numerical
approximation of pi. Small term counts run quickly but fail the application's
error threshold. It invalidates those executions and selects the fastest term
count that still satisfies the numerical requirement:

.. code-block:: console

   cmake -S . -B build -DalpakaTune_BUILD_EXAMPLES=ON
   cmake --build build --target alpakaTune_postEvaluation
   ./build/example/postEvaluation/alpakaTune_postEvaluation

The complete source is
``example/postEvaluation/src/postEvaluation.cpp``. The same pattern applies to
fixed exhaustive searches over approximation levels, sample counts, or other
algorithmic parameters whose validity is known only after execution.

Validity is independent of the minimized objective. A tuner constructed with
:doc:`custom_metrics` can attach its application objective through
``provideMetric()`` and still invalidate the same last execution when a
separate correctness requirement fails.
