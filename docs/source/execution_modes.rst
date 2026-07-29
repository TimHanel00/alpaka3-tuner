Execution modes
===============

The execution mode owns candidate admission, queue residency, measurement
lifetime, and the transition to production launches. The strategy only
proposes a normalized parameter vector. This separation applies uniformly to
exhaustive, random, simulated annealing, Bayesian optimization, learned hybrid,
and custom strategies.

Application lifetime
--------------------

The application owns the number of kernel launches. It can use a fixed loop,
simulation time steps, convergence of its own result, a wall-time limit, or any
other application-level condition. Tuner limits and horizons do not impose an
application lifetime.

``Tuner::completed()`` is optional policy information an application may
consult. In ``online_fixed`` and ``offline`` it identifies a terminal tuner
state. In ``online_adaptive`` with a configured ``horizon`` it becomes true at
that horizon only to indicate that the admission sigmoid and cooling schedule
reached their final state. It does not stop adaptation, measurement, revisits,
or residual-adapter updates. Without a horizon it remains false indefinitely.

The examples configure a horizon and demonstrate an application-owned combined
condition: at least 50,000 executions and ``completed()``. Their shipped YAML
uses an adaptive horizon of 40,000, intentionally leaving 10,000 launches at
the final schedule state.
Those numbers are independent. Applications that require exactly N launches
should simply execute exactly N launches and need not inspect ``completed()``.
``isTuningComplete()`` reports the same policy completion. It becomes true at
the adaptive horizon but does not stop the adaptive scheduler. A terminal
reason exists only for offline and fixed-mode terminal states;
``completionReason()`` therefore remains unavailable after an adaptive
horizon.

One history across runs
-----------------------

Every candidate owns exactly one rolling timing history. Loading a compatible
history restores those retained timings in place; there is no separate
strategy-only copy. Strategies immediately see the restored estimates, and a
compatible learned residual adapter is restored into the new strategy
instance.

Starting either online mode creates a new run lifecycle around that same
history:

* each candidate's run-local sample count returns to zero;
* retained timing samples, robust statistics, and rolling-window order remain;
* warm-up and candidate-residency state restart;
* scheduler, execution, retirement, retry, and completion state restart; and
* old terminal reasons and consumed execution budgets are not inherited.

Consequently, restored samples inform decisions but do not satisfy the new
run's ``runs_per_candidate`` requirement. New measurements append to the same
rolling window and replace its oldest timings when
``history_window_size`` is exceeded.

.. list-table::
   :header-rows: 1

   * - Mode
     - First run without readable history
     - Second run with readable history
   * - ``online_fixed``
     - Builds timings and adapter state, then reaches a terminal guard.
     - Retains timings and adapter, resets run counts and tunes again to a new
       terminal guard.
   * - ``online_adaptive``
     - Builds timings while advancing a fresh horizon schedule.
     - Retains timings and adapter, resets run counts, starts at the configured
       history offset, and receives a complete new horizon.
   * - ``offline``
     - Fails because no measured configuration is available.
     - Loads the robust best configuration and replays it without measurement
       or strategy construction.

``online_fixed``
----------------

``online_fixed`` is the finite tuning mode and preserves the original queue
lifecycle. Admitted candidates remain in the active queue and are interleaved.
Every activation runs up to ``max_consecutive_runs`` launches, of which the
first ``warmup_runs`` are not recorded. A candidate retires at its confidence
criterion, Mann-Whitney early-stop criterion, or ``runs_per_candidate`` cap.

The tuner finishes after all legal candidates retire, or when either
``maximum_executions`` or ``maximum_retired_configurations`` is reached. At
least one of these two limits is required. Once tuning finishes, later
``enqueue`` calls launch the best measured configuration without collecting
more samples. The application may still continue until its external run count
is reached.

``maximum_executions`` and ``maximum_retired_configurations`` are exclusive to
this mode. ``runs_per_candidate`` is the maximum number of new retained
measurements contributed by each candidate in the current run;
``minimum_runs_per_candidate`` and ``ci_check_interval`` use that same
run-local count. Statistical estimates and confidence intervals still use the
single retained timing window, including samples loaded at startup.

When a second fixed run reads a history produced by an earlier terminal run,
the old winner and completion reason do not enter replay. All candidates become
eligible for a new measurement lifecycle, the execution guards start from
zero, and the strategy can use the retained timings and adapter while proposing
the new schedule. Only a terminal condition reached in the current run starts
winner replay.

``online_adaptive``
-------------------

``online_adaptive`` is a continuous mode. A horizon is optional. One admission
gives a candidate one queue residency and therefore one activation burst. The
burst still follows the queue configuration exactly. For example:

.. code-block:: yaml

   tuning:
     mode: online_adaptive
     horizon: 40000
     warmup_runs: 1
     max_consecutive_runs: 4

This records three timings per admitted residency: the first launch is a
warm-up and the remaining three are measurements. The candidate leaves the
active queue after that burst. It may be admitted again later; the number of
visits is not limited by the number of retained timings.

Each candidate retains only its newest ``history_window_size`` measurements.
When another measurement exceeds that capacity, the oldest measurement is
discarded. Robust estimates and learned residuals are consequently based on
the current rolling window rather than an indefinitely growing history.

On a second adaptive run, retained timings make candidates revisits from the
first proposal onward, and the restored adapter supplies learned context.
Candidate run counts and both execution counters restart at zero. The loaded
history activates ``horizon_offset_with_active_history`` when a horizon is
configured but never shortens the new run's configured ``horizon``.

Without ``horizon``, every legal revisit proposed by the strategy is reopened
directly after active-queue and restriction checks. The tuner does not apply
the sigmoid revisit gate or relative-score Boltzmann gate; exploration and
exploitation are entirely strategy-driven. This is often the clearest choice
for ``learned_hybrid`` because the learned strategy already ranks candidates
and adapts from runtime observations. Measurement, rolling-history updates,
and residual-adapter updates continue for the application's full lifetime,
while ``completed()`` and ``isTuningComplete()`` remain false. Applications
must not use either query as an exit condition in this horizon-less form.

With ``horizon`` configured, unseen legal candidates are admitted directly. An
already measured candidate must pass two independent gates after active-queue
duplicates and restrictions have been rejected:

.. math::

   q = \operatorname{clamp}\left(\frac{n}{H}, 0, 1\right)

.. math::

   x =
   \begin{cases}
     q, & \text{without active history}\\
     h + (1-h)q, & \text{with active history}
   \end{cases}

.. math::

   p_{revisit}(x) =
   \frac{\sigma(k(x-\tfrac12))-\sigma(-k/2)}
        {\sigma(k/2)-\sigma(-k/2)}

Here, ``n`` is the launch count in the current tuner process run, ``H`` is
``horizon``, ``h`` is
``horizon_offset_with_active_history`` (0.8 by default), and ``k`` is
``revisit_admission_steepness`` (16 by default). Active history means that a
compatible cache containing at least one measured configuration initialized
the tuner. Merely configuring a file or finding an empty context does not
activate the offset.

Without active history, the normalized revisit probability still starts
exactly at zero. With active history, the same unmodified sigmoid and
Boltzmann-temperature functions start at ``h``. The interval from ``h`` to one
is stretched over all ``H`` new launches. Loading history never advances the
new run's execution count or prematurely finishes its horizon.
``TunerInfo::executionCount``, ``adaptiveHorizonExecutionCount``, and
``adaptiveHorizonProgress`` all describe the current process run. Persisted
timings remain in each candidate's rolling history, but online run counters
restart at zero.

The second gate prefers candidates close to the current best robust runtime:

.. math::

   T(x) = T_0\left(\frac{T_1}{T_0}\right)^x

.. math::

   p_{score} = \exp\left(
     -\frac{\max(0, s/s_{best}-1)}{T(x)}\right)

``score_temperature_start`` and ``score_temperature_end`` default to 0.25 and
0.05. The current best therefore always passes the score gate, while slower
configurations become less likely as the temperature cools.

When present, ``horizon`` is the admission and temperature schedule inside the
tuner. After that many launches in the current tuner process run,
``completed()``, ``isTuningComplete()``, ``TunerInfo::tuningComplete``, and the
corresponding ``LaunchObservation`` field become true, but the horizon itself
does not enter the internal terminal replay state. ``maximum_executions`` and
``maximum_retired_configurations`` are rejected because they belong exclusively
to ``online_fixed``.

At and after the horizon, normalized revisit admission remains exactly one.
Relative-score admission continues at ``score_temperature_end`` rather than
collapsing to a winner-only rule. Slightly slower configurations therefore
retain a nonzero Boltzmann probability; for example, with the default final
temperature of 0.05, a candidate that is two percent slower than the current
best is accepted by the score gate with probability
``exp(-0.02 / 0.05)``, approximately 0.67.

Rejected recommendations are retried synchronously. This includes active
duplicates, restrictions, the adaptive revisit gate, and the relative-score
gate. ``maximum_consecutive_strategy_retries`` defaults to 20 and bounds that
work in one refill attempt. Admission resets the streak. When active queue
entries remain, reaching the limit pauses refill until an activation completes
and changes the scheduler context.

If the adaptive queue is empty after one bounded refill attempt, the tuner
reopens the current best measured history entry as a progress fallback. This is
not a terminal winner: it is scheduled through the normal queue, synchronized,
measured, and written into the rolling window. Its execution advances the
horizon, and subsequent refills call the strategy again with access to all
current runtime observations. ``adaptiveRetryFallbackCount`` in ``TunerInfo``
reports how often this path was required. If no configuration has ever been
accepted or restored, there is no safe fallback and the triggering call throws
an explicit initialization error without marking the adaptive tuner complete.

In ``online_fixed`` only, an empty queue at the same retry limit remains a
terminal state with
``TunerCompletionReason::maximumConsecutiveStrategyRetries``. Later fixed-mode
calls replay its best measured configuration without instrumentation.

``offline``
-----------

``offline`` requires a compatible persistent history containing at least one
measured configuration. The history does not need a terminal completion reason
and may have been produced by any strategy or online mode. The tuner selects
the best robust estimate from that history and launches only that
configuration. It does not instantiate a strategy, perform warm-ups, measure
the launch, update the adapter, or update persistence. Because offline mode
does not begin a new measurement lifecycle, it simply consumes the persisted
timing window. An offline first run without readable measured history is an
error.

Strategy and queue boundary
---------------------------

Every strategy recommendation is mapped once to the exact nearest discrete
candidate. The shared tuner admission policy then reports one disposition back
to the strategy: scheduled, accepted active duplicate, restriction rejection,
revisit rejection, or score rejection. A candidate already resident in the
active queue satisfies the recommendation, so refill stops and queue execution
continues without asking the strategy for a substitute. A newly scheduled
candidate likewise ends the refill attempt. Only an actual policy rejection
requests an entirely new proposal; the tuner does not mutate it or search
locally for a nearby substitute.

Learned hybrid commits its selection cycle only when a recommendation starts a
new queue activation. Active duplicates do not consume selection slots, while
rejections advance to another candidate within the same phase. Its default
cycle contains ten predicted-fast activations followed by one
uncertainty/diversity activation.

Strategies may deliberately recommend previously measured points. They must
not hide duplicate ownership inside their own candidate bookkeeping. Regression
tests also require every built-in strategy, including learned hybrid, to expose
at least ten distinct raw candidates among 132 admitted recommendations in a
100-candidate space. Fixed seeds make this strategy contract deterministic.
This is proposal diversity, not a promise to exhaust the entire space.
