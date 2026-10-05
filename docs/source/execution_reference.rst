Exploration and selection reference
===================================

The exploration and selection policies own candidate admission, measurement
lifetime, and the transition to known-configuration reuse. During online
exploration, a strategy proposes a normalized parameter vector for a manual
space or an exact registered ID for an automatic space. Every proposal follows
the same ordered pipeline: mandatory
constraints, optional adaptive horizon rejection, optional queue scheduling,
and one launch. For policy selection and everyday loops, start with
:doc:`execution_modes`.
This reference explains the detailed admission and restart contract.

This separation applies uniformly to
exhaustive, random, simulated annealing, Bayesian optimization, learned hybrid,
and custom strategies.

Application lifetime
--------------------

The application owns the number of kernel launches. It can use a fixed loop,
simulation time steps, convergence of its own result, a wall-time limit, or any
other application-level condition. Tuner limits and horizons do not impose an
application lifetime.

``Tuner::completed()`` and ``isTuningComplete()`` report exploration completion.
Offline initialization completes exploration immediately. Online limits and
search exhaustion end exploration; ``completionReason()`` then explains why.
Fixed selection locks its winner, while adaptive selection continues measuring
known configurations. ``info().selectionLocked`` distinguishes these outcomes.
Reaching the admission/cooling horizon alone does not report completion.

One history across runs
-----------------------

Every candidate owns exactly one rolling timing history. Loading a compatible
history restores those retained timings in place; there is no separate
strategy-only copy. During online exploration, strategies see the restored
estimates, and a compatible learned residual adapter is restored into the new
strategy instance. Offline reuse uses the same estimates without constructing
a strategy.

Starting online exploration with either selection policy creates a new run
lifecycle around that same history:

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

   * - Exploration / selection
     - First run without readable history
     - Second run with readable history
   * - ``online/fixed``
     - Builds timings and adapter state, then reaches a terminal guard.
     - Retains timings and adapter, resets run counts and tunes again to a new
       terminal guard.
   * - ``online/adaptive``
     - Builds timings and optionally advances a horizon schedule; after
       exploration completes, adapts among measured configurations.
     - Retains timings and adapter, resets run counts and exploration budgets.
       If configured, a fresh horizon starts at the active-history offset.
   * - ``offline/fixed``
     - Fails because no measured configuration is available.
     - Replays the robust best without measurement or strategy construction.
   * - ``offline/adaptive``
     - Fails because no measured configuration is available.
     - Remeasures known configurations and switches without constructing a strategy.

``online/fixed``
----------------

``online/fixed`` is the finite tuning mode. With a ``queue`` section, admitted
candidates remain resident and are interleaved. Every activation runs up to
``max_consecutive_runs`` launches, of which the first ``warmup_runs`` are not
recorded. Without a queue, each accepted strategy recommendation launches and
records directly; the strategy must recommend a candidate again when its
fixed-selection record needs more samples. In either form, a candidate retires at
its confidence criterion, Mann-Whitney early-stop criterion, or
``runs_per_candidate`` cap.

For a manual space, exploration finishes after all legal candidates retire, or
when either
``maximum_executions`` or ``maximum_retired_configurations`` is reached. At
least one of these two limits is required. Once tuning finishes, later
``enqueue`` calls launch the best measured configuration without collecting
more samples. The application may still continue until its external run count
is reached.

Setting ``tuning.replay_fast_path: true`` (or
``TunerConfig::replayFastPath = true``) makes those post-completion launches a
production fast path. They may use a timing-disabled queue and bypass timing
events, execution counters, strategy and scheduler work, history mutation, and
persistence staging. The option does not change the measured learning phase.

``maximum_executions`` and ``maximum_retired_configurations`` bound online
exploration under either selection policy. ``runs_per_candidate`` is the maximum number of new retained
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
For an automatic space, finishing the registered pool can trigger more
generation. Exploration ends when generation stops and the pool is resolved,
or a guard fires; see :doc:`automatic_spaces` for its completion reasons.

``online/adaptive``
-------------------

``online/adaptive`` adapts during exploration and can continue adapting after
exploration ends. Exploration limits, the horizon, and the queue are optional.
With a queue, one admission gives a candidate one residency and
therefore one activation burst. The following is a fragment to add to a
complete schema-4 configuration:

.. code-block:: yaml

   tuning:
     exploration: online
     selection: adaptive
     runs_per_candidate: 3
     horizon: 40000
   queue:
     warmup_runs: 1
     max_consecutive_runs: 4

This records three timings per admitted residency: the first launch is a
warm-up and the remaining three are measurements. The candidate leaves the
active queue after that burst. It may be admitted again later; the number of
visits is not limited by the number of retained timings.

Without a queue, each accepted recommendation is one measured adaptive visit.
Queue activation parameters are not applied. An absent queue and
``queue: {disable: true}`` are equivalent for execution; the latter retains
saved parameters for easy switching.

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
directly after mandatory constraint checks. The tuner does not apply
the sigmoid revisit gate or relative-score Boltzmann gate; exploration and
exploitation are entirely strategy-driven. This is often the clearest choice
for ``learned_hybrid`` because the learned strategy already ranks candidates
and adapts from runtime observations. Measurement, rolling-history updates,
and residual-adapter updates continue while exploration remains active.
Without a completion limit or search exhaustion, ``completed()`` and
``isTuningComplete()`` remain false. Use an application-owned loop bound when
you want adaptation to continue, including after a finite exploration phase.

With ``horizon`` configured, unseen legal candidates are admitted directly. An
already measured candidate must pass two independent gates after constraints
have accepted it and before an enabled queue handles active duplicates:

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

Here, ``n`` is the online adaptive exploration launch count in the current
tuner process run, ``H`` is
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
``TunerInfo::executionCount`` describes tracked launches in the current run,
including later adaptive reuse. ``adaptiveHorizonExecutionCount`` and
``adaptiveHorizonProgress`` describe the exploration schedule and stop advancing
once exploration ends. Persisted timings remain in each candidate's rolling
history, but online run counters restart at zero.

The second gate prefers candidates close to the current best robust runtime:

.. math::

   T(x) = T_0\left(\frac{T_1}{T_0}\right)^x

.. math::

   p_{score} = \exp\left(
     -\frac{\max(0, s/s_{best}-1)}{T(x)}\right)

``score_temperature_start`` and ``score_temperature_end`` default to 0.25 and
0.05. The current best therefore always passes the score gate, while slower
configurations become less likely as the temperature cools.

When present, ``horizon`` schedules admission and temperature during online
adaptive exploration. It does not end exploration or make ``completed()`` true.
An optional ``maximum_executions`` or ``maximum_retired_configurations`` ends
exploration independently; adaptive selection then uses measured known candidates.

At and after the horizon, normalized revisit admission remains exactly one.
Relative-score admission continues at ``score_temperature_end`` rather than
collapsing to a winner-only rule. Slightly slower configurations therefore
retain a nonzero Boltzmann probability; for example, with the default final
temperature of 0.05, a candidate that is two percent slower than the current
best is accepted by the score gate with probability
``exp(-0.02 / 0.05)``, approximately 0.67.

Rejected recommendations are retried synchronously. This includes constraints,
the adaptive revisit gate, and the relative-score gate. An active queue
duplicate is accepted instead. ``maximum_consecutive_strategy_retries``
defaults to 20 and bounds that work in one admission attempt. Admission resets
the streak. When active queue entries remain, reaching the limit pauses refill
until an activation completes and changes the scheduler context.

If an adaptive admission attempt exhausts that bound with no candidate ready
to run, the tuner reopens the current best measured history entry as a progress
fallback. This is not a terminal winner: it is queued when the queue is active
and launched directly otherwise, then measured and written into the rolling
window. Its execution advances the horizon, and the next admission calls the
strategy again with access to current runtime observations.
``adaptiveRetryFallbackCount`` in ``TunerInfo`` reports how often this path was
required. If no configuration has ever been accepted or restored, there is no
safe fallback and the triggering call throws an explicit initialization error
without marking the adaptive tuner complete.

In ``online/fixed`` only, the same retry limit without a runnable candidate is
a terminal state with
``TunerCompletionReason::maximumConsecutiveStrategyRetries``. Later fixed-selection
calls replay its best measured configuration without instrumentation.

Offline exploration and adaptive reuse
--------------------------------------

Offline exploration requires compatible history containing at least one valid
measured configuration. History may originate from any strategy or selection
policy and need not record a completed search. No strategy is constructed and
no candidate generation takes place.

Fixed selection replays the robust best without adding measurements.
Adaptive selection reopens the existing rolling histories for direct measurement.
Ordinary launches use the current robust best; every ``adaptive_probe_interval``
successful reuse launches, an alternative is probed. Alternatives rotate by
candidate index and must have usable measurements. Horizon gates and queue
warm-ups do not apply. Both incumbent and probe measurements update the same
bounded history and are persisted only when writes are enabled.

The same reuse path handles online adaptive exploration after its budget ends
or automatic generation stops and the measured catalog is resolved. Search
queue entries are discarded, run-local probe state starts fresh,
and registered configurations without measurements cannot enter reuse. A missing
custom metric rejects its candidate under the normal submission contract. If no
valid known candidate remains, the next enqueue fails without restarting search.

Strategy and execution boundary
-------------------------------

Every normalized strategy recommendation for a manual space is mapped once
to the nearest discrete candidate. For automatic spaces the strategy returns
an exact registered candidate ID, with no coordinate remapping. Mandatory
constraints run first, followed by optional adaptive
horizon rejection and optional queue handling. The tuner then reports one
disposition back to the strategy: scheduled, accepted active duplicate,
restriction rejection, revisit rejection, or score rejection. A candidate
already resident in an enabled queue satisfies the recommendation, so refill
stops and queue execution continues without asking for a substitute. Without
a queue, a scheduled candidate launches directly. Only an actual policy
rejection requests an entirely new proposal; the tuner does not mutate it or
search locally for a nearby substitute.

Learned hybrid commits its selection cycle only when a recommendation starts a
new queue activation. Active duplicates do not consume selection slots, while
rejections advance to another candidate within the same phase. Its default
cycle contains ten predicted-fast activations followed by one
uncertainty/diversity activation.

During online exploration, strategies can recommend previously measured points. The tuner applies the
selection policy's revisit rules to those recommendations. Use ``exhaustive``
when you need to visit every candidate; the other strategies do not guarantee
full coverage. Even exhaustive coverage of an automatic catalog does not prove
full domain coverage when generation stopped at a budget, plateau, or stall.
No strategy recommendations occur during known-configuration reuse.
