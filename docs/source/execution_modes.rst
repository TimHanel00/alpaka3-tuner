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
other application-level condition. ``maximum_executions`` belongs to tuner
policy and does not impose an application lifetime.

``Tuner::completed()`` is optional policy information an application may
consult. In ``online_fixed`` and ``offline`` it identifies a terminal tuner
state. In ``online_adaptive`` it becomes true at ``maximum_executions`` only to
indicate that the admission sigmoid and cooling schedule reached their final
state. It does not stop adaptation, measurement, revisits, or residual-adapter
updates.

The examples demonstrate an application-owned combined condition: at least
50,000 executions and ``completed()``. Their default adaptive horizon is
40,000, intentionally leaving 10,000 launches at the final schedule state.
Those numbers are independent. Applications that require exactly N launches
should simply execute exactly N launches and need not inspect ``completed()``.
The stricter ``isTuningComplete()`` query reports only an actual terminal
tuner state and therefore remains false in adaptive mode.

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

``online_adaptive``
-------------------

``online_adaptive`` is a continuous mode. One admission gives a candidate one
queue residency and therefore one activation burst. The burst still follows
the queue configuration exactly. For example:

.. code-block:: yaml

   tuning:
     mode: online_adaptive
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

Unseen legal candidates are admitted directly. An already measured candidate
must pass two independent gates after active-queue duplicates and restrictions
have been rejected:

.. math::

   x = \operatorname{clamp}\left(\frac{n}{H}, 0, 1\right)

.. math::

   p_{revisit}(x) =
   \frac{\sigma(k(x-\tfrac12))-\sigma(-k/2)}
        {\sigma(k/2)-\sigma(-k/2)}

Here, ``n`` is the total launch count, ``H`` is
``maximum_executions``, and ``k`` is
``revisit_admission_steepness`` (16 by default). The normalization makes the
probability exactly zero at launch zero and exactly one at and after the
horizon.

The second gate prefers candidates close to the current best robust runtime:

.. math::

   T(x) = T_0\left(\frac{T_1}{T_0}\right)^x

.. math::

   p_{score} = \exp\left(
     -\frac{\max(0, s/s_{best}-1)}{T(x)}\right)

``score_temperature_start`` and ``score_temperature_end`` default to 0.25 and
0.05. The current best therefore always passes the score gate, while slower
configurations become less likely as the temperature cools.

In this mode ``maximum_executions`` is only the admission and temperature
horizon inside the tuner. At that point ``completed()`` becomes true as a
diagnostic, but the adaptive tuner does not enter its internal terminal state.
``maximum_retired_configurations`` is accepted for configuration compatibility
but ignored in adaptive mode; it is a completion guard only in
``online_fixed``. If a refill pass admits nothing, the tuner launches the
current best without measuring it and asks the strategy for another proposal
on the next ``enqueue`` call. Rejection never causes a synchronous
recommendation loop.

``offline``
-----------

``offline`` requires a compatible persistent history containing at least one
measured configuration. The history does not need a terminal completion reason
and may have been produced by any strategy or online mode. The tuner selects
the best robust estimate from that history and launches only that
configuration. It does not instantiate a strategy, perform warm-ups, measure
the launch, or update persistence.

Strategy and queue boundary
---------------------------

Every strategy recommendation is mapped once to the exact nearest discrete
candidate. The shared tuner admission policy then reports one disposition back
to the strategy: scheduled, active duplicate, restriction rejection, revisit
rejection, or score rejection. The tuner does not search synchronously for a
nearby unscheduled substitute.

Strategies may deliberately recommend previously measured points. They must
not hide duplicate ownership inside their own candidate bookkeeping. Regression
tests also require every built-in strategy, including learned hybrid, to expose
at least ten distinct raw candidates among 100 recommendations in a
100-candidate space. Fixed seeds make this strategy contract deterministic.
This is proposal diversity, not a promise to exhaust the entire space.
