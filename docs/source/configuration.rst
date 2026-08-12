Configuration
=============

``alpakaTune::TunerConfig`` is a mutable value containing every tuner setting.
Construct it directly, load it with ``TunerConfig::fromYaml(path)``, or obtain
the installed default with ``alpakaTune::tunerConfig()``. The default path is
selected by ``ALPAKA_TUNE_CONFIG`` when set.

.. code-block:: cpp

   auto config = alpakaTune::TunerConfig::fromYaml("tuning.yaml");
   config.mode = alpakaTune::TuningMode::onlineAdaptive;
   config.strategy = alpakaTune::StrategyKind::random;
   config.queue = alpakaTune::QueueConfig{
       .disable = false,
       .warmupRuns = 1u,
       .noiseCancellationWindow = 20u,
       .maxConsecutiveRuns = 4u};
   config.horizon = 4'000u;
   config.horizonOffsetWithActiveHistory = 0.8;
   config.maximumConsecutiveStrategyRetries = 20u;
   config.historyWindowSize = 10u;
   config.history.file = ".my-tuning-cache/history.json";
   config.history.read = true;
   config.history.write = false;
   config.history.sampleCount = 100u;
   config.completeHistory.file = ".my-tuning-cache/complete-history.json";
   config.completeHistory.read = true;
   config.completeHistory.write = false;

The type is an aggregate, so direct construction is also supported:

.. code-block:: cpp

   auto config = alpakaTune::TunerConfig{
       .mode = alpakaTune::TuningMode::onlineFixed,
       .queue = alpakaTune::QueueConfig{
           .disable = false,
           .warmupRuns = 0u,
           .noiseCancellationWindow = 20u,
           .maxConsecutiveRuns = 2u},
       .runsPerCandidate = 10u,
       .minimumRunsPerCandidate = 3u,
       .maximumConsecutiveStrategyRetries = 20u,
       .maximumExecutions = 100u,
       .strategy = alpakaTune::StrategyKind::exhaustive,
       .history = {.file = ".alpakaTune/vector-add-history.json"},
       .completeHistory =
           {.file = ".alpakaTune/vector-add-complete-history.json"}};

``makeTuner`` snapshots its configuration. Later mutations affect only tuners
created afterward. The same ``TunerConfig`` can create multiple tuners. They
keep independent candidates, measurements, and winners; when
the corresponding history file and access policy are equal, they share that
store and their fingerprinted records coexist in the file. Compact and
complete stores are independent. With no file, compatible tuners still share
that store's staged records in memory but perform no filesystem I/O.

YAML schema version 3 adds independent compact and complete histories. Schemas
1 and 2 remain accepted only when they do not configure persistence:

.. code-block:: yaml

   schema_version: 3
   tuning:
     mode: online_adaptive
     strategy: exhaustive
     random_seed: 0
     runs_per_candidate: 20
     minimum_runs_per_candidate: 5
     ci_check_interval: 10
     ci_z_score: 2.576
     ci_relative_width: 0.05
     outlier_mad_scale: 3.5
     mann_whitney_early_stop: true
     mann_whitney_min_samples: 8
     mann_whitney_alpha: 0.05
     maximum_consecutive_strategy_retries: 20
     horizon: 40000
     history_window_size: 20
     revisit_admission_steepness: 16
     score_temperature_start: 0.25
     score_temperature_end: 0.05
     horizon_offset_with_active_history: 0.8
   queue:
     # A present section is enabled unless disable is true.
     disable: false
     warmup_runs: 1
     noise_cancellation_window: 50
     max_consecutive_runs: 3
   history:
     file: .alpakaTune/history.json
     read: true
     write: true
     sample_count: 100
   complete_history:
     file: .alpakaTune/complete-history.json
     read: true
     write: true
   learning:
     # Omit model to use the bundled artifact when one was installed.
     model: /path/to/alternative-model.atml
     fallback: random
     candidate_pool_size: 4096
     candidate_batch_size: 256

Each store independently uses this access matrix:

.. list-table::
   :header-rows: 1

   * - ``read``
     - ``write``
     - Behaviour
   * - ``false``
     - ``true``
     - Ignore any old file and replace it once at normal shutdown with only
       the histories staged by this run.
   * - ``true``
     - ``false``
     - Load compatible history, but leave the file unchanged. This is the
       direct offline-replay or adaptive-with-context mode.
   * - ``true``
     - ``true``
     - Load compatible history and merge the newest staged contexts once at
       normal shutdown. This preserves the previous behaviour.
   * - ``false``
     - ``false``
     - Use only process-local staged history.

Both flags default to ``true``. Either store map or its ``file`` key may be
omitted. Without a file its flags have no filesystem effect and that history
stays process-local. ``sample_count`` is optional, positive, and affects only
compact-history writes. The two stores cannot use the same path. See
:doc:`history_workflows` for collection, adaptive continuation, and replay
configurations.

Unknown keys and invalid values are rejected. ``online_adaptive`` is the
default. In ``online_fixed``, ``runsPerCandidate`` is the hard measurement cap
and must not exceed ``historyWindowSize``. Lowering
``minimumRunsPerCandidate`` enables confidence-interval retirement.
The queue is opt-in for both online modes. A present ``queue`` map constructs
the scheduler and defaults ``disable`` to ``false``. Omitting the map, setting
``disable: true``, resetting ``TunerConfig::queue``, or setting
``QueueConfig::disable`` bypasses it. The resulting path is strategy,
mandatory constraints, optional horizon rejection, then one direct launch.
When the queue is active, it follows horizon rejection and precedes the launch.
Disabled queue parameters are retained but ignored. Legacy
``warmup_runs``, ``noise_cancellation_window``, and ``max_consecutive_runs``
keys under ``tuning`` remain accepted for compatibility but do not enable or
configure a queue. In an active queue, ``maxConsecutiveRuns`` must exceed
``warmupRuns``. In ``online_fixed``,
``maximumExecutions`` counts warm-up and measured launches, while
``maximumRetiredConfigurations`` limits completed candidate histories. These
two guards are invalid in the other modes. ``online_adaptive`` optionally
accepts ``horizon``, which spans the current tuner process run without becoming
a terminal budget. With a horizon and compatible measured history,
``horizonOffsetWithActiveHistory`` selects the initial point in the normalized
sigmoid/Boltzmann schedule. It is inclusive in ``[0, 1]`` and defaults to
``0.8``; the remaining interval is stretched over all ``horizon`` new
launches.
Without ``horizon``, adaptive mode continuously accepts legal strategy
revisits without the sigmoid or Boltzmann gates and never reports tuning-policy
completion. ``horizon_offset_with_active_history`` is consequently invalid in
YAML unless ``horizon`` is present.
``maximumConsecutiveStrategyRetries`` applies to both online modes and defaults
to ``20``. Its YAML spelling is
``maximum_consecutive_strategy_retries``. A rejected strategy proposal is
immediately replaced by a fresh strategy call. An accepted proposal resets the
retry streak. If active queue entries still exist, reaching the limit pauses
refill until a completed activation changes the admission context. Reaching
the limit with an empty ``online_fixed`` queue enters a terminal state with
completion reason ``maximum_consecutive_strategy_retries``. In
``online_adaptive``, the tuner instead reopens the best measured history entry
as a measured progress fallback and retries the strategy during later refills.
This keeps the host-side work bounded without creating a winner or ending
adaptation. ``TunerInfo`` exposes the configured maximum, the current
``consecutiveStrategyRetries`` value, the cumulative
``strategyRetryLimitReachedCount``, and the adaptive subset
``adaptiveRetryFallbackCount``.
Set ``maximum_executions: null`` in YAML when a fixed run should use only the
retired-configuration guard.
This does not configure the surrounding application's loop. Applications own
their launch count independently and may optionally inspect
``tuner.completed()``. In adaptive mode that method reports arrival at the
admission/cooling horizon without stopping adaptation. See
:doc:`execution_modes` for the complete lifecycle, ownership boundary, and
probability definitions.

``randomSeed`` defaults to the reproducible base seed zero. Each tuner mixes
that base with its stable context fingerprint before constructing its strategy
and admission random generator. Repeating the same context and numeric seed is
therefore deterministic, while distinct kernels, launch specifications, or
identity entries do not receive correlated random streams. YAML may use
``random_seed: nondeterministic`` to request process-local entropy explicitly;
the equivalent C++ opt-in is ``config.randomSeed.reset()``. The
nondeterministic mode is intended for independent experiments and cannot
reproduce an earlier proposal order from configuration alone.

Selecting ``learned_hybrid`` leaves ``makeTuner`` and ``enqueue`` unchanged.
If its model is missing, incompatible, or outside its supported feature
contract, the configured non-learned fallback is used explicitly.
The learned strategy scores at most ``candidate_batch_size`` new candidates at
a time and retains at most ``candidate_pool_size`` active candidates. The
batch size must not exceed the pool size.
