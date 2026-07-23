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
   config.horizon = 4'000u;
   config.horizonOffsetWithActiveHistory = 0.8;
   config.historyWindowSize = 10u;
   config.persistenceFile = ".my-tuning-cache/history.json";
   config.persistenceRead = true;
   config.persistenceWrite = false; // Reuse history without changing it.

The type is an aggregate, so direct construction is also supported:

.. code-block:: cpp

   auto config = alpakaTune::TunerConfig{
       .mode = alpakaTune::TuningMode::onlineFixed,
       .warmupRuns = 0u,
       .runsPerCandidate = 10u,
       .minimumRunsPerCandidate = 3u,
       .noiseCancellationWindow = 20u,
       .maxConsecutiveRuns = 2u,
       .maximumExecutions = 100u,
       .strategy = alpakaTune::StrategyKind::exhaustive,
       .persistenceFile = ".alpakaTune/vector-add.json"};

``makeTuner`` snapshots its configuration. Later mutations affect only tuners
created afterward. The same ``TunerConfig`` can create multiple tuners. They
keep independent candidates, measurements, and winners; when
``persistenceFile`` and its access policy are equal, they share the internal
persistence store and their fingerprinted records coexist in that file. With
no ``persistenceFile``, compatible tuners in the same process still share
staged records in memory, but perform no filesystem I/O.

YAML schema version 2 adds the optional learned-model section. Schema version
1 remains accepted for existing non-learned configurations:

.. code-block:: yaml

   schema_version: 2
   tuning:
     mode: online_adaptive
     strategy: exhaustive
     random_seed: 0
     warmup_runs: 1
     runs_per_candidate: 20
     minimum_runs_per_candidate: 5
     ci_check_interval: 10
     ci_z_score: 2.576
     ci_relative_width: 0.05
     outlier_mad_scale: 3.5
     mann_whitney_early_stop: true
     mann_whitney_min_samples: 8
     mann_whitney_alpha: 0.05
     noise_cancellation_window: 50
     max_consecutive_runs: 3
     horizon: 40000
     history_window_size: 20
     revisit_admission_steepness: 16
     score_temperature_start: 0.25
     score_temperature_end: 0.05
     horizon_offset_with_active_history: 0.8
   persistence:
     file: .alpakaTune/history.json
     read: true
     write: true
   learning:
     # Omit model to use the bundled artifact when one was installed.
     model: /path/to/alternative-model.atml
     fallback: random
     candidate_pool_size: 4096
     candidate_batch_size: 256

The complete persistence access matrix is:

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

Both flags default to ``true`` for backward compatibility when ``file`` is
present. The entire ``persistence`` map, or just its ``file`` key, may be
omitted. Without a file the flags have no filesystem effect and history stays
process-local. See :doc:`history_workflows` for complete fresh-collection,
read-only adaptive continuation, and offline-replay configurations.

Unknown keys and invalid values are rejected. ``online_adaptive`` is the
default. In ``online_fixed``, ``runsPerCandidate`` is the hard measurement cap
and must not exceed ``historyWindowSize``. Lowering
``minimumRunsPerCandidate`` enables confidence-interval retirement.
``maxConsecutiveRuns`` must exceed ``warmupRuns``. In ``online_fixed``,
``maximumExecutions`` counts warm-up and measured launches, while
``maximumRetiredConfigurations`` limits completed candidate histories. These
two guards are invalid in the other modes. ``online_adaptive`` instead requires
``horizon``, which spans the current tuner process run without becoming a
terminal budget. When compatible measured history is loaded,
``horizonOffsetWithActiveHistory`` selects the initial point in the normalized
sigmoid/Boltzmann schedule. It is inclusive in ``[0, 1]`` and defaults to
``0.8``; the remaining interval is stretched over all
``horizon`` new launches.
Set ``maximum_executions: null`` in YAML when a fixed run should use only the
retired-configuration guard.
This does not configure the surrounding application's loop. Applications own
their launch count independently and may optionally inspect
``tuner.completed()``. In adaptive mode that method reports arrival at the
admission/cooling horizon without stopping adaptation. See
:doc:`execution_modes` for the complete lifecycle, ownership boundary, and
probability definitions.

Selecting ``learned_hybrid`` leaves ``makeTuner`` and ``enqueue`` unchanged.
If its model is missing, incompatible, or outside its supported feature
contract, the configured non-learned fallback is used explicitly.
The learned strategy scores at most ``candidate_batch_size`` new candidates at
a time and retains at most ``candidate_pool_size`` active candidates. The
batch size must not exceed the pool size.
