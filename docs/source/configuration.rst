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
   config.maximumExecutions = 4'000u; // Adaptive admission horizon.
   config.historyWindowSize = 10u;
   config.persistenceFile = ".my-tuning-cache/history.json";

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
``persistenceFile`` is equal, they share the internal persistence store and
their fingerprinted records coexist in that file.

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
     maximum_executions: 40000
     history_window_size: 20
     revisit_admission_steepness: 16
     score_temperature_start: 0.25
     score_temperature_end: 0.05
   persistence:
     file: .alpakaTune/history.json
   learning:
     # Omit model to use the bundled artifact when one was installed.
     model: /path/to/alternative-model.atml
     fallback: random
     candidate_pool_size: 4096
     candidate_batch_size: 256

Unknown keys and invalid values are rejected. ``online_adaptive`` is the
default. In ``online_fixed``, ``runsPerCandidate`` is the hard measurement cap
and must not exceed ``historyWindowSize``. Lowering
``minimumRunsPerCandidate`` enables confidence-interval retirement.
``maxConsecutiveRuns`` must exceed ``warmupRuns``. ``maximumExecutions`` counts
warm-up and measured launches, while ``maximumRetiredConfigurations`` limits
completed candidate histories. In ``online_adaptive``, the former is an
admission horizon rather than a terminal budget and the latter is ignored.
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
