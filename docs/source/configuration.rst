Configuration
=============

``alpakaTune::TunerConfig`` is a mutable value containing every tuner setting.
Construct it directly, load it with ``TunerConfig::fromYaml(path)``, or obtain
the installed default with ``alpakaTune::tunerConfig()``. The default path is
selected by ``ALPAKA_TUNE_CONFIG`` when set.

.. code-block:: cpp

   auto config = alpakaTune::TunerConfig::fromYaml("tuning.yaml");
   config.strategy = alpakaTune::StrategyKind::random;
   config.runsPerCandidate = 10u;
   config.minimumRunsPerCandidate = 3u;
   config.maximumExecutions = 100u;
   config.persistenceFile = ".my-tuning-cache/history.json";

The type is an aggregate, so direct construction is also supported:

.. code-block:: cpp

   auto config = alpakaTune::TunerConfig{
       .warmupRuns = 0u,
       .runsPerCandidate = 10u,
       .minimumRunsPerCandidate = 3u,
       .noiseCancellationWindow = 20u,
       .maxConsecutiveRuns = 2u,
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
     maximum_executions: 100000
     maximum_retired_configurations: 100000
   persistence:
     file: .alpakaTune/history.json
   learning:
     # Omit model to use the bundled artifact when one was installed.
     model: /path/to/alternative-model.atml
     fallback: random

Unknown keys and invalid values are rejected. ``runsPerCandidate`` is the hard
measurement cap. Lowering ``minimumRunsPerCandidate`` enables confidence-
interval retirement. ``maxConsecutiveRuns`` must exceed ``warmupRuns``.
``maximumExecutions`` counts warm-up and measured launches, while
``maximumRetiredConfigurations`` limits completed candidate histories.
Selecting ``learned_hybrid`` leaves ``makeTuner`` and ``enqueue`` unchanged.
If its model is missing, incompatible, or outside its supported feature
contract, the configured non-learned fallback is used explicitly.
