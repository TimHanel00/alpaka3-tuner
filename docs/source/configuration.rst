Configuration
=============

``alpakaTune::session()`` returns a copy of the process-wide default session,
which reads the YAML file named by ``ALPAKA_TUNE_CONFIG`` on first use. If the
variable is not set, it uses the installed ``alpakaTune.yaml`` default.
``alpakaTune::session(path)`` and ``alpakaTune::contextBuilder(path)`` select
another file explicitly.

Each session handle and copy owns independent contexts and measurements; only
the parsed immutable defaults are shared internally.

.. code-block:: yaml

   schema_version: 1
   tuning:
     strategy: exhaustive       # exhaustive, random, simulated_annealing, or bayesian_optimization
     random_seed: 0
     warmup_runs: 1
     runs_per_candidate: 20           # hard maximum of measured runs
     minimum_runs_per_candidate: 5    # CI may retire only after this many
     ci_check_interval: 10
     ci_z_score: 2.576                # 99 % CI, matching the legacy tuner
     ci_relative_width: 0.05
     outlier_mad_scale: 3.5
     mann_whitney_early_stop: true
     mann_whitney_min_samples: 8
     mann_whitney_alpha: 0.05
     noise_cancellation_window: 50
     max_consecutive_runs: 3
   persistence:
     directory: .alpakaTune

Unknown keys and invalid values are rejected. The cache directory stores
versioned JSON records for completed contexts, including every raw timing
history. ``runs_per_candidate`` is the hard cap; by default
``minimum_runs_per_candidate`` is equal to it, preserving a fixed-size
benchmark. Set it lower to enable the periodic CI stopping criterion.

``warmup_runs`` is applied whenever the interleaving queue activates a record.
Therefore ``max_consecutive_runs`` must be greater than ``warmup_runs`` so an
activation can produce at least one measured sample.
