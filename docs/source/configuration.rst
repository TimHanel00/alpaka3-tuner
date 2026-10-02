Configure a run
===============

Start with an explicit ``TunerConfig`` for a small integration, as in
:doc:`getting_started`. Use YAML when users need to change the tuning policy
without rebuilding. Candidate values remain application-owned C++ data;
configuration controls how the tuner explores and measures them.

.. code-block:: cpp

   auto config = alpakaTune::TunerConfig::fromYaml("tuning.yaml");
   auto tuner = alpakaTune::makeTuner(config, tunables, device, "my-workload");

``makeTuner`` snapshots the configuration. Changing ``config`` later affects
only new tuners. The shorter ``makeTuner(tunables, device, ...)`` form loads
``tunerConfig()``. That function selects ``ALPAKA_TUNE_CONFIG`` or the
installed default once per process and returns a mutable copy on each call.
Set the environment variable before the first use:

.. code-block:: sh

   ALPAKA_TUNE_CONFIG=tuning.yaml ./my_app

A small, complete YAML file
---------------------------

This is the tutorial's fixed-mode collection policy:

.. literalinclude:: ../examples/first_tuner/collect.yaml
   :language: yaml

``schema_version: 3``, a ``tuning`` map, and positive ``runs_per_candidate``
are required by the YAML loader, **including in adaptive and offline files**.
Some sampling settings are unused in those modes, but still need valid values.
Unknown keys, mixed mode-specific limits, and invalid combinations are errors.

Settings you are likely to change
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Setting
     - Use
   * - ``mode``
     - Pick the lifecycle in :doc:`execution_modes`.
   * - ``strategy``
     - Start with ``exhaustive`` for a small space. Choose another
       :doc:`strategy <strategies>` for a larger search.
   * - ``runs_per_candidate`` / ``minimum_runs_per_candidate``
     - Fixed-mode cap and minimum before confidence retirement. The minimum
       defaults to the cap when loaded from YAML.
   * - ``maximum_executions``
     - Fixed-mode limit including warm-ups. It does not bound your application
       loop. An alternative guard is ``maximum_retired_configurations``.
   * - ``history_window_size``
     - Number of newest samples retained per candidate. It must be at least
       ``runs_per_candidate`` in fixed mode.
   * - ``history``
     - Optional saved measurements; see :doc:`history_workflows`.
   * - ``random_seed``
     - Numeric base seed, zero by default. The same context and seed reproduce
       proposal randomness, not measured runtimes. ``nondeterministic`` opts
       into a fresh stream.

Queue scheduling is optional
----------------------------

The YAML ``queue`` section configures the tuner's **candidate scheduler**, not
an Alpaka device queue. Omitting it launches accepted recommendations directly.
To interleave candidates and omit initial warm-up timings, add:

.. code-block:: yaml

   queue:
     warmup_runs: 1
     noise_cancellation_window: 20
     max_consecutive_runs: 3

This section is a fragment to append to the complete file above. An activation
contains three launches, of which the first is a warm-up. The active window
interleaves up to 20 candidates. In adaptive mode a residency ends after this
burst; in fixed mode candidates can be activated again until retirement.
``max_consecutive_runs`` must exceed ``warmup_runs``. Set ``disable: true``
to retain the scheduler settings but bypass scheduling, or use
``config.queue.reset()`` in C++.

Defaults and less common settings
---------------------------------

``TunerConfig{}`` uses adaptive mode, exhaustive proposals, no scheduler, no
horizon, and no history file. The shipped YAML instead enables a scheduler,
a 40,000-launch adaptive horizon, and two history files. These are different
starting policies; explicitly choose the one suitable for your application.

The shipped file, with comments for statistics, horizon admission, and learned
models, is the configuration reference:

.. literalinclude:: ../../config/alpakaTune.yaml
   :language: yaml
   :class: configuration-reference

The fixed-mode statistical controls affect retirement, not the correctness of
the kernel. See :doc:`strategies` for the robust estimate and early-stop rules.
Legacy schema 1/2 files remain accepted without persistence, but new files
should use schema 3 and the independent ``history`` / ``complete_history``
maps. The old ``persistence`` map is rejected.
