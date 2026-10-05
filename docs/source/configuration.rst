Configure a run
===============

Start with an explicit ``TunerConfig`` for a small integration, as in
:doc:`getting_started`. Use YAML when users need to change the tuning policy
without rebuilding. Candidate lists and domain declarations remain
application-owned C++ data; configuration controls generation, exploration,
measurement, and selection.

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

This is the tutorial's online/fixed collection policy:

.. literalinclude:: ../examples/first_tuner/collect.yaml
   :language: yaml

``schema_version: 4``, a ``tuning`` map, and positive ``runs_per_candidate``
are required by the YAML loader, **including in adaptive and offline files**.
Some sampling settings are unused under those policies, but still need valid values.
Unknown keys and invalid policy combinations are errors. The removed ``mode``
key and older YAML schemas require migration to the two independent fields.

Settings you are likely to change
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Setting
     - Use
   * - ``exploration`` / ``selection``
     - Choose the independent policies in :doc:`execution_modes`.
   * - ``strategy``
     - Start with ``exhaustive`` for a small space. Choose another
       :doc:`strategy <strategies>` for a larger search.
   * - ``runs_per_candidate`` / ``minimum_runs_per_candidate``
     - Online/fixed cap and minimum before confidence retirement. The minimum
       defaults to the cap when loaded from YAML.
   * - ``maximum_executions``
     - Online exploration limit including warm-ups. It does not bound your
       application loop or later adaptive reuse. An alternative guard is
       ``maximum_retired_configurations``.
   * - ``adaptive_probe_interval``
     - During reuse, probe a measured alternative every this many successful
       launches. Defaults to 10; must be positive. Applies after online adaptive
       exploration ends and throughout offline adaptive selection.
   * - ``horizon``
     - Optional admission and cooling schedule for online/adaptive exploration.
       Reaching it does not complete exploration.
   * - ``space``
     - Bound automatic candidate generation independently of execution budgets;
       see :doc:`automatic_spaces`.
   * - ``history_window_size``
     - Number of newest samples retained per candidate. It must be at least
       ``runs_per_candidate`` with fixed selection.
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
interleaves up to 20 candidates. With adaptive selection a residency ends after
this burst; with fixed selection candidates can be activated again until retirement.
``max_consecutive_runs`` must exceed ``warmup_runs``. Set ``disable: true``
to retain the scheduler settings but bypass scheduling, or use
``config.queue.reset()`` in C++.
This scheduler operates during online exploration. Known-configuration reuse
launches directly, without queued activations or warm-ups.

Defaults and less common settings
---------------------------------

``TunerConfig{}`` uses online exploration with adaptive selection, exhaustive
proposals, no scheduler, no exploration limit, no horizon, and no history file.
The shipped YAML also uses online/adaptive, but enables a scheduler,
a 40,000-launch exploration limit and cooling horizon, and two history files.
Choose explicitly which starting configuration suits your application.

The shipped file, with comments for statistics, horizon admission, and learned
models, is the configuration reference:

.. literalinclude:: ../../config/alpakaTune.yaml
   :language: yaml
   :class: configuration-reference

The fixed-selection statistical controls affect retirement, not the correctness of
the kernel. See :doc:`strategies` for the robust estimate and early-stop rules.

Migrate combined modes
----------------------

YAML schema 4 replaces ``tuning.mode`` with ``tuning.exploration`` and
``tuning.selection``. C++ replaces ``config.mode`` and ``TuningMode`` with
``config.exploration`` (``ExplorationPolicy``) and ``config.selection``
(``SelectionPolicy``). Use this mapping to preserve the old behaviour:

.. list-table::
   :header-rows: 1

   * - Former YAML / C++ mode
     - Exploration
     - Selection
   * - ``online_fixed`` / ``TuningMode::onlineFixed``
     - ``online``
     - ``fixed``
   * - ``online_adaptive`` / ``TuningMode::onlineAdaptive``
     - ``online``
     - ``adaptive``
   * - ``offline`` / ``TuningMode::offline``
     - ``offline``
     - ``fixed``

Set ``schema_version: 4`` and remove ``mode``. Offline/adaptive is the new
combination for adapting from saved measurements without further exploration.
Online/adaptive can now also use exploration completion limits. Offline files
must omit ``maximum_executions`` and ``maximum_retired_configurations``;
``horizon`` is valid only for online/adaptive. ``replay_fast_path`` requires
fixed selection.

``completed()`` now means exploration has ended. Adaptive selection can continue
measuring and switching configurations after it becomes true; use your own
application loop bound for this workflow. ``info().selectionLocked`` reports a
fixed replay winner. See :doc:`execution_modes` for the public status contract.

Existing compact schema-2 and complete schema-13 JSON histories remain readable;
the YAML schema change does not require recollection. Legacy combined-mode
metadata in these histories does not select the current policies. The older
``persistence`` map is still rejected; use ``history`` and ``complete_history``.
