Save a result and replay it
===========================

Use this workflow when a dedicated training run can collect measurements and
later runs should avoid online exploration. It uses the same executable and
candidate space as :doc:`getting_started`.

Collect once
------------

Run from the repository root:

.. code-block:: sh

   ./build-tutorial/first_tuner docs/examples/first_tuner/collect.yaml

.. literalinclude:: ../examples/first_tuner/collect.yaml
   :language: yaml

``read: false`` excludes any old file; ``write: true`` stages this run's
measurements and writes them at normal process shutdown. You should see
``completed=1; loaded=0``. The saved compact history is
``.alpakaTune/tutorial.json``, relative to the working directory.

Replay in another process
-------------------------

.. code-block:: sh

   ./build-tutorial/first_tuner docs/examples/first_tuner/replay.yaml

.. literalinclude:: ../examples/first_tuner/replay.yaml
   :language: yaml

Expect ``completed=1; loaded=1``. Offline mode selects the best compatible
measured candidate, performs no new measurements, and leaves the input file
unchanged. A history need not contain a terminal fixed-mode winner; measured
history from adaptive mode also works. With no compatible measured candidate,
offline initialization fails instead of silently training.

Continue adapting
-----------------

To load the same observations while exploring further:

.. code-block:: sh

   ./build-tutorial/first_tuner docs/examples/first_tuner/adaptive.yaml

.. literalinclude:: ../examples/first_tuner/adaptive.yaml
   :language: yaml

This intentionally has no horizon. Expect ``completed=0; loaded=1`` after the
application's 20 launches. The measurements guide new decisions, but the input
file stays unchanged. Set ``write: true`` to save updated observations.
Starting either online mode resets run counters and scheduling state, while
retaining compatible samples. A new fixed run tunes again before entering
winner replay.

Keep histories meaningful
-------------------------

A tuner identifies history from its device, kernel-bundle type, launch
prototype, candidates, restrictions, objective, and extra identity entries.
Strategy and mode can change between collection and replay. A changing buffer
address is ordinary launch data, not a new workload identity.

Supply a stable workload label, problem size, precision or algorithm version,
and other inputs that change performance or validity:

.. code-block:: cpp

   auto tuner = alpakaTune::makeTuner(
       config, tunables, device, executor, "solver-v2", problemSize);

Ordinary argument **values** and captured scoring-function state cannot be
inferred as semantic identity automatically. Update the identity when their
meaning changes. Create a separate tuner for a materially different workload;
compatible context records can coexist in one file. Keep identity entries
consistent in type as well as value between collection and replay.

Which history file?
-------------------

Use ``history`` for normal reuse. Add a distinct ``complete_history.file``
only when you need raw samples and detailed diagnostics; see
:doc:`persistence` for precedence and file schemas.

For either store, ``read: true, write: true`` loads and merges,
``read: true, write: false`` loads without changing the file, and
``read: false, write: true`` replaces it with this run's staged contexts.
Omitting ``file`` keeps that store process-local. Abrupt termination cannot
guarantee a final write. Disable persistence at build time with
``alpakaTune_ENABLE_PERSISTENCE=OFF`` when file reuse is unnecessary.
