alpakaTune
==========

alpakaTune chooses a fast configuration for an Alpaka3 kernel while your
application runs. You supply a small set of meaningful choices, such as batch
sizes or launch shapes. The tuner tries them, measures their cost, and uses
those measurements to select later launches. By default it minimizes runtime;
you can instead provide an application-defined score.

Start with :doc:`getting_started`: build a complete CPU example, tune three
batch sizes, and check the output. Then follow :doc:`launch_tuning` to tune
launch geometry or :doc:`history_workflows` to reuse a winner in another run.
The core guide describes the ``dev`` API. GPU availability depends on the Alpaka
backends enabled in your build.

Five concepts to keep in mind
-----------------------------

* **Tunable:** a named choice the tuner may change. ``RVals`` supplies runtime
  values; ``CVals`` and ``CTypes`` supply compiled alternatives.
* **TunableBundle:** the candidate space and its restrictions. The tuner
  snapshots it when constructed.
* **KernelBundle:** the launch prototype. Tunable markers in its arguments
  are replaced with selected values; ordinary arguments are passed through.
* **Tuner:** state for one device, kernel-bundle type, and launch prototype.
  Keep it alive across calls so it can learn.
* **Mode:** tune once and replay, keep adapting, or replay saved history.
  Your application owns the loop in every mode.

.. toctree::
   :maxdepth: 2
   :caption: Start here

   getting_started
   launch_tuning
   execution_modes
   configuration
   history_workflows

.. toctree::
   :maxdepth: 2
   :caption: Everyday extensions

   compile_time_tuning
   post_evaluation_validation
   custom_metrics
   instrumentation
   troubleshooting

.. toctree::
   :maxdepth: 1
   :caption: Further reading

   strategies
   persistence
   execution_reference
   advanced_objectives

Experimental metrics
--------------------

The ``tunerEnhancedMetrics`` feature branch integrates the optional
`alpakaMetrics <https://github.com/TimHanel00/alpakaMetrics>`_ dependency and
adds counter-based objectives. Its :doc:`experimental_metrics` guide explains the CMake option, pinned Alpaka
ownership, and metric availability. The callable scoring
API in :doc:`custom_metrics` is already part of ``dev`` and needs no metrics
dependency.

.. toctree::
   :maxdepth: 1

   experimental_metrics
