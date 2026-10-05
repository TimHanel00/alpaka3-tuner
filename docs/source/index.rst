alpakaTune
==========

alpakaTune chooses a fast configuration for an Alpaka3 kernel while your
application runs. You supply meaningful choices, such as batch sizes or launch
shapes, as explicit alternatives or automatic domains. The tuner tries
configurations, measures their cost, and uses
those measurements to select later launches. By default it minimizes runtime;
you can instead provide an application-defined score.

Start with :doc:`getting_started`: build a complete CPU example, tune three
batch sizes, and check the output. Then follow :doc:`launch_tuning` to tune
launch geometry. Use :doc:`automatic_spaces` to grow a candidate catalog from
feedback, and :doc:`execution_modes` to choose exploration and selection
independently. :doc:`history_workflows` shows how to replay or adapt from saved
measurements in another run.
The core guide describes the ``dev`` API. GPU availability depends on the Alpaka
backends enabled in your build.

Five concepts to keep in mind
-----------------------------

* **Tunable:** a named choice the tuner may change. ``RVals`` supplies runtime
  values; ``CVals`` and ``CTypes`` supply compiled alternatives.
  ``autoCandidates`` declares domains for progressive candidate generation.
* **TunableBundle:** the domains and their restrictions. The tuner snapshots
  the declarations when constructed; an automatic catalog can grow within them.
* **KernelBundle:** the launch prototype. Tunable markers in its arguments
  are replaced with selected values; ordinary arguments are passed through.
* **Tuner:** state for one device, kernel-bundle type, and launch prototype.
  Keep it alive across calls so it can learn.
* **Policies:** online/offline controls exploration; adaptive/fixed controls
  whether configuration selection changes after exploration ends. Adaptive
  selection can reuse measured configurations without searching again.
  Your application owns the loop with every policy combination.

.. toctree::
   :maxdepth: 2
   :caption: Start here

   getting_started
   launch_tuning
   automatic_spaces
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
adds counter-based objectives. Its `branch-specific guide
<https://github.com/TimHanel00/alpaka3-tuner/blob/tunerEnhancedMetrics/docs/source/experimental_metrics.rst>`_
explains the CMake option, pinned Alpaka ownership, and metric availability. The callable scoring
API in :doc:`custom_metrics` is already part of ``dev`` and needs no metrics
dependency.
