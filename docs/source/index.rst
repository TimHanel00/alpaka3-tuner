alpakaTune
==========

alpakaTune tunes Alpaka3 ``KernelBundle`` launches with a mutable
``TunerConfig`` and a device-bound ``Tuner``. A tuner represents one particular
tuning context. Depending on its execution mode, it either performs finite
tuning, adapts continuously, or replays the best compatible persisted result.

Alpaka3 provides two host-side launch descriptions:

* ``FrameSpec`` for portable logical work decomposition;
* ``ThreadSpec`` for an exact physical block/thread launch shape.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   getting_started
   configuration
   history_workflows
   execution_modes
   strategies
   launch_tuning
   compile_time_tuning
   instrumentation
   persistence
