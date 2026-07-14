alpakaTune
==========

alpakaTune tunes Alpaka3 ``KernelBundle`` launches through YAML-backed,
device-bound contexts. A context replaces a normal queue enqueue while it
measures candidates, then enqueues its cached winner directly.

Alpaka3 provides two host-side launch descriptions:

* ``FrameSpec`` for portable logical work decomposition;
* ``ThreadSpec`` for an exact physical block/thread launch shape.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   getting_started
   configuration
   strategies
   launch_tuning
   compile_time_tuning
   instrumentation
   persistence
   api
