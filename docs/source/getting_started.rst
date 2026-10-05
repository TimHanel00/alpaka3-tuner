Your first tuner
================

Tune the batch size of a kernel that adds one to each array element.
The example runs on the host CPU and needs CMake 3.25+, a C++20 compiler,
and Git. CMake fetches the pinned dependencies.
For Alpaka basics, see the `getting-started tutorial
<https://alpaka3.readthedocs.io/en/latest/tutorial/motivation.html>`_.

Build and run
-------------

From the repository root:

.. code-block:: sh

   cmake -S docs/examples/first_tuner -B build-tutorial -DCMAKE_BUILD_TYPE=Release
   cmake --build build-tutorial --target first_tuner --parallel 4
   ./build-tutorial/first_tuner

The program checks 20 launches and prints backend runtime statistics and the
batch size from the fastest timed launch. Example output (times vary):

.. code-block:: text

   20 launches checked
   Kernel runtimes (us; 9 timed launches): min=51.200; max=78.400; median=62.100; avg=63.500
   Minimum-runtime configuration: batchSize=64
   Tuning: complete; history: no saved measurements restored

``executionRuntimeSummary()`` covers timed launches in this process;
untimed replay and saved samples are excluded. The fastest single launch
may differ from the tuner's statistical winner. ``Tuning: complete`` means
exploration has ended; the history message reports whether saved measurements
were restored. See :doc:`instrumentation` for details.

A short-kernel warning is expected for this small workload.

Read the working source
-----------------------

.. literalinclude:: ../examples/first_tuner/main.cpp
   :language: cpp
   :start-at: #include

1. **Declare choices:** ``batchSize`` names the parameter;
   ``RVals{32u, 64u, 128u}`` lists its candidates.
2. **Create one tuner:** ``makeTuner`` binds the configuration, candidates,
   device, and workload identity. This example takes three samples per candidate.
3. **Launch:** ``tuner.enqueue(queue, frame, bundle)`` runs once per call.
   After exploration, it replays the measured winner.

Fills, copies, and correctness checks stay outside the timed kernel interval.
Copies use the same queue, followed by ``wait`` before reading results.
Every candidate must produce the correct answer; see
:doc:`post_evaluation_validation`.

Use it in your project
----------------------

FetchContent adds the dependency and its exported target:

.. code-block:: cmake

   cmake_minimum_required(VERSION 3.25)
   project(myApp LANGUAGES CXX)
   include(FetchContent)
   FetchContent_Declare(
     alpakaTune
     GIT_REPOSITORY https://github.com/TimHanel00/alpaka3-tuner.git
     GIT_TAG dev)
   FetchContent_MakeAvailable(alpakaTune)
   add_executable(my_app main.cpp)
   target_link_libraries(my_app PRIVATE alpakaTune::alpakaTune)
   alpaka_finalize(my_app)

``dev`` follows development; replace it with a tested commit for reproducible
builds. With a local checkout, ``add_subdirectory(path/to/alpaka3-tuner)`` can
replace FetchContent. On ``dev``, an existing ``alpaka::alpaka`` target is
reused; otherwise the tuner fetches its pinned Alpaka revision.

For an installed package, replace the FetchContent block with
``find_package(alpakaTune CONFIG REQUIRED)``. Keep the link and finalization
calls. Point ``CMAKE_PREFIX_PATH`` at the install prefix. To create one:

.. code-block:: sh

   cmake -S . -B build-install -DCMAKE_INSTALL_PREFIX="$PWD/install"
   cmake --build build-install --parallel 4
   cmake --install build-install

The install includes the default YAML and fetched dependency packages;
externally supplied dependencies must also be discoverable by the consumer.
The tutorial CMake project tries an installed package first, so you can test it
with ``-DCMAKE_PREFIX_PATH="$PWD/install"`` in a fresh build directory.

Every target using Alpaka headers must call ``alpaka_finalize`` after linking.
Include ``<alpakaTune/alpakaTune.hpp>`` in your source.

Next, try :doc:`history_workflows` with this same executable. You can replay
a fixed winner or adapt among the saved configurations without another search.
:doc:`execution_modes` explains these independent choices. For larger spaces,
:doc:`automatic_spaces` generates candidates from declared domains and measured
feedback. To compile batch alternatives ahead of time, see
:doc:`compile_time_tuning`.
