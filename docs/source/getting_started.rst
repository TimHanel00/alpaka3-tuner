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

Run the README example
----------------------

The README example visits every available device for each enabled executor.
For each combination it increments one device buffer in place, using a launch
specification derived from the buffer extent, device, and executor. Across
launches, the tuner explores different frame counts and frame extents and uses
measured runtimes to guide later configuration choices. alpaka3 maps each
selected ``FrameSpec`` to thread blocks and threads according to the device
and executor.
From the repository root:

.. code-block:: sh

   cmake -S docs/examples/readme -B build-readme -DCMAKE_BUILD_TYPE=Release
   cmake --build build-readme --target readme_example
   ./build-readme/readme_example

For each device/executor combination, the program increments a buffer of 1024
elements over 100 tuned launches. The enabled backends depend on the alpaka
CMake options in your build; see the `alpaka backend configuration
<https://alpaka3.readthedocs.io/en/latest/advanced/cmake.html>`_.
A short-kernel warning is expected for this small workload.
``makeTuner(tunables, device)`` loads the process-default YAML configuration;
set ``ALPAKA_TUNE_CONFIG`` before starting the program to use a different policy.
See :doc:`configuration`.

Verify the README snippet
~~~~~~~~~~~~~~~~~~~~~~~~

The dedicated ``alpakaTune_readme_example`` test extracts the C++ code directly
from the README, builds it, and runs it with the default YAML policy. CMake also
checks that the standalone source matches the snippet exactly. After enabling
repository tests, build and run just this example:

.. code-block:: sh

   cmake -S . -B build -DalpakaTune_BUILD_TESTING=ON
   cmake --build build --target alpakaTune_readme_example
   ctest --test-dir build -R '^alpakaTune_readme_example$' --output-on-failure

Build the repository examples and tests
---------------------------------------

From the repository root, enable the optional examples and tests:

.. code-block:: sh

   cmake -S . -B build -DalpakaTune_BUILD_EXAMPLES=ON -DalpakaTune_BUILD_TESTING=ON
   cmake --build build
   ctest --test-dir build --output-on-failure

CMake uses its default generator; Ninja is optional. Add
``-DalpakaTune_HEADER_CHECKS=ON`` to also check that public headers compile
independently.
