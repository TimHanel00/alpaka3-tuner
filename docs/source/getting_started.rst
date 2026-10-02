Your first tuner
================

This tutorial adds one to every element of an array. Its batch size changes
how much work a worker handles at once; all candidates must produce the same
answer. We use the host CPU so you can try the API without a GPU toolchain.
For the underlying device and buffer vocabulary, see the
`Alpaka getting-started tutorial
<https://alpaka3.readthedocs.io/en/latest/tutorial/motivation.html>`_.
You need CMake 3.25 or newer, a C++20 compiler, and Git. CMake fetches the
pinned dependencies on the first configure.

Build and run
-------------

From the repository root:

.. code-block:: sh

   cmake -S docs/examples/first_tuner -B build-tutorial -DCMAKE_BUILD_TYPE=Release
   cmake --build build-tutorial --target first_tuner --parallel 4
   ./build-tutorial/first_tuner

The program checks every result after each of 20 launches and prints
``candidates=3; completed=1; loaded=0``. A short-kernel warning is expected:
this small example demonstrates the interface, not a guaranteed speedup.

Read the working source
-----------------------

.. literalinclude:: ../examples/first_tuner/main.cpp
   :language: cpp
   :start-at: #include

There are three steps to the integration:

1. **Describe the choices.** ``batchSize`` is a marker, and
   ``RVals{32u, 64u, 128u}`` is an explicit list of three values. The kernel
   accepts the selected value as an ordinary argument. Its bounds check keeps
   the final partial batch correct.
2. **Keep one tuner.** ``makeTuner(config, tunables, device, ...)`` snapshots
   both the settings and the candidate space. Here ``onlineFixed`` tries a
   finite search, with three measurements per candidate and a 100-launch
   guard. Later calls replay the best measured candidate. The workload label
   and array size help identify compatible history.
3. **Replace the launch.** Use ``tuner.enqueue(queue, frame, bundle)`` where
   you would normally use ``queue.enqueue(frame, bundle)``. Each call launches
   once; it does not run a complete search internally. The application still
   asks for exactly 20 launches.

The timed non-blocking queue measures online launches. Copies use that same
queue, followed by ``wait``, so results are ready even during asynchronous
winner replay. Copies and the host correctness check are outside the measured
kernel interval. See :doc:`instrumentation` before integrating multiple queues.

Changing choices does not automatically preserve correctness: every candidate
must be valid for your kernel, input, and executor. Express known conditions
with restrictions, or validate produced results as described in
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

Next, try :doc:`history_workflows` with this same executable, or change its
batch values to compile-time candidates in :doc:`compile_time_tuning`.
