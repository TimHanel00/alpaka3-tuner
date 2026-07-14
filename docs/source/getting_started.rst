Getting started
===============

alpakaTune exposes a header-only C++20 API and links its YAML dependency through
its CMake target. A standalone checkout downloads the pinned Alpaka3 revision
when CMake configures it.

FetchContent or subproject use
------------------------------

.. code-block:: cmake

   include(FetchContent)
   FetchContent_Declare(
     alpakaTune
     GIT_REPOSITORY https://github.com/TimHanel00/alpaka3-tuner.git
     GIT_TAG dev)
   FetchContent_MakeAvailable(alpakaTune)

   target_link_libraries(my_app PRIVATE alpakaTune::alpakaTune)
   alpaka_finalize(my_app)

Installed package use
---------------------

Installing alpakaTune also installs the default YAML file and resolves the
Alpaka3, yaml-cpp, and optional nlohmann/json dependencies. Consumers use the
exported CMake target:

.. code-block:: cmake

   find_package(alpakaTune CONFIG REQUIRED)
   target_link_libraries(my_app PRIVATE alpakaTune::alpakaTune)
   alpaka_finalize(my_app)

Every target that includes Alpaka headers must call ``alpaka_finalize`` after
linking its dependencies.
