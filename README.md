# alpakaTune

alpakaTune is a header-only C++20 library that chooses a fast configuration for
an Alpaka3 kernel. Supply candidate batch sizes, launch shapes, or algorithm
parameters; the tuner measures their cost and selects values for later launches.
By default it minimizes runtime. You can also provide an application-defined
score, including a weighted combination of metrics.

Tune once and replay the winner, keep adapting as the application runs, or
reuse a saved configuration. Runtime (`RVals`) and compile-time (`CVals`)
choices can share one candidate space. Your application controls the kernel
launches and its main loop.

Start with [Your first tuner](https://alpaka3-tuner.readthedocs.io/en/latest/getting_started.html)
to build a minimal CPU example. The [full guide](https://alpaka3-tuner.readthedocs.io/en/latest/)
covers installation, launch tuning, saved history, and custom objectives.
Its Sphinx sources are under [`docs/source`](docs/source/index.rst).
See [`example/README.md`](example/README.md) for the larger examples and their
tuning-collection loops.

The optional `learned_hybrid` strategy uses an offline-trained model with online
adaptation. Model training, datasets, and HPC campaigns are handled by
[`alpakaTune-ml`](https://github.com/TimHanel00/alpakaTune-ml).

## Build the example and tests

```sh
cmake -S . -B build -G Ninja \
  -DalpakaTune_BUILD_TESTING=ON \
  -DalpakaTune_BUILD_EXAMPLES=ON \
  -DalpakaTune_HEADER_CHECKS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

Alpaka3 is downloaded through CMake FetchContent at the pinned upstream
revision recorded in `CMakeLists.txt`.
