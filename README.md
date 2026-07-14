# alpakaTune

alpakaTune is a C++20 tuner for Alpaka3 `KernelBundle` launches. YAML-backed
sessions create device-bound contexts; each `context.tune(queue, frameSpec,
prototypeBundle)` call performs one noise-cancelled tuning launch, then later
calls enqueue the persistent winner. Runtime (`RVals`) and compile-time
(`CVals`) candidates share one named `Tunables` configuration.

The full guide, including FetchContent, installed-package, launch-tuning, and
compile-time-tuning examples, is built with Sphinx for Read the Docs under
[`docs/source`](docs/source/index.rst).

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
