# alpakaTune

alpakaTune is a header-only C++20 tuner for Alpaka3 `KernelBundle` launches.
A mutable `TunerConfig` creates device-bound `Tuner` objects through
`makeTuner`; each `tuner.enqueue(queue, frameSpec, prototypeBundle)` call
performs one noise-cancelled tuning launch, then later calls enqueue the
persistent winner. Runtime (`RVals`) and compile-time
(`CVals`) candidates share one named `Tunables` configuration.
`makeTuner` has exactly two forms: one taking an explicit `TunerConfig`, and
one using `tunerConfig()`; both require the tunable bundle and device before
any identity-only names or Alpaka objects.

The full guide, including FetchContent, installed-package, launch-tuning, and
compile-time-tuning examples, is built with Sphinx for Read the Docs under
[`docs/source`](docs/source/index.rst).

Large exhaustive datasets, offline model training, evaluation, and HPC job
orchestration are intentionally maintained in the separate `alpakaTune-ml`
repository. This header-only runtime contains only automatic feature/history
contracts, native learned-model inference, online adaptation, and an optional
promoted model artifact.

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
