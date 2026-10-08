# GEMM optimization

Tune a tiled FP32 matmul and compare it with the default and vendor GEMM through
alpakaVendor. All paths compute row-major `C = A * B`, with `alpha=1`, `beta=0`.
For the measured CPU/GPU winners, direct library call, and persistent-handle
strict FP32 cuBLAS comparison, see the [compiled matmul example](../../example/matmul/README.md).

## Build and run

From the repository root, with CUDA and cuBLAS installed:

```sh
cmake -S . -B build/gemm -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DalpakaTune_BUILD_BENCHMARKS=ON -Dalpaka_DEP_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build/gemm --target alpakaTune_gemm -j 2
build/gemm/benchmarks/matmul/alpakaTune_gemm \
  --backend cuda:nvidiaGpu --executor gpuCuda \
  --shape 1024x1024x1024 --shape 2048x256x1024 \
  --csv benchmarks/results/gemm.csv
```

Repeat `--shape MxNxK` to select shapes; `--device INDEX` chooses the device.
Use `--self-test` for full-output checks on small aligned and irregular matrices.
Normal runs validate edges and deterministic samples against FP64 dot products.

The pinned alpakaVendor dependency is fetched only for enabled benchmarks.
CUDA enables cuBLAS and HIP enables rocBLAS; CPU and oneAPI require their BLAS
libraries and `alpakaV_DEP_OPENBLAS` or `alpakaV_DEP_ONEMKL` respectively.
Use `--kernel-only` to skip an unavailable vendor backend, or configure
`alpakaTune_GEMM_VENDOR=OFF` to omit the dependency.

## Tuning space

The bounded catalog has 12 compatible layouts:

| Block M × N | Registers per thread | K tile / shared buffers |
|---|---|---|
| 64 × 64 | 4 × 4 | 16 / 1, 16 / 2, 32 / 1 |
| 64 × 128 | 4 × 8 | 16 / 1, 16 / 2, 32 / 1 |
| 128 × 64 | 8 × 4 | 16 / 1, 16 / 2, 32 / 1 |
| 128 × 128 | 8 × 8 | 16 / 1, 16 / 2, 32 / 1 |

Layouts use 256 logical workers and at most 32 KiB of shared memory.
The default is 64 × 64 × 16 with one shared buffer. Exhaustive exploration
measures every legal layout, using one warmup and seven samples by default
(`--samples-per-candidate N`). Evaluation launches the selected specialization
directly, with three warmups and 15 samples per path (`--repetitions N`).
Implementation order rotates between repetitions.

`--history FILE --reuse-history` reuses a compatible shape/device/executor
context and collects fresh evaluation timings; missing contexts fail.
`--csv FILE` writes the summary, evaluation samples, and tuner observations.
Reported speedups divide default or vendor time by tuned time; below one
means slower. Break-even divides tuning cost by the measured time saving.
Paired-bootstrap 95% intervals require at least five repetitions.
Allocation, correctness checks, and transfers are outside timing.

Vendor GEMM requests `Precision::exact`, which uses pedantic FP32 on CUDA.
**The pinned alpakaVendor wrapper creates and destroys a cuBLAS handle per call**:
its result includes wrapper and submission overhead, rather than isolated native
cuBLAS kernel throughput. The compiled example's persistent-handle comparison
provides the separate native reference.
