# GEMM optimization

Compare a default tiled Alpaka FP32 matrix multiplication, the same kernel
optimized by alpakaTune, and vendor GEMM through alpakaVendor. All paths compute
`C = A * B` with row-major inputs, `alpha = 1`, and `beta = 0`.

Build and run on an NVIDIA GPU with CUDA and cuBLAS installed:

```sh
cmake -S . -B build/gemm -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DalpakaTune_BUILD_BENCHMARKS=ON \
  -Dalpaka_DEP_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build/gemm --target alpakaTune_gemm -j 2
build/gemm/benchmarks/matmul/alpakaTune_gemm \
  --backend cuda:nvidiaGpu --executor gpuCuda \
  --shape 1024x1024x1024 --shape 2048x256x1024 \
  --csv benchmarks/results/gemm.csv
```

`--shape MxNxK` is repeatable: A has shape M by K, B has shape K by N,
and C has shape M by N. Without `--shape`, the program runs three square
problems (512, 1024, and 2048) and two rectangular problems. Select a device
with `--device INDEX` (default 0).

The optional vendor dependency is pinned and fetched only when benchmarks
are enabled. CUDA enables cuBLAS by default; HIP enables rocBLAS. For CPU
comparisons, enable `alpakaV_DEP_OPENBLAS`; for oneAPI, enable
`alpakaV_DEP_ONEMKL`. Those libraries must be installed. A backend without
a vendor BLAS fails unless `--kernel-only` is supplied. Set
`alpakaTune_GEMM_VENDOR=OFF` to build without alpakaVendor.

## Bounded tuning

The search contains **12 compatible tile layouts**, rather than independent
lists of tile sizes, register sizes, and launch dimensions:

| Block M by N | Register tile per thread | K tile / shared buffers |
|---|---|---|
| 64 by 64 | 4 by 4 | 16 / 1, 16 / 2, 32 / 1 |
| 64 by 128 | 4 by 8 | 16 / 1, 16 / 2, 32 / 1 |
| 128 by 64 | 8 by 4 | 16 / 1, 16 / 2, 32 / 1 |
| 128 by 128 | 8 by 8 | 16 / 1, 16 / 2, 32 / 1 |

Every layout uses 256 logical workers and at most 32 KiB of shared memory.
Block counts are derived from the shape and tile; restrictions reject
incompatible launch/layout pairs and unsupported device resource requirements.
The default kernel is the 64 by 64, K=16, single-buffer layout.

Exhaustive exploration measures every legal layout. At this scale it needs
no surrogate model or stochastic search, gives reproducible coverage, and
has a bounded cost: by default one warm-up plus seven measured launches per
layout. `--samples-per-candidate N` changes the measured sample count.
Selection is fixed after exploration. The evaluation then launches the selected
specialization directly, without tuner instrumentation or dispatch bookkeeping.

The kernel uses cooperative four-float global loads when row pitches and base
addresses permit, transposes A into shared memory, keeps a register output tile,
reuses shared operands across fused multiply-adds, and interleaves thread output
coordinates for coalesced access and shared-memory broadcasts. Double buffering
uses separate shared arrays and one barrier per K tile. It does not use CUDA
asynchronous-copy instructions. Bounds checks handle incomplete tiles.

Store shape/device/executor-specific measurements and reuse the selected layout:

```sh
build/gemm/benchmarks/matmul/alpakaTune_gemm \
  --backend cuda:nvidiaGpu --executor gpuCuda \
  --shape 1024x1024x1024 --history benchmarks/results/gemm-history.json
build/gemm/benchmarks/matmul/alpakaTune_gemm \
  --backend cuda:nvidiaGpu --executor gpuCuda \
  --shape 1024x1024x1024 --history benchmarks/results/gemm-history.json \
  --reuse-history
```

Reuse requires a compatible saved context; a missing context fails rather than
silently selecting a default. Re-evaluation always collects new timing samples.

## Interpret the results

The program checks all legal kernel variants against independently accumulated
FP64 dot products before tuning, then checks the default, selected, and vendor
outputs after evaluation. Normal runs check edges and 64 deterministic sampled
positions; `--self-test` checks every output element on small aligned and
irregular shapes. Validation and transfers are outside the timed sections.

Evaluation uses three warm-ups and 15 timed single-GEMM samples per
implementation by default; change the latter with `--repetitions N`.
Implementation order rotates between repetitions. The console reports median
queue time, GFLOP/s, execution-time reduction, speedup, and tuning wall time.
CUDA/HIP use device timing events; host executors use host-clock timing.
With at least five repetitions, the report also includes a paired-bootstrap
95% speedup interval. An interval containing 1 does not establish an improvement.

- Time reduction: `100 * (1 - tuned_time / default_time)` percent.
- Default speedup: `default_time / tuned_time`.
- Vendor speedup: `vendor_time / tuned_time`; below 1 means the tuned kernel
  is slower than the vendor call.
- Break-even reuses: `ceil(tuning_wall_time / (default_time - tuned_time))`.
  No break-even is reported if tuning does not improve the measured time.
  The CSV also reports break-even using synchronized host wall times.

`--csv FILE` writes a summary, `FILE.samples.csv` with all evaluation samples,
and `FILE.tuning.csv` with current-run tuner launches. Blank vendor or break-even
fields mean unavailable, not zero. The tuning cost excludes allocation,
correctness checking, transfers, and compilation; these costs are common to
both custom-kernel paths. Reused-history cost describes the new process only.

Vendor GEMM requests `Precision::exact`: on CUDA this selects pedantic FP32
compute, excluding TF32 arithmetic. The custom kernel uses FP32 FMA and does
not use tensor cores. These results therefore compare strict FP32 GEMM, rather
than the maximum throughput of reduced-precision matrix multiplication.

**The pinned alpakaVendor CUDA implementation creates and destroys a cuBLAS
handle per GEMM call.** Both queue-event intervals and wall times can include
handle setup and host submission gaps. Consequently the vendor result describes
cuBLAS *through this wrapper*, and must not be presented as isolated native
cuBLAS kernel throughput. Retain the raw measurements and compare on the same
device, compute mode, and shapes; closeness to cuBLAS is a measurement goal,
not a guaranteed result.

## Correctness without a GPU

```sh
cmake -S . -B build/gemm-check -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DalpakaTune_BUILD_BENCHMARKS=ON \
  -DalpakaTune_GEMM_VENDOR=OFF \
  -DalpakaTune_BUILD_TESTING=ON -Dalpaka_DEP_OMP=OFF
cmake --build build/gemm-check --target alpakaTune_gemm -j 2
ctest --test-dir build/gemm-check -R '^alpakaTune_gemm_correctness$' \
  --output-on-failure
```

Serial CPU execution emulates the GPU's logical workers and is intended for
correctness checks. It does not establish GPU performance or concurrent
shared-memory correctness. Run `--self-test --backend cuda:nvidiaGpu
--executor gpuCuda` on a GPU to exercise concurrent execution.
