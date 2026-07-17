# Tuned matrix multiplication

This example computes a row-major `M x K` by `K x N` single-precision matrix
multiplication. It uses a two-level output tile, cooperatively loaded shared-memory
tiles, register accumulators, and Alpaka `Simd` packs. The independently tunable
compile-time parameters are:

- `blockRows` and `blockColumns`: the two output-tile directions;
- `kTile`: the reduction tile and therefore one direction of both shared arrays;
- `rowsPerThread`: register-level row tiling;
- `simdWidth`: adjacent output columns calculated together.

`blockRows`, `blockColumns`, and `kTile` directly determine the compile-time
`declareSharedMdArray` extents. The B tile has one padded column to avoid common
shared-memory bank-conflict patterns. The compile-time choices are deliberately
bounded to `3 * 3 * 2 * 2 * 2 = 72` kernel variants.

`numFrames`, `frameExtent`, and `tilesPerGroup` are runtime tunables.
`tilesPerGroup` chooses `1`, `2`, `4`, `8`, or `16` adjacent output tiles for a
scheduled block to process sequentially. The runtime search therefore contains
`4 * 3 * 5 = 60` schedules without producing more kernel types, for exactly
`72 * 60 = 4320` legal configurations.

The structure follows the reuse, coalescing, and shared-memory recommendations in
the [NVIDIA CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-in-matrix-multiplication-c-ab)
and the hierarchical block/warp/thread tiling described by
[CUTLASS efficient GEMM](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md).

```bash
build/example/matrixMultiplication/alpakaTune_matrixMultiplication \
  --backend cuda:nvidiaGpu --executor gpuCuda -m 512 -n 512 -k 512
```
