// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include "MatmulKernel.hpp"

#include <memory>
#include <stdexcept>
#include <string>

#if ALPAKA_TUNE_MATMUL_CUBLAS && ALPAKA_LANG_CUDA
#include <cublas_v2.h>
#endif

namespace alpakaTune::example::matmul {
// One context per benchmark queue; initialization and destruction are outside
// every timed interval. The queue retains ownership of native submission.
template <typename Api> class NativeCublas {
public:
  explicit NativeCublas(auto const &) {}
  void gemm(auto const &, auto const &, auto const &, auto &, Shape) const {
    throw std::invalid_argument{"Native cuBLAS requires the CUDA backend"};
  }
};

#if ALPAKA_TUNE_MATMUL_CUBLAS && ALPAKA_LANG_CUDA
template <> class NativeCublas<alpaka::api::Cuda> {
  struct Handle {
    cublasHandle_t value{};
    ~Handle() {
      if (value)
        static_cast<void>(cublasDestroy(value));
    }
  };
  std::shared_ptr<Handle> m_handle = std::make_shared<Handle>();

  static void check(cublasStatus_t status, char const *operation) {
    if (status != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error{std::string{operation} + ": cuBLAS status " +
                               std::to_string(static_cast<int>(status))};
  }

public:
  explicit NativeCublas(auto const &queue) {
    queue.enqueueNativeFn([handle = m_handle](cudaStream_t stream) {
      check(cublasCreate(&handle->value), "cublasCreate");
      check(cublasSetStream(handle->value, stream), "cublasSetStream");
      check(cublasSetMathMode(handle->value, CUBLAS_PEDANTIC_MATH),
            "cublasSetMathMode");
      check(cublasSetAtomicsMode(handle->value, CUBLAS_ATOMICS_NOT_ALLOWED),
            "cublasSetAtomicsMode");
    });
    alpaka::onHost::wait(queue);
  }

  void gemm(auto const &queue, auto const &a, auto const &b, auto &c,
            Shape shape) const {
    auto const *pa = alpaka::onHost::data(a);
    auto const *pb = alpaka::onHost::data(b);
    auto *pc = alpaka::onHost::data(c);
    auto const lda =
        static_cast<int>(alpaka::onHost::getPitches(a).y() / sizeof(float));
    auto const ldb =
        static_cast<int>(alpaka::onHost::getPitches(b).y() / sizeof(float));
    auto const ldc =
        static_cast<int>(alpaka::onHost::getPitches(c).y() / sizeof(float));
    queue.enqueueNativeFn([=, handle = m_handle](cudaStream_t) {
      constexpr float alpha = 1.0f, beta = 0.0f;
      check(cublasGemmEx(handle->value, CUBLAS_OP_N, CUBLAS_OP_N,
                         static_cast<int>(shape.n), static_cast<int>(shape.m),
                         static_cast<int>(shape.k), &alpha, pb, CUDA_R_32F, ldb,
                         pa, CUDA_R_32F, lda, &beta, pc, CUDA_R_32F, ldc,
                         CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT),
            "cublasGemmEx");
    });
  }
};
#endif
} // namespace alpakaTune::example::matmul
