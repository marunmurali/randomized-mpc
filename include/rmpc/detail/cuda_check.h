// Copyright (c) 2026 Arun Muraleedharan.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#pragma once

#include <cuda_runtime.h>
#include <curand.h>

/**
 * Return the CUDA error string from the enclosing function on failure. The
 * enclosing function must return const char* or a Result<T>.
 */
#define RMPC_CUDA_TRY(expr)                                     \
  do {                                                          \
    if (const cudaError_t rmpc_err_ = (expr);                   \
        rmpc_err_ != cudaSuccess) {                             \
      return cudaGetErrorString(rmpc_err_);                     \
    }                                                           \
  } while (0)

/** As RMPC_CUDA_TRY, for cuRAND calls. */
#define RMPC_CURAND_TRY(expr)                                   \
  do {                                                          \
    if ((expr) != CURAND_STATUS_SUCCESS) {                      \
      return "cuRAND call failed: " #expr;                      \
    }                                                           \
  } while (0)
