// Copyright (c) 2026 Arun Muraleedharan.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#pragma once

#include <math.h>

#include "rmpc/problem.h"

namespace rmpc::detail {

/** Device buffers are laid out step-major: element (k, sample j) is at
 * k * stride + j, so neighbouring threads touch neighbouring memory. */

/**
 * @brief Orthonormal IDCT (DCT-III) basis value.
 *
 * Column i of the N x N basis is the i-th cosine mode over the horizon:
 * sqrt(2/N) * c_i * cos(pi * i * (k + 0.5) / N), with c_0 = 1/sqrt(2) and
 * c_i = 1 otherwise. Mode 0 is constant, higher modes oscillate faster, so
 * drawing only the first n_basis coefficients gives smooth sequences.
 */
RMPC_HD float basisValue(int k, int i, int n_horizon) {
  const float c = (i == 0) ? 0.70710678f : 1.0f;
  return sqrtf(2.0f / static_cast<float>(n_horizon)) * c *
         cosf(static_cast<float>(M_PI) * static_cast<float>(i) *
              (static_cast<float>(k) + 0.5f) / static_cast<float>(n_horizon));
}

/** basis[k * n_basis + i] for k < n_horizon, i < n_basis */
static __global__ void buildBasis(float* basis, int n_horizon, int n_basis) {
  const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= n_horizon * n_basis) {
    return;
  }
  basis[idx] = basisValue(idx / n_basis, idx % n_basis, n_horizon);
}

RMPC_HD float clampf(float x, float lo, float hi) {
  return fmaxf(lo, fminf(x, hi));
}

/**
 * @brief Turn random IDCT coefficients into bounded input sequences.
 *
 * One thread per sample. u_cmd[0] = u0 (the input already applied); for
 * k >= 1 the increment is clamped to +/- du_step and the input to
 * [u_min, u_max]. Sample 0 holds u0. With a warm start, sample 1 follows the
 * previous best sequence shifted by one step, through the same clamps.
 *
 * @param randoms uniform (0, 1] numbers, laid out [i * n_samples + j]
 */
static __global__ void buildInputs(const float* basis, const float* randoms,
                            const float* warm, float* u_cmd, int n_horizon,
                            int n_basis, int n_samples, float u0, float u_min,
                            float u_max, float du_step, bool warm_start) {
  const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (j >= n_samples) {
    return;
  }

  const bool is_warm = warm_start && j == 1;

  float u = u0;
  u_cmd[j] = u0;

  for (int k = 1; k < n_horizon; ++k) {
    float du = 0.0f;
    if (is_warm) {
      du = warm[min(k + 1, n_horizon - 1)] - u;
    } else if (j != 0) {
      for (int i = 0; i < n_basis; ++i) {
        const float coeff = 2.0f * randoms[i * n_samples + j] - 1.0f;
        du += basis[k * n_basis + i] * coeff;
      }
      du *= du_step;
    }
    u = clampf(u + clampf(du, -du_step, du_step), u_min, u_max);
    u_cmd[k * n_samples + j] = u;
  }
}

/**
 * @brief Roll out sequences through the problem model and add up the cost.
 *
 * One thread per sample (n_active of them, buffers with the given stride).
 * Applies the first-order lag u_app[k] = a * u_app[k-1] + (1 - a) * u_cmd[k]
 * with u_app[0] = u0 (a = 0 means no lag).
 */
template <Problem P>
__global__ void rollout(typename P::State x0, float u0, const float* u_cmd,
                        float* u_app, typename P::State* states, float* cost,
                        const typename P::Reference* ref,
                        typename P::Params params, int n_horizon, int n_active,
                        int stride, float dt, float lag_a) {
  const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (j >= n_active) {
    return;
  }

  typename P::State x = x0;
  float u_prev = u0;
  float total = 0.0f;

  for (int k = 0; k < n_horizon; ++k) {
    const int idx = k * stride + j;
    const float u =
        (k == 0) ? u0 : lag_a * u_prev + (1.0f - lag_a) * u_cmd[idx];

    states[idx] = x;
    u_app[idx] = u;

    total += P::stageCost(x, u, u_prev, ref[k], params, k, n_horizon);

    if (k + 1 < n_horizon) {
      x = P::step(x, u, ref[k], params, dt);
    }
    u_prev = u;
  }

  total += P::terminalCost(x, ref[n_horizon - 1], params, n_horizon);
  cost[j] = total;
}

}  // namespace rmpc::detail
