// Copyright (c) 2026 Arun Muraleedharan.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#pragma once

#include <cuda_runtime.h>

#include <concepts>
#include <type_traits>

/**
 * Marks a function as callable from both host and device code. Problem
 * functions use it so the same model runs inside the GPU rollout and on the
 * host (closed-loop plant, tests, plotting).
 */
#define RMPC_HD __host__ __device__ __forceinline__

namespace rmpc {

/**
 * @brief A single-input optimal control problem.
 *
 * A problem is a struct with three trivially copyable types and three static
 * RMPC_HD functions:
 *
 * - State:     the integrated state x.
 * - Params:    constants (model parameters, cost weights), set once.
 * - Reference: what to track, one value per horizon step.
 *
 * - step(x, u, ref, params, dt) -> State
 *     One discrete step x_{k+1} = f(x_k, u_k). The integrator is up to the
 *     problem (Euler, RK4, ...).
 * - stageCost(x, u, u_prev, ref, params, k, n_horizon) -> float
 *     Cost of state x_k with applied input u_k at step k. u_prev is the
 *     applied input at step k-1 (u0 at k = 0), for rate penalties. Any
 *     weighting over the horizon is up to the problem.
 * - terminalCost(x, ref, params, n_horizon) -> float
 *     Cost of the last predicted state.
 */
template <class P>
concept Problem =
    std::is_trivially_copyable_v<typename P::State> &&
    std::is_trivially_copyable_v<typename P::Params> &&
    std::is_trivially_copyable_v<typename P::Reference> &&
    requires(const typename P::State& x, float u, float u_prev,
             const typename P::Reference& ref, const typename P::Params& p,
             float dt, int k, int n) {
      { P::step(x, u, ref, p, dt) } -> std::same_as<typename P::State>;
      { P::stageCost(x, u, u_prev, ref, p, k, n) } -> std::convertible_to<float>;
      { P::terminalCost(x, ref, p, n) } -> std::convertible_to<float>;
    };

}  // namespace rmpc
