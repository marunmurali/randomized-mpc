// Copyright (c) 2026 Arun Muraleedharan.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#pragma once

#include <cuda_runtime.h>
#include <curand.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <vector>

#include "rmpc/config.h"
#include "rmpc/detail/cuda_check.h"
#include "rmpc/detail/device_buffer.h"
#include "rmpc/detail/kernels.cuh"
#include "rmpc/problem.h"
#include "rmpc/result.h"
#include "rmpc/solution.h"

namespace rmpc {

/**
 * @brief Randomized MPC for a single-input problem P.
 *
 * Each solve draws n_samples smooth, bounded input sequences, rolls them all
 * out in parallel on the GPU through P::step, scores them with P::stageCost
 * and P::terminalCost, and returns the cheapest one.
 */
template <Problem P>
class Controller {
 public:
  using State = typename P::State;
  using Params = typename P::Params;
  using Reference = typename P::Reference;

  /**
   * @brief Validate the config and allocate all GPU resources.
   */
  static Result<std::unique_ptr<Controller>> create(
      const Config& config, const Params& params) noexcept {
    if (const char* err = validate(config)) {
      return err;
    }
    std::unique_ptr<Controller> ctrl{new (std::nothrow)
                                         Controller(config, params)};
    if (!ctrl) {
      return "out of host memory";
    }
    if (const char* err = ctrl->init()) {
      return err;
    }
    return std::move(ctrl);
  }

  ~Controller() {
    if (generator_ != nullptr) {
      curandDestroyGenerator(generator_);
    }
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
    }
  }

  // owns a CUDA stream and RNG handle
  Controller(const Controller&) = delete;
  Controller& operator=(const Controller&) = delete;

  /**
   * @brief Find the best input sequence from state x0.
   *
   * @param x0  current state
   * @param u0  input currently being applied
   * @param ref one reference per horizon step (size n_horizon)
   */
  Result<Solution<State>> solve(const State& x0, float u0,
                                std::span<const Reference> ref) noexcept {
    const int nh = config_.n_horizon;
    const int ns = config_.n_samples;

    if (ref.size() != static_cast<std::size_t>(nh)) {
      return "reference size must equal n_horizon";
    }
    RMPC_CUDA_TRY(copyReference(ref));

    RMPC_CURAND_TRY(curandSetPseudoRandomGeneratorSeed(
        generator_, config_.seed + call_index_));
    RMPC_CURAND_TRY(curandGenerateUniform(generator_, randoms_.data(),
                                          randoms_.size()));

    detail::buildInputs<<<blocks(ns), kThreads, 0, stream_>>>(
        basis_.data(), randoms_.data(), warm_.data(), u_cmd_.data(), nh,
        config_.n_basis, ns, u0, config_.u_min, config_.u_max,
        config_.du_max * config_.dt, config_.warm_start && has_warm_);
    RMPC_CUDA_TRY(cudaGetLastError());

    RMPC_CUDA_TRY(launchRollout(x0, u0, ns));

    RMPC_CUDA_TRY(cudaMemcpyAsync(cost_host_.data(), cost_.data(),
                                  ns * sizeof(float), cudaMemcpyDeviceToHost,
                                  stream_));
    RMPC_CUDA_TRY(cudaStreamSynchronize(stream_));

    // NaN costs never compare less, so broken rollouts are never picked
    int best = -1;
    float best_cost = std::numeric_limits<float>::infinity();
    for (int j = 0; j < ns; ++j) {
      if (cost_host_[j] < best_cost) {
        best_cost = cost_host_[j];
        best = j;
      }
    }
    if (best < 0) {
      return "no sample produced a finite cost";
    }

    Solution<State> sol;
    sol.best_index = best;
    sol.dt = config_.dt;
    RMPC_CUDA_TRY(copyTrajectory(best, sol.best));
    sol.best.cost = best_cost;

    sol.samples.resize(config_.n_debug_samples);
    for (int j = 0; j < config_.n_debug_samples; ++j) {
      RMPC_CUDA_TRY(copyTrajectory(j, sol.samples[j]));
      sol.samples[j].cost = cost_host_[j];
    }

    if (config_.warm_start) {
      RMPC_CUDA_TRY(cudaMemcpy2DAsync(
          warm_.data(), sizeof(float), u_cmd_.data() + best,
          ns * sizeof(float), sizeof(float), nh, cudaMemcpyDeviceToDevice,
          stream_));
      has_warm_ = true;
    }

    RMPC_CUDA_TRY(cudaStreamSynchronize(stream_));
    ++call_index_;
    return sol;
  }

  /**
   * @brief Roll out one given input sequence (same lag, model and cost as
   * solve). u_cmd[0] is ignored: the input at k = 0 is u0.
   */
  Result<Trajectory<State>> evaluate(const State& x0, float u0,
                                     std::span<const float> u_cmd,
                                     std::span<const Reference> ref) noexcept {
    const int nh = config_.n_horizon;
    const int ns = config_.n_samples;

    if (ref.size() != static_cast<std::size_t>(nh) ||
        u_cmd.size() != static_cast<std::size_t>(nh)) {
      return "u_cmd and reference sizes must equal n_horizon";
    }
    RMPC_CUDA_TRY(copyReference(ref));
    RMPC_CUDA_TRY(cudaMemcpy2DAsync(u_cmd_.data(), ns * sizeof(float),
                                    u_cmd.data(), sizeof(float), sizeof(float),
                                    nh, cudaMemcpyHostToDevice, stream_));
    RMPC_CUDA_TRY(launchRollout(x0, u0, 1));

    Trajectory<State> traj;
    RMPC_CUDA_TRY(copyTrajectory(0, traj));
    RMPC_CUDA_TRY(cudaMemcpyAsync(&traj.cost, cost_.data(), sizeof(float),
                                  cudaMemcpyDeviceToHost, stream_));
    RMPC_CUDA_TRY(cudaStreamSynchronize(stream_));
    traj.u_cmd[0] = u0;
    return traj;
  }

  /** Forget the warm start and restart the seed sequence. */
  void reset() noexcept {
    has_warm_ = false;
    call_index_ = 0;
  }

  /** Change problem parameters (e.g. cost weights) between solves. */
  void setParams(const Params& params) noexcept { params_ = params; }

  [[nodiscard]] const Config& config() const noexcept { return config_; }
  [[nodiscard]] const Params& params() const noexcept { return params_; }

 private:
  static constexpr int kThreads = 256;

  Controller(const Config& config, const Params& params)
      : config_(config), params_(params) {}

  static int blocks(int n) { return (n + kThreads - 1) / kThreads; }

  const char* init() noexcept {
    const auto nh = static_cast<std::size_t>(config_.n_horizon);
    const auto ns = static_cast<std::size_t>(config_.n_samples);
    const auto nb = static_cast<std::size_t>(config_.n_basis);

    RMPC_CUDA_TRY(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    RMPC_CURAND_TRY(
        curandCreateGenerator(&generator_, CURAND_RNG_PSEUDO_MTGP32));
    RMPC_CURAND_TRY(curandSetStream(generator_, stream_));

    RMPC_CUDA_TRY(basis_.allocate(nh * nb));
    RMPC_CUDA_TRY(randoms_.allocate(nb * ns));
    RMPC_CUDA_TRY(warm_.allocate(nh));
    RMPC_CUDA_TRY(u_cmd_.allocate(nh * ns));
    RMPC_CUDA_TRY(u_app_.allocate(nh * ns));
    RMPC_CUDA_TRY(states_.allocate(nh * ns));
    RMPC_CUDA_TRY(cost_.allocate(ns));
    RMPC_CUDA_TRY(ref_.allocate(nh));
    cost_host_.resize(ns);

    const int n_basis_values = config_.n_horizon * config_.n_basis;
    detail::buildBasis<<<blocks(n_basis_values), kThreads, 0, stream_>>>(
        basis_.data(), config_.n_horizon, config_.n_basis);
    RMPC_CUDA_TRY(cudaGetLastError());
    RMPC_CUDA_TRY(cudaStreamSynchronize(stream_));
    return nullptr;
  }

  cudaError_t copyReference(std::span<const Reference> ref) noexcept {
    return cudaMemcpyAsync(ref_.data(), ref.data(), ref.size_bytes(),
                           cudaMemcpyHostToDevice, stream_);
  }

  cudaError_t launchRollout(const State& x0, float u0, int n_active) noexcept {
    const float lag_a =
        config_.lag_tau > 0.0f ? std::exp(-config_.dt / config_.lag_tau)
                               : 0.0f;
    detail::rollout<P><<<blocks(n_active), kThreads, 0, stream_>>>(
        x0, u0, u_cmd_.data(), u_app_.data(), states_.data(), cost_.data(),
        ref_.data(), params_, config_.n_horizon, n_active, config_.n_samples,
        config_.dt, lag_a);
    return cudaGetLastError();
  }

  /** Copy column j (one sample over the horizon) of the device buffers. */
  cudaError_t copyTrajectory(int j, Trajectory<State>& out) noexcept {
    const auto nh = static_cast<std::size_t>(config_.n_horizon);
    const auto ns = static_cast<std::size_t>(config_.n_samples);
    out.u_cmd.resize(nh);
    out.u_applied.resize(nh);
    out.states.resize(nh);

    cudaError_t err = cudaMemcpy2DAsync(
        out.u_cmd.data(), sizeof(float), u_cmd_.data() + j, ns * sizeof(float),
        sizeof(float), nh, cudaMemcpyDeviceToHost, stream_);
    if (err != cudaSuccess) return err;
    err = cudaMemcpy2DAsync(out.u_applied.data(), sizeof(float),
                            u_app_.data() + j, ns * sizeof(float),
                            sizeof(float), nh, cudaMemcpyDeviceToHost,
                            stream_);
    if (err != cudaSuccess) return err;
    return cudaMemcpy2DAsync(out.states.data(), sizeof(State),
                             states_.data() + j, ns * sizeof(State),
                             sizeof(State), nh, cudaMemcpyDeviceToHost,
                             stream_);
  }

  Config config_;
  Params params_;

  cudaStream_t stream_{nullptr};
  curandGenerator_t generator_{nullptr};

  detail::DeviceBuffer<float> basis_;
  detail::DeviceBuffer<float> randoms_;
  detail::DeviceBuffer<float> warm_;
  detail::DeviceBuffer<float> u_cmd_;
  detail::DeviceBuffer<float> u_app_;
  detail::DeviceBuffer<State> states_;
  detail::DeviceBuffer<float> cost_;
  detail::DeviceBuffer<Reference> ref_;
  std::vector<float> cost_host_;

  bool has_warm_{false};
  std::uint64_t call_index_{0};
};

}  // namespace rmpc
