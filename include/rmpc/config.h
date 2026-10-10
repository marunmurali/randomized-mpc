// Copyright (c) 2026 Arun Muraleedharan.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#pragma once

#include <cstdint>

namespace rmpc {

/**
 * @brief Engine settings, the same for every problem.
 */
struct Config {
  int n_horizon{100};   /**< prediction steps N */
  int n_samples{1000};  /**< random input sequences per solve */
  int n_basis{7};       /**< IDCT coefficients drawn per sample (<= N) */
  float dt{0.02f};      /**< step length [s] */
  float u_min{-1.0f};   /**< hard lower input bound */
  float u_max{1.0f};    /**< hard upper input bound */
  float du_max{1.0f};   /**< hard input rate limit [unit/s] */
  float lag_tau{0.0f};  /**< first-order actuator lag [s], 0 = off */
  bool warm_start{false};  /**< reuse last best sequence as a sample */
  std::uint64_t seed{0};   /**< RNG seed, call i uses seed + i */
  int n_debug_samples{0};  /**< copy back this many sample rollouts */
};

/**
 * @brief Check a config.
 *
 * @return nullptr if valid, otherwise a description of the first problem
 */
inline const char* validate(const Config& c) noexcept {
  if (c.n_horizon < 2) return "n_horizon must be >= 2";
  if (c.n_samples < 2) return "n_samples must be >= 2";
  if (c.n_basis < 1 || c.n_basis > c.n_horizon) {
    return "n_basis must be in [1, n_horizon]";
  }
  if (!(c.dt > 0.0f)) return "dt must be > 0";
  if (!(c.u_min < c.u_max)) return "u_min must be < u_max";
  if (!(c.du_max > 0.0f)) return "du_max must be > 0";
  if (!(c.lag_tau >= 0.0f)) return "lag_tau must be >= 0";
  if (c.n_debug_samples < 0 || c.n_debug_samples > c.n_samples) {
    return "n_debug_samples must be in [0, n_samples]";
  }
  return nullptr;
}

}  // namespace rmpc
