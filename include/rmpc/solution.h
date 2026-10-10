// Copyright (c) 2026 Arun Muraleedharan.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#pragma once

#include <vector>

namespace rmpc {

/**
 * @brief One rollout over the horizon.
 *
 * Index k runs over 0..N-1. states[0] is the initial state x0, and u_cmd[0]
 * is the input already being applied (u0); the first decided input is
 * u_cmd[1]. u_applied is u_cmd after the actuator lag (equal to u_cmd when
 * the lag is off); the model and cost see u_applied.
 */
template <class State>
struct Trajectory {
  std::vector<float> u_cmd;
  std::vector<float> u_applied;
  std::vector<State> states;
  float cost{0.0f};
};

/**
 * @brief Result of a solve: the best rollout plus optional debug samples.
 */
template <class State>
struct Solution {
  Trajectory<State> best;
  int best_index{0};
  float dt{0.0f};
  /** first Config::n_debug_samples rollouts (sample 0 holds u0, sample 1 is
   * the warm start when enabled) */
  std::vector<Trajectory<State>> samples;

  /** input to command for the next step */
  [[nodiscard]] float nextInput() const noexcept { return best.u_cmd[1]; }
};

}  // namespace rmpc
