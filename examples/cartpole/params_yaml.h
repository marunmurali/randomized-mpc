// Copyright (c) 2026 Arun Muraleedharan.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#pragma once

#include "config_yaml.h"
#include "problem.cuh"

namespace cartpole {

/**
 * @brief Load the `model:` and `cost:` sections.
 */
inline rmpc::Result<CartPole::Params> loadParams(
    const YAML::Node& root) noexcept {
  using examples::read;
  CartPole::Params p{};
  const YAML::Node m = root["model"];
  const YAML::Node c = root["cost"];

  const bool ok =
      m && c && read(m, "m_cart", p.m_cart) && read(m, "m_pole", p.m_pole) &&
      read(m, "half_length", p.half_length) && read(m, "g", p.g) &&
      read(m, "track_limit", p.track_limit) &&
      read(c, "angle", p.w.angle) && read(c, "position", p.w.position) &&
      read(c, "velocity", p.w.velocity) &&
      read(c, "ang_velocity", p.w.ang_velocity) &&
      read(c, "effort", p.w.effort) && read(c, "rate", p.w.rate) &&
      read(c, "track", p.w.track) &&
      read(c, "terminal_angle", p.w.terminal_angle) &&
      read(c, "terminal_ang_velocity", p.w.terminal_ang_velocity);

  if (!ok) {
    return "cartpole config: missing or invalid key in model/cost sections";
  }
  return p;
}

}  // namespace cartpole
