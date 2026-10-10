// Copyright (c) 2026 Arun Muraleedharan.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#pragma once

#include "config_yaml.h"
#include "problem.cuh"

namespace paddle_juggling {

/**
 * @brief Load the `model:` and `cost:` sections.
 */
inline rmpc::Result<PaddleJuggling::Params> loadParams(
    const YAML::Node& root) noexcept {
  using examples::read;
  PaddleJuggling::Params p{};
  const YAML::Node m = root["model"];
  const YAML::Node c = root["cost"];

  const bool ok = m && c && read(m, "g", p.g) &&
                  read(m, "restitution", p.restitution) &&
                  read(m, "stroke", p.stroke) && read(c, "apex", p.w.apex) &&
                  read(c, "paddle_pos", p.w.paddle_pos) &&
                  read(c, "paddle_vel", p.w.paddle_vel) &&
                  read(c, "effort", p.w.effort) &&
                  read(c, "terminal_apex", p.w.terminal_apex);
  if (!ok) {
    return "paddle config: missing or invalid key in model/cost sections";
  }
  return p;
}

}  // namespace paddle_juggling
