// Copyright (c) 2026 Arun Muraleedharan.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#pragma once

#include <yaml-cpp/yaml.h>

#include <string>

#include "rmpc/config.h"
#include "rmpc/result.h"

namespace examples {

/**
 * @brief Read node[key] into out.
 *
 * @return false if the key is missing or has the wrong type
 */
template <class T>
bool read(const YAML::Node& node, const char* key, T& out) noexcept {
  try {
    if (!node[key]) {
      return false;
    }
    out = node[key].as<T>();
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

/**
 * @brief Load the `engine:` section shared by every example.
 */
inline rmpc::Result<rmpc::Config> loadEngineConfig(
    const YAML::Node& root) noexcept {
  const YAML::Node e = root["engine"];
  rmpc::Config c;
  if (!e || !read(e, "n_horizon", c.n_horizon) ||
      !read(e, "n_samples", c.n_samples) || !read(e, "n_basis", c.n_basis) ||
      !read(e, "dt", c.dt) || !read(e, "u_min", c.u_min) ||
      !read(e, "u_max", c.u_max) || !read(e, "du_max", c.du_max)) {
    return "engine: missing or invalid required key (n_horizon, n_samples, "
           "n_basis, dt, u_min, u_max, du_max)";
  }
  // optional keys keep their defaults
  read(e, "lag_tau", c.lag_tau);
  read(e, "warm_start", c.warm_start);
  read(e, "seed", c.seed);
  read(e, "n_debug_samples", c.n_debug_samples);

  if (const char* err = rmpc::validate(c)) {
    return err;
  }
  return c;
}

/** Load a YAML file without throwing; returns a null node on failure. */
inline YAML::Node loadFile(const std::string& path) noexcept {
  try {
    return YAML::LoadFile(path);
  } catch (const std::exception&) {
    return YAML::Node{};
  }
}

}  // namespace examples
