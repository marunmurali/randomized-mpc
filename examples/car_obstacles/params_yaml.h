// Copyright (c) 2026 Arun Muraleedharan.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#pragma once

#include "../car_steering/params_yaml.h"
#include "problem.cuh"

namespace car_obstacles {

/**
 * @brief Load the car sections (`vehicle:`, `cost:`) plus `collision:` and
 * the `obstacles:` list.
 */
inline rmpc::Result<CarObstacles::Params> loadParams(
    const YAML::Node& root) noexcept {
  using examples::child;
  using examples::read;
  auto car = car_steering::loadParams(root);
  if (car.isError()) {
    return car.error();
  }

  CarObstacles::Params p{};
  p.car = car.value();

  const YAML::Node c = child(root, "collision");
  if (!c || !read(c, "car_radius", p.car_radius) ||
      !read(c, "clearance", p.clearance) ||
      !read(c, "hit_cost", p.hit_cost)) {
    return "obstacles config: missing or invalid key in collision section";
  }

  try {
    const YAML::Node list = child(root, "obstacles");
    if (!list || !list.IsSequence() ||
        list.size() > static_cast<std::size_t>(MAX_OBSTACLES)) {
      return "obstacles config: obstacles must be a list of at most "
             "MAX_OBSTACLES entries";
    }
    for (const auto& o : list) {
      Obstacle& dst = p.obstacles[p.n_obstacles++];
      if (!read(o, "s", dst.s) || !read(o, "d", dst.d) ||
          !read(o, "radius", dst.radius)) {
        return "obstacles config: each obstacle needs s, d and radius";
      }
    }
  } catch (const std::exception&) {
    return "obstacles config: cannot read the obstacles list";
  }
  return p;
}

}  // namespace car_obstacles
