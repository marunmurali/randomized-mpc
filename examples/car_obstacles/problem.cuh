// Copyright (c) 2026 Arun Muraleedharan.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#pragma once

#include <math.h>

#include "../car_steering/problem.cuh"  // CarSteering, reused as-is
#include "rmpc/problem.h"

namespace car_obstacles {

using car_steering::CarSteering;

inline constexpr int MAX_OBSTACLES = 8;

/** round obstacle at Frenet position (s, d) */
struct Obstacle {
  float s;
  float d;
  float radius;
};

/**
 * @brief The car steering problem plus obstacles on the road.
 *
 * Model and tracking cost are CarSteering's, unchanged; this adds a flat
 * penalty for every predicted step in which the car touches an obstacle.
 * That cost is piecewise constant (zero gradient almost everywhere), and
 * each obstacle can be passed on either side, so a gradient-based solver
 * has nothing to follow and several local minima to get stuck in.
 *
 * The obstacles live in Params, not Reference: each sample reaches a given
 * obstacle at a different horizon step, so the cost checks the sample's own
 * predicted position against the whole list.
 */
struct CarObstacles {
  using State = CarSteering::State;
  using Reference = CarSteering::Reference;

  struct Params {
    CarSteering::Params car;
    Obstacle obstacles[MAX_OBSTACLES];
    int n_obstacles;
    float car_radius;  /**< car footprint: two discs, front and rear axle */
    float clearance;   /**< extra margin around each obstacle [m] */
    float hit_cost;    /**< cost per step in contact, flat */
  };

  RMPC_HD static State step(const State& x, float steer, const Reference& ref,
                            const Params& p, float dt) {
    return CarSteering::step(x, steer, ref, p.car, dt);
  }

  /** true if either footprint disc overlaps an obstacle */
  RMPC_HD static bool collides(const State& x, const Params& p) {
    const float c = cosf(x.yaw);
    const float sn = sinf(x.yaw);
    const float axle[2] = {p.car.lf, -p.car.lr};
    for (int i = 0; i < p.n_obstacles; ++i) {
      const Obstacle& o = p.obstacles[i];
      const float reach = o.radius + p.car_radius + p.clearance;
      for (float a : axle) {
        const float ds = x.s + a * c - o.s;
        const float dd = x.d + a * sn - o.d;
        if (ds * ds + dd * dd < reach * reach) {
          return true;
        }
      }
    }
    return false;
  }

  RMPC_HD static float stageCost(const State& x, float steer,
                                 float steer_prev, const Reference& ref,
                                 const Params& p, int k, int n) {
    const float hit = collides(x, p) ? p.hit_cost / n : 0.0f;
    return CarSteering::stageCost(x, steer, steer_prev, ref, p.car, k, n) +
           hit;
  }

  RMPC_HD static float terminalCost(const State& x, const Reference& ref,
                                    const Params& p, int n) {
    return CarSteering::terminalCost(x, ref, p.car, n);
  }
};

static_assert(rmpc::Problem<CarObstacles>);

}  // namespace car_obstacles
