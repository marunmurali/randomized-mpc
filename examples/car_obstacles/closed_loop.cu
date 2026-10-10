// Copyright (c) 2026 Arun Muraleedharan.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

/**
 * Closed-loop slalom: the plant is CarObstacles::step (the car model), the
 * same function the controller rolls out on the GPU. Writes CSV logs for
 * animate.py.
 *
 * Usage: car_obstacles_closed_loop [config.yaml] [out_dir]
 */

#include <cmath>
#include <cstdio>
#include <vector>

#include "../car_steering/track.h"
#include "csv_log.h"
#include "params_yaml.h"
#include "rmpc/rmpc.cuh"

using car_obstacles::CarObstacles;
using car_steering::Track;

namespace {

constexpr float BOUND = 4.0f;      // track half width [m]
constexpr int DEBUG_SAMPLES = 16;  // sample rollouts logged per step
constexpr int SAMPLE_K_STRIDE = 4; // log every 4th horizon step

/** smallest gap between the car footprint and any obstacle [m] */
float gap(const CarObstacles::State& x, const CarObstacles::Params& p) {
  float best = INFINITY;
  const float axle[2] = {p.car.lf, -p.car.lr};
  for (int i = 0; i < p.n_obstacles; ++i) {
    const auto& o = p.obstacles[i];
    for (float a : axle) {
      const float ds = x.s + a * std::cos(x.yaw) - o.s;
      const float dd = x.d + a * std::sin(x.yaw) - o.d;
      best = std::fmin(best, std::hypot(ds, dd) - o.radius - p.car_radius);
    }
  }
  return best;
}

}  // namespace

int main(int argc, char** argv) {
  const char* path = argc > 1 ? argv[1] : RMPC_EXAMPLE_DIR "/config.yaml";
  const std::filesystem::path out = argc > 2 ? argv[2] : "logs/car_obstacles";

  const YAML::Node root = examples::loadFile(path);
  auto cfg = examples::loadEngineConfig(root);
  auto params = car_obstacles::loadParams(root);
  if (cfg.isError() || params.isError()) {
    std::fprintf(stderr, "config error: %s\n",
                 cfg.isError() ? cfg.error() : params.error());
    return 1;
  }

  rmpc::Config c = cfg.value();
  c.n_debug_samples = DEBUG_SAMPLES;
  const CarObstacles::Params p = params.value();

  auto created = rmpc::Controller<CarObstacles>::create(c, p);
  if (created.isError()) {
    std::fprintf(stderr, "create failed: %s\n", created.error());
    return 1;
  }
  auto& ctrl = *created.value();

  // straight, gentle S-bend, straight
  const float kappa = 1.0f / 120.0f;
  const Track track(
      {{60.0f, 0.0f}, {60.0f, kappa}, {60.0f, -kappa}, {140.0f, 0.0f}});

  examples::CsvLog track_log(out, "track.csv", "s,cx,cy,lx,ly,rx,ry");
  examples::CsvLog obstacle_log(out, "obstacles.csv", "x,y,radius");
  examples::CsvLog run_log(out, "run.csv",
                           "step,t,s,d,yaw,vy,r,steer_cmd,steer_applied,"
                           "cost,d_target,x,y,psi,gap");
  examples::CsvLog pred_log(out, "predictions.csv", "step,k,x,y");
  examples::CsvLog sample_log(out, "samples.csv", "step,sample,k,x,y,cost");
  if (!run_log.ok()) {
    std::fprintf(stderr, "cannot write logs to %s\n", out.c_str());
    return 1;
  }

  for (float s = 0.0f; s <= track.length(); s += 0.5f) {
    const auto cl = track.pose(s);
    const auto l = track.toCartesian(s, BOUND, 0.0f);
    const auto r = track.toCartesian(s, -BOUND, 0.0f);
    track_log.row("%g,%g,%g,%g,%g,%g,%g", s, cl.x, cl.y, l.x, l.y, r.x, r.y);
  }
  for (int i = 0; i < p.n_obstacles; ++i) {
    const auto& o = p.obstacles[i];
    const auto pos = track.toCartesian(o.s, o.d, 0.0f);
    obstacle_log.row("%g,%g,%g", pos.x, pos.y, o.radius);
  }

  std::vector<CarObstacles::Reference> ref(c.n_horizon);
  const float lookahead = p.car.vx * c.dt * static_cast<float>(c.n_horizon);
  const float lag_a = c.lag_tau > 0.0f ? std::exp(-c.dt / c.lag_tau) : 0.0f;

  CarObstacles::State x{0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  float steer = 0.0f;  // actual (applied) steer angle
  float min_gap = INFINITY;
  int hits = 0;
  int step = 0;

  for (; x.s + lookahead < track.length(); ++step) {
    for (int k = 0; k < c.n_horizon; ++k) {
      const float s_k = x.s + p.car.vx * c.dt * static_cast<float>(k);
      ref[k] = {track.curvature(s_k), 0.0f, BOUND, -BOUND};
    }

    const auto res = ctrl.solve(x, steer, ref);
    if (res.isError()) {
      std::fprintf(stderr, "solve failed at step %d: %s\n", step, res.error());
      return 1;
    }
    const auto& sol = res.value();

    const float g = gap(x, p);
    min_gap = std::fmin(min_gap, g);
    hits += g < 0.0f ? 1 : 0;

    const float t = static_cast<float>(step) * c.dt;
    const auto pose = track.toCartesian(x.s, x.d, x.yaw);
    run_log.row("%d,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g", step, t, x.s,
                x.d, x.yaw, x.vy, x.r, sol.nextInput(), steer, sol.best.cost,
                0.0f, pose.x, pose.y, pose.heading, g);
    for (int k = 0; k < c.n_horizon; ++k) {
      const auto& xk = sol.best.states[k];
      const auto pk = track.toCartesian(xk.s, xk.d, xk.yaw);
      pred_log.row("%d,%d,%g,%g", step, k, pk.x, pk.y);
    }
    for (int j = 0; j < c.n_debug_samples; ++j) {
      const auto& traj = sol.samples[j];
      for (int k = 0; k < c.n_horizon; k += SAMPLE_K_STRIDE) {
        const auto& xk = traj.states[k];
        const auto pk = track.toCartesian(xk.s, xk.d, xk.yaw);
        sample_log.row("%d,%d,%d,%g,%g,%g", step, j, k, pk.x, pk.y,
                       traj.cost);
      }
    }

    // plant: same model, then the actuator follows the next command
    x = CarObstacles::step(x, steer, ref[0], p, c.dt);
    steer = lag_a * steer + (1.0f - lag_a) * sol.nextInput();
  }

  std::printf("steps: %d (%.1f s), steps in contact: %d, "
              "min gap to obstacles: %.2f m\n",
              step, static_cast<float>(step) * c.dt, hits, min_gap);
  std::printf("logs written to %s\n", out.c_str());
  return hits == 0 ? 0 : 2;
}
