// Copyright (c) 2026 Arun Muraleedharan.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

/**
 * Closed-loop run: the plant is CarSteering::step, the same function the
 * controller rolls out on the GPU. The car starts 1 m off the line, drives
 * an S-curve, then a lane change. Writes CSV logs for animate.py.
 *
 * Usage: car_closed_loop [config.yaml] [out_dir]
 */

#include <cmath>
#include <cstdio>
#include <vector>

#include "csv_log.h"
#include "params_yaml.h"
#include "rmpc/rmpc.cuh"
#include "track.h"

using car_steering::CarSteering;
using car_steering::Track;

namespace {

constexpr float BOUND = 4.0f;           // track half width [m]
constexpr float LANE_OFFSET = 2.5f;     // lane change target [m]
constexpr float LANE_START = 215.0f;    // [m]
constexpr float LANE_END = 270.0f;      // [m]
constexpr int DEBUG_SAMPLES = 16;       // sample rollouts logged per step
constexpr int SAMPLE_K_STRIDE = 4;      // log every 4th horizon step

float laneTarget(float s) {
  return (s >= LANE_START && s < LANE_END) ? LANE_OFFSET : 0.0f;
}

void fillReference(const Track& track, float s, float vx, float dt,
                   std::vector<CarSteering::Reference>& ref) {
  for (std::size_t k = 0; k < ref.size(); ++k) {
    const float s_k = s + vx * dt * static_cast<float>(k);
    ref[k] = {track.curvature(s_k), laneTarget(s_k), BOUND, -BOUND};
  }
}

}  // namespace

int main(int argc, char** argv) {
  const char* path = argc > 1 ? argv[1] : RMPC_EXAMPLE_DIR "/config.yaml";
  const std::filesystem::path out = argc > 2 ? argv[2] : "logs/car_steering";

  const YAML::Node root = examples::loadFile(path);
  auto cfg = examples::loadEngineConfig(root);
  auto params = car_steering::loadParams(root);
  if (cfg.isError() || params.isError()) {
    std::fprintf(stderr, "config error: %s\n",
                 cfg.isError() ? cfg.error() : params.error());
    return 1;
  }

  rmpc::Config c = cfg.value();
  c.n_debug_samples = DEBUG_SAMPLES;
  const CarSteering::Params p = params.value();

  auto created = rmpc::Controller<CarSteering>::create(c, p);
  if (created.isError()) {
    std::fprintf(stderr, "create failed: %s\n", created.error());
    return 1;
  }
  auto& ctrl = *created.value();

  // straight, S-curve (+45, -90, +45 deg on 50 m radius), straight
  const float kappa = 1.0f / 50.0f;
  const float arc45 = 50.0f * static_cast<float>(M_PI) / 4.0f;
  const Track track({{40.0f, 0.0f},
                     {arc45, kappa},
                     {2.0f * arc45, -kappa},
                     {arc45, kappa},
                     {120.0f, 0.0f}});

  examples::CsvLog track_log(out, "track.csv", "s,cx,cy,lx,ly,rx,ry");
  examples::CsvLog run_log(out, "run.csv",
                           "step,t,s,d,yaw,vy,r,steer_cmd,steer_applied,"
                           "cost,d_target,x,y,psi");
  examples::CsvLog pred_log(out, "predictions.csv", "step,k,x,y");
  examples::CsvLog sample_log(out, "samples.csv", "step,sample,k,x,y,cost");
  if (!run_log.ok()) {
    std::fprintf(stderr, "cannot write logs to %s\n", out.c_str());
    return 1;
  }

  const int n_track = static_cast<int>(track.length() / 0.5f);
  for (int i = 0; i <= n_track; ++i) {
    const float s = 0.5f * static_cast<float>(i);
    const auto cl = track.pose(s);
    const auto l = track.toCartesian(s, BOUND, 0.0f);
    const auto r = track.toCartesian(s, -BOUND, 0.0f);
    track_log.row("%g,%g,%g,%g,%g,%g,%g", s, cl.x, cl.y, l.x, l.y, r.x, r.y);
  }

  std::vector<CarSteering::Reference> ref(c.n_horizon);
  const float lookahead = p.vx * c.dt * static_cast<float>(c.n_horizon);
  const float lag_a = c.lag_tau > 0.0f ? std::exp(-c.dt / c.lag_tau) : 0.0f;

  CarSteering::State x{0.0f, 1.0f, 0.15f, 0.0f, 0.0f};
  float steer = 0.0f;  // actual (applied) steer angle
  float sq_err = 0.0f;
  float settled_max_err = 0.0f;
  float last_change_s = 0.0f;  // where curvature or lane target last changed
  CarSteering::Reference prev_ref{};
  int settled_steps = 0;
  int step = 0;

  for (; x.s + lookahead < track.length(); ++step) {
    fillReference(track, x.s, p.vx, c.dt, ref);
    const auto res = ctrl.solve(x, steer, ref);
    if (res.isError()) {
      std::fprintf(stderr, "solve failed at step %d: %s\n", step, res.error());
      return 1;
    }
    const auto& sol = res.value();

    const float t = static_cast<float>(step) * c.dt;
    const float e = x.d - ref[0].d_target;
    const auto pose = track.toCartesian(x.s, x.d, x.yaw);
    run_log.row("%d,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g", step, t, x.s,
                x.d, x.yaw, x.vy, x.r, sol.nextInput(), steer, sol.best.cost,
                ref[0].d_target, pose.x, pose.y, pose.heading);

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

    sq_err += e * e;
    // settled: reference constant for 30 m behind and 20 m (1 s) ahead
    if (step == 0 || ref[0].curvature != prev_ref.curvature ||
        ref[0].d_target != prev_ref.d_target) {
      last_change_s = x.s;
    }
    prev_ref = ref[0];
    const auto& ahead = ref[ref.size() / 2];
    const bool constant_ahead = ahead.curvature == ref[0].curvature &&
                                ahead.d_target == ref[0].d_target;
    if (x.s - last_change_s > 30.0f && constant_ahead) {
      ++settled_steps;
      settled_max_err = std::fmax(settled_max_err, std::fabs(e));
    }

    // plant: same model, then the actuator follows the next command
    x = CarSteering::step(x, steer, ref[0], p, c.dt);
    steer = lag_a * steer + (1.0f - lag_a) * sol.nextInput();
  }

  std::printf("steps: %d (%.1f s), rms lateral error: %.3f m, "
              "max settled lateral error: %.3f m (%d settled steps)\n",
              step, static_cast<float>(step) * c.dt,
              std::sqrt(sq_err / static_cast<float>(step)), settled_max_err,
              settled_steps);
  std::printf("logs written to %s\n", out.c_str());
  return 0;
}
