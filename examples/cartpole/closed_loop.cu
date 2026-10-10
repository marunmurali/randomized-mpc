// Copyright (c) 2026 Arun Muraleedharan.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

/**
 * Closed-loop swing-up: the plant is CartPole::step, the same function the
 * controller rolls out on the GPU. Starts hanging down at rest and runs for
 * a fixed time. Writes CSV logs for animate.py.
 *
 * Usage: cartpole_closed_loop [config.yaml] [out_dir] [seconds]
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "csv_log.h"
#include "params_yaml.h"
#include "rmpc/rmpc.cuh"

using cartpole::CartPole;

namespace {

constexpr int DEBUG_SAMPLES = 16;   // sample rollouts logged per step
constexpr int SAMPLE_K_STRIDE = 4;  // log every 4th horizon step
constexpr float UPRIGHT = 0.1f;     // |theta| counted as balanced [rad]

/** angle wrapped to [-pi, pi) */
float wrap(float a) {
  const auto pi = static_cast<float>(M_PI);
  return a - 2.0f * pi * std::floor((a + pi) / (2.0f * pi));
}

}  // namespace

int main(int argc, char** argv) {
  const char* path = argc > 1 ? argv[1] : RMPC_EXAMPLE_DIR "/config.yaml";
  const std::filesystem::path out = argc > 2 ? argv[2] : "logs/cartpole";
  const float duration = argc > 3 ? std::strtof(argv[3], nullptr) : 10.0f;

  const YAML::Node root = examples::loadFile(path);
  auto cfg = examples::loadEngineConfig(root);
  auto params = cartpole::loadParams(root);
  if (cfg.isError() || params.isError()) {
    std::fprintf(stderr, "config error: %s\n",
                 cfg.isError() ? cfg.error() : params.error());
    return 1;
  }

  rmpc::Config c = cfg.value();
  c.n_debug_samples = DEBUG_SAMPLES;
  const CartPole::Params p = params.value();

  auto created = rmpc::Controller<CartPole>::create(c, p);
  if (created.isError()) {
    std::fprintf(stderr, "create failed: %s\n", created.error());
    return 1;
  }
  auto& ctrl = *created.value();

  examples::CsvLog run_log(out, "run.csv",
                           "step,t,x,x_dot,theta,theta_dot,u_cmd,u_applied,"
                           "cost,x_target");
  examples::CsvLog pred_log(out, "predictions.csv", "step,k,x,theta");
  examples::CsvLog sample_log(out, "samples.csv", "step,sample,k,x,theta,cost");
  examples::CsvLog meta_log(out, "meta.csv",
                            "dt,half_length,track_limit,u_min,u_max");
  if (!run_log.ok()) {
    std::fprintf(stderr, "cannot write logs to %s\n", out.c_str());
    return 1;
  }
  meta_log.row("%g,%g,%g,%g,%g", c.dt, p.half_length, p.track_limit, c.u_min,
               c.u_max);

  const std::vector<CartPole::Reference> ref(c.n_horizon, {0.0f});
  const float lag_a = c.lag_tau > 0.0f ? std::exp(-c.dt / c.lag_tau) : 0.0f;
  const int steps = static_cast<int>(std::lround(duration / c.dt));

  CartPole::State x{0.0f, 0.0f, static_cast<float>(M_PI), 0.0f};
  float force = 0.0f;  // actual (applied) force

  float upright_since = -1.0f;  // start of the current balanced stretch
  float first_balanced = -1.0f; // start of the first stretch held >= 2 s
  float max_abs_x = 0.0f;

  for (int step = 0; step < steps; ++step) {
    const auto res = ctrl.solve(x, force, ref);
    if (res.isError()) {
      std::fprintf(stderr, "solve failed at step %d: %s\n", step, res.error());
      return 1;
    }
    const auto& sol = res.value();
    const float t = static_cast<float>(step) * c.dt;

    run_log.row("%d,%g,%g,%g,%g,%g,%g,%g,%g,%g", step, t, x.x, x.x_dot,
                x.theta, x.theta_dot, sol.nextInput(), force, sol.best.cost,
                ref[0].x_target);
    for (int k = 0; k < c.n_horizon; ++k) {
      const auto& xk = sol.best.states[k];
      pred_log.row("%d,%d,%g,%g", step, k, xk.x, xk.theta);
    }
    for (int j = 0; j < c.n_debug_samples; ++j) {
      const auto& traj = sol.samples[j];
      for (int k = 0; k < c.n_horizon; k += SAMPLE_K_STRIDE) {
        sample_log.row("%d,%d,%d,%g,%g,%g", step, j, k, traj.states[k].x,
                       traj.states[k].theta, traj.cost);
      }
    }

    if (std::fabs(wrap(x.theta)) < UPRIGHT) {
      if (upright_since < 0.0f) upright_since = t;
      if (first_balanced < 0.0f && t - upright_since >= 2.0f) {
        first_balanced = upright_since;
      }
    } else {
      upright_since = -1.0f;
    }
    max_abs_x = std::fmax(max_abs_x, std::fabs(x.x));

    // plant: same model, then the actuator follows the next command
    x = CartPole::step(x, force, ref[0], p, c.dt);
    force = lag_a * force + (1.0f - lag_a) * sol.nextInput();
  }

  if (first_balanced >= 0.0f) {
    std::printf("balanced (|theta| < %.2f rad for >= 2 s) from t = %.2f s\n",
                UPRIGHT, first_balanced);
  } else {
    std::printf("not balanced within %.1f s\n", duration);
  }
  std::printf("final theta: %.3f rad, max |x|: %.2f m (limit %.2f)\n",
              wrap(x.theta), max_abs_x, p.track_limit);
  std::printf("logs written to %s\n", out.c_str());
  return first_balanced >= 0.0f ? 0 : 2;
}
