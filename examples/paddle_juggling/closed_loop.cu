// Copyright (c) 2026 Arun Muraleedharan.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

/**
 * Closed-loop juggling: the plant is PaddleJuggling::step, the same
 * function the controller rolls out on the GPU. The ball is dropped from
 * rest; the target apex steps down halfway through. Writes CSV logs for
 * animate.py and reports the apex reached after every bounce.
 *
 * Usage: paddle_closed_loop [config.yaml] [out_dir]
 */

#include <cmath>
#include <cstdio>
#include <vector>

#include "csv_log.h"
#include "params_yaml.h"
#include "rmpc/rmpc.cuh"

using paddle_juggling::PaddleJuggling;

namespace {

constexpr int DEBUG_SAMPLES = 16;   // sample rollouts logged per step
constexpr int SAMPLE_K_STRIDE = 3;  // log every 3rd horizon step
constexpr float SETTLE_TIME = 3.0f; // ignore bounces this soon after a
                                    // target change [s]

struct Scenario {
  float ball_start;
  float apex_target;
  float apex_target_2;
  float duration;
};

}  // namespace

int main(int argc, char** argv) {
  const char* path = argc > 1 ? argv[1] : RMPC_EXAMPLE_DIR "/config.yaml";
  const std::filesystem::path out = argc > 2 ? argv[2] : "logs/paddle_juggling";

  const YAML::Node root = examples::loadFile(path);
  auto cfg = examples::loadEngineConfig(root);
  auto params = paddle_juggling::loadParams(root);
  Scenario sc{};
  const YAML::Node s = examples::child(root, "scenario");
  if (cfg.isError() || params.isError() || !s ||
      !examples::read(s, "ball_start", sc.ball_start) ||
      !examples::read(s, "apex_target", sc.apex_target) ||
      !examples::read(s, "apex_target_2", sc.apex_target_2) ||
      !examples::read(s, "duration", sc.duration)) {
    std::fprintf(stderr, "config error: %s\n",
                 cfg.isError()      ? cfg.error()
                 : params.isError() ? params.error()
                                    : "missing scenario section");
    return 1;
  }

  rmpc::Config c = cfg.value();
  c.n_debug_samples = DEBUG_SAMPLES;
  const PaddleJuggling::Params p = params.value();

  auto created = rmpc::Controller<PaddleJuggling>::create(c, p);
  if (created.isError()) {
    std::fprintf(stderr, "create failed: %s\n", created.error());
    return 1;
  }
  auto& ctrl = *created.value();

  examples::CsvLog run_log(out, "run.csv",
                           "step,t,ball_z,ball_v,paddle_z,paddle_v,u_cmd,"
                           "u_applied,cost,apex,apex_target,bounce");
  examples::CsvLog pred_log(out, "predictions.csv", "step,k,ball_z,paddle_z");
  examples::CsvLog sample_log(out, "samples.csv",
                              "step,sample,k,ball_z,paddle_z,cost");
  examples::CsvLog meta_log(out, "meta.csv",
                            "dt,n_horizon,stroke,u_min,u_max");
  if (!run_log.ok()) {
    std::fprintf(stderr, "cannot write logs to %s\n", out.c_str());
    return 1;
  }
  meta_log.row("%g,%d,%g,%g,%g", c.dt, c.n_horizon, p.stroke, c.u_min,
               c.u_max);

  std::vector<PaddleJuggling::Reference> ref(c.n_horizon);
  const int steps = static_cast<int>(std::lround(sc.duration / c.dt));
  const float switch_time = 0.5f * sc.duration;

  PaddleJuggling::State x{sc.ball_start, 0.0f, 0.0f, 0.0f};
  float accel = 0.0f;
  int bounces = 0;
  int settled_bounces = 0;
  float settled_err_sum = 0.0f;
  float settled_err_max = 0.0f;

  for (int step = 0; step < steps; ++step) {
    const float t = static_cast<float>(step) * c.dt;
    for (int k = 0; k < c.n_horizon; ++k) {
      const float t_k = t + c.dt * static_cast<float>(k);
      ref[k] = {t_k < switch_time ? sc.apex_target : sc.apex_target_2};
    }

    const auto res = ctrl.solve(x, accel, ref);
    if (res.isError()) {
      std::fprintf(stderr, "solve failed at step %d: %s\n", step, res.error());
      return 1;
    }
    const auto& sol = res.value();

    // plant: same model; a bounce is a jump in the ball velocity
    const auto next = PaddleJuggling::step(x, accel, ref[0], p, c.dt);
    const bool bounce = next.ball_v > 0.0f && x.ball_v <= 0.0f &&
                        next.ball_z <= next.paddle_z + 1e-6f;

    run_log.row("%d,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%d", step, t, x.ball_z,
                x.ball_v, x.paddle_z, x.paddle_v, sol.nextInput(), accel,
                sol.best.cost, PaddleJuggling::apex(x, p), ref[0].apex_target,
                bounce ? 1 : 0);
    for (int k = 0; k < c.n_horizon; ++k) {
      const auto& xk = sol.best.states[k];
      pred_log.row("%d,%d,%g,%g", step, k, xk.ball_z, xk.paddle_z);
    }
    for (int j = 0; j < c.n_debug_samples; ++j) {
      const auto& traj = sol.samples[j];
      for (int k = 0; k < c.n_horizon; k += SAMPLE_K_STRIDE) {
        sample_log.row("%d,%d,%d,%g,%g,%g", step, j, k, traj.states[k].ball_z,
                       traj.states[k].paddle_z, traj.cost);
      }
    }

    if (bounce) {
      ++bounces;
      // judge each bounce against the target in force when the ball peaks
      // (the controller previews target changes and acts on them early)
      const float t_apex = t + c.dt + next.ball_v / p.g;
      const float target =
          t_apex < switch_time ? sc.apex_target : sc.apex_target_2;
      const float since_change =
          t_apex < switch_time ? t_apex : t_apex - switch_time;
      if (since_change > SETTLE_TIME && t_apex < sc.duration) {
        const float err = std::fabs(PaddleJuggling::apex(next, p) - target);
        ++settled_bounces;
        settled_err_sum += err;
        settled_err_max = std::fmax(settled_err_max, err);
      }
    }

    x = next;
    accel = sol.nextInput();
  }

  const float mean_err =
      settled_bounces > 0 ? settled_err_sum / settled_bounces : INFINITY;
  std::printf("bounces: %d, settled bounces: %d, apex error mean %.3f m, "
              "max %.3f m\n",
              bounces, settled_bounces, mean_err, settled_err_max);
  std::printf("logs written to %s\n", out.c_str());
  return settled_bounces >= 4 && settled_err_max < 0.1f ? 0 : 2;
}
