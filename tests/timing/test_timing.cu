// Copyright (c) 2026 Arun Muraleedharan.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

/**
 * Per-solve latency of the car example (median/P95/mean over 10k warm
 * calls). Informational, not pass/fail. Usage: rmpc_timing [config.yaml]
 */

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <numeric>
#include <vector>

#include "params_yaml.h"
#include "rmpc/rmpc.cuh"

using car_steering::CarSteering;

int main(int argc, char** argv) {
  const char* path = argc > 1 ? argv[1] : RMPC_EXAMPLE_DIR "/config.yaml";
  const YAML::Node root = examples::loadFile(path);
  auto cfg = examples::loadEngineConfig(root);
  auto params = car_steering::loadParams(root);
  if (cfg.isError() || params.isError()) {
    std::fprintf(stderr, "config error: %s\n",
                 cfg.isError() ? cfg.error() : params.error());
    return 1;
  }
  auto ctrl = rmpc::Controller<CarSteering>::create(cfg.value(),
                                                    params.value());
  if (ctrl.isError()) {
    std::fprintf(stderr, "create failed: %s\n", ctrl.error());
    return 1;
  }

  const auto& c = cfg.value();
  std::printf("Config: %s\nn_horizon=%d n_samples=%d\n", path, c.n_horizon,
              c.n_samples);

  const std::vector<CarSteering::Reference> ref(
      c.n_horizon, {0.0f, 0.0f, 3.0f, -3.0f});
  const CarSteering::State x0{0.0f, 1.0f, 0.15f, 0.0f, 0.0f};

  constexpr int WARMUP = 100;
  constexpr int RUNS = 10000;
  std::vector<double> times;
  times.reserve(RUNS);

  for (int i = 0; i < WARMUP + RUNS; ++i) {
    const auto start = std::chrono::steady_clock::now();
    const auto res = ctrl.value()->solve(x0, 0.0f, ref);
    const auto end = std::chrono::steady_clock::now();
    if (res.isError()) {
      std::fprintf(stderr, "solve failed: %s\n", res.error());
      return 1;
    }
    if (i >= WARMUP) {
      times.push_back(
          std::chrono::duration<double, std::micro>(end - start).count());
    }
  }

  std::sort(times.begin(), times.end());
  const double mean =
      std::accumulate(times.begin(), times.end(), 0.0) / RUNS;
  std::printf("Median: %.1f us\nP95   : %.1f us\nMean  : %.1f us\n",
              times[RUNS / 2], times[RUNS * 95 / 100], mean);
  return 0;
}
