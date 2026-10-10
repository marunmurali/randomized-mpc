// Copyright (c) 2026 Arun Muraleedharan.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

/**
 * One solve from an offset start on a straight road; prints the predicted
 * best trajectory as CSV. Usage: car_solve_once [config.yaml]
 */

#include <cstdio>
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

  const int n = cfg.value().n_horizon;
  const std::vector<CarSteering::Reference> ref(n, {0.0f, 0.0f, 3.0f, -3.0f});

  // 1 m left of the line, heading 0.15 rad away from it
  const CarSteering::State x0{0.0f, 1.0f, 0.15f, 0.0f, 0.0f};
  const auto res = ctrl.value()->solve(x0, 0.0f, ref);
  if (res.isError()) {
    std::fprintf(stderr, "solve failed: %s\n", res.error());
    return 1;
  }

  const auto& best = res.value().best;
  std::printf("t,s,d,yaw,vy,r,steer_cmd,steer_applied\n");
  for (int k = 0; k < n; ++k) {
    const auto& x = best.states[k];
    std::printf("%g,%g,%g,%g,%g,%g,%g,%g\n", k * cfg.value().dt, x.s, x.d,
                x.yaw, x.vy, x.r, best.u_cmd[k], best.u_applied[k]);
  }
  return 0;
}
