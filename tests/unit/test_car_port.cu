// Copyright (c) 2026 Arun Muraleedharan.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

/**
 * Checks the car model and cost port against the legacy controller: the
 * golden file is the legacy best rollout (actuator delay set to identity),
 * re-evaluated here through Controller::evaluate.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "params_yaml.h"
#include "rmpc/rmpc.cuh"

namespace {

using car_steering::CarSteering;

struct Golden {
  float cost{0.0f};
  std::vector<float> u;
  std::vector<CarSteering::State> states;
};

Golden loadGolden(const std::string& path) {
  Golden g;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("# cost=", 0) == 0) {
      g.cost = std::stof(line.substr(7));
      continue;
    }
    if (line.empty() || line[0] == 'k') {
      continue;
    }
    std::stringstream ss(line);
    std::string cell;
    std::vector<float> v;
    while (std::getline(ss, cell, ',')) {
      v.push_back(std::stof(cell));
    }
    // k,u,s,d,yaw,vy,r
    g.u.push_back(v[1]);
    g.states.push_back({v[2], v[3], v[4], v[5], v[6]});
  }
  return g;
}

TEST(CarPortTest, rolloutMatchesLegacyController) {
  const Golden golden =
      loadGolden(RMPC_SOURCE_DIR "/tests/data/car_rollout_golden.csv");
  ASSERT_EQ(golden.u.size(), 100u);

  const YAML::Node root =
      examples::loadFile(RMPC_SOURCE_DIR "/examples/car_steering/config.yaml");
  auto cfg = examples::loadEngineConfig(root);
  ASSERT_FALSE(cfg.isError()) << cfg.error();
  auto params = car_steering::loadParams(root);
  ASSERT_FALSE(params.isError()) << params.error();

  // legacy smoke scenario: straight road, 3 m to each side, 730 kg, 20 m/s
  rmpc::Config c = cfg.value();
  c.lag_tau = 0.0f;
  CarSteering::Params p = params.value();
  p.mass = 730.0f;
  p.vx = 20.0f;
  const std::vector<CarSteering::Reference> ref(100, {0.0f, 0.0f, 3.0f, -3.0f});

  auto ctrl = rmpc::Controller<CarSteering>::create(c, p);
  ASSERT_FALSE(ctrl.isError()) << ctrl.error();

  const CarSteering::State x0{0.0f, 1.0f, 0.15f, 0.0f, 0.0f};
  const auto res = ctrl.value()->evaluate(x0, 0.0f, golden.u, ref);
  ASSERT_FALSE(res.isError()) << res.error();
  const auto& traj = res.value();

  for (std::size_t k = 0; k < golden.u.size(); ++k) {
    const auto& a = traj.states[k];
    const auto& b = golden.states[k];
    EXPECT_NEAR(a.s, b.s, 1e-4f) << "k=" << k;
    EXPECT_NEAR(a.d, b.d, 1e-4f) << "k=" << k;
    EXPECT_NEAR(a.yaw, b.yaw, 1e-4f) << "k=" << k;
    EXPECT_NEAR(a.vy, b.vy, 1e-4f) << "k=" << k;
    EXPECT_NEAR(a.r, b.r, 1e-4f) << "k=" << k;
  }
  EXPECT_NEAR(traj.cost, golden.cost, 1e-5f * std::fabs(golden.cost));
}

TEST(CarPortTest, termsAddUpToStageCost) {
  const YAML::Node root =
      examples::loadFile(RMPC_SOURCE_DIR "/examples/car_steering/config.yaml");
  auto params = car_steering::loadParams(root);
  ASSERT_FALSE(params.isError()) << params.error();

  const CarSteering::State x{5.0f, 0.4f, 0.05f, 0.2f, 0.1f};
  const CarSteering::Reference ref{0.01f, 0.0f, 3.0f, -3.0f};
  const auto terms =
      CarSteering::stageTerms(x, 0.05f, 0.04f, ref, params.value(), 3, 100);
  float sum = 0.0f;
  for (float t : terms) {
    sum += t;
  }
  EXPECT_FLOAT_EQ(sum, CarSteering::stageCost(x, 0.05f, 0.04f, ref,
                                              params.value(), 3, 100));
}

}  // namespace
