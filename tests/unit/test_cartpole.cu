// Copyright (c) 2026 Arun Muraleedharan.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "params_yaml.h"
#include "rmpc/rmpc.cuh"

namespace {

using cartpole::CartPole;

CartPole::Params loadParams() {
  const YAML::Node root =
      examples::loadFile(RMPC_SOURCE_DIR "/examples/cartpole/config.yaml");
  auto params = cartpole::loadParams(root);
  EXPECT_FALSE(params.isError()) << params.error();
  return params.value();
}

TEST(CartPoleTest, unforcedMotionConservesEnergy) {
  const CartPole::Params p = loadParams();
  CartPole::State x{0.0f, 0.3f, 1.0f, -0.5f};
  const float e0 = CartPole::energy(x, p);

  for (int k = 0; k < 100; ++k) {  // 2 s
    x = CartPole::step(x, 0.0f, {0.0f}, p, 0.02f);
  }
  EXPECT_NEAR(CartPole::energy(x, p), e0, 1e-3f);
}

TEST(CartPoleTest, hangingAtRestStaysAtRest) {
  const CartPole::Params p = loadParams();
  CartPole::State x{0.0f, 0.0f, static_cast<float>(M_PI), 0.0f};
  for (int k = 0; k < 100; ++k) {
    x = CartPole::step(x, 0.0f, {0.0f}, p, 0.02f);
  }
  EXPECT_NEAR(x.x, 0.0f, 1e-5f);
  EXPECT_NEAR(x.theta, static_cast<float>(M_PI), 1e-4f);
}

TEST(CartPoleSlowTest, swingsUpAndBalances) {
  const YAML::Node root =
      examples::loadFile(RMPC_SOURCE_DIR "/examples/cartpole/config.yaml");
  auto cfg = examples::loadEngineConfig(root);
  ASSERT_FALSE(cfg.isError()) << cfg.error();
  const CartPole::Params p = loadParams();

  auto ctrl = rmpc::Controller<CartPole>::create(cfg.value(), p);
  ASSERT_FALSE(ctrl.isError()) << ctrl.error();
  const auto& c = cfg.value();
  const std::vector<CartPole::Reference> ref(c.n_horizon, {0.0f});

  CartPole::State x{0.0f, 0.0f, static_cast<float>(M_PI), 0.0f};
  float force = 0.0f;
  float max_late_angle = 0.0f;
  float max_abs_x = 0.0f;

  for (int step = 0; step < 400; ++step) {  // 8 s
    const auto res = ctrl.value()->solve(x, force, ref);
    ASSERT_FALSE(res.isError()) << res.error();
    x = CartPole::step(x, force, ref[0], p, c.dt);
    force = res.value().nextInput();

    const float angle = std::remainder(x.theta, 2.0f * static_cast<float>(M_PI));
    if (step >= 300) {  // last 2 s
      max_late_angle = std::fmax(max_late_angle, std::fabs(angle));
    }
    max_abs_x = std::fmax(max_abs_x, std::fabs(x.x));
  }

  EXPECT_LT(max_late_angle, 0.1f);
  EXPECT_LT(max_abs_x, p.track_limit);
}

}  // namespace
