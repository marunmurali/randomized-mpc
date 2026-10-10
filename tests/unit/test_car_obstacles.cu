// Copyright (c) 2026 Arun Muraleedharan.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "params_yaml.h"
#include "rmpc/rmpc.cuh"

namespace {

using car_obstacles::CarObstacles;

constexpr const char* CONFIG =
    RMPC_SOURCE_DIR "/examples/car_obstacles/config.yaml";

CarObstacles::Params loadParams() {
  auto p = car_obstacles::loadParams(examples::loadFile(CONFIG));
  EXPECT_FALSE(p.isError()) << p.error();
  return p.value();
}

TEST(CarObstaclesTest, configHasObstacles) {
  EXPECT_EQ(loadParams().n_obstacles, 4);
}

TEST(CarObstaclesTest, collisionUsesBothAxleDiscs) {
  const CarObstacles::Params p = loadParams();
  const auto& o = p.obstacles[0];

  // centre of the car on the obstacle
  EXPECT_TRUE(CarObstacles::collides({o.s, o.d, 0.0f, 0.0f, 0.0f}, p));
  // only the front axle disc on the obstacle
  EXPECT_TRUE(
      CarObstacles::collides({o.s - p.car.lf, o.d, 0.0f, 0.0f, 0.0f}, p));
  // only the rear axle disc on the obstacle
  EXPECT_TRUE(
      CarObstacles::collides({o.s + p.car.lr, o.d, 0.0f, 0.0f, 0.0f}, p));
  // well beside it
  EXPECT_FALSE(
      CarObstacles::collides({o.s, o.d + 3.0f, 0.0f, 0.0f, 0.0f}, p));
}

TEST(CarObstaclesTest, collisionCostIsFlat) {
  const CarObstacles::Params p = loadParams();
  const auto& o = p.obstacles[0];
  const CarObstacles::Reference ref{0.0f, 0.0f, 4.0f, -4.0f};

  // two positions inside the obstacle: the collision term is identical,
  // so the cost has no gradient pointing out of it
  const CarObstacles::State a{o.s, o.d, 0.0f, 0.0f, 0.0f};
  const CarObstacles::State b{o.s, o.d + 0.3f, 0.0f, 0.0f, 0.0f};
  const float hit_a = CarObstacles::stageCost(a, 0, 0, ref, p, 1, 100) -
                      car_steering::CarSteering::stageCost(a, 0, 0, ref,
                                                           p.car, 1, 100);
  const float hit_b = CarObstacles::stageCost(b, 0, 0, ref, p, 1, 100) -
                      car_steering::CarSteering::stageCost(b, 0, 0, ref,
                                                           p.car, 1, 100);
  EXPECT_NEAR(hit_a, p.hit_cost / 100.0f, 1e-2f);
  EXPECT_NEAR(hit_a, hit_b, 1e-2f);
}

TEST(CarObstaclesSlowTest, slalomWithoutContact) {
  const YAML::Node root = examples::loadFile(CONFIG);
  auto cfg = examples::loadEngineConfig(root);
  ASSERT_FALSE(cfg.isError()) << cfg.error();
  const CarObstacles::Params p = loadParams();
  const rmpc::Config& c = cfg.value();

  auto ctrl = rmpc::Controller<CarObstacles>::create(c, p);
  ASSERT_FALSE(ctrl.isError()) << ctrl.error();

  // straight road through all obstacles
  const std::vector<CarObstacles::Reference> ref(c.n_horizon,
                                                 {0.0f, 0.0f, 4.0f, -4.0f});
  CarObstacles::State x{0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  float steer = 0.0f;
  const float lag_a = std::exp(-c.dt / c.lag_tau);

  while (x.s < 260.0f) {
    const auto res = ctrl.value()->solve(x, steer, ref);
    ASSERT_FALSE(res.isError()) << res.error();
    x = CarObstacles::step(x, steer, ref[0], p, c.dt);
    steer = lag_a * steer + (1.0f - lag_a) * res.value().nextInput();

    // the plant never touches an obstacle (the clearance margin may be used)
    CarObstacles::Params no_margin = p;
    no_margin.clearance = 0.0f;
    ASSERT_FALSE(CarObstacles::collides(x, no_margin)) << "s=" << x.s;
    ASSERT_LT(std::fabs(x.d), 4.0f) << "s=" << x.s;
  }
}

}  // namespace
