// Copyright (c) 2026 Arun Muraleedharan.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "params_yaml.h"
#include "rmpc/rmpc.cuh"

namespace {

using paddle_juggling::PaddleJuggling;

constexpr const char* CONFIG =
    RMPC_SOURCE_DIR "/examples/paddle_juggling/config.yaml";
constexpr float DT = 0.01f;

PaddleJuggling::Params loadParams() {
  auto p = paddle_juggling::loadParams(examples::loadFile(CONFIG));
  EXPECT_FALSE(p.isError()) << p.error();
  return p.value();
}

TEST(PaddleTest, freeFlightKeepsApex) {
  const PaddleJuggling::Params p = loadParams();
  PaddleJuggling::State s{0.5f, 2.0f, 0.0f, 0.0f};
  const float apex = PaddleJuggling::apex(s, p);
  for (int k = 0; k < 40; ++k) {  // up and partly down, no contact
    s = PaddleJuggling::step(s, 0.0f, {1.0f}, p, DT);
    EXPECT_NEAR(PaddleJuggling::apex(s, p), apex, 1e-4f) << k;
  }
}

TEST(PaddleTest, bounceOnStillPaddleUsesRestitution) {
  const PaddleJuggling::Params p = loadParams();
  PaddleJuggling::State s{0.01f, -3.0f, 0.0f, 0.0f};
  s = PaddleJuggling::step(s, 0.0f, {1.0f}, p, DT);
  const float v_impact = -3.0f - p.g * DT;
  EXPECT_FLOAT_EQ(s.ball_z, 0.0f);
  EXPECT_NEAR(s.ball_v, -p.restitution * v_impact, 1e-5f);
}

TEST(PaddleTest, movingPaddleAddsEnergy) {
  const PaddleJuggling::Params p = loadParams();
  const PaddleJuggling::State still{0.01f, -3.0f, 0.0f, 0.0f};
  const PaddleJuggling::State rising{0.01f, -3.0f, 0.0f, 1.0f};
  const auto a = PaddleJuggling::step(still, 0.0f, {1.0f}, p, DT);
  const auto b = PaddleJuggling::step(rising, 0.0f, {1.0f}, p, DT);
  EXPECT_GT(b.ball_v, a.ball_v + 1.0f);
}

TEST(PaddleTest, endStopHoldsThePaddle) {
  const PaddleJuggling::Params p = loadParams();
  PaddleJuggling::State s{1.0f, 0.0f, 0.0f, 0.0f};
  for (int k = 0; k < 100; ++k) {
    s = PaddleJuggling::step(s, 40.0f, {1.0f}, p, DT);
  }
  EXPECT_FLOAT_EQ(s.paddle_z, p.stroke);
  EXPECT_FLOAT_EQ(s.paddle_v, 0.0f);
}

TEST(PaddleSlowTest, juggleToTargetApex) {
  const YAML::Node root = examples::loadFile(CONFIG);
  auto cfg = examples::loadEngineConfig(root);
  ASSERT_FALSE(cfg.isError()) << cfg.error();
  const PaddleJuggling::Params p = loadParams();
  const rmpc::Config& c = cfg.value();

  auto ctrl = rmpc::Controller<PaddleJuggling>::create(c, p);
  ASSERT_FALSE(ctrl.isError()) << ctrl.error();
  const std::vector<PaddleJuggling::Reference> ref(c.n_horizon, {1.0f});

  PaddleJuggling::State x{0.5f, 0.0f, 0.0f, 0.0f};
  float accel = 0.0f;
  int late_bounces = 0;

  for (int step = 0; step < 700; ++step) {  // 7 s
    const auto res = ctrl.value()->solve(x, accel, ref);
    ASSERT_FALSE(res.isError()) << res.error();
    const auto next = PaddleJuggling::step(x, accel, ref[0], p, c.dt);
    const bool bounce = next.ball_v > 0.0f && x.ball_v <= 0.0f;
    if (bounce && step * c.dt > 3.0f) {
      ++late_bounces;
      EXPECT_NEAR(PaddleJuggling::apex(next, p), 1.0f, 0.1f)
          << "t=" << step * c.dt;
    }
    x = next;
    accel = res.value().nextInput();
  }
  EXPECT_GE(late_bounces, 3);
}

}  // namespace
