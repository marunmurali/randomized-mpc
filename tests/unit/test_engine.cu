// Copyright (c) 2026 Arun Muraleedharan.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "rmpc/rmpc.cuh"

namespace {

/** x' = u, track a target position. */
struct Integrator {
  struct State {
    float x;
  };
  struct Params {
    float q;
    float r;
  };
  struct Reference {
    float target;
  };

  RMPC_HD static State step(const State& s, float u, const Reference&,
                            const Params&, float dt) {
    return {s.x + dt * u};
  }

  RMPC_HD static float stageCost(const State& s, float u, float,
                                 const Reference& ref, const Params& p, int,
                                 int) {
    const float e = s.x - ref.target;
    return p.q * e * e + p.r * u * u;
  }

  RMPC_HD static float terminalCost(const State& s, const Reference& ref,
                                    const Params& p, int) {
    const float e = s.x - ref.target;
    return p.q * e * e;
  }
};

static_assert(rmpc::Problem<Integrator>);

using Ctrl = rmpc::Controller<Integrator>;

rmpc::Config makeConfig() {
  rmpc::Config c;
  c.n_horizon = 50;
  c.n_samples = 512;
  c.n_basis = 6;
  c.dt = 0.05f;
  c.u_min = -0.5f;
  c.u_max = 0.5f;
  c.du_max = 2.0f;
  c.seed = 1234;
  c.n_debug_samples = 512;
  return c;
}

constexpr Integrator::Params PARAMS{1.0f, 0.01f};

std::vector<Integrator::Reference> refs(int n, float target) {
  return std::vector<Integrator::Reference>(static_cast<std::size_t>(n),
                                            {target});
}

std::unique_ptr<Ctrl> make(const rmpc::Config& c) {
  auto res = Ctrl::create(c, PARAMS);
  EXPECT_FALSE(res.isError()) << res.error();
  return std::move(res).value();
}

TEST(EngineTest, rejectsInvalidConfig) {
  rmpc::Config c = makeConfig();
  c.n_basis = c.n_horizon + 1;
  EXPECT_TRUE(Ctrl::create(c, PARAMS).isError());

  c = makeConfig();
  c.u_min = c.u_max;
  EXPECT_TRUE(Ctrl::create(c, PARAMS).isError());
}

TEST(EngineTest, rejectsWrongReferenceSize) {
  auto ctrl = make(makeConfig());
  const auto r = refs(10, 0.0f);
  EXPECT_TRUE(ctrl->solve({0.0f}, 0.0f, r).isError());
}

TEST(EngineTest, basisIsOrthonormal) {
  constexpr int N = 50;
  for (int a = 0; a < N; ++a) {
    for (int b = 0; b < N; ++b) {
      double dot = 0.0;
      for (int k = 0; k < N; ++k) {
        dot += static_cast<double>(rmpc::detail::basisValue(k, a, N)) *
               rmpc::detail::basisValue(k, b, N);
      }
      EXPECT_NEAR(dot, a == b ? 1.0 : 0.0, 1e-5) << a << "," << b;
    }
  }
}

TEST(EngineTest, everySampleRespectsBoundsAndRate) {
  const auto c = makeConfig();
  auto ctrl = make(c);
  const float u0 = 0.45f;
  const auto res = ctrl->solve({0.0f}, u0, refs(c.n_horizon, 1.0f));
  ASSERT_FALSE(res.isError()) << res.error();

  const float du_step = c.du_max * c.dt;
  for (const auto& s : res.value().samples) {
    EXPECT_EQ(s.u_cmd[0], u0);
    for (int k = 1; k < c.n_horizon; ++k) {
      EXPECT_GE(s.u_cmd[k], c.u_min);
      EXPECT_LE(s.u_cmd[k], c.u_max);
      EXPECT_LE(std::fabs(s.u_cmd[k] - s.u_cmd[k - 1]), du_step + 1e-6f);
    }
  }
}

TEST(EngineTest, samplesExploreTheInputRange) {
  const auto c = makeConfig();
  auto ctrl = make(c);
  const auto res = ctrl->solve({0.0f}, 0.0f, refs(c.n_horizon, 0.0f));
  ASSERT_FALSE(res.isError()) << res.error();

  float lo = 0.0f;
  float hi = 0.0f;
  for (const auto& s : res.value().samples) {
    lo = std::min(lo, s.u_cmd.back());
    hi = std::max(hi, s.u_cmd.back());
  }
  EXPECT_LT(lo, -0.25f);
  EXPECT_GT(hi, 0.25f);
}

TEST(EngineTest, sampleZeroHoldsCurrentInput) {
  const auto c = makeConfig();
  auto ctrl = make(c);
  const auto res = ctrl->solve({0.0f}, 0.3f, refs(c.n_horizon, 1.0f));
  ASSERT_FALSE(res.isError()) << res.error();
  for (float u : res.value().samples[0].u_cmd) {
    EXPECT_EQ(u, 0.3f);
  }
}

TEST(EngineTest, appliedEqualsCommandedWithoutLag) {
  const auto c = makeConfig();
  auto ctrl = make(c);
  const auto res = ctrl->solve({0.0f}, 0.1f, refs(c.n_horizon, 1.0f));
  ASSERT_FALSE(res.isError()) << res.error();
  for (const auto& s : res.value().samples) {
    EXPECT_EQ(s.u_cmd, s.u_applied);
  }
}

TEST(EngineTest, lagIsFirstOrder) {
  auto c = makeConfig();
  c.lag_tau = 0.2f;
  auto ctrl = make(c);

  std::vector<float> step(static_cast<std::size_t>(c.n_horizon), 1.0f);
  const auto res = ctrl->evaluate({0.0f}, 0.0f, step, refs(c.n_horizon, 0.0f));
  ASSERT_FALSE(res.isError()) << res.error();

  const float a = std::exp(-c.dt / c.lag_tau);
  for (int k = 0; k < c.n_horizon; ++k) {
    EXPECT_NEAR(res.value().u_applied[k], 1.0f - std::pow(a, k), 1e-5f) << k;
  }
}

TEST(EngineTest, bestIsTheCheapestSample) {
  const auto c = makeConfig();
  auto ctrl = make(c);
  const auto res = ctrl->solve({0.0f}, 0.0f, refs(c.n_horizon, 1.0f));
  ASSERT_FALSE(res.isError()) << res.error();
  const auto& sol = res.value();

  float min_cost = sol.samples[0].cost;
  for (const auto& s : sol.samples) {
    min_cost = std::min(min_cost, s.cost);
  }
  EXPECT_EQ(sol.best.cost, min_cost);
  EXPECT_EQ(sol.samples[sol.best_index].cost, min_cost);
  EXPECT_EQ(sol.samples[sol.best_index].u_cmd, sol.best.u_cmd);
}

TEST(EngineTest, sameSeedSameResult) {
  const auto c = makeConfig();
  auto a = make(c);
  auto b = make(c);
  const auto r = refs(c.n_horizon, 1.0f);
  const auto ra = a->solve({0.0f}, 0.0f, r);
  const auto rb = b->solve({0.0f}, 0.0f, r);
  ASSERT_FALSE(ra.isError() || rb.isError());
  EXPECT_EQ(ra.value().best.u_cmd, rb.value().best.u_cmd);
  EXPECT_EQ(ra.value().best.cost, rb.value().best.cost);
}

TEST(EngineTest, warmStartIsShiftedPreviousBest) {
  auto c = makeConfig();
  c.warm_start = true;
  auto ctrl = make(c);
  const auto r = refs(c.n_horizon, 1.0f);

  const auto first = ctrl->solve({0.0f}, 0.0f, r);
  ASSERT_FALSE(first.isError()) << first.error();
  const auto& prev = first.value().best;

  const float u0 = prev.u_cmd[1];
  const auto second = ctrl->solve(prev.states[1], u0, r);
  ASSERT_FALSE(second.isError()) << second.error();
  const auto& warm = second.value().samples[1].u_cmd;

  EXPECT_EQ(warm[0], u0);
  for (int k = 1; k < c.n_horizon; ++k) {
    EXPECT_NEAR(warm[k], prev.u_cmd[std::min(k + 1, c.n_horizon - 1)], 1e-6f);
  }
}

TEST(EngineTest, evaluateMatchesSolve) {
  auto c = makeConfig();
  c.lag_tau = 0.1f;
  auto ctrl = make(c);
  const auto r = refs(c.n_horizon, 1.0f);
  const auto sol = ctrl->solve({0.2f}, 0.1f, r);
  ASSERT_FALSE(sol.isError()) << sol.error();
  const auto& best = sol.value().best;

  const auto ev = ctrl->evaluate({0.2f}, 0.1f, best.u_cmd, r);
  ASSERT_FALSE(ev.isError()) << ev.error();
  EXPECT_FLOAT_EQ(ev.value().cost, best.cost);
  for (int k = 0; k < c.n_horizon; ++k) {
    EXPECT_FLOAT_EQ(ev.value().states[k].x, best.states[k].x);
    EXPECT_FLOAT_EQ(ev.value().u_applied[k], best.u_applied[k]);
  }
}

TEST(EngineTest, closedLoopReachesTarget) {
  auto c = makeConfig();
  c.n_debug_samples = 0;
  c.warm_start = true;
  auto ctrl = make(c);
  const auto r = refs(c.n_horizon, 1.0f);

  Integrator::State x{0.0f};
  float u = 0.0f;
  for (int t = 0; t < 200; ++t) {
    const auto res = ctrl->solve(x, u, r);
    ASSERT_FALSE(res.isError()) << res.error();
    x = Integrator::step(x, u, r[0], PARAMS, c.dt);  // plant: same model
    u = res.value().nextInput();
  }
  EXPECT_NEAR(x.x, 1.0f, 0.02f);
}

}  // namespace
