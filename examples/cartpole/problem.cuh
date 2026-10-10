// Copyright (c) 2026 Arun Muraleedharan.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#pragma once

#include <math.h>

#include "rmpc/problem.h"

namespace cartpole {

/**
 * @brief Cart-pole swing-up and balance.
 *
 * A cart on a rail carries a pole (uniform rod) on a frictionless pivot.
 * theta = 0 is upright, theta = pi hanging down; the input is the
 * horizontal force on the cart [N]. Underactuated and nonconvex: from
 * hanging, the controller has to pump energy in with back-and-forth moves
 * before it can balance.
 */
struct CartPole {
  struct State {
    float x;          /**< cart position [m] */
    float x_dot;
    float theta;      /**< pole angle from upright [rad] */
    float theta_dot;
  };

  struct Weights {
    float angle;          /**< (1 - cos theta) */
    float position;       /**< (x - x_target)^2 */
    float velocity;       /**< x_dot^2 */
    float ang_velocity;   /**< theta_dot^2 */
    float effort;         /**< u^2 */
    float rate;           /**< (u - u_prev)^2 */
    float track;          /**< barrier beyond track_limit */
    float terminal_angle;
    float terminal_ang_velocity;
  };

  struct Params {
    float m_cart;
    float m_pole;
    float half_length;  /**< pivot to pole centre of mass [m] */
    float g;
    float track_limit;  /**< |x| beyond this is penalized [m] */
    Weights w;
  };

  struct Reference {
    float x_target;
  };

  /** continuous-time dynamics (rigid rod, frictionless) */
  RMPC_HD static State derivative(const State& s, float force,
                                  const Params& p) {
    const float total_mass = p.m_cart + p.m_pole;
    const float sin_t = sinf(s.theta);
    const float cos_t = cosf(s.theta);
    const float pole_ml = p.m_pole * p.half_length;

    const float temp =
        (force + pole_ml * s.theta_dot * s.theta_dot * sin_t) / total_mass;
    const float theta_acc =
        (p.g * sin_t - cos_t * temp) /
        (p.half_length *
         (4.0f / 3.0f - p.m_pole * cos_t * cos_t / total_mass));
    const float x_acc = temp - pole_ml * theta_acc * cos_t / total_mass;

    return {s.x_dot, x_acc, s.theta_dot, theta_acc};
  }

  RMPC_HD static State axpy(const State& s, float h, const State& d) {
    return {s.x + h * d.x, s.x_dot + h * d.x_dot, s.theta + h * d.theta,
            s.theta_dot + h * d.theta_dot};
  }

  /** classic 4th-order Runge-Kutta step */
  RMPC_HD static State step(const State& s, float force, const Reference&,
                            const Params& p, float dt) {
    const State k1 = derivative(s, force, p);
    const State k2 = derivative(axpy(s, 0.5f * dt, k1), force, p);
    const State k3 = derivative(axpy(s, 0.5f * dt, k2), force, p);
    const State k4 = derivative(axpy(s, dt, k3), force, p);
    return {
        s.x + dt / 6.0f * (k1.x + 2.0f * k2.x + 2.0f * k3.x + k4.x),
        s.x_dot + dt / 6.0f *
                      (k1.x_dot + 2.0f * k2.x_dot + 2.0f * k3.x_dot + k4.x_dot),
        s.theta + dt / 6.0f * (k1.theta + 2.0f * k2.theta + 2.0f * k3.theta +
                               k4.theta),
        s.theta_dot + dt / 6.0f * (k1.theta_dot + 2.0f * k2.theta_dot +
                                   2.0f * k3.theta_dot + k4.theta_dot)};
  }

  /** total mechanical energy, zero potential at the pivot height */
  RMPC_HD static float energy(const State& s, const Params& p) {
    const float total_mass = p.m_cart + p.m_pole;
    const float l = p.half_length;
    const float kinetic =
        0.5f * total_mass * s.x_dot * s.x_dot +
        p.m_pole * l * cosf(s.theta) * s.x_dot * s.theta_dot +
        (2.0f / 3.0f) * p.m_pole * l * l * s.theta_dot * s.theta_dot;
    return kinetic + p.m_pole * p.g * l * cosf(s.theta);
  }

  RMPC_HD static float trackCost(float x, const Params& p) {
    const float over = fabsf(x) - p.track_limit;
    return over > 0.0f ? p.w.track * over * over : 0.0f;
  }

  RMPC_HD static float stageCost(const State& s, float force, float force_prev,
                                 const Reference& ref, const Params& p, int,
                                 int) {
    const Weights& w = p.w;
    const float dx = s.x - ref.x_target;
    const float du = force - force_prev;
    return w.angle * (1.0f - cosf(s.theta)) + w.position * dx * dx +
           w.velocity * s.x_dot * s.x_dot +
           w.ang_velocity * s.theta_dot * s.theta_dot +
           w.effort * force * force + w.rate * du * du + trackCost(s.x, p);
  }

  RMPC_HD static float terminalCost(const State& s, const Reference&,
                                    const Params& p, int) {
    return p.w.terminal_angle * (1.0f - cosf(s.theta)) +
           p.w.terminal_ang_velocity * s.theta_dot * s.theta_dot;
  }
};

static_assert(rmpc::Problem<CartPole>);

}  // namespace cartpole
