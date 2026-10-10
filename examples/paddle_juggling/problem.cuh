// Copyright (c) 2026 Arun Muraleedharan.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#pragma once

#include <math.h>

#include "rmpc/problem.h"

namespace paddle_juggling {

/**
 * @brief Bounce a ball on a paddle to a target apex height (1-D).
 *
 * A ball falls under gravity onto a paddle that moves up and down; the
 * input is the paddle acceleration. At impact the ball's velocity relative
 * to the paddle reverses and shrinks by the restitution e, so energy can
 * only be added or removed by moving the paddle at the moment of contact.
 *
 * Why this is hard for gradient-based MPC: the bounce is a discontinuous
 * jump in the state, and the main cost is on the ball's mechanical energy,
 * which only changes at impacts. Between impacts the cost does not depend
 * on the input at all, so its gradient is zero almost everywhere.
 */
struct PaddleJuggling {
  struct State {
    float ball_z;    /**< ball height [m] */
    float ball_v;    /**< ball velocity, up positive [m/s] */
    float paddle_z;  /**< paddle height, 0 = rest [m] */
    float paddle_v;  /**< paddle velocity [m/s] */
  };

  struct Weights {
    float apex;          /**< (apex height - target)^2 */
    float paddle_pos;    /**< paddle_z^2, stay near rest */
    float paddle_vel;    /**< paddle_v^2 */
    float effort;        /**< a^2 */
    float terminal_apex;
  };

  struct Params {
    float g;
    float restitution;  /**< 0..1, ball/paddle impact */
    float stroke;       /**< paddle travel limit |paddle_z| [m] */
    Weights w;
  };

  struct Reference {
    float apex_target;  /**< desired apex height above paddle rest [m] */
  };

  /**
   * One step: ballistic flight, paddle double integrator with a hard end
   * stop, then an instantaneous bounce if the ball has reached the paddle
   * while moving towards it.
   */
  RMPC_HD static State step(const State& s, float accel, const Reference&,
                            const Params& p, float dt) {
    State n;
    n.ball_v = s.ball_v - p.g * dt;
    n.ball_z = s.ball_z + dt * s.ball_v - 0.5f * p.g * dt * dt;

    n.paddle_v = s.paddle_v + accel * dt;
    n.paddle_z = s.paddle_z + dt * s.paddle_v + 0.5f * accel * dt * dt;
    if (fabsf(n.paddle_z) > p.stroke) {  // end stop
      n.paddle_z = copysignf(p.stroke, n.paddle_z);
      n.paddle_v = 0.0f;
    }

    const float rel_v = n.ball_v - n.paddle_v;
    if (n.ball_z <= n.paddle_z && rel_v < 0.0f) {
      n.ball_z = n.paddle_z;
      n.ball_v = n.paddle_v - p.restitution * rel_v;
    }
    return n;
  }

  /** apex height the ball would reach in free flight from this state */
  RMPC_HD static float apex(const State& s, const Params& p) {
    return s.ball_z + 0.5f * s.ball_v * s.ball_v / p.g;
  }

  RMPC_HD static float stageCost(const State& s, float accel, float,
                                 const Reference& ref, const Params& p, int,
                                 int n) {
    const Weights& w = p.w;
    const float e = apex(s, p) - ref.apex_target;
    return (w.apex * e * e + w.paddle_pos * s.paddle_z * s.paddle_z +
            w.paddle_vel * s.paddle_v * s.paddle_v +
            w.effort * accel * accel) /
           n;
  }

  RMPC_HD static float terminalCost(const State& s, const Reference& ref,
                                    const Params& p, int) {
    const float e = apex(s, p) - ref.apex_target;
    return p.w.terminal_apex * e * e;
  }
};

static_assert(rmpc::Problem<PaddleJuggling>);

}  // namespace paddle_juggling
