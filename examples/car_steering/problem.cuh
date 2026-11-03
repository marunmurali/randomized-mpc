// Copyright (c) 2026 Arun Muraleedharan.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#pragma once

#include <math.h>

#include <array>
#include <cstdint>

#include "rmpc/problem.h"

namespace car_steering {

/** below this speed the slip angles use a floor to avoid dividing by ~0 */
inline constexpr float MIN_VX_FOR_SLIP = 7.0f;

/** Pacejka Magic Formula: F = D*fz*sin(C*atan(B*x - E*(B*x - atan(B*x)))) */
struct TireParams {
  float B;
  float C;
  float D;
  float E;
};

/** min(quad_gain * x^2, quad_clamp) + lin_gain * |x| */
struct ErrorCost {
  float quad_gain;
  float quad_clamp;
  float lin_gain;
};

/** soft barrier on |fy| / grip limit, sharpness f_exp */
struct ForceCost {
  float f_exp;
  float q_force;
};

struct Weights {
  ErrorCost lateral;          /**< lateral error d */
  ErrorCost heading;          /**< heading error */
  ErrorCost terminal_lateral;
  ErrorCost terminal_heading;
  float progress;             /**< reward on distance s */
  float actuation;            /**< steer^2 */
  float rate;                 /**< (steer - steer_prev)^2 */
  float decay;                /**< fraction of stage cost dropped by horizon end */
  ForceCost force_f;
  ForceCost force_r;
  float grip_limit_scaler;
  float bound;
  float bound_margin;
  float max_bound_cost;
  float slip;
  float slip_threshold_f;
  float slip_threshold_r;
  float max_slip_cost;
};

/**
 * @brief Steering a car along a reference path at constant speed.
 *
 * Dynamic single-track model with Pacejka lateral tire forces and aero
 * downforce, in Frenet coordinates along the path: s is the distance along
 * the path, d the lateral offset, yaw the heading relative to the path.
 * The input is the front wheel steer angle [rad].
 */
struct CarSteering {
  struct State {
    float s;
    float d;
    float yaw;
    float vy;
    float r;
  };

  struct Params {
    float vx;    /**< constant forward speed [m/s] */
    float mass;
    float lf;    /**< CoG to front axle */
    float lr;    /**< CoG to rear axle */
    float Iz;
    float g;
    float rho;   /**< air density */
    float cl_f;  /**< lift coefficient x area, front */
    float cl_r;  /**< lift coefficient x area, rear */
    float car_width;
    TireParams tire_f;
    TireParams tire_r;
    Weights w;
  };

  struct Reference {
    float curvature;
    float d_target;     /**< desired lateral offset (0 = on the path) */
    float left_bound;   /**< > 0, lateral offset of the left edge */
    float right_bound;  /**< < 0, lateral offset of the right edge */
  };

  struct TireForces {
    float alpha_f;
    float alpha_r;
    float fyf;
    float fyr;
    float fzf;
    float fzr;
  };

  enum Term : std::uint8_t {
    LATERAL = 0,
    HEADING,
    ACTUATION,
    PROGRESS,
    FORCE_F,
    FORCE_R,
    RATE,
    LEFT_BOUND,
    RIGHT_BOUND,
    SLIP_F,
    SLIP_R,
    NUM_TERMS,
  };
  using Terms = std::array<float, NUM_TERMS>;

  static constexpr std::array<const char*, NUM_TERMS> TERM_NAMES{
      "lateral", "heading",     "actuation",    "progress",
      "force_f", "force_r",     "rate",         "left_bound",
      "right_bound", "slip_f",  "slip_r"};

  RMPC_HD static float latForcePacejka(float fz, float alpha,
                                       const TireParams& t) {
    const float d = t.D * fz;
    return d * sinf(t.C * atanf(t.B * alpha -
                                t.E * (t.B * alpha - atanf(t.B * alpha))));
  }

  /** slip angles and tire forces for state x with steer applied */
  RMPC_HD static TireForces tireForces(const State& x, float steer,
                                       const Params& p) {
    const float vx_slip = fmaxf(MIN_VX_FOR_SLIP, p.vx);
    const float alpha_f = atanf((x.vy + (p.lf * x.r)) / vx_slip) - steer;
    const float alpha_r = atanf((x.vy - (p.lr * x.r)) / vx_slip);

    const float sq_air_speed = p.vx * p.vx;
    const float fz_aero_f = -0.5f * p.rho * p.cl_f * sq_air_speed;
    const float fz_aero_r = -0.5f * p.rho * p.cl_r * sq_air_speed;

    // static load split on a flat road
    const float m_f = p.mass * p.lr / (p.lf + p.lr);
    const float m_r = p.mass * p.lf / (p.lf + p.lr);
    const float gz = -p.g;

    const float fzf = -(m_f * gz + fz_aero_f);
    const float fzr = -(m_r * gz + fz_aero_r);

    return {alpha_f,
            alpha_r,
            latForcePacejka(fzf, alpha_f, p.tire_f),
            latForcePacejka(fzr, alpha_r, p.tire_r),
            fzf,
            fzr};
  }

  RMPC_HD static State step(const State& x, float steer, const Reference& ref,
                            const Params& p, float dt) {
    const TireForces f = tireForces(x, steer, p);

    // lateral and yaw dynamics, front force rotated by the steer angle
    const float yf = f.fyf * cosf(steer);
    const float yr = f.fyr;
    const float ay = (yf + yr) / p.mass;
    const float vy_dot = ay - p.vx * x.r;
    const float r_dot = (p.lf * yf - p.lr * yr) / p.Iz;

    // Frenet frame along the reference path
    const float k = ref.curvature;
    const float v_t = p.vx * cosf(x.yaw) - x.vy * sinf(x.yaw);
    const float v_n = p.vx * sinf(x.yaw) + x.vy * cosf(x.yaw);
    const float s_dot = v_t / (1 - x.d * k);
    const float d_dot = v_n;
    const float yaw_dot = x.r - s_dot * k;

    return {x.s + dt * s_dot, x.d + dt * d_dot, x.yaw + dt * yaw_dot,
            x.vy + dt * vy_dot, x.r + dt * r_dot};
  }

  RMPC_HD static float clampedQuadratic(const ErrorCost& c, float x) {
    return fminf(c.quad_gain * powf(x, 2), c.quad_clamp) +
           c.lin_gain * fabsf(x);
  }

  RMPC_HD static float forceCost(float fy, float fz, const TireParams& tire,
                                 const ForceCost& c, float grip_scaler) {
    const float limit = tire.D * fz * grip_scaler;
    const float norm = fabsf(fy / limit);
    // soft barrier around the lateral grip limit
    return c.q_force * (logf(1.0f + expf(c.f_exp * (norm - 1.0f))) -
                        logf(1.0f + expf(-c.f_exp)));
  }

  RMPC_HD static float boundCost(float d, float bound, const Params& p) {
    const float margin = 0.5f * p.car_width + p.w.bound_margin;
    const float side = copysignf(1.0f, bound);  // +1 left, -1 right
    const float limit = side * bound - margin;  // last safe offset
    const float violation = side * d - limit;   // > 0 only when outside
    return fminf(p.w.max_bound_cost, p.w.bound * fmaxf(0.0f, violation));
  }

  RMPC_HD static float slipCost(float slip, float threshold, const Params& p) {
    const float over = fabsf(slip) - threshold;
    return fminf(p.w.max_slip_cost, over > 0.0f ? over * p.w.slip : 0.0f);
  }

  /** weighted stage cost terms; their sum is stageCost */
  RMPC_HD static Terms stageTerms(const State& x, float steer,
                                  float steer_prev, const Reference& ref,
                                  const Params& p, int k, int n) {
    const Weights& w = p.w;
    const TireForces f = tireForces(x, steer, p);

    Terms t{};
    t[LATERAL] = clampedQuadratic(w.lateral, x.d - ref.d_target);
    t[HEADING] = clampedQuadratic(w.heading, x.yaw);
    t[ACTUATION] = w.actuation * powf(steer, 2);
    t[PROGRESS] = -1 * w.progress * x.s;
    t[FORCE_F] = forceCost(f.fyf, f.fzf, p.tire_f, w.force_f,
                           w.grip_limit_scaler);
    t[FORCE_R] = forceCost(f.fyr, f.fzr, p.tire_r, w.force_r,
                           w.grip_limit_scaler);
    t[RATE] = k > 0 ? w.rate * powf(steer - steer_prev, 2) : 0.0f;
    t[LEFT_BOUND] = boundCost(x.d, ref.left_bound, p);
    t[RIGHT_BOUND] = boundCost(x.d, ref.right_bound, p);
    t[SLIP_F] = slipCost(f.alpha_f, w.slip_threshold_f, p);
    t[SLIP_R] = slipCost(f.alpha_r, w.slip_threshold_r, p);

    // later steps weigh less, dropping linearly by `decay` over the horizon
    const float weight = (1 - (k * w.decay / (n - 1))) / n;
    for (float& term : t) {
      term *= weight;
    }
    return t;
  }

  RMPC_HD static float stageCost(const State& x, float steer,
                                 float steer_prev, const Reference& ref,
                                 const Params& p, int k, int n) {
    const Terms t = stageTerms(x, steer, steer_prev, ref, p, k, n);
    float total = 0.0f;
    for (float term : t) {
      total += term;
    }
    return total;
  }

  RMPC_HD static float terminalCost(const State& x, const Reference& ref,
                                    const Params& p, int n) {
    return clampedQuadratic(p.w.terminal_lateral, x.d - ref.d_target) / n +
           clampedQuadratic(p.w.terminal_heading, x.yaw) / n;
  }
};

static_assert(rmpc::Problem<CarSteering>);

}  // namespace car_steering
