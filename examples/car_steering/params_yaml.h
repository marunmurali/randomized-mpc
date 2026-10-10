// Copyright (c) 2026 Arun Muraleedharan.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#pragma once

#include "config_yaml.h"
#include "problem.cuh"

namespace car_steering {

inline bool readTire(const YAML::Node& n, TireParams& t) noexcept {
  return n && examples::read(n, "B", t.B) && examples::read(n, "C", t.C) &&
         examples::read(n, "D", t.D) && examples::read(n, "E", t.E);
}

inline bool readError(const YAML::Node& n, ErrorCost& c) noexcept {
  return n && examples::read(n, "quad_gain", c.quad_gain) &&
         examples::read(n, "quad_clamp", c.quad_clamp) &&
         examples::read(n, "lin_gain", c.lin_gain);
}

inline bool readForce(const YAML::Node& n, ForceCost& c) noexcept {
  return n && examples::read(n, "f_exp", c.f_exp) &&
         examples::read(n, "q_force", c.q_force);
}

/**
 * @brief Load the `vehicle:` and `cost:` sections.
 */
inline rmpc::Result<CarSteering::Params> loadParams(
    const YAML::Node& root) noexcept {
  using examples::read;
  CarSteering::Params p{};
  const YAML::Node v = root["vehicle"];
  const YAML::Node c = root["cost"];

  const bool ok =
      v && c && read(v, "vx", p.vx) && read(v, "mass", p.mass) &&
      read(v, "lf", p.lf) && read(v, "lr", p.lr) && read(v, "Iz", p.Iz) &&
      read(v, "g", p.g) && read(v, "rho", p.rho) &&
      read(v, "cl_f", p.cl_f) && read(v, "cl_r", p.cl_r) &&
      read(v, "car_width", p.car_width) && readTire(v["tire_f"], p.tire_f) &&
      readTire(v["tire_r"], p.tire_r) &&
      readError(c["lateral"], p.w.lateral) &&
      readError(c["heading"], p.w.heading) &&
      readError(c["terminal_lateral"], p.w.terminal_lateral) &&
      readError(c["terminal_heading"], p.w.terminal_heading) &&
      read(c, "progress", p.w.progress) &&
      read(c, "actuation", p.w.actuation) && read(c, "rate", p.w.rate) &&
      read(c, "decay", p.w.decay) && readForce(c["force_f"], p.w.force_f) &&
      readForce(c["force_r"], p.w.force_r) &&
      read(c, "grip_limit_scaler", p.w.grip_limit_scaler) &&
      read(c, "bound", p.w.bound) &&
      read(c, "bound_margin", p.w.bound_margin) &&
      read(c, "max_bound_cost", p.w.max_bound_cost) &&
      read(c, "slip", p.w.slip) &&
      read(c, "slip_threshold_f", p.w.slip_threshold_f) &&
      read(c, "slip_threshold_r", p.w.slip_threshold_r) &&
      read(c, "max_slip_cost", p.w.max_slip_cost);

  if (!ok) {
    return "car config: missing or invalid key in vehicle/cost sections";
  }
  return p;
}

}  // namespace car_steering
