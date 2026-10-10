// Copyright (c) 2026 Arun Muraleedharan.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

namespace car_steering {

/**
 * @brief Reference path given by curvature over arc length (host only).
 *
 * The centerline is built once by integrating the curvature, so Frenet
 * states (s, d, yaw) can be converted to x/y for drawing.
 */
class Track {
 public:
  struct Segment {
    float length;
    float curvature;
  };

  struct Pose {
    float x;
    float y;
    float heading;
  };

  explicit Track(const std::vector<Segment>& segments, float ds = 0.05f)
      : ds_(ds) {
    float x = 0.0f;
    float y = 0.0f;
    float heading = 0.0f;
    for (const auto& seg : segments) {
      const int n = static_cast<int>(std::lround(seg.length / ds));
      for (int i = 0; i < n; ++i) {
        kappa_.push_back(seg.curvature);
        poses_.push_back({x, y, heading});
        // midpoint rule along the arc
        const float mid = heading + 0.5f * ds * seg.curvature;
        x += ds * std::cos(mid);
        y += ds * std::sin(mid);
        heading += ds * seg.curvature;
      }
    }
    kappa_.push_back(kappa_.empty() ? 0.0f : kappa_.back());
    poses_.push_back({x, y, heading});
  }

  [[nodiscard]] float length() const {
    return ds_ * static_cast<float>(poses_.size() - 1);
  }

  [[nodiscard]] float curvature(float s) const { return kappa_[index(s)]; }

  /** centerline pose at s */
  [[nodiscard]] Pose pose(float s) const {
    const std::size_t i = index(s);
    const Pose& p = poses_[i];
    const float rest = s - ds_ * static_cast<float>(i);
    return {p.x + rest * std::cos(p.heading), p.y + rest * std::sin(p.heading),
            p.heading + rest * kappa_[i]};
  }

  /** point at arc length s, offset d to the left; heading + yaw */
  [[nodiscard]] Pose toCartesian(float s, float d, float yaw) const {
    const Pose c = pose(s);
    return {c.x - d * std::sin(c.heading), c.y + d * std::cos(c.heading),
            c.heading + yaw};
  }

 private:
  [[nodiscard]] std::size_t index(float s) const {
    const float i = std::clamp(s / ds_, 0.0f,
                               static_cast<float>(poses_.size() - 1));
    return static_cast<std::size_t>(i);
  }

  float ds_;
  std::vector<float> kappa_;
  std::vector<Pose> poses_;
};

}  // namespace car_steering
