// Copyright (c) 2026 Arun Muraleedharan.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#pragma once

#include <utility>

namespace rmpc {

/**
 * @brief Value-or-error wrapper, so that the public API never throws.
 *
 * Holds either a value or a static error message. Check isError() before
 * calling value().
 */
template <typename T>
class Result {
 public:
  Result(T value) noexcept : value_(std::move(value)) {}
  Result(const char* error) noexcept : error_(error) {}

  [[nodiscard]] bool isError() const noexcept { return error_ != nullptr; }
  [[nodiscard]] const T& value() const& noexcept { return value_; }
  [[nodiscard]] T& value() & noexcept { return value_; }
  [[nodiscard]] T&& value() && noexcept { return std::move(value_); }
  [[nodiscard]] const char* error() const noexcept { return error_; }

 private:
  T value_{};
  const char* error_{nullptr};
};

}  // namespace rmpc
