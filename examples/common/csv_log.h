// Copyright (c) 2026 Arun Muraleedharan.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#pragma once

#include <cstdio>
#include <filesystem>
#include <string>

namespace examples {

/**
 * @brief Minimal CSV file writer for simulation logs (host only).
 */
class CsvLog {
 public:
  CsvLog(const std::filesystem::path& dir, const char* name,
         const char* header) noexcept {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    file_ = std::fopen((dir / name).string().c_str(), "w");
    if (file_ != nullptr) {
      std::fprintf(file_, "%s\n", header);
    }
  }
  ~CsvLog() {
    if (file_ != nullptr) {
      std::fclose(file_);
    }
  }
  CsvLog(const CsvLog&) = delete;
  CsvLog& operator=(const CsvLog&) = delete;

  [[nodiscard]] bool ok() const noexcept { return file_ != nullptr; }

  /** fprintf one row (format without the trailing newline) */
  template <class... Args>
  void row(const char* fmt, Args... args) noexcept {
    if (file_ != nullptr) {
      std::fprintf(file_, fmt, args...);
      std::fputc('\n', file_);
    }
  }

 private:
  std::FILE* file_{nullptr};
};

}  // namespace examples
