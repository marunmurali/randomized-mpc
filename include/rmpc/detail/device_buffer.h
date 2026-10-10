// Copyright (c) 2026 Arun Muraleedharan.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace rmpc::detail {

/**
 * @brief Owning, non-throwing device array.
 */
template <class T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  ~DeviceBuffer() { release(); }

  // owns a raw device pointer, a copy would double free on destruct
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  [[nodiscard]] cudaError_t allocate(std::size_t size) noexcept {
    release();
    const cudaError_t err =
        cudaMalloc(reinterpret_cast<void**>(&data_), size * sizeof(T));
    if (err != cudaSuccess) {
      data_ = nullptr;
      return err;
    }
    size_ = size;
    return cudaSuccess;
  }

  [[nodiscard]] T* data() noexcept { return data_; }
  [[nodiscard]] const T* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }

 private:
  void release() noexcept {
    if (data_ != nullptr) {
      cudaFree(data_);
      data_ = nullptr;
      size_ = 0;
    }
  }

  T* data_{nullptr};
  std::size_t size_{0};
};

}  // namespace rmpc::detail
