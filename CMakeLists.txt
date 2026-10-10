cmake_minimum_required(VERSION 3.27)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(rmpc
  DESCRIPTION "Randomized model predictive control for single-input problems on the GPU"
  VERSION 0.1.0
  LANGUAGES CUDA CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(RMPC_BUILD_EXAMPLES "Build the example problems" ON)

include(CTest)
include(GNUInstallDirs)

find_package(CUDAToolkit REQUIRED)

# Header-only engine: code including it is compiled as CUDA (.cu).
add_library(rmpc INTERFACE)
add_library(rmpc::rmpc ALIAS rmpc)

target_include_directories(rmpc
  INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

target_compile_features(rmpc INTERFACE cuda_std_20)

# Problem code is plain C++ running on host and device. Allow constexpr
# std helpers (std::array, std::min, ...) in device code, and make calls
# to host-only functions a compile error: nvcc only warns by default and
# the kernel then silently does nothing at runtime.
target_compile_options(rmpc
  INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--diag-error=20011,20012,20013,20014,20015>
)

target_link_libraries(rmpc
  INTERFACE
    CUDA::cudart
    CUDA::curand
)

# Strict host-side warnings for this repo's own examples and tests (not
# propagated to projects consuming the rmpc target).
add_compile_options(
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wshadow,-Werror>
)

if(RMPC_BUILD_EXAMPLES OR BUILD_TESTING)
  add_subdirectory("examples")
endif()

if(BUILD_TESTING)
  add_subdirectory("tests")
endif()

install(TARGETS rmpc EXPORT rmpc-targets)

install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

install(EXPORT rmpc-targets
  FILE "rmpc-targets.cmake"
  NAMESPACE rmpc::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/rmpc"
)
