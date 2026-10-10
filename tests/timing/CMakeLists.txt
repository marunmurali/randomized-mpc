add_executable(rmpc_timing
  "test_timing.cu"
)

target_link_libraries(rmpc_timing
  PRIVATE
    car_steering
)
