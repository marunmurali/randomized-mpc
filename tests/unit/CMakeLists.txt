add_executable(test_engine
  "test_engine.cu"
)

target_link_libraries(test_engine
  PRIVATE
    rmpc::rmpc
    GTest::gtest
    GTest::gtest_main
)

gtest_discover_tests(test_engine
  DISCOVERY_MODE PRE_TEST
)

add_executable(test_car_port
  "test_car_port.cu"
)

target_link_libraries(test_car_port
  PRIVATE
    car_steering
    GTest::gtest
    GTest::gtest_main
)

target_compile_definitions(test_car_port
  PRIVATE
    RMPC_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)

gtest_discover_tests(test_car_port
  DISCOVERY_MODE PRE_TEST
)

# Tests for an example problem: fast suite "<suite>Test", closed-loop suite
# "<suite>SlowTest" labeled slow (ctest -LE slow skips it).
function(add_example_test test_name example suite)
  add_executable(${test_name} "${test_name}.cu")

  target_link_libraries(${test_name}
    PRIVATE
      ${example}
      GTest::gtest
      GTest::gtest_main
  )

  target_compile_definitions(${test_name}
    PRIVATE
      RMPC_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
  )

  gtest_discover_tests(${test_name}
    TEST_FILTER "${suite}Test.*"
    DISCOVERY_MODE PRE_TEST
  )

  gtest_discover_tests(${test_name}
    TEST_FILTER "${suite}SlowTest.*"
    DISCOVERY_MODE PRE_TEST
    PROPERTIES LABELS slow
  )
endfunction()

add_example_test(test_cartpole cartpole CartPole)
add_example_test(test_car_obstacles car_obstacles CarObstacles)
add_example_test(test_paddle paddle_juggling Paddle)
