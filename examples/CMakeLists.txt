find_package(yaml-cpp REQUIRED)

# YAML config loading shared by the examples (the engine itself has no
# YAML dependency).
add_library(examples_common INTERFACE)

target_include_directories(examples_common
  INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/common"
)

target_link_libraries(examples_common
  INTERFACE
    rmpc::rmpc
    yaml-cpp
)

# Car steering at constant speed.
add_library(car_steering INTERFACE)

target_include_directories(car_steering
  INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/car_steering"
)

target_link_libraries(car_steering
  INTERFACE
    examples_common
)

target_compile_definitions(car_steering
  INTERFACE
    RMPC_EXAMPLE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/car_steering"
)

add_executable(car_solve_once "car_steering/solve_once.cu")
target_link_libraries(car_solve_once PRIVATE car_steering)

add_executable(car_closed_loop "car_steering/closed_loop.cu")
target_link_libraries(car_closed_loop PRIVATE car_steering)

# Cart-pole swing-up.
add_library(cartpole INTERFACE)

target_include_directories(cartpole
  INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/cartpole"
)

target_link_libraries(cartpole
  INTERFACE
    examples_common
)

target_compile_definitions(cartpole
  INTERFACE
    RMPC_EXAMPLE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/cartpole"
)

add_executable(cartpole_closed_loop "cartpole/closed_loop.cu")
target_link_libraries(cartpole_closed_loop PRIVATE cartpole)

# Car slalom: car_steering's model plus obstacles.
add_library(car_obstacles INTERFACE)

target_include_directories(car_obstacles
  INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/car_obstacles"
)

target_link_libraries(car_obstacles
  INTERFACE
    examples_common
)

target_compile_definitions(car_obstacles
  INTERFACE
    RMPC_EXAMPLE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/car_obstacles"
)

add_executable(car_obstacles_closed_loop "car_obstacles/closed_loop.cu")
target_link_libraries(car_obstacles_closed_loop PRIVATE car_obstacles)

# Paddle juggling: hybrid dynamics with impacts.
add_library(paddle_juggling INTERFACE)

target_include_directories(paddle_juggling
  INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/paddle_juggling"
)

target_link_libraries(paddle_juggling
  INTERFACE
    examples_common
)

target_compile_definitions(paddle_juggling
  INTERFACE
    RMPC_EXAMPLE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/paddle_juggling"
)

add_executable(paddle_closed_loop "paddle_juggling/closed_loop.cu")
target_link_libraries(paddle_closed_loop PRIVATE paddle_juggling)
