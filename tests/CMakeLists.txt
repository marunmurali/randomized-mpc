include(GoogleTest)

find_package(GTest REQUIRED)

# Unit tests: gtest-based, registered with ctest. The closed-loop cart-pole
# swing-up carries the label "slow" (ctest -LE slow skips it).
add_subdirectory(unit)

# Timing test: per-solve latency report, run manually.
add_subdirectory(timing)
