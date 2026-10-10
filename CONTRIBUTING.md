# Contributing

## Code practices

### Exceptions-free code
Exceptions do not allow a clear tracking of failure modes, neither they enforce error checking, which is central in the realization of safety-critical systems.

To keep code exceptions-free, every function should
- not throw any exception, and
- catch and handle any possible exception thrown from external libraries.

Such a function should be marked `noexcept`. This is a convention checked by review and static analysis (see `.clang-tidy`'s `bugprone-exception-escape`), not by a compiler flag, so a `noexcept` function that can still throw is a bug to fix, not a build error.

### Functional paradigm
To enable a full traceability of the processes, minimizing the side-effects, the functional paradigm should be followed, reducing as much as possible the use of classes.

Whenever a class is needed, effort should be put in reducing side-effects and exposing any information needed to determine the class state.


## C++ code style
Code is written according to the [Google C++ Style Guide](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwj32NbcztX8AhUCTaQEHXb6CUcQFnoECBgQAQ&url=https%3A%2F%2Fgoogle.github.io%2Fstyleguide%2Fcppguide.html&usg=AOvVaw21RHeXr-wJRXW065nKkziR).

| Naming rules |                |
| ------------ | -------------- |
| Constants    | SNAKE_CASE     |
| Variables    | snake_case     |
| Functions    | lowerCamelCase |
| Classes      | UpperCamelCase |

In the development environment the style is enforced by means of VS Code extensions and static analysis (`.clang-tidy`). Host code of the CUDA sources in this repository builds with `-Wall -Wextra -Wshadow -Werror`.


## Namespaces and file hierarchy
The engine lives in the single namespace `rmpc`, with implementation details in `rmpc::detail`. Each example problem gets its own namespace named after its directory, e.g. `cartpole` for `examples/cartpole/`.

Directories in the include folder mirror the namespaces: `include/rmpc/` for `rmpc`, `include/rmpc/detail/` for `rmpc::detail`.


## Testing
Tests are split by level under `tests/`, each in its own subdirectory with its own `CMakeLists.txt`. Test sources are CUDA (`.cu`), since problem code is compiled for the GPU.

| Level              | Directory        | Purpose                                                        |
| ------------------ | ---------------- | -------------------------------------------------------------- |
| Unit Test (UT)     | `tests/unit/`    | gtest-based, ctest-registered: engine, car port, cart-pole     |
| Timing Test        | `tests/timing/`  | per-solve latency report (`rmpc_timing`), run manually         |

Closed-loop tests that take more than a moment carry the ctest label `slow`; `ctest -LE slow` skips them.

`tests/data/car_rollout_golden.csv` is the best rollout of the original (pre-generic) controller. `test_car_port` replays its input sequence through `Controller::evaluate`, so a change to the car model or cost that alters its numbers fails there.

gtest suite names are `UpperCamelCase`, matching the `Classes` rule above (e.g. `EngineTest`); individual test case names are `lowerCamelCase`, matching the `Functions` rule (e.g. `basisIsOrthonormal`).


## CMake
Refer to [Modern CMake](https://cliutils.gitlab.io/modern-cmake/).


## Branching and review process
Cannot push directly to `main` nor to `dev`.

| Branches       |                             |
| -------------- | --------------------------- |
| main           |                             |
| dev            | &#8593; *if works (tested)* |
| feature-branch | &#8593; *if compiles*       |
