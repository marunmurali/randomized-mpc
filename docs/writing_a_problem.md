# Writing your own problem

This guide shows how to control a new system with `rmpc`. You write one
struct that describes the model and the cost; the engine does the sampling,
the input constraints, the parallel rollout on the GPU, and the selection.

The running example is the smallest possible problem: a point that moves
with the commanded velocity (`x' = u`) and should reach a target position.
It is the problem used in `tests/unit/test_engine.cu`.

## 1. The problem struct

```cpp
#include "rmpc/rmpc.cuh"

struct Integrator {
  struct State     { float x; };          // what you integrate
  struct Params    { float q; float r; }; // constants: model + cost weights
  struct Reference { float target; };     // what to track, one per step

  // one discrete step x_{k+1} = f(x_k, u_k)
  RMPC_HD static State step(const State& s, float u, const Reference&,
                            const Params&, float dt) {
    return {s.x + dt * u};
  }

  // cost of x_k with applied input u_k (u_prev = input at k-1)
  RMPC_HD static float stageCost(const State& s, float u, float u_prev,
                                 const Reference& ref, const Params& p,
                                 int k, int n_horizon) {
    const float e = s.x - ref.target;
    return p.q * e * e + p.r * u * u;
  }

  // cost of the last predicted state
  RMPC_HD static float terminalCost(const State& s, const Reference& ref,
                                    const Params& p, int n_horizon) {
    const float e = s.x - ref.target;
    return p.q * e * e;
  }
};

static_assert(rmpc::Problem<Integrator>);  // readable error if anything is off
```

That is the whole interface:

| Member | Meaning |
|---|---|
| `State` | The integrated state. Any trivially copyable struct. |
| `Params` | Everything constant: model parameters and cost weights. Passed by value to the GPU on every solve, so keep it to plain numbers (a few hundred bytes is fine). Change it between solves with `Controller::setParams`. |
| `Reference` | Whatever the cost compares against, one value per horizon step. Use an empty-ish struct if you only regulate to a fixed point. |
| `step` | Your model. You choose the integrator: Euler, RK4, sub-stepping, all fine. |
| `stageCost` | Called for every step `k = 0 .. N-1`. `k` and `n_horizon` let you weight steps differently (the car example drops later steps linearly). |
| `terminalCost` | Called once on the last predicted state. |

### Rules for `RMPC_HD` functions

`RMPC_HD` means *host and device*: the same function runs inside the GPU
rollout and on the CPU (your simulation, tests, plotting). Inside it:

- Plain math only: `sinf`, `cosf`, `expf`, `sqrtf`, `fminf`, `fabsf`, ...
  `constexpr` standard helpers such as `std::array` and `std::min` also work.
- No heap allocation, `std::vector`, exceptions, virtual calls or I/O.
- Calling a function that is not `RMPC_HD` (or `constexpr`) is a compile
  error. This is deliberate: by default nvcc only warns, and the GPU kernel
  then silently does nothing.

## 2. Solving

```cpp
rmpc::Config cfg;
cfg.n_horizon = 50;     // N prediction steps
cfg.n_samples = 1024;   // random input sequences per solve
cfg.n_basis   = 6;      // smoothness: cosine modes drawn per sample
cfg.dt        = 0.05f;
cfg.u_min = -0.5f;  cfg.u_max = 0.5f;   // hard input bounds
cfg.du_max = 2.0f;                      // hard rate limit [unit/s]
cfg.warm_start = true;

auto created = rmpc::Controller<Integrator>::create(cfg, {1.0f, 0.01f});
if (created.isError()) { /* created.error() says why */ }
auto& ctrl = *created.value();

std::vector<Integrator::Reference> ref(cfg.n_horizon, {1.0f});

Integrator::State x{0.0f};
float u = 0.0f;                       // input currently applied
for (int t = 0; t < 200; ++t) {
  auto sol = ctrl.solve(x, u, ref);   // plan from the current state
  // ... apply to your system for one step; here the model is the plant:
  x = Integrator::step(x, u, ref[0], {1.0f, 0.01f}, cfg.dt);
  u = sol.value().nextInput();
}
```

The API never throws: `create`, `solve` and `evaluate` return a
`Result<T>`; check `isError()` before `value()`.

### Timing convention

`solve(x0, u0, ref)` assumes `u0` is the input **already being applied**
during the coming step: computing the plan takes time, so it cannot change
the current step. Every sampled sequence therefore starts with
`u_cmd[0] = u0`, and the decision is `u_cmd[1]`, returned by
`nextInput()`. The loop above is the matching pattern: advance the plant
with the current input, then switch to `nextInput()`.

### What `solve` returns

`Solution::best` is a `Trajectory` over the full horizon:

- `u_cmd[k]`: commanded inputs (bounded and rate-limited),
- `u_applied[k]`: the same after the actuator lag (equal when `lag_tau = 0`);
  this is what `step` and the cost saw,
- `states[k]`: predicted states, `states[0] = x0`,
- `cost`: total cost.

With `Config::n_debug_samples = K`, `Solution::samples` also holds the
first `K` sample trajectories, useful for plotting what the controller
considered. `Controller::evaluate(x0, u0, u_cmd, ref)` rolls out a sequence
of your choice through the same model and cost.

## 3. How the input sequences are generated

Each solve draws `n_samples` sequences:

1. For each sample, `n_basis` coefficients are drawn uniformly in
   `[-1, 1]` and mapped through an orthonormal inverse DCT (cosine modes
   over the horizon) to a smooth sequence of input *increments*, scaled by
   `du_max * dt`. Fewer modes give smoother sequences.
2. The increments are summed from `u0`. Every increment is clamped to
   `±du_max * dt` and every input to `[u_min, u_max]`, so **every sample
   satisfies the bounds and the rate limit**.
3. Sample 0 holds `u0` constant: a safe baseline that is always available.
4. With `warm_start`, sample 1 is the previous solution shifted by one step,
   through the same clamps, so a good plan is never lost between solves.
5. With `lag_tau > 0`, a first-order actuator lag
   `u_app[k] = a u_app[k-1] + (1 - a) u_cmd[k]`, `a = exp(-dt / lag_tau)`,
   is applied before the rollout.

Then every sample is rolled out through your `step` on its own GPU thread,
costed with your `stageCost` and `terminalCost`, and the cheapest wins.

State constraints are expressed as soft costs (see the track limit in the
cart-pole example and the track bounds and grip limit in the car example).

## 4. Tuning tips

- **Unstable systems need short horizons.** Random open-loop sequences
  diverge on an unstable system, and over a long horizon every sample's
  cost is dominated by how it falls apart at the end. The cart-pole
  balances with a 0.8 s horizon and fails with 2.5 s.
- **Keep cost terms on comparable scales.** The ranking of samples is all
  that matters, so one huge term drowns the others.
- **`du_max` sets how far samples can move.** The sampled increments are
  usually well below the limit, so a rate limit that is too tight makes
  every sample look alike.
- **More samples help less than a better cost.** Doubling `n_samples`
  doubles the GPU work; try reshaping the cost first.
- Fix `Config::seed` for reproducible runs; solve `i` uses `seed + i`.

## 5. Building on an existing problem

A problem is just a struct, so a new one can wrap an existing one instead
of copying it. `examples/car_obstacles` reuses the car unchanged and only
adds a collision term:

```cpp
struct CarObstacles {
  using State = CarSteering::State;          // same state and reference
  using Reference = CarSteering::Reference;
  struct Params {
    CarSteering::Params car;                 // the car's own parameters
    Obstacle obstacles[MAX_OBSTACLES];       // fixed size: Params goes to
    int n_obstacles;                         // the GPU by value
    float car_radius, clearance, hit_cost;
  };

  RMPC_HD static State step(const State& x, float u, const Reference& r,
                            const Params& p, float dt) {
    return CarSteering::step(x, u, r, p.car, dt);
  }
  RMPC_HD static float stageCost(...) {
    return CarSteering::stageCost(..., p.car, ...) + (collides(x, p) ? hit : 0);
  }
  ...
};
```

Two things to note:

- **Fixed-size data in `Params`.** Use arrays with a maximum count, not
  `std::vector`; the struct is copied to the GPU on every solve.
- **Where to put time-independent data.** The obstacles are in `Params`,
  not `Reference`, because every sample reaches a given obstacle at a
  different horizon step. The cost checks each sample's own predicted
  position against the whole list. Use `Reference` for things indexed by
  time (a speed profile, a target that changes at a known time).

Non-smooth models work the same way: `examples/paddle_juggling` handles a
bounce with an `if` inside `step`.

## 6. Making it a new example

The fastest start is to copy `examples/cartpole/`:

| File | What to change |
|---|---|
| `problem.cuh` | your struct |
| `params_yaml.h` | read your `Params` from YAML |
| `config.yaml` | the shared `engine:` section, plus your own sections |
| `closed_loop.cu` | initial state, references, what to log |
| `animate.py` | `draw_frame(i)`: how to draw your system |

Register it in `examples/CMakeLists.txt` like the cart-pole. The
animation helpers in `examples/common/anim.py` give you playback, saving,
and time-series side panels with a moving cursor; you only draw the scene.
Keep the rule the examples follow: Python only draws what the C++ run
logged, and the C++ plant is your own `step`, so the animation always
shows the model the controller used.
