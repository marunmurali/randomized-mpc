# Copyright (c) 2026 Arun Muraleedharan.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Animate the car-steering closed-loop run.

Runs build/examples/car_closed_loop (unless --no-run), then plays the
logged run top-down with a camera following the car: the car with its
front wheels at the actual steer angle, the predicted path of the chosen
plan, and a few of the random sample plans behind it.

    python examples/car_steering/animate.py                 # run + show
    python examples/car_steering/animate.py --save car.gif
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
sys.path.insert(0, str(HERE))
import anim  # noqa: E402
from car_scene import CarScene, load_logs  # noqa: E402


def main():
    args = anim.parse_args(__doc__.splitlines()[0], "car_steering")
    if not args.no_run:
        anim.run_sim(args.build / "examples" / "car_closed_loop",
                     HERE / "config.yaml", args.logs)

    run, track, pred, samp = load_logs(args.logs)
    t = run["t"]

    anim.style()
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 6.2))
    grid = fig.add_gridspec(3, 2, width_ratios=[1.6, 1], hspace=0.65,
                            wspace=0.18, left=0.05, right=0.97, top=0.9,
                            bottom=0.09)
    fig.suptitle("Car steering at constant speed with randomized MPC",
                 x=0.05, ha="left", fontsize=13, fontweight="bold",
                 color=anim.TEXT)
    scene = CarScene(fig.add_subplot(grid[:, 0]), run, track, pred, samp)

    err = run["d"] - run["d_target"]
    cursors = [
        anim.time_panel(fig.add_subplot(grid[0, 1]), t,
                        [(run["d_target"], anim.BLUE, "target"),
                         (run["d"], anim.ORANGE, "actual")],
                        "Lateral offset", "d [m]"),
        anim.time_panel(fig.add_subplot(grid[1, 1]), t,
                        [(err, anim.BLUE, None)], "Lateral error",
                        "d − target [m]"),
        anim.time_panel(fig.add_subplot(grid[2, 1]), t,
                        [(run["steer_cmd"], anim.BLUE, "commanded"),
                         (run["steer_applied"], anim.ORANGE, "applied")],
                        "Steer angle", "δ [rad]", bounds=(-0.3, 0.3)),
    ]
    fig.axes[-1].set_xlabel("time [s]")

    def draw_frame(i):
        scene.draw(i)
        for cur in cursors:
            anim.set_cursor(cur, t[i])

    anim.play(fig, draw_frame, len(run), args)


if __name__ == "__main__":
    main()
