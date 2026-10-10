# Copyright (c) 2026 Arun Muraleedharan.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Animate the car slalom closed-loop run.

Runs build/examples/car_obstacles_closed_loop (unless --no-run), then plays
the logged run top-down: the car, the obstacles, the chosen plan and the
random samples, which visibly split around each obstacle.

    python examples/car_obstacles/animate.py                # run + show
    python examples/car_obstacles/animate.py --save slalom.gif
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
sys.path.insert(0, str(HERE.parent / "car_steering"))
import anim  # noqa: E402
from car_scene import CarScene, load_logs  # noqa: E402


def main():
    args = anim.parse_args(__doc__.splitlines()[0], "car_obstacles")
    if not args.no_run:
        anim.run_sim(args.build / "examples" / "car_obstacles_closed_loop",
                     HERE / "config.yaml", args.logs)

    run, track, pred, samp = load_logs(args.logs)
    obstacles = anim.load(args.logs / "obstacles.csv")
    t = run["t"]

    anim.style()
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 6.2))
    grid = fig.add_gridspec(3, 2, width_ratios=[1.6, 1], hspace=0.65,
                            wspace=0.18, left=0.05, right=0.97, top=0.9,
                            bottom=0.09)
    fig.suptitle("Car slalom with randomized MPC", x=0.05, ha="left",
                 fontsize=13, fontweight="bold", color=anim.TEXT)
    scene = CarScene(fig.add_subplot(grid[:, 0]), run, track, pred, samp,
                     obstacles=obstacles)

    gap_ax = fig.add_subplot(grid[1, 1])
    cursors = [
        anim.time_panel(fig.add_subplot(grid[0, 1]), t,
                        [(run["d"], anim.BLUE, None)],
                        "Lateral offset", "d [m]", bounds=(-4.0, 4.0)),
        anim.time_panel(gap_ax, t, [(run["gap"], anim.BLUE, None)],
                        "Gap to nearest obstacle", "gap [m]"),
        anim.time_panel(fig.add_subplot(grid[2, 1]), t,
                        [(run["steer_cmd"], anim.BLUE, "commanded"),
                         (run["steer_applied"], anim.ORANGE, "applied")],
                        "Steer angle", "δ [rad]", bounds=(-0.3, 0.3)),
    ]
    gap_ax.axhline(0.0, color=anim.TEXT_2, linewidth=1)
    gap_ax.set_ylim(-0.5, 6.0)
    fig.axes[-1].set_xlabel("time [s]")

    def draw_frame(i):
        scene.draw(i)
        for cur in cursors:
            anim.set_cursor(cur, t[i])

    anim.play(fig, draw_frame, len(run), args)


if __name__ == "__main__":
    main()
