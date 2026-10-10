# Copyright (c) 2026 Arun Muraleedharan.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Animate the paddle-juggling closed-loop run.

Runs build/examples/paddle_closed_loop (unless --no-run), then plays the
logged run as a scrolling height-over-time view: the ball and paddle so
far, the chosen plan over the horizon with its predicted bounces, and a
few of the random sample plans. The ball and paddle are drawn at "now".

    python examples/paddle_juggling/animate.py              # run + show
    python examples/paddle_juggling/animate.py --save paddle.gif
"""

import sys
from pathlib import Path

import numpy as np
from matplotlib.patches import Circle, Rectangle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import anim  # noqa: E402

PAST = 1.5          # history shown left of "now" [s]
PADDLE_W = 0.12     # drawn paddle width, in seconds of the time axis


def main():
    args = anim.parse_args(__doc__.splitlines()[0], "paddle_juggling")
    if not args.no_run:
        anim.run_sim(args.build / "examples" / "paddle_closed_loop",
                     HERE / "config.yaml", args.logs)

    run = anim.load(args.logs / "run.csv")
    meta = anim.load(args.logs / "meta.csv")
    n = len(run)
    dt = float(meta["dt"])
    horizon = int(meta["n_horizon"]) * dt
    stroke = float(meta["stroke"])
    pred = anim.by_step(anim.load(args.logs / "predictions.csv"), n,
                        ["k", "ball_z", "paddle_z"])
    samples = anim.load(args.logs / "samples.csv")
    n_samples = int(samples["sample"].max()) + 1
    samp = anim.by_step(samples, n, ["k", "ball_z"]).reshape(
        n, n_samples, -1, 2)
    t = run["t"]

    anim.style()
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 6.2))
    grid = fig.add_gridspec(3, 2, width_ratios=[1.6, 1], hspace=0.65,
                            wspace=0.18, left=0.06, right=0.97, top=0.9,
                            bottom=0.09)
    ax = fig.add_subplot(grid[:, 0])
    fig.suptitle("Paddle juggling with randomized MPC", x=0.05, ha="left",
                 fontsize=13, fontweight="bold", color=anim.TEXT)

    ax.set_ylim(-stroke - 0.15, 1.35)
    ax.set_xlabel("time [s]  (now at the paddle)")
    ax.set_ylabel("height [m]")
    target_line, = ax.plot([], [], color=anim.TEXT_2, linewidth=1,
                           linestyle="--", label="target apex")
    sample_lines = [ax.plot([], [], color=anim.MUTED, linewidth=0.8,
                            alpha=0.6)[0] for _ in range(n_samples)]
    ball_past, = ax.plot([], [], color=anim.ORANGE, label="ball")
    paddle_past, = ax.plot([], [], color=anim.TEXT_2, linewidth=1.5,
                           label="paddle")
    ball_plan, = ax.plot([], [], color=anim.BLUE, label="chosen plan")
    paddle_plan, = ax.plot([], [], color=anim.BLUE, linewidth=1.5,
                           linestyle=(0, (4, 3)))
    ax.plot([], [], color=anim.MUTED, linewidth=1,
            label=f"{n_samples} of the random samples")
    now_line = ax.axvline(0, color=anim.GRID, linewidth=1)
    ball = Circle((0, 0), 0.035, color=anim.ORANGE, zorder=5)
    ax.add_patch(ball)
    paddle = Rectangle((0, 0), PADDLE_W, 0.03, color=anim.TEXT, zorder=5)
    ax.add_patch(paddle)
    ax.legend(loc="upper left", ncol=5)

    cursors = [
        anim.time_panel(fig.add_subplot(grid[0, 1]), t,
                        [(run["apex_target"], anim.TEXT_2, "target"),
                         (run["apex"], anim.ORANGE, "ball")],
                        "Apex height", "apex [m]"),
        anim.time_panel(fig.add_subplot(grid[1, 1]), t,
                        [(run["paddle_z"], anim.BLUE, None)],
                        "Paddle position", "z [m]",
                        bounds=(-stroke, stroke)),
        anim.time_panel(fig.add_subplot(grid[2, 1]), t,
                        [(run["u_cmd"], anim.BLUE, "commanded"),
                         (run["u_applied"], anim.ORANGE, "applied")],
                        "Paddle acceleration", "a [m/s²]",
                        bounds=(float(meta["u_min"]), float(meta["u_max"]))),
    ]
    fig.axes[-1].set_xlabel("time [s]")

    def draw_frame(i):
        now = t[i]
        ax.set_xlim(now - PAST, now + horizon)
        past = slice(max(0, i - int(PAST / dt)), i + 1)
        ball_past.set_data(t[past], run["ball_z"][past])
        paddle_past.set_data(t[past], run["paddle_z"][past])

        t_pred = now + pred[i, :, 0] * dt
        ball_plan.set_data(t_pred, pred[i, :, 1])
        paddle_plan.set_data(t_pred, pred[i, :, 2])
        for j, line in enumerate(sample_lines):
            line.set_data(now + samp[i, j, :, 0] * dt, samp[i, j, :, 1])

        lo = max(0, i - int(PAST / dt))
        hi = min(n - 1, i + int(horizon / dt))
        target_line.set_data(t[lo:hi + 1], run["apex_target"][lo:hi + 1])
        now_line.set_xdata([now, now])
        ball.center = (now, run["ball_z"][i])
        paddle.set_xy((now - PADDLE_W / 2, run["paddle_z"][i] - 0.03))
        for cur in cursors:
            anim.set_cursor(cur, now)

    anim.play(fig, draw_frame, n, args)


if __name__ == "__main__":
    main()
