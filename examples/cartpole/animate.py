# Copyright (c) 2026 Arun Muraleedharan.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Animate the cart-pole closed-loop run.

Runs build/examples/cartpole_closed_loop (unless --no-run), then plays the
logged run: cart and pole, the predicted pole-tip path of the chosen plan,
and a few of the random sample plans behind it.

    python examples/cartpole/animate.py                 # run + show
    python examples/cartpole/animate.py --save cartpole.gif
"""

import sys
from pathlib import Path

import numpy as np
from matplotlib.patches import Circle, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import anim  # noqa: E402

HERE = Path(__file__).resolve().parent
CART_W, CART_H = 0.4, 0.2


def main():
    args = anim.parse_args(__doc__.splitlines()[0], "cartpole")
    if not args.no_run:
        anim.run_sim(args.build / "examples" / "cartpole_closed_loop",
                     HERE / "config.yaml", args.logs)

    run = anim.load(args.logs / "run.csv")
    meta = anim.load(args.logs / "meta.csv")
    n = len(run)
    pole = 2.0 * float(meta["half_length"])
    limit = float(meta["track_limit"])

    pred = anim.by_step(anim.load(args.logs / "predictions.csv"), n,
                        ["x", "theta"])
    samples = anim.load(args.logs / "samples.csv")
    n_samples = int(samples["sample"].max()) + 1
    samp = anim.by_step(samples, n, ["x", "theta"]).reshape(
        n, n_samples, -1, 2)

    def tip(x, theta):
        return x + pole * np.sin(theta), pole * np.cos(theta)

    t = run["t"]
    theta = np.remainder(run["theta"] + np.pi, 2 * np.pi) - np.pi

    anim.style()
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 6.2))
    grid = fig.add_gridspec(3, 2, width_ratios=[1.6, 1], hspace=0.65,
                            wspace=0.18, left=0.05, right=0.97, top=0.9,
                            bottom=0.09)
    ax = fig.add_subplot(grid[:, 0])
    fig.suptitle("Cart-pole swing-up with randomized MPC", x=0.05,
                 ha="left", fontsize=13, fontweight="bold", color=anim.TEXT)

    # scene
    span = limit + 0.6
    ax.set_xlim(-span, span)
    ax.set_ylim(-pole - 0.4, pole + 0.5)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_yticks([])
    ax.set_xlabel("cart position x [m]")
    ax.plot([-span, span], [0, 0], color=anim.MUTED, linewidth=2)
    for side in (-limit, limit):
        ax.plot([side, side], [-0.15, 0.15], color=anim.TEXT_2, linewidth=2)
    ax.text(limit, 0.22, "track limit", ha="center", fontsize=8,
            color=anim.TEXT_2)

    sample_lines = [ax.plot([], [], color=anim.MUTED, linewidth=0.8,
                            alpha=0.45)[0] for _ in range(n_samples)]
    plan_line, = ax.plot([], [], color=anim.BLUE, linewidth=2,
                         label="chosen plan (pole tip)")
    cart = Rectangle((0, 0), CART_W, CART_H, color=anim.TEXT_2, zorder=3)
    ax.add_patch(cart)
    rod, = ax.plot([], [], color=anim.ORANGE, linewidth=4,
                   solid_capstyle="round", zorder=4, label="pole")
    bob = Circle((0, 0), 0.06, color=anim.ORANGE, zorder=5)
    ax.add_patch(bob)
    ax.plot([], [], color=anim.MUTED, linewidth=1,
            label=f"{n_samples} of the random samples")
    ax.legend(loc="lower left", ncol=3)
    clock = ax.text(0.98, 0.97, "", transform=ax.transAxes, ha="right",
                    va="top", fontsize=10, color=anim.TEXT)

    # side panels
    cursors = [
        anim.time_panel(fig.add_subplot(grid[0, 1]), t,
                        [(theta, anim.BLUE, None)], "Pole angle",
                        "θ [rad]", bounds=(-0.1, 0.1)),
        anim.time_panel(fig.add_subplot(grid[1, 1]), t,
                        [(run["x"], anim.BLUE, None)], "Cart position",
                        "x [m]", bounds=(-limit, limit)),
        anim.time_panel(fig.add_subplot(grid[2, 1]), t,
                        [(run["u_cmd"], anim.BLUE, "commanded"),
                         (run["u_applied"], anim.ORANGE, "applied")],
                        "Cart force", "u [N]",
                        bounds=(float(meta["u_min"]), float(meta["u_max"]))),
    ]
    fig.axes[-1].set_xlabel("time [s]")

    def draw_frame(i):
        x, th = run["x"][i], run["theta"][i]
        cart.set_xy((x - CART_W / 2, -CART_H / 2))
        tx, ty = tip(x, th)
        rod.set_data([x, tx], [0, ty])
        bob.center = (tx, ty)
        plan_line.set_data(*tip(pred[i, :, 0], pred[i, :, 1]))
        for j, line in enumerate(sample_lines):
            line.set_data(*tip(samp[i, j, :, 0], samp[i, j, :, 1]))
        clock.set_text(f"t = {t[i]:.2f} s")
        for c in cursors:
            anim.set_cursor(c, t[i])

    anim.play(fig, draw_frame, n, args)


if __name__ == "__main__":
    main()
