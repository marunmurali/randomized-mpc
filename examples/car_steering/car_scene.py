# Copyright (c) 2026 Arun Muraleedharan.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Top-down car scene shared by the car animations.

Draws the track, the car with its front wheels at the actual steer angle,
the chosen plan, the logged random samples and a trail, with a camera that
follows the car. Everything comes from the closed-loop CSV logs.
"""

import numpy as np
from matplotlib.patches import Circle, Polygon

import anim

LF, LR = 1.8, 1.315           # CoG to axles, as in config.yaml
LENGTH, WIDTH = 4.8, 1.92     # drawn body
WHEEL_L, WHEEL_W = 0.7, 0.3
VIEW_W, VIEW_H = 70.0, 34.0   # camera window [m]


def box(cx, cy, heading, length, width):
    """Corners of a length x width rectangle centred at (cx, cy)."""
    c, s = np.cos(heading), np.sin(heading)
    hx, hy = length / 2, width / 2
    local = np.array([[hx, hy], [-hx, hy], [-hx, -hy], [hx, -hy]])
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + [cx, cy]


def load_logs(logs):
    """run, track, predictions [step, k, xy] and samples [step, j, k, xy]."""
    run = anim.load(logs / "run.csv")
    n = len(run)
    pred = anim.by_step(anim.load(logs / "predictions.csv"), n, ["x", "y"])
    samples = anim.load(logs / "samples.csv")
    n_samples = int(samples["sample"].max()) + 1
    samp = anim.by_step(samples, n, ["x", "y"]).reshape(n, n_samples, -1, 2)
    return run, anim.load(logs / "track.csv"), pred, samp


class CarScene:
    """The top-down view; call draw(i) once per frame."""

    def __init__(self, ax, run, track, pred, samp, obstacles=None):
        self.ax, self.run, self.pred, self.samp = ax, run, pred, samp

        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("l", "r"):
            ax.plot(track[f"{side}x"], track[f"{side}y"], color=anim.TEXT_2,
                    linewidth=1.5)
        ax.plot(track["cx"], track["cy"], color=anim.MUTED, linewidth=1,
                linestyle=(0, (6, 6)))
        if obstacles is not None:
            for x, y, r in zip(obstacles["x"], obstacles["y"],
                               obstacles["radius"]):
                ax.add_patch(Circle((x, y), r, color=anim.TEXT, zorder=3))
            ax.plot([], [], "o", color=anim.TEXT, label="obstacle")

        n_samples = samp.shape[1]
        self.sample_lines = [ax.plot([], [], color=anim.MUTED, linewidth=0.8,
                                     alpha=0.6)[0] for _ in range(n_samples)]
        self.trail, = ax.plot([], [], color=anim.ORANGE, linewidth=1.5,
                              alpha=0.5)
        self.plan, = ax.plot([], [], color=anim.BLUE, linewidth=2,
                             label="chosen plan (2 s)")
        self.body = Polygon(np.zeros((4, 2)), closed=True, color=anim.ORANGE,
                            zorder=4, label="car")
        ax.add_patch(self.body)
        self.wheels = [Polygon(np.zeros((4, 2)), closed=True, color=anim.TEXT,
                               zorder=5) for _ in range(4)]
        for w in self.wheels:
            ax.add_patch(w)
        ax.plot([], [], color=anim.MUTED, linewidth=1,
                label=f"{n_samples} of the random samples")
        ax.legend(loc="lower left", ncol=4)
        self.clock = ax.text(0.98, 0.97, "", transform=ax.transAxes,
                             ha="right", va="top", fontsize=10,
                             color=anim.TEXT)

    def draw(self, i):
        run = self.run
        x, y, psi = run["x"][i], run["y"][i], run["psi"][i]
        steer = run["steer_applied"][i]
        c, s = np.cos(psi), np.sin(psi)
        # body centred between the axles
        mid = (LF - LR) / 2
        self.body.set_xy(box(x + mid * c, y + mid * s, psi, LENGTH, WIDTH))
        for w, (along, side, heading) in zip(self.wheels, [
                (LF, 0.8, psi + steer), (LF, -0.8, psi + steer),
                (-LR, 0.8, psi), (-LR, -0.8, psi)]):
            wx = x + along * c - side * s
            wy = y + along * s + side * c
            w.set_xy(box(wx, wy, heading, WHEEL_L, WHEEL_W))
        self.plan.set_data(self.pred[i, :, 0], self.pred[i, :, 1])
        for j, line in enumerate(self.sample_lines):
            line.set_data(self.samp[i, j, :, 0], self.samp[i, j, :, 1])
        self.trail.set_data(run["x"][:i + 1], run["y"][:i + 1])

        # camera: centred a little ahead of the car
        cx, cy = x + 15 * c, y + 15 * s
        self.ax.set_xlim(cx - VIEW_W / 2, cx + VIEW_W / 2)
        self.ax.set_ylim(cy - VIEW_H / 2, cy + VIEW_H / 2)
        self.clock.set_text(f"t = {run['t'][i]:.2f} s   s = {run['s'][i]:.0f} m")
