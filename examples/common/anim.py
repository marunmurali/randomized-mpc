# Copyright (c) 2026 Arun Muraleedharan.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Shared helpers for the example animations.

The animations never compute dynamics: every frame is drawn from the CSV
logs a C++ closed-loop run writes, where the plant is the same `step`
function the controller rolls out on the GPU.

A new example needs only a `draw_frame(i)` function; `play()` handles
playback, the time cursor on the side panels, and saving.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# reference palette (light): two categorical slots, text, recessive greys
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
MUTED = "#a3a29c"
GRID = "#e6e5e0"
BLUE = "#2a78d6"    # chosen plan / commanded input
ORANGE = "#eb6834"  # applied input / plant


def style():
    """Apply the shared figure style."""
    matplotlib.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_2,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 9,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": TEXT_2,
        "ytick.color": TEXT_2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 2.0,
        "font.size": 9,
    })


def parse_args(description, default_out):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--logs", type=Path, default=ROOT / "logs" / default_out,
                   help="log directory (default: %(default)s)")
    p.add_argument("--no-run", action="store_true",
                   help="reuse existing logs instead of running the simulation")
    p.add_argument("--build", type=Path, default=ROOT / "build",
                   help="CMake build directory (default: %(default)s)")
    p.add_argument("--save", type=Path,
                   help="write a .gif or .mp4 instead of showing a window")
    p.add_argument("--stride", type=int, default=2,
                   help="draw every n-th control step (default: %(default)s)")
    p.add_argument("--fps", type=int, default=25,
                   help="playback frame rate (default: %(default)s)")
    return p.parse_args()


def run_sim(binary, config, logs, *extra):
    """Run a closed-loop example binary, writing logs into `logs`."""
    if not binary.exists():
        sys.exit(f"missing {binary}: build the project first "
                 f"(cmake --build build)")
    result = subprocess.run([str(binary), str(config), str(logs),
                             *map(str, extra)],
                            capture_output=True, text=True, check=False)
    print(result.stdout.strip())
    if result.returncode not in (0, 2):  # 2 = ran but missed its target
        sys.exit(result.stderr.strip() or f"{binary.name} failed")


def load(path):
    """CSV with a header row -> numpy structured array."""
    return np.genfromtxt(path, delimiter=",", names=True)


def by_step(table, n_steps, field_names):
    """Rows logged step-major -> array [step, row_in_step, field]."""
    cols = np.stack([table[f] for f in field_names], axis=-1)
    return cols.reshape(n_steps, -1, len(field_names))


def time_panel(ax, t, series, title, ylabel, bounds=None):
    """A side panel: full history plus a moving time cursor.

    series: list of (values, color, label); bounds: (lo, hi) or None.
    Returns the cursor line, to be moved with set_cursor().
    """
    for values, color, label in series:
        ax.plot(t, values, color=color, label=label)
    if bounds is not None:
        for b in bounds:
            ax.axhline(b, color=MUTED, linewidth=1, linestyle="--")
    if len(series) > 1:
        # above the plot, right-aligned opposite the title
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0),
                  ncol=len(series), borderaxespad=0.2)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlim(t[0], t[-1])
    return ax.axvline(t[0], color=TEXT_2, linewidth=1)


def set_cursor(cursor, t):
    cursor.set_xdata([t, t])


def play(fig, draw_frame, n_steps, args):
    """Animate draw_frame(step) over every args.stride-th step."""
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

    frames = range(0, n_steps, max(1, args.stride))
    anim = FuncAnimation(fig, draw_frame, frames=frames,
                         interval=1000 / args.fps, blit=False)
    if args.save is None:
        import matplotlib.pyplot as plt
        plt.show()
        return
    args.save.parent.mkdir(parents=True, exist_ok=True)
    writer = (FFMpegWriter(fps=args.fps) if args.save.suffix == ".mp4"
              else PillowWriter(fps=args.fps))
    anim.save(args.save, writer=writer, dpi=80)
    print(f"saved {args.save}")
