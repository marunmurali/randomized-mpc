#!/usr/bin/env python3
# Copyright (c) 2026 Arun Muraleedharan.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Rewrite CMake's nvcc compile_commands.json into one clang-tidy can parse.

clang has its own CUDA front end but does not understand nvcc's flags
(--options-file, --expt-relaxed-constexpr, --diag-error, -arch=...). Keep
only what analysis needs (include paths, defines, language standard) and
add clang's CUDA flags.

usage: clang_compile_db.py <build/compile_commands.json> <out_dir>
                           [--cuda-path /usr/local/cuda] [--gpu-arch sm_80]
"""

import argparse
import json
import os
import shlex


def expandOptionFiles(args, directory):
    """Inline nvcc --options-file response files (CMake puts -I there)."""
    out = []
    it = iter(args)
    for a in it:
        if a == "--options-file":
            with open(os.path.join(directory, next(it))) as f:
                out += shlex.split(f.read())
        else:
            out.append(a)
    return out


def analysisFlags(args):
    """Keep include paths, defines and -std; drop everything nvcc-specific."""
    keep = []
    it = iter(args)
    for a in it:
        if a in ("-I", "-isystem", "-D"):
            keep += [a, next(it)]
        elif a.startswith(("-I", "-isystem", "-D", "-std=")):
            keep.append(a)
    return keep


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("compile_db")
    ap.add_argument("out_dir")
    ap.add_argument("--cuda-path", default="/usr/local/cuda")
    ap.add_argument("--gpu-arch", default="sm_80")
    opts = ap.parse_args()

    with open(opts.compile_db) as f:
        entries = json.load(f)

    out = []
    for e in entries:
        args = e["arguments"] if "arguments" in e else shlex.split(e["command"])
        flags = analysisFlags(expandOptionFiles(args[1:], e["directory"]))
        flags += [
            "-x", "cuda",
            f"--cuda-gpu-arch={opts.gpu_arch}",
            f"--cuda-path={opts.cuda_path}",
            # clang may not know the newest CUDA release yet; parsing is
            # unaffected
            "-Wno-unknown-cuda-version",
            "-c", e["file"],
        ]
        out.append({"directory": e["directory"], "file": e["file"],
                    "arguments": ["clang++"] + flags})

    os.makedirs(opts.out_dir, exist_ok=True)
    with open(os.path.join(opts.out_dir, "compile_commands.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {len(out)} entries to {opts.out_dir}/compile_commands.json")


if __name__ == "__main__":
    main()
