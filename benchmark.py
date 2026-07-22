"""Benchmark detectors on YOUR machine — mean latency, FPS, parameter count.

Numbers like these belong in every README you write. Extend with --include-rtdetr
for the YOLO-vs-DETR comparison (slow on CPU but worth one run).

    python benchmark.py
    python benchmark.py --include-rtdetr --runs 10
"""
import argparse
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO, RTDETR

OUT = Path(__file__).parent / "outputs"


def bench(model, image, runs):
    for _ in range(3):
        model(image, verbose=False)  # warmup: first calls pay one-time setup costs
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(image, verbose=False)
        times.append((time.perf_counter() - t0) * 1000)
    return np.mean(times), np.std(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--include-rtdetr", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    image = "https://ultralytics.com/images/bus.jpg"

    candidates = [("YOLO26n", YOLO, "yolo26n.pt"), ("YOLO11n", YOLO, "yolo11n.pt")]
    if args.include_rtdetr:
        candidates.append(("RT-DETR-L", RTDETR, "rtdetr-l.pt"))

    rows = []
    for name, cls, weights in candidates:
        try:
            model = cls(weights)
        except Exception as e:
            print(f"skipping {name}: {e}")
            continue
        params = sum(p.numel() for p in model.model.parameters()) / 1e6
        mean, std = bench(model, image, args.runs)
        rows.append((name, params, mean, std, 1000 / mean))
        print(f"{name}: {mean:.1f}±{std:.1f} ms  ({1000 / mean:.1f} FPS, {params:.1f}M params)")

    table = ["| Model | Params (M) | Latency (ms) | FPS |", "|---|---|---|---|"]
    table += [f"| {n} | {p:.1f} | {m:.1f} ± {s:.1f} | {f:.1f} |" for n, p, m, s, f in rows]
    md = "\n".join(table)
    (OUT / "benchmark.md").write_text(f"CPU inference, {args.runs} runs, 640px:\n\n{md}\n")
    print(f"\nmarkdown table -> {OUT / 'benchmark.md'} (paste into README/blog)")


if __name__ == "__main__":
    main()
