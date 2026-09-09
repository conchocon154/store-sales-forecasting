#!/usr/bin/env python3
"""Run the queued experiments that have no result yet, and record what they do.

Designed to be interrupted. Every experiment appends one row per fold to
`reports/experiments.csv` as soon as it finishes, so stopping the process loses
at most the experiment in flight, and starting it again skips everything
already measured.

    python3 scripts/experiment.py                 # run whatever is left
    python3 scripts/experiment.py --only residual # run one
    python3 scripts/experiment.py --status        # what is done, what is not
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from storesales import data as D            # noqa: E402
from storesales import evaluate as E        # noqa: E402
from storesales.configs import QUEUE        # noqa: E402
from storesales.models import DirectGBM     # noqa: E402

RESULTS = ROOT / "reports" / "experiments.csv"


def done() -> set[str]:
    if not RESULTS.exists():
        return set()
    df = pd.read_csv(RESULTS)
    counts = df.groupby("experiment")["fold"].nunique()
    return set(counts[counts >= 3].index)


def record(rows: list[dict]) -> None:
    RESULTS.parent.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS, mode="a", header=not RESULTS.exists(), index=False)


def leaderboard() -> pd.DataFrame:
    if not RESULTS.exists():
        return pd.DataFrame()
    df = pd.read_csv(RESULTS)
    return (df.pivot_table(index="experiment", columns="fold", values="rmsle")
              .assign(mean=lambda t: t.mean(axis=1))
              .sort_values("mean").round(5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--folds", type=int, default=3)
    args = ap.parse_args()

    finished = done()
    if args.status:
        print("done:   ", ", ".join(sorted(finished)) or "(none)")
        print("pending:", ", ".join(k for k in QUEUE if k not in finished) or "(none)")
        if not leaderboard().empty:
            print("\n" + leaderboard().to_string())
        return 0

    frames = D.load()
    panel = D.build_panel(frames)
    train = panel[panel["split"] == "train"]
    folds = E.folds(train["date"].max(), n=args.folds)

    for name, cfg in QUEUE.items():
        if name in finished or (args.only and args.only != name):
            continue
        print(f"\n=== {name}  {cfg}", flush=True)
        rows = []
        for start, end in folds:
            holdout = train[(train["date"] >= start) & (train["date"] <= end)]
            history = pd.concat([panel[panel["date"] < start],
                                 panel[panel["split"] == "test"]])
            t0 = time.perf_counter()
            model = DirectGBM(**cfg).fit(history, start, frames=frames)
            pred = np.nan_to_num(model.predict(holdout))
            score = E.rmsle(holdout["sales"], pred)
            rows.append({"experiment": name, "fold": f"{start:%Y-%m-%d}",
                         "rmsle": score, "seconds": time.perf_counter() - t0,
                         "config": str(cfg)})
            print(f"  {start:%Y-%m-%d}  {score:.5f}"
                  f"  ({rows[-1]['seconds']:.0f}s)", flush=True)
        record(rows)
        print(f"  {name} mean {np.mean([r['rmsle'] for r in rows]):.5f}", flush=True)

    print("\n" + leaderboard().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
