#!/usr/bin/env python3
"""Measure every model on the same folds and write the table.

The folds are sixteen-day blocks ending at the last day of training, which is
the question the leaderboard asks. A random train/test split would ask a much
easier one — predict a Tuesday given the Wednesday after it — and would rank
these models in a different order.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from storesales import data as D          # noqa: E402
from storesales import evaluate as E      # noqa: E402
from storesales import features as F      # noqa: E402
from storesales.models import (           # noqa: E402
    DirectGBM, DowMean, LastValue, SeasonalNaive, Zero)

ROOT = Path(__file__).resolve().parents[1]


def build_models():
    return [Zero(), LastValue(), SeasonalNaive(),
            DowMean(4), DowMean(8), DowMean(16), DirectGBM()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--only", default=None, help="run one model by name")
    args = ap.parse_args()

    frames = D.load()
    panel = D.build_panel(frames)
    train = panel[panel["split"] == "train"]
    last = train["date"].max()

    results = []
    for start, end in E.folds(last, n=args.folds):
        holdout = train[(train["date"] >= start) & (train["date"] <= end)]
        history = panel[panel["date"] < start]
        dead = F.dead_series(panel, start)

        for model in build_models():
            if args.only and args.only not in model.name:
                continue
            t0 = time.perf_counter()
            kw = {"frames": frames} if isinstance(model, DirectGBM) else {}
            model.fit(pd.concat([history, panel[panel["split"] == "test"]]),
                      start, **kw)
            pred = np.nan_to_num(model.predict(holdout))
            if not isinstance(model, DirectGBM):
                # Every model gets the free win, so the table compares the
                # modelling and not who remembered the rule.
                is_dead = holdout.set_index(["store_nbr", "family"]).index.isin(dead)
                pred[is_dead] = 0.0
            score = E.rmsle(holdout["sales"], pred)
            results.append({"fold": f"{start:%Y-%m-%d}", "model": model.name,
                            "rmsle": score, "seconds": time.perf_counter() - t0})
            print(f"  {start:%Y-%m-%d}  {model.name:16s} {score:.5f}"
                  f"  ({results[-1]['seconds']:.1f}s)", flush=True)

    df = pd.DataFrame(results)
    table = (df.pivot_table(index="model", columns="fold", values="rmsle")
               .assign(mean=lambda t: t.mean(axis=1))
               .sort_values("mean"))
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "backtest_raw.csv", index=False)
    table.round(5).to_csv(out / "backtest.csv")
    print("\n" + table.round(5).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
