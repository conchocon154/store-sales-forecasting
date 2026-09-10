#!/usr/bin/env python3
"""Where the loss actually is.

RMSLE is a mean over 28,512 rows and it hides everything. Adding features until
the aggregate moves is guessing; this splits the same number by family, by
store and by day so the next change can be aimed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from storesales import data as D            # noqa: E402
from storesales import evaluate as E        # noqa: E402
from storesales.models import DirectGBM     # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=2, help="0 oldest, 2 newest")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    frames = D.load()
    panel = D.build_panel(frames)
    train = panel[panel["split"] == "train"]
    start, end = E.folds(train["date"].max(), n=3)[args.fold]

    holdout = train[(train["date"] >= start) & (train["date"] <= end)].copy()
    history = pd.concat([panel[panel["date"] < start],
                         panel[panel["split"] == "test"]])
    model = DirectGBM().fit(history, start, frames=frames)
    holdout["pred"] = np.nan_to_num(model.predict(holdout))
    holdout["err2"] = (np.log1p(holdout["pred"].clip(lower=0))
                       - np.log1p(holdout["sales"])) ** 2

    total = holdout["err2"].sum()
    print(f"fold {start:%Y-%m-%d} → {end:%Y-%m-%d}   "
          f"RMSLE {np.sqrt(holdout['err2'].mean()):.5f}   rows {len(holdout)}\n")

    for by in ("family", "store_nbr", "date"):
        g = holdout.groupby(by, observed=True).agg(
            rmsle=("err2", lambda s: float(np.sqrt(s.mean()))),
            share=("err2", lambda s: float(s.sum() / total)),
            rows=("err2", "size"),
            mean_sales=("sales", "mean"))
        g = g.sort_values("share", ascending=False).head(args.top)
        print(f"--- worst by {by} (share = fraction of total squared error) ---")
        print(g.assign(share=lambda t: (100 * t.share).round(1)).round(3).to_string())
        print()

    out = ROOT / "reports" / f"diagnose_fold{args.fold}.csv"
    holdout[["date", "store_nbr", "family", "sales", "pred", "err2"]].to_csv(
        out, index=False)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
