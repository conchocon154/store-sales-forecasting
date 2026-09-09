#!/usr/bin/env python3
"""Fit on everything and write the competition submission.

The only difference from a backtest fold is the origin: 2017-08-16, the first
day the leaderboard asks about, with every row of train behind it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from storesales import data as D            # noqa: E402
from storesales import features as F        # noqa: E402
from storesales.models import DirectGBM     # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "submission.csv"))
    args = ap.parse_args()

    frames = D.load()
    panel = D.build_panel(frames)
    test = panel[panel["split"] == "test"].copy()
    origin = test["date"].min()

    model = DirectGBM().fit(panel, origin, frames=frames)
    test["sales"] = np.clip(np.nan_to_num(model.predict(test)), 0.0, None)

    sub = (test[["id", "sales"]].astype({"id": int})
                                .sort_values("id", ignore_index=True))
    expected = frames.test["id"].nunique()
    assert len(sub) == expected, f"{len(sub)} rows, expected {expected}"
    assert sub["sales"].notna().all() and (sub["sales"] >= 0).all()

    sub.to_csv(args.out, index=False)
    dead = len(F.dead_series(panel, origin))
    print(f"wrote {args.out}: {len(sub)} rows, "
          f"{(sub['sales'] == 0).mean():.1%} zeros "
          f"({dead} series answered by the dead-series rule), "
          f"mean {sub['sales'].mean():.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
