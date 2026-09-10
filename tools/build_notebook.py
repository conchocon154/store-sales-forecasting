#!/usr/bin/env python3
"""Generate the Kaggle notebook from one place, so it cannot drift from the repo.

A Kaggle notebook cannot import from `src/`, so the model has to be inlined.
Writing it by hand means two copies of the same code and, sooner or later, two
different answers to the same question. This builds it instead.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "store-sales-forecasting.ipynb"


def source(name: str) -> str:
    """A module's code, with the package-relative imports the notebook lacks."""
    text = (ROOT / "src" / "storesales" / name).read_text()
    keep = []
    for line in text.splitlines():
        if line.startswith(("from .", "from __future__")):
            continue
        if line.startswith(("import ", "from dataclasses", "from pathlib",
                            "from sklearn")):
            continue
        # A notebook has no `__file__`, and it sets DATA itself in the setup
        # cell, so the module-level default has nothing to resolve against.
        if "__file__" in line:
            continue
        keep.append(line)
    # The modules annotate with `X | Y`, which is evaluated at runtime once the
    # code is pasted into a cell rather than imported from a module. The future
    # import turns annotations back into strings, so the cell runs on any
    # interpreter rather than only on 3.10 and later.
    body = "from __future__ import annotations\n\n" + "\n".join(keep).strip()
    if name == "evaluate.py":
        # `folds` collides with a local in the notebook's backtest runner.
        body = body.replace("def folds(", "def make_folds(")
    if name == "models.py":
        # `features` is imported as `F` in the package; in a notebook every
        # name is already in the one namespace.
        body = body.replace("F.", "")
    return body


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip().splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.strip().splitlines(keepends=True)}


HEADER = """
# Store Sales — forecasting 1,782 series sixteen days ahead

Daily sales for 54 Corporación Favorita stores × 33 product families in
Ecuador, forecast sixteen days out and scored on RMSLE.

**What this notebook is honest about.** It reaches **0.40695** on the public
leaderboard, rank ~92 of 642. The top is 0.37294 and the top 5% is 0.38369, so
the gap is 0.023 and this is not a winning solution. What it does have is a
measurement protocol that says *why* — including one result that surprised me:
two substantially different models, 0.021 apart on the fold immediately before
the test window, scored 0.40713 and 0.40695 on the leaderboard. Local gains are
not reaching it.

The reading behind this: [Ekrem Bayar's comprehensive
guide](https://www.kaggle.com/code/ekrembayar/store-sales-ts-forecasting-a-comprehensive-guide),
the most-voted solution notebook on this competition, is where the
zero-forecasting rule comes from; [Ryan Holbrook's Time Series
course](https://www.kaggle.com/learn/time-series) is where the
deterministic-plus-learned framing comes from.

Code, tests and the full write-up:
**[github.com/conchocon154/store-sales-forecasting](https://github.com/conchocon154/store-sales-forecasting)**
"""

DECISIONS = """
## Three decisions that set the result

**Work in log space, throughout.** RMSLE is RMSE on `log1p`, so a model fitted
on raw sales optimises a loss the competition does not use. On series running
from 0 to 124,717 that is not a rounding difference. Every model here — the
baselines' averages included — is computed on `log1p` and converted back once,
at the end.

**Forecast directly, not recursively.** On the day the forecast is made the
last known sale is yesterday, and day sixteen still has to be predicted. A
one-step model that feeds its own output back in compounds its error sixteen
times. Here each training row is *(series, origin, horizon)*: the per-series
history is summarised strictly before the origin, the calendar features describe
the target day, and "sixteen days out" is learned as its own problem.

**Answer the dead series with a rule.** Ninety-six of the 1,782 series sold
nothing at all in the last sixty days, and they cover 1,536 of the 28,512 rows
to predict. Their forecast is zero — that is the answer, not an estimate — so
they are dropped from the fit and filled in afterwards.
"""

FOLDS_NOTE = """
The folds are sixteen-day blocks ending at the last day of training, stepping
backwards without overlap:

```
… train ……………………│ 06-29 → 07-14 │ 07-15 → 07-30 │ 07-31 → 08-15 │  test: 08-16 → 08-31
```

A random train/test split would ask a much easier question — predict a Tuesday
given the Wednesday after it — and would rank these models differently.
"""

HOLIDAY_NOTE = """
Three things the holiday file makes easy to get wrong, all of them documented in
the competition's own data description:

* `transferred` is `True` on the day that was **worked** — a separate `Transfer`
  row carries the day people actually took off;
* `Work Day` is the *opposite* of a holiday: a Saturday worked to make up for a
  bridge;
* `locale` scopes a holiday to the country, a region or one city, so a Local
  holiday closes only the stores in that city.

Getting any of the three backwards is invisible in the output, which is why the
repo has a test for each.
"""

DIAGNOSIS_NOTE = """
Five families out of thirty-three carry about 40% of the squared error on 15% of
the rows, and the worst is the one with the sharpest annual season: Ecuadorean
term starts in August and school supplies go up several-fold for a few weeks.
The year-ago seasonal index in the features above was built for exactly that and
moved the August fold from 0.42002 to 0.41305 — real, and small.

Error also climbs across the horizon, from about 0.40 on a fold's first days to
0.46 on its last. That matters for what comes next.
"""

FINDING = """
## The result I did not expect

Two submissions — a plain gradient booster, and a three-way blend with
cross-sectional features, explicit lags, a residual target and recency-weighted
origins — differ by **0.021** on the third fold, the sixteen days immediately
before the test window. On the public leaderboard they scored **0.40713** and
**0.40695**. Two hundredths of a percent apart.

The local improvement is real and measured. It is simply not reaching the
leaderboard, and until that is understood, more feature work is guessing.

The hypothesis worth testing first is in the per-day table above: error climbs
steadily across the horizon, so if the public leaderboard is scored on the
earlier part of the test window, it is measuring exactly the stretch where every
one of these models agrees. The test is cheap — score a fold on its first eight
days and its last eight separately, and see which half tracks the leaderboard.

If your own local gains do transfer to the leaderboard, I would genuinely like
to know what you are doing differently.

## Submission
"""

LADDER_NOTE = """
The weekday mean is the number to beat, and it is three lines of pandas: it
captures the level of each series and its weekly shape, which is most of what
there is to capture here. Everything after that is worth about a tenth.

### Where the loss actually is

RMSLE is a mean over 28,512 rows and it hides everything. Adding features until
the aggregate moves is guessing; splitting the same number says where to aim.
"""

SETUP = """
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

DATA = Path("/kaggle/input/store-sales-time-series-forecasting")
pd.set_option("display.width", 120)
"""

FACTS = """
frames = load(DATA)
panel = build_panel(frames)
train = panel[panel["split"] == "train"]
test = panel[panel["split"] == "test"]

print("train ", train.date.min().date(), "→", train.date.max().date())
print("series", train.groupby(["store_nbr", "family"]).ngroups,
      "  test days", test.date.nunique(), "  test rows", len(test))
print("zero rows in train: %.1f%%" % (100 * (train.sales == 0).mean()))

# The dead-series fact, computed here rather than with the helper below,
# because it is the observation that motivates the rule.
origin = test.date.min()
last60 = train[train.date > origin - pd.Timedelta(days=60)]
gone = last60.groupby(["store_nbr", "family"]).sales.sum().pipe(lambda s: s[s == 0].index)
covered = test.set_index(["store_nbr", "family"]).index.isin(set(gone))
print(f"sold nothing in the last 60 days: {len(gone)} series, covering "
      f"{covered.sum()} of {covered.size} rows to predict")
"""

BACKTEST = """
def run(builders, n_folds=3):
    rows = []
    for start, end in make_folds(train["date"].max(), n=n_folds):
        holdout = train[(train.date >= start) & (train.date <= end)]
        history = pd.concat([panel[panel.date < start], test])
        absent = dead_series(panel, start)
        for build in builders:
            model, t0 = build(), time.perf_counter()
            learned = isinstance(model, (DirectGBM, Blend))
            model.fit(history, start, **({"frames": frames} if learned else {}))
            pred = np.nan_to_num(model.predict(holdout))
            if not learned:
                # Every model gets the free win, so the table compares the
                # modelling and not who remembered the rule.
                is_dead = holdout.set_index(["store_nbr", "family"]).index.isin(absent)
                pred[is_dead] = 0.0
            rows.append({"fold": f"{start:%m-%d}", "model": model.name,
                         "rmsle": rmsle(holdout.sales, pred),
                         "sec": round(time.perf_counter() - t0)})
            print(f"  {start:%m-%d}  {model.name:16s} {rows[-1]['rmsle']:.5f}"
                  f"  ({rows[-1]['sec']}s)", flush=True)
    return pd.DataFrame(rows)


scores = run([Zero, LastValue, SeasonalNaive,
              lambda: DowMean(4), lambda: DowMean(8), lambda: DowMean(16),
              DirectGBM])
(scores.pivot_table(index="model", columns="fold", values="rmsle")
       .assign(mean=lambda t: t.mean(axis=1))
       .sort_values("mean").round(5))
"""

DIAGNOSE = """
start, end = make_folds(train.date.max(), n=3)[-1]
holdout = train[(train.date >= start) & (train.date <= end)].copy()
history = pd.concat([panel[panel.date < start], test])

model = DirectGBM().fit(history, start, frames=frames)
holdout["pred"] = np.nan_to_num(model.predict(holdout))
holdout["err2"] = (np.log1p(holdout.pred.clip(lower=0))
                   - np.log1p(holdout.sales)) ** 2
total = holdout.err2.sum()

(holdout.groupby("family")
        .agg(rmsle=("err2", lambda s: float(np.sqrt(s.mean()))),
             share_pct=("err2", lambda s: 100 * s.sum() / total),
             mean_sales=("sales", "mean"))
        .sort_values("share_pct", ascending=False)
        .head(8).round(3))
"""

PER_DAY = """
(holdout.groupby("date").err2
        .apply(lambda s: float(np.sqrt(s.mean())))
        .round(4).to_frame("rmsle").T)
"""

SUBMIT = """
origin = test.date.min()
final = Blend().fit(panel, origin, frames=frames)

out = test.copy()
out["sales"] = np.clip(np.nan_to_num(final.predict(out)), 0.0, None)
sub = out[["id", "sales"]].astype({"id": int}).sort_values("id", ignore_index=True)

assert len(sub) == 28512, len(sub)
assert sub.sales.notna().all() and (sub.sales >= 0).all()
sub.to_csv("submission.csv", index=False)

print(f"{len(sub)} rows, {(sub.sales == 0).mean():.1%} zeros, "
      f"mean {sub.sales.mean():.1f}")
sub.head()
"""


def build() -> list[dict]:
    return [
        md(HEADER),
        md(DECISIONS),
        code(SETUP),
        md("## The metric, and the folds it has to be measured on"),
        code(source("evaluate.py")),
        md(FOLDS_NOTE),
        md("## Loading and joining the seven files"),
        code(source("data.py")),
        md(HOLIDAY_NOTE),
        code(FACTS),
        md("## Features"),
        code(source("features.py")),
        md("## The model, and the ladder it has to beat"),
        code(source("models.py")),
        md("### Backtest"),
        code(BACKTEST),
        md(LADDER_NOTE),
        code(DIAGNOSE),
        md(DIAGNOSIS_NOTE),
        code(PER_DAY),
        md(FINDING),
        code(SUBMIT),
    ]


def main() -> int:
    nb = {
        "cells": build(),
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1))
    print(f"wrote {OUT} ({len(nb['cells'])} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
