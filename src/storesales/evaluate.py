"""The metric, and the only honest way to measure against it here.

RMSLE is RMSE on `log1p`, so every model in this repo predicts in log space and
is only converted back at the end. That is not a convenience: fitting on raw
sales optimises the wrong loss, and on a series whose values run from 0 to
124,717 the difference is not small.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON = 16          # the test period, 2017-08-16 to 2017-08-31


def rmsle(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0.0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def rmse_log(y_true_log, y_pred_log) -> float:
    """The same number, when both sides are already in log space."""
    d = np.asarray(y_pred_log, float) - np.asarray(y_true_log, float)
    return float(np.sqrt(np.mean(d ** 2)))


def folds(last_train_date: pd.Timestamp, n: int = 3,
          horizon: int = HORIZON) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Backtest origins, each holding out the next `horizon` days.

    The folds are the same length as the competition's test window and sit
    directly before it, so a fold answers the question the leaderboard asks —
    "sixteen days ahead, from a standing start" — rather than the much easier
    one-day-ahead question a random split would ask.
    """
    out = []
    end = pd.Timestamp(last_train_date)
    for _ in range(n):
        start = end - pd.Timedelta(days=horizon - 1)
        out.append((start, end))
        end = start - pd.Timedelta(days=1)
    return list(reversed(out))
