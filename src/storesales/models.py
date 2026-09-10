"""The model ladder, from the answer you get for free to the one worth fitting.

Every model predicts in log space and returns sales in the original units, and
every one of them is measured on the same backtest, so the table in the README
is a like-for-like comparison rather than a set of numbers from different
protocols.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import features as F
from .data import holiday_flags, oil_series

KEYS = ["store_nbr", "family"]


def _days_to_national(holidays: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """Days to the nearest national day off, in either direction.

    Shops empty out the day after a holiday and fill up the day before it, and
    a flag on the day itself cannot say either.
    """
    h = holidays[(holidays["locale"] == "National") &
                 (holidays["type"].isin(["Holiday", "Transfer", "Bridge",
                                         "Additional"])) &
                 (~holidays["transferred"].astype(bool))]
    days = np.sort(pd.DatetimeIndex(h["date"].unique())
                     .to_numpy(dtype="datetime64[D]").astype(np.int64))
    if len(days) == 0:
        return pd.Series(99, index=index, dtype="int16")
    target = index.to_numpy(dtype="datetime64[D]").astype(np.int64)
    pos = np.searchsorted(days, target)
    before = days[np.clip(pos - 1, 0, len(days) - 1)]
    after = days[np.clip(pos, 0, len(days) - 1)]
    nearest = np.minimum(np.abs(target - before), np.abs(target - after))
    return pd.Series(np.clip(nearest, 0, 30).astype("int16"), index=index)


class Baseline:
    """Interface: fit on everything before `origin`, predict the frame given."""

    name = "baseline"

    def fit(self, panel: pd.DataFrame, origin: pd.Timestamp) -> "Baseline":
        self.origin = pd.Timestamp(origin)
        self.past = panel[panel["date"] < self.origin]
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class Zero(Baseline):
    """Predict nothing ever sells. The floor every other number is read against."""

    name = "zero"

    def predict(self, frame):
        return np.zeros(len(frame))


class LastValue(Baseline):
    """Carry the last observed day forward for sixteen days."""

    name = "last_value"

    def fit(self, panel, origin):
        super().fit(panel, origin)
        last_day = self.past["date"].max()
        self.value = (self.past[self.past["date"] == last_day]
                      .set_index(KEYS)["sales"])
        return self

    def predict(self, frame):
        return frame.set_index(KEYS).index.map(self.value).to_numpy(float)


class SeasonalNaive(Baseline):
    """Carry the same weekday from the most recent complete week."""

    name = "seasonal_naive"

    def fit(self, panel, origin):
        super().fit(panel, origin)
        last = self.past["date"].max()
        week = self.past[self.past["date"] > last - pd.Timedelta(days=7)].copy()
        week["dayofweek"] = week["date"].dt.dayofweek
        self.value = week.set_index(KEYS + ["dayofweek"])["sales"]
        return self

    def predict(self, frame):
        idx = frame.assign(dayofweek=frame["date"].dt.dayofweek) \
                   .set_index(KEYS + ["dayofweek"]).index
        return idx.map(self.value).to_numpy(float)


class DowMean(Baseline):
    """Mean of log1p sales on the same weekday over the last `weeks` weeks.

    The one to beat. It is three lines of pandas and it captures the two things
    that carry this dataset — the level of each series and its weekly shape —
    which is most of what there is to capture.
    """

    name = "dow_mean"

    def __init__(self, weeks: int = 8):
        self.weeks = weeks
        self.name = f"dow_mean_{weeks}w"

    def fit(self, panel, origin):
        super().fit(panel, origin)
        last = self.past["date"].max()
        window = self.past[self.past["date"] >
                           last - pd.Timedelta(days=7 * self.weeks)].copy()
        window["dayofweek"] = window["date"].dt.dayofweek
        window["y"] = np.log1p(window["sales"].to_numpy(float))
        self.value = window.groupby(KEYS + ["dayofweek"], observed=True)["y"].mean()
        return self

    def predict(self, frame):
        idx = frame.assign(dayofweek=frame["date"].dt.dayofweek) \
                   .set_index(KEYS + ["dayofweek"]).index
        return np.expm1(np.nan_to_num(idx.map(self.value).to_numpy(float)))


class DirectGBM(Baseline):
    """One gradient booster over every series, trained to forecast directly.

    Direct rather than recursive: a row is (series, origin, horizon), so the
    model learns "sixteen days out" as its own problem instead of feeding its
    own predictions back in sixteen times and compounding the error.

    Global rather than per-series: 1,782 series share one model, so a quiet
    series borrows the weekly shape and holiday response of the 1,781 others.
    Fitting 1,782 separate models on four years of daily data is the obvious
    alternative and it is worse — most of these series do not have enough
    signal of their own to estimate a holiday effect.
    """

    name = "direct_gbm"

    # The column the residual target is measured against: the series' own mean
    # for this weekday over the last eight weeks.
    ANCHOR = "dow_mean_8w"

    def __init__(self, n_origins: int = 52, origin_step: int = 7,
                 horizon: int = 16, seed: int = 0, residual: bool = False,
                 half_life: float | None = None, **kw):
        self.n_origins = n_origins
        self.origin_step = origin_step
        self.horizon = horizon
        # Predict the gap from the weekday mean rather than the level itself.
        # The level is the easy part and it dominates the loss, so a tree spends
        # most of its splits rediscovering that this store sells more bread than
        # that one sells car parts. Subtracting it first spends every split on
        # the part that is hard.
        self.residual = residual
        # Weight an origin by how recent it is. The sixteen days to forecast sit
        # immediately after the training data, so last month's shopping is worth
        # more evidence than last year's.
        self.half_life = half_life
        self.params = dict(
            max_iter=800, learning_rate=0.05, max_leaf_nodes=63,
            min_samples_leaf=40, l2_regularization=1.0,
            early_stopping=False, random_state=seed)
        self.params.update(kw)

    # -- feature assembly ---------------------------------------------------

    def _context(self, panel: pd.DataFrame) -> None:
        # The calendar has to run past the last row the panel holds: in a
        # backtest the panel stops at the fold origin, but the rows to be
        # predicted are the sixteen days after it. Building it from the panel
        # alone leaves those days unmatched and every calendar feature NaN.
        index = pd.date_range(panel["date"].min(),
                              max(panel["date"].max(),
                                  self.origin + pd.Timedelta(days=self.horizon)),
                              freq="D")
        self._oil = oil_series(self._frames.oil, index)
        self._hol = holiday_flags(self._frames.holidays, self._frames.stores, index)
        self._cal = F.calendar(pd.Series(index)).assign(date=index)
        self._to_holiday = _days_to_national(self._frames.holidays, index)
        self._season = F.seasonal_index(panel[panel["split"] == "train"])

    def _design(self, panel: pd.DataFrame, origin: pd.Timestamp,
                target: pd.DataFrame) -> pd.DataFrame:
        hist = F.series_history(panel, origin)
        x = target.merge(hist, on=KEYS, how="left")
        x = x.merge(self._cal, on="date", how="left")
        x = x.merge(self._hol, on=["date", "store_nbr"], how="left")
        x = x.merge(self._season, on=KEYS + ["date"], how="left")
        x["oil"] = x["date"].map(self._oil)
        x["horizon"] = (x["date"] - origin).dt.days.astype("int16") + 1
        x["days_to_holiday"] = x["date"].map(self._to_holiday)
        # A promotion only means something against what this series usually
        # runs: `onpromotion = 4` is a push for one family and a quiet day for
        # another.
        x["promo_vs_hist"] = x["onpromotion"] - x["promo_mean_28"] * 1.0
        # The same-weekday history for *this* row's weekday, picked out of the
        # seven columns the history builder produced.
        for weeks in F.DOW_WEEKS:
            cols = [f"dow{i}_mean_{weeks}w" for i in range(7)]
            for c in cols:
                if c not in x:
                    x[c] = 0.0
            x[f"dow_mean_{weeks}w"] = x[cols].to_numpy()[
                np.arange(len(x)), x["dayofweek"].to_numpy()]
            x = x.drop(columns=cols)
        return x

    FEATURES_DROP = {"date", "id", "sales", "split", "city", "state", "type"}

    def _matrix(self, x: pd.DataFrame):
        cols = [c for c in x.columns if c not in self.FEATURES_DROP]
        m = x[cols].copy()
        m["family"] = m["family"].map(self._family_code).astype("int16")
        return m[sorted(cols)], sorted(cols)

    # -- fit / predict ------------------------------------------------------

    def fit(self, panel: pd.DataFrame, origin: pd.Timestamp, frames=None):
        self.origin = pd.Timestamp(origin)
        self._frames = frames
        self._family_code = {f: i for i, f in
                             enumerate(sorted(panel["family"].unique()))}
        self._context(panel)

        rows = []
        for k in range(self.n_origins):
            o = self.origin - pd.Timedelta(days=self.origin_step * (k + 1))
            end = o + pd.Timedelta(days=self.horizon - 1)
            target = panel[(panel["date"] >= o) & (panel["date"] <= end) &
                           (panel["split"] == "train")]
            if target.empty:
                continue
            rows.append(self._design(panel, o, target))
        design = pd.concat(rows, ignore_index=True)

        # Dead series are answered by rule, so they are not in the fit: their
        # truth is a constant and they would only pull the loss around.
        alive = ~design.set_index(KEYS).index.isin(
            F.dead_series(panel, self.origin))
        design = design[alive]

        # Nor are the months before a series was first stocked. Those zeros are
        # a fact about the store's range, not a forecasting problem, and they
        # teach the model to predict zero for a series that is now healthy.
        started = F.first_sale(panel[panel["split"] == "train"])
        launch = design.set_index(KEYS).index.map(started)
        design = design[design["date"].to_numpy() >=
                        (pd.to_datetime(launch) +
                         pd.Timedelta(days=14)).to_numpy()]

        X, self._cols = self._matrix(design)
        y = np.log1p(design["sales"].to_numpy(float))
        if self.residual:
            y = y - design[self.ANCHOR].to_numpy(float)

        weight = None
        if self.half_life:
            age = (self.origin - design["date"]).dt.days.to_numpy(float)
            weight = 0.5 ** (age / self.half_life)

        cat = [self._cols.index("family"), self._cols.index("store_nbr"),
               self._cols.index("cluster")]
        self.model = HistGradientBoostingRegressor(
            categorical_features=cat, **self.params).fit(X, y, sample_weight=weight)

        self._panel = panel
        self._dead = F.dead_series(panel, self.origin)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        x = self._design(self._panel, self.origin, frame)
        X, _ = self._matrix(x)
        pred_log = self.model.predict(X[self._cols])
        if self.residual:
            pred_log = pred_log + x[self.ANCHOR].to_numpy(float)
        out = np.expm1(pred_log)
        dead = x.set_index(KEYS).index.isin(self._dead)
        out[dead] = 0.0
        return np.clip(out, 0.0, None)


class Blend(Baseline):
    """Average several variants in log space.

    Not an ensemble for its own sake. Every variant tried here lands within
    0.003 of every other on the aggregate while disagreeing on which rows it
    gets wrong — the residual target wins the June fold and loses the August
    one, the lag features do the reverse. Averaging keeps the agreement and
    cancels part of the disagreement, which is the one reliable gain left once
    a feature search has stopped paying.

    Averaged in log space because that is the space the metric lives in;
    averaging the sales and taking the log afterwards is a different, worse
    estimator on a series that is zero a third of the time.
    """

    name = "blend"

    def __init__(self, variants: list[dict] | None = None):
        self.variants = variants or [dict(), dict(residual=True),
                                     dict(half_life=180)]
        self.name = f"blend_{len(self.variants)}"

    def fit(self, panel, origin, frames=None):
        self.origin = pd.Timestamp(origin)
        self.models = [DirectGBM(**v).fit(panel, origin, frames=frames)
                       for v in self.variants]
        return self

    def predict(self, frame):
        logs = [np.log1p(np.clip(m.predict(frame), 0.0, None))
                for m in self.models]
        return np.expm1(np.mean(logs, axis=0))
