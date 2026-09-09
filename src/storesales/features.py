"""Features for a direct sixteen-day-ahead forecast.

The horizon is what shapes this file. On the day the forecast is made, the last
known sale is the day before; the model must still predict day sixteen. So no
feature may depend on anything after the origin, and a lag of one day is
unavailable for fifteen of the sixteen targets.

The way out is a *direct* model: one row per (series, origin, horizon), where
the series statistics are computed strictly up to the origin and the calendar
features describe the target day. Recursive one-step models avoid the lag
problem by feeding predictions back in, which compounds error across sixteen
steps; this does not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import PAYDAYS, QUAKE, QUAKE_WEEKS

# Windows for the per-series history summaries, in days before the origin.
WINDOWS = (7, 14, 28, 56, 112)
# How many same-weekday observations to average.
DOW_WEEKS = (4, 8)


def calendar(dates: pd.Series) -> pd.DataFrame:
    d = pd.to_datetime(dates)
    out = pd.DataFrame({
        "dayofweek": d.dt.dayofweek.astype("int8"),
        "day": d.dt.day.astype("int8"),
        "month": d.dt.month.astype("int8"),
        "year": d.dt.year.astype("int16"),
        "dayofyear": d.dt.dayofyear.astype("int16"),
        "weekofyear": d.dt.isocalendar().week.astype("int16").to_numpy(),
    })
    out["is_weekend"] = (out["dayofweek"] >= 5).astype("int8")
    # Wages land on the 15th and the last day of the month; the shops feel it.
    out["is_payday"] = (d.dt.day.isin(PAYDAYS) | d.dt.is_month_end).astype("int8")
    out["days_from_payday"] = _days_from_payday(d).astype("int8")
    weeks_since_quake = (d - QUAKE).dt.days / 7.0
    out["quake_window"] = ((weeks_since_quake >= 0) &
                           (weeks_since_quake <= QUAKE_WEEKS)).astype("int8")
    # A smooth trend beats a raw date for a tree: it can split it, and it does
    # not explode when the test period sits past every value it has seen.
    out["t"] = (d - pd.Timestamp("2013-01-01")).dt.days.astype("int32")
    return out


def _days_from_payday(d: pd.Series) -> pd.Series:
    day = d.dt.day
    month_end = d.dt.days_in_month
    to_15 = (15 - day).abs()
    to_end = (month_end - day).abs()
    return np.minimum(to_15, to_end)


# Days before the origin to read off directly. The first week gives the recent
# shape, 14/21/28 give the weekly rhythm, and 364 is the same weekday a year ago
# — which is how a tree gets at an annual season from four years of data.
LAGS = (1, 2, 3, 4, 5, 6, 7, 14, 21, 28, 35, 364)

# The longest window anything below looks back over. Slicing the past to this
# once, instead of grouping over four years for every origin, is what makes a
# year of training origins affordable.
MAX_LOOKBACK = 365


def series_history(panel: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    """Per-series summaries of everything known strictly before `origin`.

    Computed in log space, because that is the space the model is scored in:
    the mean of log1p sales is a far better predictor of log1p sales than the
    log of the mean, on a series that is zero a third of the time.
    """
    past = panel[panel["date"] < origin]
    if past.empty:
        raise ValueError(f"no history before {origin!r}")
    past = past[past["date"] > origin - pd.Timedelta(days=MAX_LOOKBACK + 1)]

    past = past.assign(y=np.log1p(past["sales"].to_numpy(dtype=float)))
    past = past.assign(lag=(origin - past["date"]).dt.days)
    keys = ["store_nbr", "family"]
    out = past.groupby(keys, observed=True)[["y"]].mean().rename(
        columns={"y": "mean_365"})

    last = past["date"].max()
    for w in WINDOWS:
        window = past[past["date"] > last - pd.Timedelta(days=w)]
        agg = window.groupby(keys, observed=True)["y"].agg(["mean", "std", "max"])
        agg.columns = [f"mean_{w}", f"std_{w}", f"max_{w}"]
        out = out.join(agg)
        # Share of days with no sale: separates "quiet" from "dead".
        zero = window.assign(z=(window["sales"] == 0).astype(float))
        out[f"zero_rate_{w}"] = zero.groupby(keys, observed=True)["z"].mean()

    # Same weekday as the target, which is where most of the signal lives.
    for weeks in DOW_WEEKS:
        window = past[past["date"] > last - pd.Timedelta(days=7 * weeks)]
        dow = (window.assign(dayofweek=window["date"].dt.dayofweek)
                     .groupby(keys + ["dayofweek"], observed=True)["y"].mean()
                     .rename(f"dow_mean_{weeks}w"))
        out = out.join(dow.reset_index().pivot_table(
            index=keys, columns="dayofweek", values=f"dow_mean_{weeks}w"
        ).rename(columns=lambda c: f"dow{c}_mean_{weeks}w"))

    # Explicit lags, not just window means. A mean says what the level is; the
    # last seven days in order say what shape it is in — a series that sold
    # 0,0,0,40,40,40,40 and one that sold 20 every day have the same mean and
    # very different futures.
    lag_frame = past.pivot_table(index=keys, columns="lag", values="y",
                                 observed=True)
    for k in LAGS:
        out[f"lag_{k}"] = lag_frame[k] if k in lag_frame.columns else np.nan

    out["promo_mean_28"] = (
        past[past["date"] > last - pd.Timedelta(days=28)]
        .groupby(keys, observed=True)["onpromotion"].mean())

    # Recent slope: the last four weeks against the four before them.
    recent = past[past["date"] > last - pd.Timedelta(days=28)]
    prior = past[(past["date"] <= last - pd.Timedelta(days=28)) &
                 (past["date"] > last - pd.Timedelta(days=56))]
    out["trend_28"] = (recent.groupby(keys, observed=True)["y"].mean()
                       - prior.groupby(keys, observed=True)["y"].mean())

    # Cross-sectional context. A tree cannot derive "this family is having a
    # good month everywhere" or "this store is quiet" from the series' own
    # columns, and both carry signal a single series is too noisy to show.
    for w in (28, 112):
        window = past[past["date"] > last - pd.Timedelta(days=w)]
        fam = window.groupby("family", observed=True)["y"].mean()
        store = window.groupby("store_nbr", observed=True)["y"].mean()
        out[f"fam_mean_{w}"] = out.index.get_level_values("family").map(fam)
        out[f"store_mean_{w}"] = out.index.get_level_values("store_nbr").map(store)
        # Where this series sits against both, which is what actually matters.
        out[f"rel_fam_{w}"] = out[f"mean_{28 if w == 28 else 112}"] - out[f"fam_mean_{w}"]
        out[f"rel_store_{w}"] = out[f"mean_{28 if w == 28 else 112}"] - out[f"store_mean_{w}"]

    return out.fillna(0.0).reset_index()


def first_sale(panel: pd.DataFrame) -> pd.Series:
    """The day each series first sold anything.

    Many series are all zero for months at the start — the family had not been
    stocked yet. Those rows are not a forecasting problem, they are a fact
    about the store's range, and training on them teaches the model to predict
    zero for a series that is now perfectly healthy.
    """
    sold = panel[panel["sales"] > 0]
    return sold.groupby(["store_nbr", "family"], observed=True)["date"].min()


def dead_series(panel: pd.DataFrame, origin: pd.Timestamp,
                lookback: int = 60) -> set[tuple[int, str]]:
    """Series that sold nothing at all in the `lookback` days before the origin.

    Ninety-six of the 1,782 series are in this state at the end of training,
    and they cover 1,536 of the 28,512 rows to be predicted. Forecasting them
    at zero is not a modelling choice, it is the answer — and it keeps a
    gradient booster from spending capacity on rows whose truth is a constant.
    """
    past = panel[(panel["date"] < origin) &
                 (panel["date"] >= origin - pd.Timedelta(days=lookback))]
    total = past.groupby(["store_nbr", "family"], observed=True)["sales"].sum()
    return set(total[total == 0].index)
