"""Tests for the things that would be wrong silently.

A forecast that leaks the future scores well in a backtest and badly on the
leaderboard, and nothing in the output looks unusual either way. Most of what
is pinned down here is that boundary.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from storesales import data as D          # noqa: E402
from storesales import evaluate as E      # noqa: E402
from storesales import features as F      # noqa: E402
from storesales.models import DowMean, SeasonalNaive, Zero  # noqa: E402


def panel(days=120, stores=(1, 2), families=("A", "B"), seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2017-01-01", periods=days, freq="D")
    idx = pd.MultiIndex.from_product([dates, stores, families],
                                     names=["date", "store_nbr", "family"])
    df = idx.to_frame(index=False)
    # A weekly shape plus a level per series, so the weekday baselines have
    # something real to find.
    level = {(s, f): 10 * (i + 1) for i, (s, f) in
             enumerate((s, f) for s in stores for f in families)}
    df["sales"] = [level[(s, f)] * (1.5 if d.dayofweek >= 5 else 1.0)
                   for d, s, f in zip(df["date"], df["store_nbr"], df["family"])]
    df["sales"] += rng.normal(0, 0.5, len(df))
    df["onpromotion"] = 0.0
    df["split"] = "train"
    df["id"] = np.arange(len(df))
    return df


# --------------------------------------------------------------------------
# the metric
# --------------------------------------------------------------------------

def test_rmsle_is_zero_for_a_perfect_forecast():
    y = np.array([0.0, 3.0, 1000.0])
    assert E.rmsle(y, y) == pytest.approx(0.0)


def test_rmsle_punishes_relative_error_not_absolute():
    """Which is the whole reason this repo works in log space: missing 10 by 5
    costs more than missing 10,000 by 5."""
    small = E.rmsle([10.0], [15.0])
    large = E.rmsle([10000.0], [10005.0])
    assert small > large


def test_rmsle_treats_a_negative_prediction_as_zero():
    assert np.isfinite(E.rmsle([0.0], [-5.0]))


# --------------------------------------------------------------------------
# the backtest split
# --------------------------------------------------------------------------

def test_folds_are_the_length_of_the_competition_horizon():
    fs = E.folds(pd.Timestamp("2017-08-15"), n=3)
    for start, end in fs:
        assert (end - start).days + 1 == E.HORIZON


def test_folds_do_not_overlap_and_run_forwards():
    fs = E.folds(pd.Timestamp("2017-08-15"), n=3)
    assert fs[-1][1] == pd.Timestamp("2017-08-15")
    for (s1, e1), (s2, _) in zip(fs, fs[1:]):
        assert e1 < s2, "a fold may not contain a later fold's history"


# --------------------------------------------------------------------------
# leakage, which is the failure that looks like success
# --------------------------------------------------------------------------

def test_history_features_never_see_the_origin_or_later():
    p = panel()
    origin = pd.Timestamp("2017-03-01")
    tampered = p.copy()
    future = tampered["date"] >= origin
    tampered.loc[future, "sales"] = 1e6      # obvious poison, after the origin

    before = F.series_history(p, origin)
    after = F.series_history(tampered, origin)
    pd.testing.assert_frame_equal(before, after)


def test_history_needs_something_to_look_back_at():
    with pytest.raises(ValueError):
        F.series_history(panel(), pd.Timestamp("2016-01-01"))


def test_dead_series_are_found_only_from_the_lookback_window():
    p = panel()
    origin = pd.Timestamp("2017-04-01")
    p.loc[(p["store_nbr"] == 1) & (p["family"] == "A") &
          (p["date"] >= origin - pd.Timedelta(days=60)), "sales"] = 0.0
    assert (1, "A") in F.dead_series(p, origin)
    assert (2, "B") not in F.dead_series(p, origin)


def test_a_series_that_only_died_recently_is_not_called_dead():
    """Sixty days is the window; a fortnight of zeros is a quiet spell."""
    p = panel()
    origin = pd.Timestamp("2017-04-01")
    p.loc[(p["store_nbr"] == 1) & (p["family"] == "A") &
          (p["date"] >= origin - pd.Timedelta(days=14)), "sales"] = 0.0
    assert (1, "A") not in F.dead_series(p, origin)


# --------------------------------------------------------------------------
# the panel
# --------------------------------------------------------------------------

def test_calendar_marks_both_paydays():
    cal = F.calendar(pd.Series(pd.to_datetime(
        ["2017-01-14", "2017-01-15", "2017-01-31", "2017-02-28"])))
    assert cal["is_payday"].tolist() == [0, 1, 1, 1]


def test_holiday_flags_ignore_a_transferred_holiday_on_its_original_date():
    """`transferred` is True on the day that was *worked*; a separate Transfer
    row carries the day people actually took off."""
    hol = pd.DataFrame({
        "date": pd.to_datetime(["2017-01-02", "2017-01-06"]),
        "type": ["Holiday", "Transfer"],
        "locale": ["National", "National"],
        "locale_name": ["Ecuador", "Ecuador"],
        "description": ["x", "x"],
        "transferred": [True, False],
    })
    stores = pd.DataFrame({"store_nbr": [1], "city": ["Quito"],
                           "state": ["Pichincha"], "type": ["D"], "cluster": [1]})
    idx = pd.date_range("2017-01-01", "2017-01-07", freq="D")
    flags = D.holiday_flags(hol, stores, idx).set_index("date")
    assert flags.loc["2017-01-02", "is_holiday"] == 0
    assert flags.loc["2017-01-06", "is_holiday"] == 1


def test_a_local_holiday_closes_only_its_own_city():
    hol = pd.DataFrame({
        "date": pd.to_datetime(["2017-01-03"]), "type": ["Holiday"],
        "locale": ["Local"], "locale_name": ["Quito"],
        "description": ["x"], "transferred": [False]})
    stores = pd.DataFrame({"store_nbr": [1, 2], "city": ["Quito", "Guayaquil"],
                           "state": ["Pichincha", "Guayas"], "type": ["D", "D"],
                           "cluster": [1, 1]})
    flags = D.holiday_flags(hol, stores,
                            pd.date_range("2017-01-01", "2017-01-05")) \
             .set_index(["store_nbr", "date"])
    assert flags.loc[(1, pd.Timestamp("2017-01-03")), "is_holiday"] == 1
    assert flags.loc[(2, pd.Timestamp("2017-01-03")), "is_holiday"] == 0


def test_oil_is_filled_across_the_weekends_it_has_no_quote_for():
    oil = pd.DataFrame({"date": pd.to_datetime(["2017-01-02", "2017-01-06"]),
                        "dcoilwtico": [50.0, 55.0]})
    s = D.oil_series(oil, pd.date_range("2017-01-01", "2017-01-08"))
    assert s.notna().all(), "a missing quote is a closed market, not a gap"
    assert s.loc["2017-01-01"] == 50.0        # filled backwards
    assert s.loc["2017-01-04"] == 50.0        # carried forwards


# --------------------------------------------------------------------------
# the baselines
# --------------------------------------------------------------------------

def test_the_weekday_baseline_beats_carrying_last_week_forward():
    p = panel(days=200)
    origin = pd.Timestamp("2017-06-01")
    holdout = p[(p["date"] >= origin) &
                (p["date"] < origin + pd.Timedelta(days=16))]
    scores = {}
    for m in (Zero(), SeasonalNaive(), DowMean(8)):
        m.fit(p[p["date"] < origin], origin)
        scores[m.name] = E.rmsle(holdout["sales"], np.nan_to_num(m.predict(holdout)))
    assert scores["dow_mean_8w"] < scores["seasonal_naive"] < scores["zero"]


def test_baselines_predict_one_number_per_row_and_none_of_them_negative():
    p = panel()
    origin = pd.Timestamp("2017-04-01")
    holdout = p[p["date"] >= origin]
    for m in (Zero(), SeasonalNaive(), DowMean(4)):
        m.fit(p[p["date"] < origin], origin)
        pred = np.nan_to_num(m.predict(holdout))
        assert pred.shape == (len(holdout),)
        assert (pred >= 0).all()
