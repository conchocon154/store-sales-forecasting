"""Loading and joining the seven files the competition ships.

The panel is dense on purpose: 54 stores x 33 families x every date, with the
gaps filled at zero. Every calendar-aware model below assumes a row exists for
each series on each day, and the raw train file does not guarantee that.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data"

# Ecuador's public sector is paid on the 15th and the last day of the month,
# and the competition's own data description calls this out as a driver.
PAYDAYS = (15,)

# The Manabi earthquake, 16 April 2016. Relief buying distorts the weeks after
# it, so the model gets to see the window rather than learning it as noise.
QUAKE = pd.Timestamp("2016-04-16")
QUAKE_WEEKS = 8


@dataclass(frozen=True)
class Frames:
    train: pd.DataFrame
    test: pd.DataFrame
    stores: pd.DataFrame
    oil: pd.DataFrame
    holidays: pd.DataFrame
    transactions: pd.DataFrame


def load(data_dir: Path | str = DATA) -> Frames:
    d = Path(data_dir)
    dates = ["date"]
    return Frames(
        train=pd.read_csv(d / "train.csv", parse_dates=dates),
        test=pd.read_csv(d / "test.csv", parse_dates=dates),
        stores=pd.read_csv(d / "stores.csv"),
        oil=pd.read_csv(d / "oil.csv", parse_dates=dates),
        holidays=pd.read_csv(d / "holidays_events.csv", parse_dates=dates),
        transactions=pd.read_csv(d / "transactions.csv", parse_dates=dates),
    )


def build_panel(frames: Frames) -> pd.DataFrame:
    """One row per (store, family, date) across train and test, sales NaN on test."""
    train = frames.train.assign(split="train")
    test = frames.test.assign(sales=np.nan, split="test")
    cols = ["id", "date", "store_nbr", "family", "sales", "onpromotion", "split"]
    panel = pd.concat([train[cols], test[cols]], ignore_index=True)

    full = _dense_index(panel)
    panel = full.merge(panel, on=["date", "store_nbr", "family"], how="left")
    panel["onpromotion"] = panel["onpromotion"].fillna(0.0)
    panel["split"] = panel["split"].fillna("train")
    panel.loc[panel["split"] == "train", "sales"] = \
        panel.loc[panel["split"] == "train", "sales"].fillna(0.0)

    panel = panel.merge(frames.stores, on="store_nbr", how="left")
    return panel.sort_values(["store_nbr", "family", "date"], ignore_index=True)


def _dense_index(panel: pd.DataFrame) -> pd.DataFrame:
    dates = pd.date_range(panel["date"].min(), panel["date"].max(), freq="D")
    stores = np.sort(panel["store_nbr"].unique())
    families = np.sort(panel["family"].unique())
    idx = pd.MultiIndex.from_product([dates, stores, families],
                                     names=["date", "store_nbr", "family"])
    return idx.to_frame(index=False)


def oil_series(oil: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """Daily WTI, forward-filled.

    The file has weekday quotes with gaps, and the first row is empty, so it is
    reindexed onto every calendar day and filled in both directions — a missing
    price is a market holiday, not a change in price.
    """
    s = (oil.set_index("date")["dcoilwtico"]
            .reindex(index)
            .ffill()
            .bfill())
    s.name = "oil"
    return s


def holiday_flags(holidays: pd.DataFrame, stores: pd.DataFrame,
                  index: pd.DatetimeIndex) -> pd.DataFrame:
    """A per-(date, store) view of whether the day is off.

    Three things the raw file makes easy to get wrong, all of them documented
    in the competition's data description:

      * `transferred` is True on the *original* date, which was worked, and a
        separate row of type `Transfer` carries the day people actually took.
      * `Work Day` is the opposite of a holiday: a Saturday worked to make up
        for a bridge.
      * `locale` scopes the holiday to the country, a region or a single city,
        so a Local holiday only closes the stores in that city.
    """
    h = holidays.copy()
    h = h[h["type"] != "Work Day"]
    h = h[~h["transferred"].astype(bool)]
    h = h[h["type"] != "Event"]

    national = set(h.loc[h["locale"] == "National", "date"])
    regional = set(zip(h.loc[h["locale"] == "Regional", "date"],
                       h.loc[h["locale"] == "Regional", "locale_name"]))
    local = set(zip(h.loc[h["locale"] == "Local", "date"],
                    h.loc[h["locale"] == "Local", "locale_name"]))
    workdays = set(holidays.loc[holidays["type"] == "Work Day", "date"])

    rows = []
    for _, store in stores.iterrows():
        flags = pd.DataFrame({"date": index})
        flags["store_nbr"] = store["store_nbr"]
        flags["is_national_holiday"] = flags["date"].isin(national).astype("int8")
        flags["is_regional_holiday"] = [
            (d, store["state"]) in regional for d in index]
        flags["is_local_holiday"] = [
            (d, store["city"]) in local for d in index]
        flags["is_work_day"] = flags["date"].isin(workdays).astype("int8")
        rows.append(flags)

    out = pd.concat(rows, ignore_index=True)
    for c in ("is_regional_holiday", "is_local_holiday"):
        out[c] = out[c].astype("int8")
    out["is_holiday"] = ((out[["is_national_holiday", "is_regional_holiday",
                               "is_local_holiday"]].max(axis=1) == 1)
                         & (out["is_work_day"] == 0)).astype("int8")
    return out
