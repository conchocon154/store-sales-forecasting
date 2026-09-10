#!/usr/bin/env python3
"""Emit the figures, and the tables they are drawn from.

Two outputs, in that order. `reports/tables/*.csv` holds the numbers;
`reports/figures/*.png` draws them, light and dark. Nothing is transcribed by
hand, so any point on any chart can be traced back to the run that produced it
— and the portfolio's own chart builder reads the same tables rather than a
second copy of the arithmetic.

    python3 tools/render_charts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from storesales import data as D         # noqa: E402

TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"

# One palette per theme. GitHub serves the dark file to a reader in dark mode
# through <picture>, so both are rendered rather than compromising on a grey
# that looks washed out against one background and muddy against the other.
THEMES = {
    "light": dict(bg="#ffffff", ink="#1a1d21", dim="#5b6470", grid="#e4e7eb",
                  accent="#0f766e", warn="#c2410c", mute="#9aa3ad"),
    "dark":  dict(bg="#0d1117", ink="#e6edf3", dim="#9aa5b1", grid="#21262d",
                  accent="#2dd4bf", warn="#fb923c", mute="#6e7781"),
}

FAMILY_SPIKE = "SCHOOL AND OFFICE SUPPLIES"
FAMILY_STEADY = "GROCERY I"


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------

def build_tables() -> dict[str, pd.DataFrame]:
    frames = D.load()
    train = frames.train

    daily = (train.groupby("date", as_index=False)["sales"].sum()
                  .rename(columns={"sales": "total_sales"}))
    # A 28-day mean alongside the raw series: the raw line carries the New Year
    # closures and the spikes, the mean carries the trend they hide.
    daily["rolling_28"] = daily["total_sales"].rolling(28, min_periods=7).mean()

    dow = (train.assign(dow=train["date"].dt.dayofweek)
                .groupby("dow", as_index=False)["sales"].mean()
                .rename(columns={"sales": "mean_sales"}))
    dow["weekday"] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    # The August spike, indexed so two families of very different size can be
    # read on one axis: 1.0 is each family's own average day.
    pair = train[train["family"].isin([FAMILY_SPIKE, FAMILY_STEADY])]
    by_day = (pair.groupby(["family", "date"], as_index=False)["sales"].sum())
    by_day["doy"] = by_day["date"].dt.dayofyear
    season = (by_day.groupby(["family", "doy"], as_index=False)["sales"].mean())
    season["index"] = season.groupby("family")["sales"].transform(
        lambda s: s / s.mean())
    season = season.pivot(index="doy", columns="family", values="index")
    season = season.rolling(7, center=True, min_periods=3).mean().reset_index()

    tables = {"daily_sales": daily, "weekday_profile": dow,
              "seasonal_index": season}

    backtest = ROOT / "reports" / "backtest.csv"
    if backtest.exists():
        tables["model_ladder"] = pd.read_csv(backtest)

    fold = ROOT / "reports" / "diagnose_fold2.csv"
    if fold.exists():
        d = pd.read_csv(fold, parse_dates=["date"])
        d["horizon"] = (d["date"] - d["date"].min()).dt.days + 1
        horizon = d.groupby("horizon", as_index=False)["err2"].mean()
        horizon["rmsle"] = np.sqrt(horizon["err2"])
        tables["error_by_horizon"] = horizon[["horizon", "rmsle"]]

        total = d["err2"].sum()
        fam = d.groupby("family", as_index=False)["err2"].sum()
        fam["share_pct"] = 100 * fam["err2"] / total
        tables["error_by_family"] = fam.sort_values(
            "share_pct", ascending=False)[["family", "share_pct"]]

    TABLES.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(TABLES / f"{name}.csv", index=False)
    return tables


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def style(ax, c, title, xlabel="", ylabel=""):
    ax.set_title(title, color=c["ink"], fontsize=11, loc="left", pad=10)
    ax.set_xlabel(xlabel, color=c["dim"], fontsize=9)
    ax.set_ylabel(ylabel, color=c["dim"], fontsize=9)
    ax.tick_params(colors=c["dim"], labelsize=8.5)
    for side, spine in ax.spines.items():
        spine.set_visible(side in ("left", "bottom"))
        spine.set_color(c["grid"])
    ax.grid(True, color=c["grid"], linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)


def save(fig, name: str, theme: str, c) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}-{theme}.png", dpi=170,
                facecolor=c["bg"], bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def fig_daily(t, theme, c):
    d = t["daily_sales"]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    fig.patch.set_facecolor(c["bg"]); ax.set_facecolor(c["bg"])
    ax.plot(d["date"], d["total_sales"], color=c["mute"], linewidth=0.6,
            alpha=0.75, label="daily total")
    ax.plot(d["date"], d["rolling_28"], color=c["accent"], linewidth=2.0,
            label="28-day mean")
    # Every 1 January the shops are shut and the whole country sells nothing.
    zeros = d[d["total_sales"] == 0]
    ax.scatter(zeros["date"], zeros["total_sales"], s=26, color=c["warn"],
               zorder=3, label="closed (1 Jan)")
    style(ax, c, "Total daily sales across 54 stores, 2013–2017",
          ylabel="sales")
    leg = ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    for txt in leg.get_texts():
        txt.set_color(c["dim"])
    save(fig, "daily-sales", theme, c)


def fig_weekday(t, theme, c):
    d = t["weekday_profile"]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    fig.patch.set_facecolor(c["bg"]); ax.set_facecolor(c["bg"])
    colours = [c["accent"] if i < 5 else c["warn"] for i in range(7)]
    ax.bar(d["weekday"], d["mean_sales"], color=colours, width=0.68)
    style(ax, c, "Mean sales by weekday", ylabel="sales per series-day")
    save(fig, "weekday-profile", theme, c)


def fig_season(t, theme, c):
    d = t["seasonal_index"]
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    fig.patch.set_facecolor(c["bg"]); ax.set_facecolor(c["bg"])
    ax.plot(d["doy"], d[FAMILY_SPIKE], color=c["warn"], linewidth=2.0,
            label="school & office supplies")
    ax.plot(d["doy"], d[FAMILY_STEADY], color=c["accent"], linewidth=2.0,
            label="grocery I")
    ax.axhline(1.0, color=c["grid"], linewidth=1.0)
    # The forecast window, which is what makes this family expensive.
    ax.axvspan(228, 243, color=c["warn"], alpha=0.12)
    # Low, because the spike itself owns the top of this axis.
    ax.annotate("test window", xy=(235, ax.get_ylim()[1] * 0.055),
                color=c["warn"], fontsize=8, ha="center")
    style(ax, c, "Annual shape, indexed to each family's own average day",
          xlabel="day of year", ylabel="× average day")
    leg = ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    for txt in leg.get_texts():
        txt.set_color(c["dim"])
    save(fig, "seasonal-index", theme, c)


def fig_ladder(t, theme, c):
    if "model_ladder" not in t:
        return
    d = t["model_ladder"].sort_values("mean", ascending=False)
    d = d[d["model"] != "zero"]        # 4.42 flattens everything else
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    fig.patch.set_facecolor(c["bg"]); ax.set_facecolor(c["bg"])
    colours = [c["accent"] if m == d["mean"].min() else c["mute"]
               for m in d["mean"]]
    ax.barh(d["model"], d["mean"], color=colours, height=0.62)
    for y, v in enumerate(d["mean"]):
        ax.text(v + 0.006, y, f"{v:.4f}", va="center", color=c["dim"],
                fontsize=8.5)
    ax.set_xlim(0, d["mean"].max() * 1.18)
    style(ax, c, "RMSLE, mean of three sixteen-day folds — lower is better",
          xlabel="RMSLE")
    save(fig, "model-ladder", theme, c)


def fig_horizon(t, theme, c):
    if "error_by_horizon" not in t:
        return
    d = t["error_by_horizon"]
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    fig.patch.set_facecolor(c["bg"]); ax.set_facecolor(c["bg"])
    ax.plot(d["horizon"], d["rmsle"], color=c["warn"], linewidth=2.0,
            marker="o", markersize=3.5, label="RMSLE")
    # The zigzag is the weekday; the line under it is the claim. Drawn rather
    # than asserted, because "error climbs" is the whole argument for why the
    # public leaderboard may be measuring the easy half of the window.
    slope, intercept = np.polyfit(d["horizon"], d["rmsle"], 1)
    ax.plot(d["horizon"], intercept + slope * d["horizon"], color=c["dim"],
            linewidth=1.3, linestyle="--",
            label=f"+{slope * 15:.3f} over the window")
    style(ax, c, "Error climbs across the sixteen-day horizon",
          xlabel="days ahead", ylabel="RMSLE")
    leg = ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    for txt in leg.get_texts():
        txt.set_color(c["dim"])
    save(fig, "error-by-horizon", theme, c)


def fig_family(t, theme, c):
    if "error_by_family" not in t:
        return
    d = t["error_by_family"].head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    fig.patch.set_facecolor(c["bg"]); ax.set_facecolor(c["bg"])
    colours = [c["warn"] if f == FAMILY_SPIKE else c["accent"]
               for f in d["family"]]
    ax.barh(d["family"].str.title(), d["share_pct"], color=colours, height=0.66)
    style(ax, c, "Share of squared error, worst ten of thirty-three families",
          xlabel="% of total squared error")
    save(fig, "error-by-family", theme, c)


FIGS = (fig_daily, fig_weekday, fig_season, fig_ladder, fig_horizon, fig_family)


def main() -> int:
    tables = build_tables()
    print(f"wrote {len(tables)} tables to {TABLES.relative_to(ROOT)}")
    for theme, c in THEMES.items():
        for fn in FIGS:
            fn(tables, theme, c)
    made = sorted(p.name for p in FIGURES.glob("*.png"))
    print(f"wrote {len(made)} figures to {FIGURES.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
