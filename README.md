# Store Sales — Time Series Forecasting

[![CI](https://github.com/conchocon154/store-sales-forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/conchocon154/store-sales-forecasting/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Sixteen days of daily sales for **1,782 series** — 54 Corporación Favorita
stores in Ecuador × 33 product families — forecast from four and a half years
of history. Scored on RMSLE.

| Model | 2017-06-29 | 2017-07-15 | 2017-07-31 | **mean RMSLE** |
|---|---:|---:|---:|---:|
| **direct GBM** | 0.3746 | 0.3884 | 0.4296 | **0.3975** |
| weekday mean, 8 weeks | 0.4059 | 0.4170 | 0.5286 | 0.4505 |
| weekday mean, 4 weeks | 0.4175 | 0.4295 | 0.5398 | 0.4623 |
| weekday mean, 16 weeks | 0.4870 | 0.4649 | 0.5224 | 0.4914 |
| seasonal naive (last week) | 0.5176 | 0.5212 | 0.6170 | 0.5519 |
| last value carried forward | 0.6051 | 0.5881 | 0.6595 | 0.6175 |
| predict zero | 4.4202 | 4.4193 | 4.4195 | 4.4197 |

Lower is better. Every row is measured the same way, on the same folds, with
the same dead-series rule applied — so the gap between them is the modelling
and not the protocol.

## Three decisions that set the result

**Work in log space, throughout.** RMSLE is RMSE on `log1p`, so a model fitted
on raw sales optimises a loss the competition does not use. On series that run
from 0 to 124,717 that is not a rounding difference. Everything here — the
baselines' averages included — is computed on `log1p` and converted back once,
at the end.

**Forecast directly, not recursively.** On the day the forecast is made the
last known sale is yesterday, and day sixteen still has to be predicted. A
one-step model that feeds its own output back in compounds its error sixteen
times. Instead each training row is *(series, origin, horizon)*: the per-series
history is summarised strictly before the origin, the calendar features
describe the target day, and the model learns "sixteen days out" as its own
problem. Twenty origins, spaced a fortnight apart, give it every horizon many
times over.

**Answer the dead series with a rule.** Ninety-six of the 1,782 series sold
nothing at all in the last sixty days, and they account for 1,536 of the 28,512
rows to predict. Their forecast is zero — that is the answer, not an estimate —
so they are dropped from the fit and filled in afterwards. Leaving them in
gives a booster 5% of its rows whose truth is a constant.

## Validation

Three folds, each the **same sixteen days** the leaderboard asks for, ending at
the last day of training and stepping backwards without overlap.

```
… train …………………………………│ 06-29 → 07-14 │ 07-15 → 07-30 │ 07-31 → 08-15 │  test: 08-16 → 08-31
```

A random train/test split would ask an easier question — predict a Tuesday
given the Wednesday after it — and would rank these models differently. The
last fold is the hardest for every model, which is worth knowing before reading
too much into any single number.

## Features

Per series, computed strictly before the origin: mean, standard deviation, max
and zero-rate of `log1p` sales over 7 / 14 / 28 / 56 / 112 days; the mean for
the target's own weekday over 4 and 8 weeks; promotion rate; and a 28-day slope
against the 28 days before it.

Per target day: weekday, day, month, day-of-year, week, weekend, a linear
trend, WTI oil forward-filled across the days it has no quote, `onpromotion`
(which the competition gives for the test period), the horizon itself, and:

- **Payday.** Ecuador's public sector is paid on the 15th and the last day of
  the month. Both the flag and the distance to the nearer of the two.
- **Holidays, scoped properly.** `transferred` is `True` on the day that was
  *worked* — a separate `Transfer` row carries the day people actually took —
  and `Work Day` is the opposite of a holiday. `locale` scopes a holiday to the
  country, a region or one city, so a Local holiday closes only the stores in
  that city. There are tests for all three, because getting any of them
  backwards is invisible in the output.
- **The Manabí earthquake**, 16 April 2016, and the eight weeks of relief
  buying after it.

## Running it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
kaggle competitions download store-sales-time-series-forecasting -p data && unzip -o 'data/*.zip' -d data

.venv/bin/python scripts/backtest.py --folds 3     # the table above
.venv/bin/python scripts/submit.py                 # writes submission.csv
.venv/bin/python -m pytest tests/ -q               # 15 tests
```

The data is gitignored — it is Kaggle's to distribute, not this repo's.

## What the tests are for

A forecast that leaks the future scores well in backtest and badly on the
leaderboard, and the output looks identical either way. So the tests are aimed
at that boundary rather than at coverage: poison every value at and after the
origin and assert the history features come out byte-identical; assert the
folds neither overlap nor run backwards; assert a transferred holiday is not
treated as a day off on its original date; assert a Local holiday leaves the
other city's stores open.

## Reading I used

The public notebook with the most votes on this competition is
[Ekrem Bayar's comprehensive guide](https://www.kaggle.com/code/ekrembayar/store-sales-ts-forecasting-a-comprehensive-guide)
(2,932 votes), and the zero-forecasting rule above is his observation. Ryan
Holbrook's [Time Series course](https://www.kaggle.com/learn/time-series)
exercises are the highest-voted notebooks attached to the competition overall
(65,818 on the first) and are where the hybrid framing comes from — a
deterministic part for trend and seasonality, a learned part for the rest.

## Licence

MIT.
