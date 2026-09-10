# Where this is, and how to carry on

Written so that a session which stops mid-flight — out of context, out of quota,
closed laptop — loses nothing. Read this file first, then run the status
command; between them they say exactly what is done and what is not.

```bash
cd ~/store-sales
.venv/bin/python scripts/experiment.py --status     # the live answer
```

## The mechanism

The expensive part does not need an agent attached to it. `scripts/experiment.py`
works through a queue of named model variants in `src/storesales/configs.py`,
and **appends each result to `reports/experiments.csv` the moment a variant
finishes**. So:

* interrupting it costs at most the one variant in flight;
* starting it again skips every variant already measured;
* running it twice by accident is harmless.

`scripts/continue.sh` is the whole resume procedure — it prints what is
outstanding, drains the queue, and appends to `reports/run.log`. It can be run
by a scheduled task, by cron, or by hand:

```bash
./scripts/continue.sh
```

An agent picking this up again has three jobs and no need to re-derive anything:

1. `scripts/experiment.py --status` — read the table. **Pick the variant with
   the lowest *third fold*, not the lowest mean.** Picking by mean shipped
   `residual_wide` and it scored 0.50065 (see below).
2. Set that variant's config in `SUBMISSION` in `configs.py` (and, if it is a
   single model, `DirectGBM.__init__`), then `.venv/bin/python scripts/submit.py`
   and submit to Kaggle — only if it beats the shipped blend on the third fold.
   Nothing in the queue currently does.
3. Update the results table in `README.md` and push.

## The target, which is the thing worth being strict about

| | RMSLE | rank of 642 |
|---|---:|---:|
| leaderboard best | 0.37294 | 1 |
| top 1% | 0.37715 | 7 |
| top 5% | 0.38369 | 33 |
| **submitted so far** | **0.40695** | **~92** |

**0.40695 is not the goal.** The gap to the top 5% is 0.023 RMSLE.

## What the last run established

The queue drained — **nineteen variants measured**, `reports/experiments.csv`
has them all. Two were put in front of the leaderboard and both regressed:

| variant | why picked | local fold 3 | public |
|---|---|---:|---:|
| `residual_wide` | best three-fold mean (0.38991) | 0.41073 | **0.50065** |
| `blend_recency` | best fold 3 (0.40879), no residual | 0.40879 | **0.41139** |

Three things are now firm:

* **The three-fold mean is anti-correlated with the leaderboard.** Folds 1–2
  end in early July and reward `residual` and extra origins, which fail August.
* **A fold-3 edge of 0.0001–0.001 is noise.** `blend_recency` beat the shipped
  blend on fold 3 by 0.0001 and lost by 0.004 on the leaderboard.
* **Recency weighting helps the backtest and hurts the test set.** A 90-day
  half-life buries the year-ago seasonal signal that school-supplies season
  needs. `half_life` is off the table as a lever.

The shipped three-way blend (0.40695) is still the best the leaderboard has
seen. **Nothing measured against the current folds is worth submitting.**

## The open question, which now blocks everything

The three backtest folds all end 31 July and cannot see the August seasonal
ramp. Every variant tuned against them has stopped paying or actively backfired.
**Do not submit anything else selected on these folds.** The next move is one of:

1. **Understand the public/private split.** Hypothesis: the error climbs from
   ~0.40 on early horizons to ~0.46 on late ones (see `scripts/diagnose.py`
   grouped by date), so if the public leaderboard scores only the first stretch
   of the test window it is measuring where every model agrees — which would
   explain why nothing moves it. Cheap test: score a fold on its first eight and
   last eight days separately; the split is knowable from the competition's
   own timeline too.
2. **A lever robust by construction**, not selected on the folds — the
   two-stage zero model is the clearest (bounded fold-3 gain 0.42002 → 0.38602
   for a perfect classifier) but it needs real code and a way to validate that
   is not the July folds.

## Levers not yet pulled

Config-only variants are exhausted — the `QUEUE` in `configs.py` is drained and
nothing in it beat the shipped blend on the leaderboard. What is left needs
code, and none of it should be shipped on the strength of the July folds alone.

- **Two-stage zero handling.** A third of all rows are zero. Classify
  sells/doesn't, then regress on the rows that sell. A *perfect* zero classifier
  takes the August fold from 0.42002 to 0.38602 — the largest bounded prize
  left, and it needs real code in `models.py`, not a config entry.
- **Per-family models.** 33 families with very different dynamics share one
  model; the large families have enough rows to support their own.
- **Lag features at the exact horizon.** The design uses windows ending at the
  origin; explicit lags of 16, 17, 21, 28 days would give the trees the shape
  as well as the level.
- **Transactions.** The file is unused. It only exists up to the origin, so it
  can only enter as a store-level history feature, but store footfall is real
  signal about the level.
