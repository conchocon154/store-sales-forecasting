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

1. `scripts/experiment.py --status` — read the table, pick the best variant.
2. Set that variant's config as the default in `DirectGBM.__init__`, then
   `.venv/bin/python scripts/submit.py` and submit to Kaggle.
3. Update the results table in `README.md` and push.

## The target, which is the thing worth being strict about

| | RMSLE | rank of 642 |
|---|---:|---:|
| leaderboard best | 0.37294 | 1 |
| top 1% | 0.37715 | 7 |
| top 5% | 0.38369 | 33 |
| **submitted so far** | **0.40713** | **92** |

Local backtest and public leaderboard track each other closely — local 0.39752
came back as 0.40713 — so a local mean is worth trusting to about +0.010.

**0.40713 is not the goal.** The gap to the top 5% is 0.024 RMSLE, and the
levers not yet pulled are listed below.

## Levers not yet pulled

- **Blending.** Several variants averaged in log space, which almost always
  beats the best single one. Nothing here has been blended yet.
- **Per-family models.** 33 families with very different dynamics share one
  model; the large families have enough rows to support their own.
- **Two-stage zero handling.** A third of all rows are zero. Classify
  sells/doesn't, then regress on the rows that sell.
- **Lag features at the exact horizon.** The design uses windows ending at the
  origin; explicit lags of 16, 17, 21, 28 days would give the trees the shape
  as well as the level.
- **Transactions.** The file is unused. It only exists up to the origin, so it
  can only enter as a store-level history feature, but store footfall is real
  signal about the level.
