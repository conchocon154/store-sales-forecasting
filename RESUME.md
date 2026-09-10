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
| **submitted so far** | **0.40695** | **~92** |

**0.40695 is not the goal.** The gap to the top 5% is 0.023 RMSLE.

## The open question, which comes before any new feature

Two submissions scored 0.40713 and 0.40695 — 0.0002 apart — while differing by
0.021 on the local fold that sits immediately before the test window. Local
gains are not reaching the leaderboard. **Do not add features until this is
understood**; anything measured locally is currently unfalsifiable.

The hypothesis worth testing first: across a sixteen-day fold the error climbs
from about 0.40 on the early days to 0.46 on the late ones, so if the public
leaderboard covers the earlier part of the test window it is scoring the stretch
where every model agrees. The test is cheap — score a fold on its first eight
days and its last eight days separately, resubmit one of the two existing
models, and see which half tracks the leaderboard.

## Levers not yet pulled
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
