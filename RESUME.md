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

The queue drained — fifteen variants measured, `reports/experiments.csv` has
them all. `residual_wide` (residual target, 78 origins) had the best three-fold
mean at 0.38991, was shipped, and **came back 0.50065 on the public
leaderboard** — 0.09 worse, where every single-model gap before was ~0.010. It
also had the worst third fold of the top three variants (0.41073 against 0.40892
for the shipped blend). Two lessons, both now baked into job #1 above:

* **The three-fold mean is the wrong selection criterion.** Folds 1 and 2 end
  in early July and reward exactly the tricks (`residual`, more origins) that
  fail an August test set. Select on fold 3.
* **On fold 3, nothing in the queue beats the shipped three-way blend (0.40892).**
  So the blend is still what ships, and the leaderboard still reads 0.40695.

Recency weighting (`half_life=180`) is the one thing the sweep surfaced that
helps fold 3 specifically: `lags` 0.42002 → 0.41123, `season` 0.41305 → 0.41123.
The shipped blend already has one recency-weighted component; a blend tilted
harder toward recency and dropping the fragile `residual` component is the
cheapest thing left to try. Those variants are queued in `configs.py`.

## The open question, which comes before any new feature

Local gains measured on the mean are anti-correlated with the leaderboard once
the target leaves the plateau. **Do not add features until the local/public gap
is understood.**

The hypothesis: across a sixteen-day fold the error climbs from ~0.40 on the
early days to ~0.46 on the late ones (see `scripts/diagnose.py`), so if the
public leaderboard covers the earlier part of the test window it is scoring the
stretch where every model agrees. Cheap test — score a fold on its first eight
and last eight days separately, resubmit one existing model, see which half
tracks the leaderboard.

## Levers not yet pulled

- **Recency-tilted blends without the residual component.** Queued now
  (`blend_recency`, `blend_recency4`, `hl90`, `hl120`). Config-only, low risk,
  aimed at fold 3.
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
