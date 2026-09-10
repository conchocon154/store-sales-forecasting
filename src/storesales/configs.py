"""The experiment queue.

Each entry is a named variant of the model. `scripts/experiment.py` runs the
ones that have no result yet and appends what it finds to
`reports/experiments.csv`, so a session that stops half way loses nothing and a
session that resumes never repeats an experiment.
"""

from __future__ import annotations

# Ordered by expected value, because a run that is cut short should have spent
# its time on the experiments most likely to move the number.
QUEUE: dict[str, dict] = {
    # --- established -----------------------------------------------------
    "base_20origins":   dict(n_origins=20, origin_step=14, max_iter=500,
                             learning_rate=0.06),
    "wide_52origins":   dict(n_origins=52, origin_step=7),

    # --- the residual target ---------------------------------------------
    # Predict the gap from the weekday mean rather than the level itself. The
    # level is the easy part and it swamps the loss; subtracting it lets every
    # split in the tree be spent on the part that is actually hard.
    # Averaging what disagrees. Every variant lands within 0.003 of the others
    # while getting different rows wrong, which is the shape that blends well.
    "blend3":           dict(variants=[dict(), dict(residual=True),
                                       dict(half_life=180)]),
    "blend4":           dict(variants=[dict(), dict(residual=True),
                                       dict(half_life=180),
                                       dict(max_leaf_nodes=127,
                                            min_samples_leaf=20)]),
    "lags":             dict(),
    "season":           dict(),
    "season_recency":   dict(half_life=180),
    "lags_recency":     dict(half_life=180),
    "residual":         dict(residual=True),
    "residual_deep":    dict(residual=True, max_leaf_nodes=127,
                             min_samples_leaf=20),
    "residual_slow":    dict(residual=True, max_iter=1600, learning_rate=0.025),

    # --- recency ----------------------------------------------------------
    # The test window sits immediately after training, so an origin from last
    # month is worth more than one from last year.
    "residual_recency": dict(residual=True, half_life=180),
    "residual_recency_fast": dict(residual=True, half_life=90),

    # --- capacity ---------------------------------------------------------
    "residual_wide":    dict(residual=True, n_origins=78, origin_step=7),
    "residual_deep_slow": dict(residual=True, max_leaf_nodes=127,
                               min_samples_leaf=20, max_iter=1600,
                               learning_rate=0.025, half_life=180),

    # --- recency: measured, and a dead end ------------------------------
    # Recency weighting improves fold 3 in the backtest (lags 0.42002 -> 0.41123
    # at half_life 180) so these pushed on it and dropped the `residual` target.
    # `blend_recency` won fold 3 locally by 0.0001 and lost 0.004 on the public
    # leaderboard: a 90-day half-life buries the year-ago seasonal signal that
    # school-supplies season needs. Kept for the record; do not resubmit.
    "hl120":            dict(half_life=120),
    "hl90":             dict(half_life=90),
    "blend_recency":    dict(variants=[dict(), dict(half_life=180),
                                       dict(half_life=90)]),
    "blend_recency4":   dict(variants=[dict(), dict(half_life=270),
                                       dict(half_life=180), dict(half_life=90)]),
}

# What ships: the original three-way blend. It is still the best score the
# leaderboard has seen (0.40695), and two attempts to beat it both regressed:
#
#   variant         local fold 3   public   note
#   residual_wide   0.41073        0.50065  best local *mean*, worst top-3 fold 3
#   blend_recency   0.40879        0.41139  ~tied blend3 on fold 3, +0.004 public
#
# The pattern is now firm: at this magnitude a local fold-3 edge of 0.0001-0.001
# is noise, and recency weighting — which improves fold 3 — *hurts* the real
# August test set, because a 90-day half-life buries the year-ago seasonal
# signal that school-supplies season depends on. The three backtest folds all
# end 31 July; none of them can see the August ramp, so tuning against them has
# stopped paying. The next move is structural (understand the public/private
# split, or a lever robust by construction), not another variant.
SUBMISSION: dict = dict(variants=[dict(), dict(residual=True),
                                  dict(half_life=180)])
