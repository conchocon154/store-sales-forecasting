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
}

# What ships. Chosen on the *third* fold rather than the three-fold mean: the
# third fold is the sixteen days immediately before the test period and it is
# the only one asking the question the leaderboard asks. On that fold the blend
# scores 0.40892 against 0.41305 for the best single model and 0.42956 for the
# first submission — while on the three-fold mean it looks like a tie, which is
# how the mean misleads.
SUBMISSION: dict = dict(variants=[dict(), dict(residual=True),
                                  dict(half_life=180)])
