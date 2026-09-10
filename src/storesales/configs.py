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

    # --- recency, targeting the third fold -------------------------------
    # The queue's one clear signal: recency weighting improves fold 3 (the only
    # fold shaped like the test set) — lags 0.42002 -> 0.41123 with half_life
    # 180. These push on that and drop the `residual` target, which shipped as
    # `residual_wide` and scored 0.50065. Single models are cheap; the blends
    # keep the diversity that beat every single model on fold 3.
    "hl120":            dict(half_life=120),
    "hl90":             dict(half_life=90),
    "blend_recency":    dict(variants=[dict(), dict(half_life=180),
                                       dict(half_life=90)]),
    "blend_recency4":   dict(variants=[dict(), dict(half_life=270),
                                       dict(half_life=180), dict(half_life=90)]),
}

# What ships: the three-way blend, the best score the leaderboard has actually
# seen (0.40695). Chosen on the *third* fold, not the three-fold mean: the third
# fold is the sixteen days immediately before the test period. `residual_wide`
# won the mean (0.38991 vs 0.39163) and was shipped next — it came back 0.50065,
# which is why the mean is not the selection criterion here.
SUBMISSION: dict = dict(variants=[dict(), dict(residual=True),
                                  dict(half_life=180)])
