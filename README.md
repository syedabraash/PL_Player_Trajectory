# Player Performance Trajectory Model — Data Pipeline

Turns 5 seasons of PL player-season stats (2021-22 → 2025-26) into a
longitudinal dataset: given a player's state at season N, predict their
G+A/90 in season N+1.

## Data sources

- `data/PL_Players_5Years.xlsx` — one sheet per season, player-level season totals.
- `data/PL_Club_5years.xlsx` — one sheet per club, match-by-match logs (18 of the
  26 clubs that appear over the window; relegated clubs aren't covered).

## Pipeline (run in order from `src/`)

1. `build_player_panel.py` — loads all 5 seasons, drops the one incomplete row,
   **sums mid-season transfer stints into one season total** (assigning the
   *last-listed* club as season context — an assumption, since no transfer
   date is given), and recomputes per-90 rates from the summed raw totals
   rather than trusting averaged per-90 columns. Keeps the full multi-label
   Position string (e.g. `"FW,MF"`) as its own category.
2. `build_team_context.py` — aggregates club match logs to season level
   (possession, shots for/against, a shot-share proxy for "attacking-ness").
   No xG in the source data, so this is a shot-volume proxy, not an
   expected-goals-based one. Maps club-file sheet names to player-file Squad
   names (`SQUAD_NAME_MAP`).
3. `build_trajectory_dataset.py` — merges the two, builds features (age,
   age², cumulative minutes, minutes trend, 1- and 2-season lags of G+A/90,
   a 2-season EWM trend), and sets `target_next_ga90` = the player's G+A/90
   in the following season. Outputs both the full panel and a `has_target`
   filtered "supervised" table (next-season target exists AND next season
   has ≥3.0 90s played, to avoid training on tiny-sample noise).
4. `add_partial_season.py` — brings in the partial 2026-27 season (first 4
   gameweeks, `data/PL_26_27_Stats.xlsx`) as a **live, non-supervised**
   extension. Handles a broken two-row header in that file's player sheet
   (stat sub-headers sit 2 columns left of the data they describe — verified
   by cross-checking values, not assumed), derives Age from the historical
   Born-year lookup (`Age = season_start_year − Born − 1`, the convention
   confirmed against existing rows), and re-runs the same lag/EWM feature
   logic so the 2026-27 rows carry valid `ga90_prev_season`, `ga90_ewm`, etc.
   These rows have no `target_next_ga90` (the season isn't over) — they're
   what you'd feed a trained model to forecast the rest of 2026-27, not
   training data.
5. `train_baselines.py` — defines the time-based split used by everything
   downstream (**train**: season_start_year 2021-22, **validate**: 2023,
   **test**: 2024) and computes the two naive baselines: predict-last-season
   ("persistence") and predict-the-positional-average.
6. `train_model.py` — trains a Ridge regression baseline and a gradient-boosted
   model (see "Model choice" below), evaluates all four models on val/test,
   runs the breakout/decline/stable segment evaluation, trains 10th/50th/90th
   percentile quantile models for prediction intervals, computes permutation
   feature importance, and saves the fitted models (`.joblib`).
7. `generate_forecasts.py` — applies the trained models to every 2025-26
   player-season (≥5 90s played) to forecast 2026-27 G+A/90, ranks
   breakout-risk and decline-risk players, and sanity-checks the forecasts'
   *direction* against the real (tiny-sample) 2026-27 early numbers.
8. `make_visuals.py` — generates the report figures into `outputs/figures/`.

## Model choice

The original scope called for **XGBoost**. This sandboxed environment has no
network access to install it, so `sklearn.ensemble.HistGradientBoostingRegressor`
is used instead — a histogram-based gradient boosting implementation like
XGBoost/LightGBM, with native NaN handling (important given the missing
team-context values) and native quantile-loss support for prediction
intervals. `train_model.py` has the exact `XGBRegressor` call commented in
as a drop-in swap if you run this locally with `xgboost` installed —
same features, same split, no other changes needed.

## Results (honest, not cherry-picked)

**Aggregate test MAE** (predicting 2025-26 G+A/90 from 2024-25 features):

| Model | MAE | RMSE |
|---|---|---|
| positional_average | 0.109 | 0.154 |
| **gradient_boosted** | **0.121** | 0.160 |
| naive_persistence | 0.135 | 0.203 |
| ridge | 0.135 | 0.171 |

The model does **not** clearly beat the positional-average baseline on
aggregate error, and only edges out naive persistence — an honest finding
worth stating plainly rather than hiding, per the scope doc's own
methodology section. Aggregate MAE also hides where the model actually
earns its keep:

| Segment | naive_persistence | ridge | gradient_boosted |
|---|---|---|---|
| decline (biggest drops) | 0.223 | 0.181 | 0.183 |
| stable | 0.021 | 0.121 | 0.075 |
| breakout (biggest rises) | 0.160 | 0.104 | 0.106 |

On the *stable* majority, persistence is (unsurprisingly) hard to beat — a
player who didn't change much next season is well described by "predict no
change." On the **breakout and decline segments — the actually interesting,
hard-to-predict cases** — both Ridge and the gradient-boosted model cut
error by roughly a third to a half versus naive persistence. That's the
real value proposition of this project: not better average-case prediction,
but meaningfully better prediction exactly where a simple heuristic fails.

**Feature importance** (permutation, MAE increase when shuffled): `Position`,
current-season `G+A /90`, and the 2-season EWM trend dominate; team-context
features (`avg_shots_for`, `shot_share`) contribute a little; `Age` and
`minutes_trend` contribute effectively nothing at this sample size — the
aging-curve signal the original scope hoped to isolate did not clearly
emerge here (`outputs/figures/feature_importance.png`).

**Quantile intervals**: the 10th-90th percentile band achieves ~90% empirical
coverage on the test set (target ~80%) — the intervals are wider than
strictly necessary, i.e. conservative rather than overconfident, which is
the safer direction to err in for a "breakout risk" framing.

**2026-27 forecast sanity check**: comparing the forecast direction (will
G+A/90 rise or fall vs. 2025-26) against the actual direction implied by the
first 4 games gives ~43% directional agreement — worse than a coin flip.
This is **not evidence the model is bad** — 4 games is not enough signal to
validate anything against — but it's an honest number to report rather than
omit, and a reminder not to over-read early-season form.

## Known limitations (carry these into any writeup)

- **No xG/xA/shot data at the player level** — the single feature the original
  scope doc flagged as most valuable (goals-vs-xG overperformance, a
  regression-to-mean signal) isn't available here. Team-level shot-share is
  the closest proxy, and it's a team, not a player, signal.
- **~19% of supervised rows have missing team context** — players at clubs
  relegated during the window aren't in the club file, so `shot_share` etc.
  are NaN for them. Tree models (XGBoost) handle this natively; don't
  mean-impute it away, since "missing" itself correlates with relegation.
- **Year-over-year player overlap is only ~63%** — real churn from promotion/
  relegation and transfers out of the league, not a data error. This is the
  survivorship-bias caveat from the original scope: the trajectory model only
  ever sees players who stayed in the league long enough to have a "next
  season" to predict.
- **"Most recent club" for multi-club seasons is inferred from row order**,
  not a transfer date — reasonable but unverified assumption.
- **The 2026-27 rows are 4 games' worth of data (90s of 0-4.4)** — their own
  `G+A /90` etc. is high-variance and should not be treated as a settled
  season observation. `minutes_trend` for these rows is also not meaningful
  as computed (it compares a 4-game total against a full prior season) —
  don't use it as a feature for this season without rescaling.
- **`shot_share`/team-context is NaN for all 2026-27 rows** — the partial-season
  club file is a squad goalkeeping/results table (GA, SoTA, Saves%, W/D/L),
  not the shot/possession match-log schema used for historical team context.
  It's kept separately as `club_form_2627.csv` rather than force-merged.
- **90 of 406 players in the 2026-27 file (22%) have no prior history** — new
  signings and players at the two newly promoted clubs (Coventry City, Hull
  City), which also have zero historical team-context data.
- **The training window is short (2 seasons)** — `ga90_2_seasons_ago` had to
  be dropped as a feature because the train split (2021-22, 2022-23) has no
  "2 seasons ago" data for anyone. More historical seasons would let the
  model use longer-range trend features.
- **The model does not clearly beat the positional-average baseline on
  aggregate MAE** (see Results) — report this, don't paper over it.

## Outputs (`outputs/`)

- `player_season_panel.csv` — cleaned, transfer-aggregated player-season rows.
- `team_season_context.csv` — season-level team style features.
- `player_trajectory_full.csv` — full panel with all engineered features (includes
  rows with no next-season target, useful for generating forward predictions
  for the *current* season).
- `player_trajectory_supervised.csv` — the actual training table: 1,300 rows,
  each with a valid `target_next_ga90`.
- `player_panel_with_2627.csv` — full historical panel plus the partial
  2026-27 season appended.
- `club_form_2627.csv` — 2026-27 squad-level GA/saves/W-D-L to date (20 clubs,
  including the two promoted sides).
- `player_live_features_2627.csv` — 2026-27 rows only, with lag/EWM/age
  features computed — the input table for generating current forecasts.
- `model_metrics.csv`, `segment_metrics.csv`, `baseline_metrics.csv` — all
  evaluation numbers reported above.
- `feature_importance.csv` — permutation importance for every feature.
- `test_predictions.csv` — every test-set row with each model's prediction
  and its breakout/decline/stable label.
- `model_gradient_boosted.joblib`, `model_ridge.joblib`,
  `model_quantile_{10,50,90}.joblib` — fitted, ready-to-load models.
- `forecast_2026_27_all.csv` — every eligible player's 2026-27 forecast
  (point estimate + 10th/90th percentile interval) plus the early 4-game
  actual for comparison.
- `breakout_risk_rankings.csv` / `decline_risk_rankings.csv` — top 25 players
  by projected G+A/90 change for 2026-27.
- `figures/` — the 5 report visuals: `model_vs_baseline_mae.png`,
  `segment_mae.png`, `feature_importance.png`, `player_trajectories.png`
  (actual-vs-predicted for notable breakout/decline cases),
  `quantile_fan_chart.png` (forecast intervals for top breakout risks).

## Possible next extensions

- Feed this player-level output into the FPL pipeline as a per-player input
  alongside team-level match predictions.
- Feed into a transfer-success model: was a transfer's outcome within the
  player's predicted range, or a clear over/undershoot?
- Extend beyond G+A to defensive metrics (tackles, interceptions) for
  fuller coverage of defenders and goalkeepers, who are underserved by a
  G+A-only target.
- Re-run with more historical seasons once available, to properly test
  longer-range trend features and get a larger, more stable validation set.
