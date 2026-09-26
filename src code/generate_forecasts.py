import joblib
import numpy as np
import pandas as pd

from train_model import NUMERIC_FEATURES, CATEGORICAL_FEATURES, BOOLEAN_FEATURES

FULL_PANEL_PATH = "C:\\Users\\syedt\\Desktop\\PL_Player_Trajectory\\outputs\\player_trajectory_full.csv"
PARTIAL_STATS_PATH = "C:\\Users\\syedt\\Desktop\\PL_Player_Trajectory\\outputs\\player_panel_with_2627.csv"

MIN_90S_FOR_FORECAST = 5.0  # don't forecast for players with a token 2025-26 involvement


def load_2025_26_rows() -> pd.DataFrame:
    df = pd.read_csv(FULL_PANEL_PATH)
    rows = df[(df["Season"] == "2025-26") & (df["90s"] >= MIN_90S_FOR_FORECAST)].copy()
    return rows


def prep_X(df: pd.DataFrame) -> pd.DataFrame:
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES].copy()
    X[BOOLEAN_FEATURES] = X[BOOLEAN_FEATURES].astype(int)
    return X


def load_early_2627_actuals() -> pd.DataFrame:
    df = pd.read_csv(PARTIAL_STATS_PATH)
    early = df[df["Season"] == "2026-27"][["Player", "90s", "G+A /90"]].copy()
    early.columns = ["Player", "early_2627_90s", "early_2627_ga90"]
    return early


if __name__ == "__main__":
    rows = load_2025_26_rows()
    print(f"Forecasting for {len(rows)} players (2025-26 season, >= {MIN_90S_FOR_FORECAST} 90s)")

    X = prep_X(rows)
    hgb = joblib.load("C:\\Users\\syedt\\Desktop\\PL_Player_Trajectory\\outputs\\model_gradient_boosted.joblib")
    q10 = joblib.load("C:\\Users\\syedt\\Desktop\\PL_Player_Trajectory\\outputs\\model_quantile_10.joblib")
    q50 = joblib.load("C:\\Users\\syedt\\Desktop\\PL_Player_Trajectory\\outputs\\model_quantile_50.joblib")
    q90 = joblib.load("C:\\Users\\syedt\\Desktop\\PL_Player_Trajectory\\outputs\\model_quantile_90.joblib")

    rows["forecast_2627_ga90"] = hgb.predict(X)
    rows["forecast_2627_p10"] = q10.predict(X)
    rows["forecast_2627_p50"] = q50.predict(X)
    rows["forecast_2627_p90"] = q90.predict(X)
    rows["forecast_delta"] = rows["forecast_2627_ga90"] - rows["G+A /90"]

    early = load_early_2627_actuals()
    rows = rows.merge(early, on="Player", how="left")

    out_cols = ["Player", "Squad", "Position", "Age", "G+A /90", "ga90_ewm",
                "forecast_2627_ga90", "forecast_2627_p10", "forecast_2627_p90",
                "forecast_delta", "early_2627_90s", "early_2627_ga90"]

    breakout = rows.sort_values("forecast_delta", ascending=False).head(25)
    decline = rows.sort_values("forecast_delta", ascending=True).head(25)

    print("\nTop 10 breakout-risk (biggest projected increase):")
    print(breakout[out_cols].head(10).to_string(index=False))
    print("\nTop 10 decline-risk (biggest projected drop):")
    print(decline[out_cols].head(10).to_string(index=False))

    # Directional sanity check: of players with >=2 early 90s (i.e. played most of
    # the first 4 games), how often does the SIGN of the early trend agree with
    # the forecast direction? Small-sample, illustrative only.
    checkable = rows[rows["early_2627_90s"] >= 2].copy()
    checkable["early_delta"] = checkable["early_2627_ga90"] - checkable["G+A /90"]
    agree = np.sign(checkable["forecast_delta"]) == np.sign(checkable["early_delta"])
    print(f"\nDirectional agreement between forecast and early 4-game trend: "
          f"{agree.mean():.1%} ({agree.sum()}/{len(checkable)} players, early sample only, not a real validation)")

    rows.to_csv("C:\\Users\\syedt\\Desktop\\PL_Player_Trajectory\\outputs\\forecast_2026_27_all.csv", index=False)
    breakout[out_cols].to_csv("C:\\Users\\syedt\\Desktop\\PL_Player_Trajectory\\outputs\\breakout_risk_rankings.csv", index=False)
    decline[out_cols].to_csv("C:\\Users\\syedt\\Desktop\\PL_Player_Trajectory\\outputs\\decline_risk_rankings.csv", index=False)
    print("\nSaved -> C:\\Users\\syedt\\Desktop\\PL_Player_Trajectory\\outputs\\forecast_2026_27_all.csv")
    print("Saved -> C:\\Users\\syedt\\Desktop\\PL_Player_Trajectory\\outputs\\breakout_risk_rankings.csv")
    print("Saved -> C:\\Users\\syedt\\Desktop\\PL_Player_Trajectory\\outputs\\decline_risk_rankings.csv")
