import pandas as pd
import numpy as np

from build_player_panel import build_panel
from build_team_context import build_team_context

MIN_90S_FOR_TARGET = 3.0  # drop next-season targets built on tiny sample sizes


def add_rolling_features(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["Player", "season_start_year"]).copy()
    g = panel.groupby("Player")

    # 1-season lag (this IS the current season row, features describe player's state going in)
    panel["prev_ga90"] = g["G+A /90"].shift(0)  # alias for clarity in feature list below
    panel["career_seasons_so_far"] = g.cumcount() + 1
    panel["cum_minutes"] = g["Minutes Played"].cumsum()
    panel["minutes_prev_season"] = g["Minutes Played"].shift(1)
    panel["minutes_trend"] = panel["Minutes Played"] - panel["minutes_prev_season"]

    # 2-3 season EWM trend of G+A/90 (span=2 -> heavier weight on recent seasons)
    panel["ga90_ewm"] = g["G+A /90"].transform(lambda s: s.ewm(span=2, adjust=False).mean())
    panel["ga90_prev_season"] = g["G+A /90"].shift(1)
    panel["ga90_2_seasons_ago"] = g["G+A /90"].shift(2)

    panel["age_sq"] = panel["Age"] ** 2
    return panel


def build_dataset() -> pd.DataFrame:
    panel = build_panel()
    panel = add_rolling_features(panel)

    team_context = build_team_context()
    panel = panel.merge(team_context, on=["Squad", "Season"], how="left")

    # target: next season's G+A/90 for the SAME player
    panel = panel.sort_values(["Player", "season_start_year"])
    g = panel.groupby("Player")
    panel["target_next_ga90"] = g["G+A /90"].shift(-1)
    panel["target_next_90s"] = g["90s"].shift(-1)
    panel["target_season"] = g["Season"].shift(-1)

    panel["has_target"] = panel["target_next_ga90"].notna() & (
        panel["target_next_90s"] >= MIN_90S_FOR_TARGET
    )
    return panel


if __name__ == "__main__":
    df = build_dataset()
    print("Full panel (all player-seasons):", df.shape)

    supervised = df[df["has_target"]].copy()
    print("Supervised rows (have a valid next-season target):", supervised.shape)

    print("\nRows by season_start_year (for the time-based split):")
    print(df.groupby("season_start_year").size())

    print("\nSample of a player's trajectory (Erling Haaland if present, else first striker):")
    sample_name = "Erling Haaland" if "Erling Haaland" in df["Player"].values else df["Player"].iloc[0]
    cols = ["Player", "Season", "Age", "Squad", "Minutes Played", "90s", "G+A /90",
            "ga90_ewm", "minutes_trend", "shot_share", "target_next_ga90", "target_season"]
    print(df[df["Player"] == sample_name][cols].to_string())

    print("\nMissing team-context rate (players at non-ever-present clubs):")
    print(supervised["shot_share"].isna().mean().round(3))

    df.to_csv("C:\\Users\\syedt\\Desktop\\PL_Player_Trajectory\\outputs\\player_trajectory_full.csv", index=False)
    supervised.to_csv("C:\\Users\\syedt\\Desktop\\PL_Player_Trajectory\\outputs\\player_trajectory_supervised.csv", index=False)
    print("\nSaved -> C:\\Users\\syedt\\Desktop\\PL_Player_Trajectory\\outputs\\player_trajectory_full.csv")
    print("Saved -> C:\\Users\\syedt\\Desktop\\PL_Player_Trajectory\\outputs\\player_trajectory_supervised.csv")
