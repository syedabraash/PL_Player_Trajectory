import pandas as pd
import numpy as np

from build_player_panel import build_panel

PARTIAL_PATH = "C:\\Users\\syedt\\Desktop\\PL_Player_Trajectory\\data\\PL_26_27_Stats.xlsx"
SEASON_LABEL = "2026-27"
SEASON_START_YEAR = 2026

PLAYER_COLS = [
    "Player", "Position", "Squad", "Matches Played", "Starts", "Minutes Played", "90s",
    "Goals Scored", "Assists", "Goals + Assists", "Non-Penalty Goals", "Penalty Goals",
    "Penalty Attempts", "Yellow Card", "Red Cards",
    "Gls/90", "Ast/90", "G+A /90", "G-PK /90", "G+A-PK /90",
]


def load_partial_player_stats() -> pd.DataFrame:
    raw = pd.read_excel(PARTIAL_PATH, sheet_name="Player_Stats_2627", header=None)
    data = raw.iloc[2:].copy()          # first two rows are the broken header
    data.columns = PLAYER_COLS
    data = data.reset_index(drop=True)

    numeric_cols = [c for c in PLAYER_COLS if c not in ("Player", "Position", "Squad")]
    for c in numeric_cols:
        data[c] = pd.to_numeric(data[c], errors="coerce")

    data["Season"] = SEASON_LABEL
    data["season_start_year"] = SEASON_START_YEAR
    data["is_partial_season"] = True
    data["gameweeks_covered"] = 4
    data["multi_club_season"] = False  # too early in the season for mid-season moves
    return data


def load_club_form_2627() -> pd.DataFrame:
    """Squad-level GK/results table -- a different schema than the historical
    club file, kept separate rather than force-merged into team_season_context.csv."""
    df = pd.read_excel(PARTIAL_PATH, sheet_name="Club_Stats_2627")
    df["Season"] = SEASON_LABEL
    df["points_so_far"] = df["W"] * 3 + df["D"]
    return df


def attach_age(partial: pd.DataFrame, historical_panel: pd.DataFrame) -> pd.DataFrame:
    """No Age/Born columns in the partial file -- look up Born year from the
    5-year panel and age players forward by one year. Players with no
    history (new signings, promoted-club players) get NaN age."""
    born_lookup = (
        historical_panel.sort_values("season_start_year")
        .groupby("Player")["Born"].last()
    )
    partial = partial.copy()
    partial["Born"] = partial["Player"].map(born_lookup)
    # Age convention in the source data is (season_start_year - Born - 1), verified
    # against the historical panel (e.g. Ødegaard: born 1998, age 26 in 2025-26).
    partial["Age"] = SEASON_START_YEAR - partial["Born"] - 1  # NaN where Born is unknown
    return partial


def build_extended_panel() -> pd.DataFrame:
    historical = build_panel()
    partial = load_partial_player_stats()
    partial = attach_age(partial, historical)

    historical["is_partial_season"] = False
    extended = pd.concat([historical, partial], ignore_index=True, sort=False)
    extended = extended.sort_values(["Player", "season_start_year"]).reset_index(drop=True)
    return extended


def build_live_prediction_rows() -> pd.DataFrame:
    """Run the same lag/EWM feature logic as build_trajectory_dataset.py over
    the extended panel, then return just the 2026-27 rows: these have valid
    features describing the player's history INTO this season (age,
    ga90_prev_season, ga90_ewm, minutes_trend, etc.) and are the rows you'd
    feed a trained model to forecast the rest of 2026-27. Their own 90s/G+A
    numbers are the noisy 4-game observation, not something to train on."""
    from build_trajectory_dataset import add_rolling_features
    from build_team_context import build_team_context

    extended = build_extended_panel()
    extended = add_rolling_features(extended)
    team_context = build_team_context()
    extended = extended.merge(team_context, on=["Squad", "Season"], how="left")

    live = extended[extended["Season"] == SEASON_LABEL].copy()
    return live


if __name__ == "__main__":
    partial = load_partial_player_stats()
    print("Partial 2026-27 player rows:", partial.shape)
    print("90s range (sample size warning):", partial["90s"].min(), "-", partial["90s"].max())
    print(partial.head(3).to_string())

    club_form = load_club_form_2627()
    print("\nClub form 2026-27:", club_form.shape)
    print(club_form[["Squad", "MP", "W", "D", "L", "points_so_far", "GA90"]].head(5).to_string())

    extended = build_extended_panel()
    print("\nExtended panel shape:", extended.shape)
    print("New clubs with no history:",
          sorted(set(partial["Squad"]) - set(build_panel()["Squad"])))
    print("\nSample: a player's trajectory now including the partial season")
    sample_name = "Martin Ødegaard" if "Martin Ødegaard" in extended["Player"].values else extended["Player"].iloc[0]
    cols = ["Player", "Season", "Age", "Squad", "Minutes Played", "90s", "G+A /90", "is_partial_season"]
    print(extended[extended["Player"] == sample_name][cols].to_string())

    unmatched = set(partial["Player"]) - set(build_panel()["Player"])
    print(f"\n2026-27 players with no prior history (new arrivals/promoted clubs): {len(unmatched)} / {partial['Player'].nunique()}")

    live = build_live_prediction_rows()
    print("\nLive feature rows for 2026-27 (age/lag features filled, own stats still noisy):")
    live_cols = ["Player", "Squad", "Age", "ga90_prev_season", "ga90_ewm", "career_seasons_so_far",
                 "minutes_trend", "shot_share", "90s", "G+A /90"]
    print(live[live_cols].head(5).to_string())

    extended.to_csv("C:\\Users\\syedt\\Desktop\\PL_Player_Trajectory\\outputs\\player_panel_with_2627.csv", index=False)
    club_form.to_csv("C:\\Users\\syedt\\Desktop\\PL_Player_Trajectory\\outputs\\club_form_2627.csv", index=False)
    live.to_csv("C:\\Users\\syedt\\Desktop\\PL_Player_Trajectory\\outputs\\player_live_features_2627.csv", index=False)
    print("\nSaved -> C:\\Users\\syedt\\Desktop\\PL_Player_Trajectory\\outputs\\player_panel_with_2627.csv")
    print("Saved -> C:\\Users\\syedt\\Desktop\\PL_Player_Trajectory\\outputs\\club_form_2627.csv")
    print("Saved -> C:\\Users\\syedt\\Desktop\\PL_Player_Trajectory\\outputs\\player_live_features_2627.csv")
