import pandas as pd
import numpy as np

PLAYERS_PATH = "C:\\Users\\syedt\\Desktop\\PL_Player_Trajectory\\data\\PL_Players_5Years.xlsx"

SUM_COLS = [
    "Player", "Nation", "Position", "Squad", "Age", "Born", "Matches Played", "Starts", "Minutes Played", "90s", "Goals Scored", "Assists", "Goals + Assists", "Non-Penalty Goals", "Penalty Goals", "Penalty Attempts", "Yellow Card", "Red Cards", "Gls/90", "Ast/90", "G+A /90", "G-PK /90", "G+A-PK /90"
]

def load_season(sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(PLAYERS_PATH, sheet_name=sheet_name)

    # --- basic cleaning ---
    # one known null row per audit (2024-25): drop if core identifiers are missing
    df = df.dropna(subset=["Player", "Age", "Born"]).copy()
    df["Age"] = df["Age"].astype(int)
    df["Born"] = df["Born"].astype(int)

    # --- collapse mid-season transfers ---
    # keep row order as given (FBref lists earlier club first, later club last)
    df["_row_order"] = np.arange(len(df))

    agg_dict = {c: "sum" for c in SUM_COLS}
    agg_dict.update({
        "Nation": "first",
        "Position": "first",       # primary-position label assumed stable across a season
        "Squad": "last",           # "most recent club" = last-listed stint
        "Age": "last",
        "Born": "first",
        "_row_order": "max",
    })

    grouped = df.groupby("Player", as_index=False).agg(agg_dict)
    stint_counts = df.groupby("Player")["Squad"].nunique()
    grouped["multi_club_season"] = grouped["Player"].map(stint_counts) > 1

    grouped = grouped.drop(columns=["_row_order"])

    # --- recompute per-90 rates from SUMMED raw stats (don't trust averaged per-90 cols) ---
    nineties = grouped["90s"].replace(0, np.nan)
    grouped["Gls/90"] = (grouped["Goals Scored"] / nineties).fillna(0)
    grouped["Ast/90"] = (grouped["Assists"] / nineties).fillna(0)
    grouped["G+A /90"] = (grouped["Goals + Assists"] / nineties).fillna(0)
    grouped["G-PK /90"] = ((grouped["Goals Scored"] - grouped["Penalty Goals"]) / nineties).fillna(0)
    grouped["G+A-PK /90"] = (grouped["G+A /90"] - grouped["Penalty Goals"] / nineties).fillna(0)

    grouped["Season"] = sheet_name
    return grouped


def build_panel() -> pd.DataFrame:
    xls = pd.ExcelFile(PLAYERS_PATH)
    seasons = sorted(xls.sheet_names)  # chronological: 2021-22 ... 2025-26
    panel = pd.concat([load_season(s) for s in seasons], ignore_index=True)
    panel["season_start_year"] = panel["Season"].str[:4].astype(int)
    panel = panel.sort_values(["Player", "season_start_year"]).reset_index(drop=True)
    return panel


if __name__ == "__main__":
    panel = build_panel()
    print("Panel shape:", panel.shape)
    print("Seasons:", sorted(panel["Season"].unique()))
    print("Multi-club season rows:", panel["multi_club_season"].sum())
    print(panel[panel["multi_club_season"]][["Player", "Season", "Squad", "Minutes Played", "Goals + Assists"]].head(10))
    print()
    print(panel.head(5).to_string())
    panel.to_csv("../outputs/player_season_panel.csv", index=False)
    print("\nSaved -> ../outputs/player_season_panel.csv")