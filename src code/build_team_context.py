import pandas as pd

CLUB_PATH = "C:\\Users\\syedt\\Desktop\\PL_Player_Trajectory\\data\\PL_Club_5years.xlsx"

# club-file sheet name -> player-file Squad name
SQUAD_NAME_MAP = {
    "Arsenal": "Arsenal",
    "AstonVilla": "Aston Villa",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton": "Brighton",
    "CrystalPalace": "Crystal Palace",
    "Chelsea": "Chelsea",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Ipswich Town": "Ipswich Town",
    "Leeds": "Leeds United",
    "Liverpool": "Liverpool",
    "ManCity": "Manchester City",
    "ManUnited": "Manchester Utd",
    "Newcastle": "Newcastle",
    "Nottingham": "Nottingham",
    "Spurs": "Tottenham",
    "Sunderland": "Sunderland",
}


def season_from_date(date: pd.Timestamp) -> str:
    """PL seasons run Aug-May. Jul onward = start of a new season."""
    if date.month >= 7:
        start = date.year
    else:
        start = date.year - 1
    return f"{start}-{str(start + 1)[-2:]}"


def build_team_context() -> pd.DataFrame:
    xls = pd.ExcelFile(CLUB_PATH)
    rows = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(CLUB_PATH, sheet_name=sheet)
        df["Season"] = df["Date"].apply(season_from_date)
        df["Squad"] = SQUAD_NAME_MAP[sheet]

        season_agg = df.groupby(["Squad", "Season"]).agg(
            matches=("Date", "count"),
            avg_possession=("Possession", "mean"),
            avg_shots_for=("Shots", "mean"),
            avg_sot_for=("SoT", "mean"),
            avg_shots_against=("Opp Shots", "mean"),
            goals_for=("GF", "sum"),
            goals_against=("GA", "sum"),
        ).reset_index()
        rows.append(season_agg)

    team_context = pd.concat(rows, ignore_index=True)
    # attacking-team-ness proxy (no xG available): shots-for share of total shots in the game
    team_context["shot_share"] = team_context["avg_shots_for"] / (
        team_context["avg_shots_for"] + team_context["avg_shots_against"]
    )
    return team_context


if __name__ == "__main__":
    tc = build_team_context()
    print("Team-context shape:", tc.shape)
    print("Squads covered:", sorted(tc["Squad"].unique()))
    print(tc.head(10).to_string())
    tc.to_csv("C:\\Users\\syedt\\Desktop\\PL_Player_Trajectory\\outputs\\team_season_context.csv", index=False)
    print("\nSaved -> C:\\Users\\syedt\\Desktop\\PL_Player_Trajectory\\outputs\\team_season_context.csv")