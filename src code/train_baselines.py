import pandas as pd
import numpy as np

SUPERVISED_PATH = "C:\\Users\\syedt\\Desktop\\PL_Player_Trajectory\\outputs\\player_trajectory_supervised.csv"
TRAIN_YEARS = [2021, 2022]
VAL_YEAR = 2023
TEST_YEAR = 2024


def load_splits():
    df = pd.read_csv(SUPERVISED_PATH)
    train = df[df["season_start_year"].isin(TRAIN_YEARS)].copy()
    val = df[df["season_start_year"] == VAL_YEAR].copy()
    test = df[df["season_start_year"] == TEST_YEAR].copy()
    return train, val, test


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def naive_persistence(split: pd.DataFrame) -> np.ndarray:
    """Predict next season = this season's G+A/90 (no change)."""
    return split["G+A /90"].values


def positional_average(train: pd.DataFrame, split: pd.DataFrame) -> np.ndarray:
    """Predict next season = the training-set average target for that position."""
    pos_avg = train.groupby("Position")["target_next_ga90"].mean()
    overall_avg = train["target_next_ga90"].mean()
    return split["Position"].map(pos_avg).fillna(overall_avg).values


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {"model": name, "mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred), "n": len(y_true)}


if __name__ == "__main__":
    train, val, test = load_splits()
    print(f"Train: {len(train)} (seasons {TRAIN_YEARS}) | Val: {len(val)} ({VAL_YEAR}) | Test: {len(test)} ({TEST_YEAR})")

    results = []
    for split_name, split in [("val", val), ("test", test)]:
        y_true = split["target_next_ga90"].values

        y_persist = naive_persistence(split)
        results.append({**evaluate("naive_persistence", y_true, y_persist), "split": split_name})

        y_posavg = positional_average(train, split)
        results.append({**evaluate("positional_average", y_true, y_posavg), "split": split_name})

    results_df = pd.DataFrame(results)[["split", "model", "mae", "rmse", "n"]]
    print(results_df.to_string(index=False))
    results_df.to_csv("C:\\Users\\syedt\\Desktop\\PL_Player_Trajectory\\outputs\\baseline_metrics.csv", index=False)
    print("\nSaved -> C:\\Users\\syedt\\Desktop\\PL_Player_Trajectory\\outputs\\baseline_metrics.csv")
