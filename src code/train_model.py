import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from train_baselines import load_splits, mae, rmse, naive_persistence, positional_average

NUMERIC_FEATURES = [
    "Age", "age_sq", "career_seasons_so_far", "cum_minutes",
    "Minutes Played", "90s", "minutes_prev_season", "minutes_trend",
    "G+A /90", "ga90_prev_season", "ga90_ewm",
    "avg_possession", "avg_shots_for", "avg_shots_against", "shot_share",
]
CATEGORICAL_FEATURES = ["Position"]
BOOLEAN_FEATURES = ["multi_club_season"]
TARGET = "target_next_ga90"

QUANTILES = [0.1, 0.5, 0.9]


def make_ridge_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("bool", SimpleImputer(strategy="most_frequent"), BOOLEAN_FEATURES),
    ])
    return Pipeline([("pre", pre), ("model", Ridge(alpha=5.0))])


def make_hgb_pipeline(quantile=None) -> Pipeline:
    pre = ColumnTransformer([
        ("num", "passthrough", NUMERIC_FEATURES),          # HGB handles NaN natively
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("bool", SimpleImputer(strategy="most_frequent"), BOOLEAN_FEATURES),
    ])
    if quantile is None:
        model = HistGradientBoostingRegressor(
            max_iter=300, max_depth=4, learning_rate=0.05, random_state=42,
        )
    else:
        model = HistGradientBoostingRegressor(
            loss="quantile", quantile=quantile,
            max_iter=300, max_depth=4, learning_rate=0.05, random_state=42,
        )
    return Pipeline([("pre", pre), ("model", model)])


def prep_xy(df: pd.DataFrame):
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES].copy()
    X[BOOLEAN_FEATURES] = X[BOOLEAN_FEATURES].astype(int)
    y = df[TARGET].values
    return X, y


def evaluate(name, split, y_true, y_pred, results):
    results.append({
        "split": split, "model": name,
        "mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred), "n": len(y_true),
    })


def segment_labels(df: pd.DataFrame) -> pd.Series:
    delta = df[TARGET] - df["G+A /90"]
    q_low, q_high = delta.quantile([1 / 3, 2 / 3])
    return pd.cut(delta, bins=[-np.inf, q_low, q_high, np.inf],
                  labels=["decline", "stable", "breakout"])


if __name__ == "__main__":
    train, val, test = load_splits()
    X_train, y_train = prep_xy(train)
    X_val, y_val = prep_xy(val)
    X_test, y_test = prep_xy(test)

    results = []

    for split_name, split in [("val", val), ("test", test)]:
        y_true = split[TARGET].values
        evaluate("naive_persistence", split_name, y_true, naive_persistence(split), results)
        evaluate("positional_average", split_name, y_true, positional_average(train, split), results)

    ridge = make_ridge_pipeline().fit(X_train, y_train)
    evaluate("ridge", "val", y_val, ridge.predict(X_val), results)
    evaluate("ridge", "test", y_test, ridge.predict(X_test), results)

    hgb = make_hgb_pipeline().fit(X_train, y_train)
    evaluate("gradient_boosted", "val", y_val, hgb.predict(X_val), results)
    evaluate("gradient_boosted", "test", y_test, hgb.predict(X_test), results)

    results_df = pd.DataFrame(results)[["split", "model", "mae", "rmse", "n"]]
    print(results_df.sort_values(["split", "mae"]).to_string(index=False))
    results_df.to_csv("C:\\Users\\syedt\\Desktop\\PL_Player_Trajectory\\outputs\\model_metrics.csv", index=False)

    test = test.copy()
    test["segment"] = segment_labels(test)
    test["pred_gb"] = hgb.predict(X_test)
    test["pred_ridge"] = ridge.predict(X_test)
    test["pred_persist"] = naive_persistence(test)

    seg_rows = []
    for seg in ["decline", "stable", "breakout"]:
        sub = test[test["segment"] == seg]
        for model_name, col in [("gradient_boosted", "pred_gb"), ("ridge", "pred_ridge"), ("naive_persistence", "pred_persist")]:
            seg_rows.append({
                "segment": seg, "model": model_name,
                "mae": mae(sub[TARGET].values, sub[col].values), "n": len(sub),
            })
    seg_df = pd.DataFrame(seg_rows)
    print("\nSegment evaluation (test set, by actual breakout/decline/stable):")
    print(seg_df.to_string(index=False))
    seg_df.to_csv("C:\\Users\\syedt\\Desktop\\PL_Player_Trajectory\\outputs\\segment_metrics.csv", index=False)

    quantile_models = {}
    quantile_preds_test = {}
    for q in QUANTILES:
        m = make_hgb_pipeline(quantile=q).fit(X_train, y_train)
        quantile_models[q] = m
        quantile_preds_test[q] = m.predict(X_test)

    coverage = np.mean((y_test >= quantile_preds_test[0.1]) & (y_test <= quantile_preds_test[0.9]))
    print(f"\n10th-90th percentile interval empirical coverage on test: {coverage:.1%} (target ~80%)")

    perm = permutation_importance(hgb, X_test, y_test, n_repeats=20, random_state=42, scoring="neg_mean_absolute_error")
    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False)
    print("\nPermutation feature importance (MAE increase when shuffled):")
    print(importance_df.to_string(index=False))
    importance_df.to_csv("C:\\Users\\syedt\\Desktop\\PL_Player_Trajectory\\outputs\\feature_importance.csv", index=False)

    joblib.dump(hgb, "C:\\Users\\syedt\\Desktop\\PL_Player_Trajectory\\outputs\\model_gradient_boosted.joblib")
    joblib.dump(ridge, "C:\\Users\\syedt\\Desktop\\PL_Player_Trajectory\\outputs\\model_ridge.joblib")
    for q, m in quantile_models.items():
        joblib.dump(m, f"C:\\Users\\syedt\\Desktop\\PL_Player_Trajectory\\outputs\\model_quantile_{int(q*100)}.joblib")

    test.to_csv("C:\\Users\\syedt\\Desktop\\PL_Player_Trajectory\\outputs\\test_predictions.csv", index=False)
    print("\nSaved model metrics, segment metrics, feature importance, models (.joblib), and test_predictions.csv")
