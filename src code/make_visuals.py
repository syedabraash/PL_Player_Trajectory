import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FIG_DIR = "C:\\Users\\syedt\\Desktop\\PL_Player_Trajectory\\outputs\\figures"
os.makedirs(FIG_DIR, exist_ok=True)
plt.rcParams["font.size"] = 10


def plot_model_vs_baseline():
    metrics = pd.read_csv("C:\\Users\\syedt\\Desktop\\PL_Player_Trajectory\\outputs\\model_metrics.csv")
    test = metrics[metrics["split"] == "test"].sort_values("mae")
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#4C72B0" if m == "gradient_boosted" else "#999999" for m in test["model"]]
    ax.bar(test["model"], test["mae"], color=colors)
    ax.set_ylabel("MAE (test season 2024-25 -> 2025-26)")
    ax.set_title("Model vs. baselines: next-season G+A/90 prediction error")
    ax.set_xticklabels(test["model"], rotation=20, ha="right")
    for i, v in enumerate(test["mae"]):
        ax.text(i, v + 0.001, f"{v:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/model_vs_baseline_mae.png", dpi=150)
    plt.close(fig)


def plot_segment_mae():
    seg = pd.read_csv("C:\\Users\\syedt\\Desktop\\PL_Player_Trajectory\\outputs\\segment_metrics.csv")
    pivot = seg.pivot(index="segment", columns="model", values="mae").reindex(["decline", "stable", "breakout"])
    fig, ax = plt.subplots(figsize=(7, 4))
    pivot.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452", "#999999"])
    ax.set_ylabel("MAE")
    ax.set_title("Error by outcome type (test set) — the segments the aggregate MAE hides")
    ax.set_xticklabels(pivot.index, rotation=0)
    ax.legend(title=None)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/segment_mae.png", dpi=150)
    plt.close(fig)


def plot_feature_importance():
    imp = pd.read_csv("C:\\Users\\syedt\\Desktop\\PL_Player_Trajectory\\outputs\\feature_importance.csv").sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(imp["feature"], imp["importance_mean"], xerr=imp["importance_std"], color="#4C72B0")
    ax.set_xlabel("Permutation importance (MAE increase when shuffled)")
    ax.set_title("What drives the model's next-season prediction")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/feature_importance.png", dpi=150)
    plt.close(fig)


def plot_player_trajectories():
    full = pd.read_csv("C:\\Users\\syedt\\Desktop\\PL_Player_Trajectory\\outputs\\player_trajectory_full.csv")
    test = pd.read_csv("C:\\Users\\syedt\\Desktop\\PL_Player_Trajectory\\outputs\\test_predictions.csv")

    # pick 2 breakout and 2 decline stories from the test set, by magnitude
    breakout_players = test.sort_values("target_next_ga90", ascending=False).head(2)["Player"].tolist()
    decline_players = test[test["segment"] == "decline"].assign(
        drop=lambda d: d["G+A /90"] - d["target_next_ga90"]
    ).sort_values("drop", ascending=False).head(2)["Player"].tolist()
    players = breakout_players + decline_players

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=False)
    for ax, player in zip(axes.flat, players):
        hist = full[full["Player"] == player].sort_values("season_start_year")
        ax.plot(hist["Season"], hist["G+A /90"], marker="o", color="#4C72B0", label="Actual")

        row = test[test["Player"] == player]
        if len(row):
            target_season = row["target_season"].values[0]
            ax.scatter([target_season], row["target_next_ga90"], color="#4C72B0", marker="o", s=60, zorder=5)
            ax.scatter([target_season], row["pred_gb"], color="#DD8452", marker="x", s=80,
                       label="Model prediction", zorder=5)
        ax.set_title(player, fontsize=10)
        ax.set_ylabel("G+A / 90")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=8)

    fig.suptitle("Actual vs. predicted trajectories: breakout and decline cases (test set)")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/player_trajectories.png", dpi=150)
    plt.close(fig)


def plot_quantile_fan():
    fc = pd.read_csv("C:\\Users\\syedt\\Desktop\\PL_Player_Trajectory\\outputs\\forecast_2026_27_all.csv")
    full = pd.read_csv("C:\\Users\\syedt\\Desktop\\PL_Player_Trajectory\\outputs\\player_trajectory_full.csv")

    # a couple of the top breakout-risk forecasts, illustrated with their interval
    top = fc.sort_values("forecast_delta", ascending=False).head(3)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    for ax, (_, row) in zip(axes, top.iterrows()):
        hist = full[full["Player"] == row["Player"]].sort_values("season_start_year")
        seasons = list(hist["Season"]) + ["2026-27 (fcst)"]
        values = list(hist["G+A /90"])

        ax.plot(seasons[:-1], values, marker="o", color="#4C72B0", label="Actual")
        ax.errorbar([seasons[-1]], [row["forecast_2627_p50"]],
                    yerr=[[row["forecast_2627_p50"] - row["forecast_2627_p10"]],
                          [row["forecast_2627_p90"] - row["forecast_2627_p50"]]],
                    fmt="D", color="#DD8452", capsize=5, label="Forecast (10-90th pct)")
        ax.set_title(row["Player"], fontsize=10)
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylabel("G+A / 90")
        ax.legend(fontsize=8)

    fig.suptitle("2026-27 breakout-risk forecasts with prediction intervals")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/quantile_fan_chart.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    plot_model_vs_baseline()
    plot_segment_mae()
    plot_feature_importance()
    plot_player_trajectories()
    plot_quantile_fan()
    print("Saved 5 figures to", FIG_DIR)
    print(os.listdir(FIG_DIR))
