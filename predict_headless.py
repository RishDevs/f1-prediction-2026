import sys
import os
sys.path.insert(0, "/Users/krrish/Desktop/projects/f1")

import warnings
warnings.filterwarnings('ignore')

from src.data.data_generator import generate_historical_data, generate_2026_grid
from src.data.features import (
    compute_driver_stats,
    build_training_features,
    build_sprint_training_features,
    build_ranking_features,
    build_2026_prediction_features,
)
from src.models.sprint_model import SprintPredictionModel
from src.models.pace_model import RacePaceModel
from src.models.ranking_model import DriverRankingModel
from src.models.strategy_model import RaceStrategyModel
from src.models.monte_carlo import MonteCarloSimulator

def load_and_train(mc_runs: int = 10_000, mc_seed: int = 42):
    hist_df = generate_historical_data(seed=42)
    grid_df = generate_2026_grid(seed=99)
    driver_stats = compute_driver_stats(hist_df, circuit="Japan")
    pred_features = build_2026_prediction_features(grid_df, driver_stats)

    X_gp, y_gp = build_training_features(hist_df)
    X_sprint, y_sprint = build_sprint_training_features(hist_df)
    X_rank, y_rank, groups_rank = build_ranking_features(hist_df)

    sprint_model = SprintPredictionModel()
    sprint_model.fit(X_sprint, y_sprint)
    sprint_proba_df = sprint_model.predict_sprint_probabilities(pred_features)

    pace_model = RacePaceModel()
    gp_df_raw = hist_df[hist_df["is_sprint"] == False]
    lap_target = gp_df_raw["lap_time"].reset_index(drop=True).iloc[:len(X_gp)]
    pace_model.fit(X_gp, lap_target)
    pace_df = pace_model.predict_race_pace(pred_features)

    rank_model = DriverRankingModel()
    rank_model.fit(X_rank, y_rank, groups_rank)
    rank_df = rank_model.predict_finishing_order(pred_features)

    strat_model = RaceStrategyModel()
    strat_model.fit(X_gp, hist_df[hist_df["is_sprint"] == False].reset_index(drop=True))
    strategy_df = strat_model.predict_strategy(pred_features)

    simulator = MonteCarloSimulator(n_runs=mc_runs, seed=mc_seed)
    mc_gp_df   = simulator.run_grand_prix(pred_features, pace_df, strategy_df)
    mc_sprint_df = simulator.run_sprint(pred_features, sprint_proba_df)

    return {
        "grid_df": grid_df,
        "mc_gp": mc_gp_df,
        "mc_sprint": mc_sprint_df,
    }

print("Loading models and running simulations...")
data = load_and_train(mc_runs=2500)

print("\n==== 2026 JAPANESE GP PREDICTIONS ====")
print("\n[ QUALIFYING PROJECTED ORDER (FP1 + Pace Simulation) ]")
grid_df = data["grid_df"].sort_values("grid_position")
for i, row in grid_df.iterrows():
    print(f"P{int(row['grid_position'])}: {row['driver']} ({row['team']}) - Pace Score: {row['pace_score']:.4f}")

print("\n\n[ GRAND PRIX PROJECTED RACE RESULTS (Top 10) ]")
mc_gp = data["mc_gp"]
for i, row in mc_gp.head(10).iterrows():
    print(f"P{i+1}: {row['driver']} ({row['team']}) - Win Prob: {row['win_probability']:.1f}%")
