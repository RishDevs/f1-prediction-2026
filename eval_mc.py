import sys
import pandas as pd
from src.data.data_generator import generate_historical_data, generate_2026_grid
from src.data.features import build_training_features, build_sprint_training_features, build_2026_prediction_features
import src.config as config
from src.models.sprint_model import SprintClassifier
from src.models.pace_model import PaceRegressor
from src.models.strategy_model import StrategyModel
from src.models.monte_carlo import MonteCarloSimulator

print("Generating history...")
hist_df = generate_historical_data(seed=42)

print("Training Pace...")
gp_X, gp_y = build_training_features(hist_df)
pace_model = PaceRegressor(params=config.MODEL_PARAMS["lgbm_regressor"])
pace_model.train(gp_X, hist_df[hist_df["is_sprint"]==False]["lap_time"])

print("Training Sprint...")
sprint_X, sprint_y = build_sprint_training_features(hist_df)
sprint_model = SprintClassifier(params=config.MODEL_PARAMS["lgbm_classifier"])
sprint_model.train(sprint_X, sprint_y)

print("Generating 2026 Grid with FP1...")
grid_df = generate_2026_grid(seed=101)
from src.data.features import compute_driver_stats
driver_stats = compute_driver_stats(hist_df)
pred_features = build_2026_prediction_features(grid_df, driver_stats)

print("Running Monte Carlo (n=3000)...")
mc = MonteCarloSimulator(pace_model, strategy_model=StrategyModel(), sprint_model=sprint_model, n_runs=3000)
gp_results, sprint_results = mc.run_simulation(grid_df, pred_features)

print("\n--- SPRINT RACE ---")
print(sprint_results.head(5)[["driver", "team", "win_prob", "podium_prob"]].to_string())
print("\n--- GRAND PRIX ---")
print(gp_results.head(5)[["driver", "team", "win_prob", "podium_prob"]].to_string())
