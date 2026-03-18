import sys
import os

sys.path.insert(0, "/Users/krrish/Desktop/projects/f1")

# Suppress lightgbm warnings if possible
import warnings
warnings.filterwarnings('ignore')

try:
    from app import load_and_train

    # Run the training and prediction
    print("Running simulations (N=2500)...")
    data = load_and_train(mc_runs=2500)

    print("\n==== 2026 JAPANESE GP PREDICTIONS ====")
    print("\n[ PROJECTED QUALIFYING ORDER ]")
    grid_df = data["grid_df"].sort_values("grid_position")
    for i, row in grid_df.iterrows():
        print(f"P{int(row['grid_position']):02d}: {row['driver']} ({row['team']}) - FP1 Simulated Pace: {row['pace_score']:.4f}")

    print("\n\n[ PROJECTED GRAND PRIX RACE RESULTS (Top 10) ]")
    mc_gp = data["mc_gp"]
    for i, row in mc_gp.head(10).iterrows():
        print(f"P{i+1}: {row['driver']} ({row['team']}) - Win Prob: {row['win_probability']:.1f}% - Podium Prob: {row['podium_probability']:.1f}%")

except Exception as e:
    import traceback
    traceback.print_exc()
