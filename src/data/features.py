"""
features.py
Feature engineering for the F1 prediction models.
"""

import pandas as pd
import numpy as np


def compute_driver_stats(hist_df: pd.DataFrame, circuit: str = "Japan") -> pd.DataFrame:
    """
    Compute aggregated driver performance statistics from historical data.
    Returns a DataFrame indexed by driver name.
    """
    stats = {}

    for driver, grp in hist_df.groupby("driver"):
        total_races = len(grp)
        if total_races == 0:
            continue

        # General performance
        avg_finish = grp["finish_position"].mean()
        podium_rate = (grp["finish_position"] <= 3).mean()
        win_rate = (grp["finish_position"] == 1).mean()
        dnf_rate = grp["dnf"].mean()
        avg_grid = grp["grid_position"].mean()
        grid_finish_delta = (grp["grid_position"] - grp["finish_position"]).mean()  # positive = gained places

        # Non-sprint only
        gp_grp = grp[grp["is_sprint"] == False]
        gp_avg_finish = gp_grp["finish_position"].mean() if len(gp_grp) > 0 else avg_finish

        # Sprint performance
        sprint_grp = grp[grp["is_sprint"] == True]
        sprint_avg_finish = sprint_grp["finish_position"].mean() if len(sprint_grp) > 0 else avg_finish
        sprint_win_rate = (sprint_grp["finish_position"] == 1).mean() if len(sprint_grp) > 0 else win_rate

        # Suzuka-specific
        sh_grp = grp[grp["circuit"] == circuit]
        sh_avg_finish = sh_grp["finish_position"].mean() if len(sh_grp) > 0 else avg_finish
        sh_podium_rate = (sh_grp["finish_position"] <= 3).mean() if len(sh_grp) > 0 else podium_rate

        # Qualifying performance
        quali_overpace = (grp["grid_position"] - grp["grid_position"].mean()).mean()
        avg_laptime = grp["lap_time"].mean()

        stats[driver] = {
            "total_races": total_races,
            "avg_finish": round(avg_finish, 3),
            "avg_gp_finish": round(gp_avg_finish, 3),
            "avg_sprint_finish": round(sprint_avg_finish, 3),
            "podium_rate": round(podium_rate, 4),
            "win_rate": round(win_rate, 4),
            "sprint_win_rate": round(sprint_win_rate, 4),
            "dnf_rate": round(dnf_rate, 4),
            "avg_grid": round(avg_grid, 3),
            "grid_finish_delta": round(grid_finish_delta, 3),
            "suzuka_avg_finish": round(sh_avg_finish, 3),
            "suzuka_podium_rate": round(sh_podium_rate, 4),
            "avg_laptime": round(avg_laptime, 3),
            "skill_rating": grp["driver_skill"].iloc[0],
            "constructor_pace": grp["constructor_pace"].iloc[0],
        }

    return pd.DataFrame.from_dict(stats, orient="index")


def build_training_features(hist_df: pd.DataFrame) -> tuple:
    """
    Build feature matrix X and target y for GP race winner prediction.
    Uses only the main races (not sprints).

    Returns (X_df, y_series)
    """
    gp_df = hist_df[hist_df["is_sprint"] == False].copy()
    driver_stats = compute_driver_stats(hist_df)

    feature_rows = []
    targets = []

    for _, row in gp_df.iterrows():
        driver = row["driver"]
        if driver not in driver_stats.index:
            continue
        ds = driver_stats.loc[driver]

        features = {
            "grid_position": row["grid_position"],
            "driver_skill": row["driver_skill"],
            "constructor_pace": row["constructor_pace"],
            "avg_finish": ds["avg_finish"],
            "podium_rate": ds["podium_rate"],
            "win_rate": ds["win_rate"],
            "dnf_rate": ds["dnf_rate"],
            "grid_finish_delta": ds["grid_finish_delta"],
            "suzuka_avg_finish": ds["suzuka_avg_finish"],
            "suzuka_podium_rate": ds["suzuka_podium_rate"],
            "avg_laptime": row["lap_time"],
            "pit_stops": row["pit_stops"],
            "tire_hard": int(row["tire_compound"] == "Hard"),
            "tire_medium": int(row["tire_compound"] == "Medium"),
            "tire_soft": int(row["tire_compound"] == "Soft"),
            "weather_wet": int(row["weather"] == "Wet"),
            "safety_car": row["safety_car"],
            "tire_degradation": row["tire_degradation"],
        }
        feature_rows.append(features)
        targets.append(int(row["finish_position"] == 1))

    X = pd.DataFrame(feature_rows)
    y = pd.Series(targets)
    return X, y


def build_sprint_training_features(hist_df: pd.DataFrame) -> tuple:
    """
    Build feature matrix X and target y for sprint race prediction.
    """
    sprint_df = hist_df[hist_df["is_sprint"] == True].copy()
    driver_stats = compute_driver_stats(hist_df)

    feature_rows = []
    targets = []

    for _, row in sprint_df.iterrows():
        driver = row["driver"]
        if driver not in driver_stats.index:
            continue
        ds = driver_stats.loc[driver]

        features = {
            "sprint_grid_position": row["grid_position"],
            "driver_skill": row["driver_skill"],
            "constructor_pace": row["constructor_pace"],
            "avg_sprint_finish": ds["avg_sprint_finish"],
            "sprint_win_rate": ds["sprint_win_rate"],
            "podium_rate": ds["podium_rate"],
            "dnf_rate": ds["dnf_rate"],
            "grid_finish_delta": ds["grid_finish_delta"],
            "overtaking_ability": np.clip(row["driver_skill"] + 0.02, 0, 1),
            "consistency_score": np.clip(row["driver_skill"] - 0.03, 0, 1),
            "tire_soft": int(row["tire_compound"] == "Soft"),
            "tire_medium": int(row["tire_compound"] == "Medium"),
            "weather_wet": int(row["weather"] == "Wet"),
        }
        feature_rows.append(features)
        targets.append(int(row["finish_position"] == 1))

    X = pd.DataFrame(feature_rows)
    y = pd.Series(targets)
    return X, y


def build_ranking_features(hist_df: pd.DataFrame) -> tuple:
    """
    Build features for the learning-to-rank model (predicts finishing order).
    Returns (X_df, y_positions, group_sizes)
    """
    gp_df = hist_df[hist_df["is_sprint"] == False].copy()
    driver_stats = compute_driver_stats(hist_df)

    feature_rows = []
    targets = []
    groups = []

    for race_id, race_grp in gp_df.groupby("race_id"):
        group_size = 0
        for _, row in race_grp.iterrows():
            driver = row["driver"]
            if driver not in driver_stats.index:
                continue
            ds = driver_stats.loc[driver]

            features = {
                "grid_position": row["grid_position"],
                "driver_skill": row["driver_skill"],
                "constructor_pace": row["constructor_pace"],
                "avg_finish": ds["avg_finish"],
                "win_rate": ds["win_rate"],
                "podium_rate": ds["podium_rate"],
                "dnf_rate": ds["dnf_rate"],
                "grid_finish_delta": ds["grid_finish_delta"],
                "avg_laptime": row["lap_time"],
                "safety_car": row["safety_car"],
                "pit_stops": row["pit_stops"],
                "tire_degradation": row["tire_degradation"],
            }
            feature_rows.append(features)
            targets.append(row["finish_position"])
            group_size += 1
        if group_size > 0:
            groups.append(group_size)

    X = pd.DataFrame(feature_rows)
    y = pd.Series(targets)
    return X, y, groups


def build_2026_prediction_features(grid_df: pd.DataFrame, driver_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Build prediction feature matrix for the 2026 Japanese GP drivers.
    Merges grid data with historical driver statistics.
    """
    rows = []
    for _, row in grid_df.iterrows():
        driver = row["driver"]

        if driver in driver_stats.index:
            ds = driver_stats.loc[driver]
            avg_finish = ds["avg_finish"]
            avg_gp_finish = ds["avg_gp_finish"]
            avg_sprint_finish = ds["avg_sprint_finish"]
            podium_rate = ds["podium_rate"]
            win_rate = ds["win_rate"]
            sprint_win_rate = ds["sprint_win_rate"]
            dnf_rate = ds["dnf_rate"]
            grid_finish_delta = ds["grid_finish_delta"]
            sh_avg_finish = ds["suzuka_avg_finish"]
            sh_podium_rate = ds["suzuka_podium_rate"]
        else:
            # Rookie defaults based on skill
            avg_finish = 12.0 - row["driver_skill"] * 5
            avg_gp_finish = avg_finish
            avg_sprint_finish = avg_finish
            podium_rate = max(0.0, row["driver_skill"] - 0.7)
            win_rate = max(0.0, row["driver_skill"] - 0.85)
            sprint_win_rate = win_rate
            dnf_rate = 0.05
            grid_finish_delta = 0.0
            sh_avg_finish = avg_finish
            sh_podium_rate = podium_rate

        rows.append({
            "driver": driver,
            "team": row["team"],
            "grid_position": row["grid_position"],
            "sprint_grid_position": row["sprint_grid_position"],
            "driver_skill": row["driver_skill"],
            "constructor_pace": row["constructor_pace"],
            "pace_score": row["pace_score"],
            "avg_finish": round(avg_finish, 3),
            "avg_gp_finish": round(avg_gp_finish, 3),
            "avg_sprint_finish": round(avg_sprint_finish, 3),
            "podium_rate": round(podium_rate, 4),
            "win_rate": round(win_rate, 4),
            "sprint_win_rate": round(sprint_win_rate, 4),
            "dnf_rate": round(dnf_rate, 4),
            "grid_finish_delta": round(grid_finish_delta, 3),
            "suzuka_avg_finish": round(sh_avg_finish, 3),
            "suzuka_podium_rate": round(sh_podium_rate, 4),
            "avg_laptime": 92.0 - row["pace_score"] * 4 + np.random.normal(0, 0.1),
            "pit_stops": 1,
            "tire_hard": 0,
            "tire_medium": 1,
            "tire_soft": 0,
            "weather_wet": 0,
            "safety_car": 0,
            "tire_degradation": 0.22,
            "overtaking_ability": row["overtaking_ability"],
            "consistency_score": row["consistency_score"],
            "suzuka_track_factor": row["suzuka_track_factor"],
        })

    return pd.DataFrame(rows)
