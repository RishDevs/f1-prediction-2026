"""
strategy_model.py
XGBoost model predicting race strategy variables: pit stop count,
average stint length, and tire degradation for use in the Monte Carlo simulator.
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

from ..config import MODEL_PARAMS


STRATEGY_FEATURES = [
    "grid_position",
    "driver_skill",
    "constructor_pace",
    "avg_finish",
    "dnf_rate",
    "avg_laptime",
    "tire_degradation",
    "safety_car",
    "weather_wet",
]


class RaceStrategyModel:
    """
    XGBoost multi-output regressor that forecasts per-driver race strategy:
    - pit_stops: expected number of pit stops
    - stint_length: average stint length in laps
    - tire_deg: expected tire degradation rate
    """

    def __init__(self):
        params = MODEL_PARAMS["xgboost"]
        base = XGBRegressor(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            random_state=params["random_state"],
            verbosity=0,
        )
        self.model = MultiOutputRegressor(base)
        self.feature_cols = STRATEGY_FEATURES
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, hist_df: pd.DataFrame):
        """
        Fit strategy model from historical GP data.
        Targets: pit_stops, stint_length, tire_degradation
        """
        cols = [c for c in self.feature_cols if c in X.columns]
        X_sub = X[cols].fillna(X[cols].median())

        # Build target vector from historical data (only GP races)
        gp_df = hist_df[hist_df["is_sprint"] == False]
        y = gp_df[["pit_stops", "stint_length", "tire_degradation"]].reset_index(drop=True)

        # Align lengths (X may be slightly different due to driver stats join)
        min_len = min(len(X_sub), len(y))
        X_sub = X_sub.iloc[:min_len]
        y = y.iloc[:min_len]

        X_tr, X_val, y_tr, y_val = train_test_split(X_sub, y, test_size=0.15, random_state=42)
        self.model.fit(X_tr, y_tr)
        self.is_fitted = True
        return self

    def predict_strategy(self, pred_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict race strategy for each driver in the 2026 Chinese GP grid.
        Returns a DataFrame with predicted strategy variables.
        """
        cols = [c for c in self.feature_cols if c in pred_df.columns]
        X_sub = pred_df.copy()
        for c in self.feature_cols:
            if c not in X_sub.columns:
                X_sub[c] = 0.0
        X_sub = X_sub[self.feature_cols].fillna(0)

        preds = self.model.predict(X_sub)
        result = pred_df[["driver", "team"]].copy()
        result["pred_pit_stops"] = np.clip(np.round(preds[:, 0]).astype(int), 1, 3)
        result["pred_stint_length"] = np.clip(np.round(preds[:, 1], 1), 10, 40)
        result["pred_tire_degradation"] = np.clip(preds[:, 2], 0.05, 0.50)
        return result
