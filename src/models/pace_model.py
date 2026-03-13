"""
pace_model.py
LightGBM regressor predicting expected race lap pace (seconds).
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from ..config import MODEL_PARAMS


PACE_FEATURES = [
    "grid_position",
    "driver_skill",
    "constructor_pace",
    "avg_finish",
    "podium_rate",
    "win_rate",
    "dnf_rate",
    "avg_laptime",
    "tire_hard",
    "tire_medium",
    "tire_soft",
    "weather_wet",
    "safety_car",
    "tire_degradation",
    "shanghai_track_factor",
]


class RacePaceModel:
    """
    LightGBM regressor that predicts the expected average race lap time
    (in seconds) for each driver at the Shanghai International Circuit.
    Lower lap time = faster pace = better expected race outcome.
    """

    def __init__(self):
        params = MODEL_PARAMS["lgbm_regressor"]
        self.model = LGBMRegressor(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            num_leaves=params["num_leaves"],
            random_state=params["random_state"],
            verbose=-1,
        )
        self.feature_cols = PACE_FEATURES
        self.is_fitted = False
        self.feature_importance_ = {}
        self.val_mae_ = None

    def fit(self, X: pd.DataFrame, y_laptime: pd.Series):
        cols = [c for c in self.feature_cols if c in X.columns]
        X_sub = X[cols].fillna(X[cols].median())
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_sub, y_laptime, test_size=0.15, random_state=42
        )
        self.model.fit(X_tr, y_tr)
        self.is_fitted = True

        preds = self.model.predict(X_val)
        self.val_mae_ = mean_absolute_error(y_val, preds)

        if hasattr(self.model, "feature_importances_"):
            self.feature_importance_ = dict(zip(cols, self.model.feature_importances_))
        return self

    def predict_pace(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted lap times in seconds for each row."""
        cols = [c for c in self.feature_cols if c in X.columns]

        # Fill missing features with sensible defaults
        X_sub = X.copy()
        for c in self.feature_cols:
            if c not in X_sub.columns:
                X_sub[c] = 0.0

        X_sub = X_sub[self.feature_cols].fillna(0)
        return self.model.predict(X_sub)

    def predict_race_pace(self, pred_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict expected race lap time for each driver.
        Returns a DataFrame with driver, predicted lap time, and pace rank.
        """
        pace_sec = self.predict_pace(pred_df)
        result = pred_df[["driver", "team"]].copy()
        result["predicted_laptime"] = np.round(pace_sec, 3)
        result["laptime_delta"] = np.round(pace_sec - pace_sec.min(), 3)
        result = result.sort_values("predicted_laptime").reset_index(drop=True)
        result["pace_rank"] = range(1, len(result) + 1)
        return result
