"""
sprint_model.py
LightGBM classifier for Sprint Race win probability prediction.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

from ..config import MODEL_PARAMS


SPRINT_FEATURES = [
    "sprint_grid_position",
    "driver_skill",
    "constructor_pace",
    "avg_sprint_finish",
    "sprint_win_rate",
    "podium_rate",
    "dnf_rate",
    "grid_finish_delta",
    "overtaking_ability",
    "consistency_score",
    "tire_soft",
    "tire_medium",
    "weather_wet",
]


class SprintPredictionModel:
    """
    LightGBM-based classifier that estimates each driver's probability
    of winning a Sprint Race given pre-race features.
    """

    def __init__(self):
        params = MODEL_PARAMS["lgbm_classifier"]
        base = LGBMClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            num_leaves=params["num_leaves"],
            random_state=params["random_state"],
            class_weight="balanced",
            verbose=-1,
        )
        self.model = CalibratedClassifierCV(base, method="isotonic", cv=3)
        self.feature_cols = SPRINT_FEATURES
        self.is_fitted = False
        self.feature_importance_ = {}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        cols = [c for c in self.feature_cols if c in X.columns]
        X_sub = X[cols].fillna(X[cols].median())
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_sub, y, test_size=0.15, random_state=42, stratify=y if y.sum() > 1 else None
        )
        self.model.fit(X_tr, y_tr)
        self.is_fitted = True

        # Extract feature importance from inner estimator
        inner = self.model.calibrated_classifiers_[0].estimator
        if hasattr(inner, "feature_importances_"):
            self.feature_importance_ = dict(zip(cols, inner.feature_importances_))

        return self

    def predict_win_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability of winning for each row."""
        cols = [c for c in self.feature_cols if c in X.columns]
        X_sub = X[cols].fillna(0)
        proba = self.model.predict_proba(X_sub)
        if proba.shape[1] == 2:
            return proba[:, 1]
        return proba[:, 0]

    def predict_sprint_probabilities(self, pred_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict sprint win probabilities for the 2026 Chinese GP grid.
        Returns a DataFrame with drivers and their normalised win probabilities.
        """
        raw_proba = self.predict_win_proba(pred_df)

        # Grid position correction: front starters have higher base probability
        grid_bonus = np.exp(-0.18 * (pred_df["sprint_grid_position"].values - 1))
        adjusted = raw_proba * (0.7 + 0.3 * grid_bonus)

        # Normalise to sum to 1
        total = adjusted.sum()
        if total > 0:
            adjusted = adjusted / total

        result = pred_df[["driver", "team"]].copy()
        result["sprint_win_probability"] = np.round(adjusted * 100, 1)
        result = result.sort_values("sprint_win_probability", ascending=False).reset_index(drop=True)
        result["sprint_rank"] = range(1, len(result) + 1)
        return result
