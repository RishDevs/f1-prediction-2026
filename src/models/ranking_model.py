"""
ranking_model.py
LightGBM LambdaRank (learning-to-rank) model for full finishing order prediction.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit

from ..config import MODEL_PARAMS


RANK_FEATURES = [
    "grid_position",
    "driver_skill",
    "constructor_pace",
    "avg_finish",
    "win_rate",
    "podium_rate",
    "dnf_rate",
    "grid_finish_delta",
    "avg_laptime",
    "safety_car",
    "pit_stops",
    "tire_degradation",
]


class DriverRankingModel:
    """
    LightGBM LambdaRank model that predicts the full finishing order
    for the Grand Prix. Uses pairwise learning-to-rank objective.
    """

    def __init__(self):
        params = MODEL_PARAMS["lgbm_regressor"]
        self.params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [3, 5, 10],
            "learning_rate": params["learning_rate"],
            "num_leaves": 31,
            "max_depth": 6,
            "n_estimators": 300,
            "random_state": params["random_state"],
            "verbose": -1,
        }
        self.model = None
        self.feature_cols = RANK_FEATURES
        self.is_fitted = False
        self.feature_importance_ = {}

    def fit(self, X: pd.DataFrame, y_positions: pd.Series, groups: list):
        """
        X: feature matrix
        y_positions: finishing positions (lower = better)
        groups: list of group sizes (one per race)
        """
        self.feature_cols = [c for c in self.feature_cols if c in X.columns]
        X_sub = X[self.feature_cols].fillna(X[self.feature_cols].median())

        # LambdaRank needs relevance scores (higher = better result)
        # Convert positions so that 1st=20, 20th=1 etc.
        max_pos = y_positions.max()
        relevance = (max_pos + 1) - y_positions  # flip: winner gets highest relevance

        train_data = lgb.Dataset(
            X_sub.values, label=relevance.values, group=groups
        )

        lgb_params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [3, 5, 10],
            "learning_rate": self.params["learning_rate"],
            "num_leaves": 31,
            "max_depth": 6,
            "verbose": -1,
            "seed": 42,
        }

        self.model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=300,
        )
        self.is_fitted = True
        self.feature_importance_ = dict(zip(
            self.feature_cols, self.model.feature_importance(importance_type="gain")
        ))
        return self

    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Return relevance scores. Higher score = expected better finish."""
        cols = [c for c in self.feature_cols if c in X.columns]
        X_sub = X.copy()
        for c in self.feature_cols:
            if c not in X_sub.columns:
                X_sub[c] = 0.0
        X_sub = X_sub[self.feature_cols].fillna(0)
        return self.model.predict(X_sub)

    def predict_finishing_order(self, pred_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict full finishing order for the 2026 Japanese GP.
        Returns a DataFrame ranked from best to worst expected finish.
        """
        scores = self.predict_scores(pred_df)
        result = pred_df[["driver", "team"]].copy()
        result["rank_score"] = scores
        result = result.sort_values("rank_score", ascending=False).reset_index(drop=True)
        result["predicted_position"] = range(1, len(result) + 1)
        return result
