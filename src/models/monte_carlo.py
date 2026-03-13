"""
monte_carlo.py
Monte Carlo Race Simulator for the 2026 Chinese Grand Prix.
Runs N simulated races by sampling from pace distributions and applying
stochastic events (safety cars, DNFs, overtakes, tire degradation).
"""

import numpy as np
import pandas as pd
from ..config import CIRCUIT_INFO


class MonteCarloSimulator:
    """
    Simulates 2026 Chinese GP races using Monte Carlo sampling.

    Each simulation:
    1. Samples per-driver lap pace from a normal distribution
    2. Applies tire degradation over stint length
    3. Adds pit stop time losses
    4. Includes safety car events (neutralize gaps)
    5. Simulates DNFs probabilistically
    6. Returns the finishing order

    Results are aggregated over N_RUNS to produce win/podium probabilities.
    """

    def __init__(self, n_runs: int = 10_000, seed: int = 0):
        self.n_runs = n_runs
        self.seed = seed
        self.race_laps = CIRCUIT_INFO["race_laps"]
        self.sprint_laps = CIRCUIT_INFO["sprint_laps"]
        self.sc_prob = CIRCUIT_INFO["safety_car_probability"]
        self.rng = np.random.default_rng(seed)

    def _simulate_race(
        self,
        pace_means: np.ndarray,        # base lap time (s) per driver
        pace_stds: np.ndarray,         # lap time std dev per driver
        tire_degs: np.ndarray,         # tire degradation per driver (s/lap)
        pit_stops: np.ndarray,         # expected pit count per driver
        dnf_probs: np.ndarray,         # per-driver DNF probability for this race
        total_laps: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Returns the finishing order (0-indexed) for a single simulated race.
        """
        n = len(pace_means)

        # Sample pit stop timing (fraction through the race)
        pit_lap_fracs = []
        for ps in pit_stops:
            if ps <= 0:
                pit_lap_fracs.append([])
            else:
                fracs = sorted(rng.uniform(0.25, 0.75, int(ps)))
                pit_lap_fracs.append(fracs)

        total_times = np.zeros(n)
        current_tire_age = np.zeros(n)
        pit_done = [0] * n

        # Safety car event
        sc_lap = None
        if rng.random() < self.sc_prob:
            sc_lap = int(rng.integers(5, total_laps - 5))
            sc_duration = int(rng.integers(3, 8))  # laps under SC

        for lap in range(1, total_laps + 1):
            lap_times = rng.normal(pace_means, pace_stds)
            lap_times = np.clip(lap_times, pace_means * 0.96, pace_means * 1.08)

            # Tire degradation
            lap_times += tire_degs * current_tire_age
            current_tire_age += 1

            # Pit stops
            for i in range(n):
                pd_fracs = pit_lap_fracs[i]
                if pit_done[i] < len(pd_fracs):
                    pit_frac = pd_fracs[pit_done[i]]
                    if lap / total_laps >= pit_frac:
                        lap_times[i] += rng.uniform(20, 28)   # pit stop time loss
                        current_tire_age[i] = 0
                        pit_done[i] += 1

            # Safety car: all cars converge (no meaningful gap gains)
            if sc_lap is not None and sc_lap <= lap < sc_lap + sc_duration:
                sc_laptime = np.max(lap_times) * 1.35
                lap_times = np.full(n, sc_laptime) + rng.normal(0, 0.05, n)

            total_times += lap_times

        # Apply DNFs: assign penalty time
        dnf_mask = rng.random(n) < dnf_probs
        total_times[dnf_mask] += 1e6

        # Return finishing order (argsort ascending = fastest first)
        return np.argsort(total_times)

    def run_grand_prix(
        self,
        pred_df: pd.DataFrame,
        pace_df: pd.DataFrame,
        strategy_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Run N_RUNS Grand Prix simulations.

        Args:
            pred_df      : 2026 grid DataFrame with driver/team/skill columns
            pace_df      : Per-driver predicted lap times (from RacePaceModel)
            strategy_df  : Per-driver strategy predictions (from RaceStrategyModel)

        Returns:
            DataFrame with win_prob, podium_prob per driver (sorted by win_prob desc)
        """
        drivers = pred_df["driver"].tolist()
        n = len(drivers)

        # Merge predictions
        merged = pred_df.merge(pace_df[["driver", "predicted_laptime"]], on="driver")
        merged = merged.merge(
            strategy_df[["driver", "pred_pit_stops", "pred_tire_degradation"]],
            on="driver"
        )

        pace_means = merged["predicted_laptime"].values
        pace_stds = np.clip(pace_means * 0.003 + merged["driver_skill"].values * (-0.05) + 0.8, 0.3, 1.5)
        tire_degs = merged["pred_tire_degradation"].values * 0.08  # s/lap
        pit_stops = merged["pred_pit_stops"].values.astype(int)
        dnf_probs = np.clip(0.04 - merged["driver_skill"].values * 0.02, 0.005, 0.06)

        win_counts = np.zeros(n, dtype=int)
        podium_counts = np.zeros(n, dtype=int)

        local_rng = np.random.default_rng(self.seed)
        for _ in range(self.n_runs):
            order = self._simulate_race(
                pace_means, pace_stds, tire_degs, pit_stops, dnf_probs,
                self.race_laps, local_rng
            )
            win_counts[order[0]] += 1
            for podium_idx in order[:3]:
                podium_counts[podium_idx] += 1

        result = pred_df[["driver", "team"]].copy()
        result["win_probability"] = np.round(win_counts / self.n_runs * 100, 1)
        result["podium_probability"] = np.round(podium_counts / self.n_runs * 100, 1)
        result = result.sort_values("win_probability", ascending=False).reset_index(drop=True)
        result["mc_rank"] = range(1, len(result) + 1)
        return result

    def run_sprint(
        self,
        pred_df: pd.DataFrame,
        sprint_proba_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Run N_RUNS Sprint Race simulations (shorter, fewer laps, no pit stops typical).
        Uses sprint_proba as base pace proxy.

        Returns DataFrame with sprint_win_prob per driver sorted descending.
        """
        drivers = pred_df["driver"].tolist()
        n = len(drivers)

        merged = pred_df.merge(
            sprint_proba_df[["driver", "sprint_win_probability"]],
            on="driver"
        )

        # Convert win probability to a pace proxy (higher prob = lower lap time)
        prob = merged["sprint_win_probability"].values
        pace_means = 88 + (1 - prob / 100) * 6 + merged["grid_position"].values * 0.05
        pace_stds = np.clip(merged["driver_skill"].values * (-0.3) + 1.1, 0.3, 1.0)
        tire_degs = np.full(n, 0.04)
        pit_stops = np.zeros(n, dtype=int)      # typically no pit stops in sprint
        dnf_probs = np.clip(0.02 - merged["driver_skill"].values * 0.01, 0.003, 0.03)

        win_counts = np.zeros(n, dtype=int)
        podium_counts = np.zeros(n, dtype=int)

        local_rng = np.random.default_rng(self.seed + 1)
        for _ in range(self.n_runs):
            order = self._simulate_race(
                pace_means, pace_stds, tire_degs, pit_stops, dnf_probs,
                self.sprint_laps, local_rng
            )
            win_counts[order[0]] += 1
            for podium_idx in order[:3]:
                podium_counts[podium_idx] += 1

        result = pred_df[["driver", "team"]].copy()
        result["sprint_win_probability"] = np.round(win_counts / self.n_runs * 100, 1)
        result["sprint_podium_probability"] = np.round(podium_counts / self.n_runs * 100, 1)
        result = result.sort_values("sprint_win_probability", ascending=False).reset_index(drop=True)
        result["sprint_mc_rank"] = range(1, len(result) + 1)
        return result
