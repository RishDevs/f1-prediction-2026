# Central configuration for the F1 2026 Chinese GP Prediction Platform

DRIVERS = [
    {"name": "George Russell",    "code": "RUS", "team": "Mercedes",    "number": 63},
    {"name": "Kimi Antonelli",    "code": "ANT", "team": "Mercedes",    "number": 12},
    {"name": "Lando Norris",      "code": "NOR", "team": "McLaren",     "number": 4},
    {"name": "Oscar Piastri",     "code": "PIA", "team": "McLaren",     "number": 81},
    {"name": "Max Verstappen",    "code": "VER", "team": "Red Bull",    "number": 1},
    {"name": "Isack Hadjar",      "code": "HAD", "team": "Red Bull",    "number": 6},
    {"name": "Charles Leclerc",   "code": "LEC", "team": "Ferrari",     "number": 16},
    {"name": "Lewis Hamilton",    "code": "HAM", "team": "Ferrari",     "number": 44},
    {"name": "Fernando Alonso",   "code": "ALO", "team": "Aston Martin","number": 14},
    {"name": "Lance Stroll",      "code": "STR", "team": "Aston Martin","number": 18},
    {"name": "Esteban Ocon",      "code": "OCO", "team": "Haas",        "number": 31},
    {"name": "Oliver Bearman",    "code": "BEA", "team": "Haas",        "number": 87},
    {"name": "Pierre Gasly",      "code": "GAS", "team": "Alpine",      "number": 10},
    {"name": "Franco Colapinto",  "code": "COL", "team": "Alpine",      "number": 43},
    {"name": "Carlos Sainz",      "code": "SAI", "team": "Williams",    "number": 55},
    {"name": "Alexander Albon",   "code": "ALB", "team": "Williams",    "number": 23},
    {"name": "Nico Hulkenberg",   "code": "HUL", "team": "Audi",        "number": 27},
    {"name": "Gabriel Bortoleto", "code": "BOR", "team": "Audi",        "number": 5},
    {"name": "Liam Lawson",       "code": "LAW", "team": "Racing Bulls","number": 30},
    {"name": "Arvid Lindblad",    "code": "LIN", "team": "Racing Bulls","number": 20},
    {"name": "Valtteri Bottas",   "code": "BOT", "team": "Cadillac",    "number": 77},
    {"name": "Sergio Perez",      "code": "PER", "team": "Cadillac",    "number": 11},
]

TEAM_COLORS = {
    "Mercedes":     "#00D2BE",
    "McLaren":      "#FF8000",
    "Red Bull":     "#3671C6",
    "Ferrari":      "#E8002D",
    "Aston Martin": "#358C75",
    "Haas":         "#B6BABD",
    "Alpine":       "#FF87BC",
    "Williams":     "#37BEDD",
    "Audi":         "#F50537",
    "Racing Bulls": "#6692FF",
    "Cadillac":     "#FFCF00",
}

TEAM_SECONDARY_COLORS = {
    "Mercedes":     "#1a1a2e",
    "McLaren":      "#1a1a2e",
    "Red Bull":     "#1a1a2e",
    "Ferrari":      "#1a1a2e",
    "Aston Martin": "#1a1a2e",
    "Haas":         "#1a1a2e",
    "Alpine":       "#1a1a2e",
    "Williams":     "#1a1a2e",
    "Audi":         "#1a1a2e",
    "Racing Bulls": "#1a1a2e",
    "Cadillac":     "#1a1a2e",
}

# Constructor competitiveness index (0-1, 1 = best)
CONSTRUCTOR_PACE_2026 = {
    "Mercedes":     0.95,
    "McLaren":      0.97,
    "Red Bull":     0.91,
    "Ferrari":      0.89,
    "Aston Martin": 0.72,
    "Haas":         0.68,
    "Alpine":       0.65,
    "Williams":     0.70,
    "Audi":         0.55,
    "Racing Bulls": 0.73,
    "Cadillac":     0.50,
}

# Driver skill ratings (0-1, based on historical performance)
DRIVER_SKILL_2026 = {
    "George Russell":    0.93,
    "Kimi Antonelli":    0.80,
    "Lando Norris":      0.95,
    "Oscar Piastri":     0.90,
    "Max Verstappen":    0.97,
    "Isack Hadjar":      0.79,
    "Charles Leclerc":   0.91,
    "Lewis Hamilton":    0.92,
    "Fernando Alonso":   0.90,
    "Lance Stroll":      0.74,
    "Esteban Ocon":      0.77,
    "Oliver Bearman":    0.76,
    "Pierre Gasly":      0.79,
    "Franco Colapinto":  0.78,
    "Carlos Sainz":      0.88,
    "Alexander Albon":   0.82,
    "Nico Hulkenberg":   0.78,
    "Gabriel Bortoleto": 0.74,
    "Liam Lawson":       0.78,
    "Arvid Lindblad":    0.73,
    "Valtteri Bottas":   0.81,
    "Sergio Perez":      0.82,
}

# Shanghai International Circuit
CIRCUIT_INFO = {
    "name": "Shanghai International Circuit",
    "country": "China",
    "city": "Shanghai",
    "length_km": 5.451,
    "corners": 16,
    "race_laps": 56,
    "sprint_laps": 19,
    "race_distance_km": 305.066,
    "sprint_distance_km": 100.0,
    "drs_zones": 2,
    "overtaking_difficulty": 0.35,   # lower = easier to overtake
    "safety_car_probability": 0.42,
    "tire_degradation": "Medium-High",
    "lap_record": "1:32.238",
    "lap_record_holder": "Michael Schumacher",
    "lap_record_year": 2004,
    "first_gp": 2004,
    "surface": "Tarmac",
}

MODEL_PARAMS = {
    "lgbm_classifier": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "random_state": 42,
    },
    "lgbm_regressor": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "random_state": 42,
    },
    "xgboost": {
        "n_estimators": 200,
        "learning_rate": 0.08,
        "max_depth": 5,
        "random_state": 42,
    },
    "monte_carlo_runs": 10000,
}
