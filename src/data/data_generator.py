"""
data_generator.py
Generates realistic synthetic F1 historical data for model training (2018-2025).
Seeded from actual historical statistics to produce realistic distributions.
"""

import numpy as np
import pandas as pd
from ..config import DRIVERS, CONSTRUCTOR_PACE_2026, DRIVER_SKILL_2026

# ─────────────────────────────────────────────────────────────────────────────
# Historical constructor competitiveness by year (relative to 2026 values)
# ─────────────────────────────────────────────────────────────────────────────
HIST_CONSTRUCTOR_PACE = {
    2018: {"Mercedes": 0.99, "Ferrari": 0.95, "Red Bull": 0.93, "McLaren": 0.72,
           "Renault": 0.74, "Haas": 0.71, "Force India": 0.78, "Williams": 0.67,
           "Toro Rosso": 0.73, "Sauber": 0.65},
    2019: {"Mercedes": 0.99, "Ferrari": 0.96, "Red Bull": 0.92, "McLaren": 0.78,
           "Renault": 0.75, "Haas": 0.73, "Racing Point": 0.80, "Williams": 0.65,
           "Toro Rosso": 0.74, "Alfa Romeo": 0.70},
    2020: {"Mercedes": 0.99, "Red Bull": 0.94, "Racing Point": 0.88, "McLaren": 0.85,
           "Renault": 0.82, "Ferrari": 0.84, "AlphaTauri": 0.76, "Alfa Romeo": 0.71,
           "Haas": 0.69, "Williams": 0.64},
    2021: {"Mercedes": 0.97, "Red Bull": 0.99, "Ferrari": 0.83, "McLaren": 0.88,
           "AlphaTauri": 0.78, "Aston Martin": 0.85, "Alpine": 0.80, "Williams": 0.74,
           "Alfa Romeo": 0.72, "Haas": 0.65},
    2022: {"Red Bull": 0.99, "Ferrari": 0.96, "Mercedes": 0.90, "Alpine": 0.80,
           "McLaren": 0.82, "Alfa Romeo": 0.77, "Aston Martin": 0.78, "Haas": 0.73,
           "AlphaTauri": 0.76, "Williams": 0.72},
    2023: {"Red Bull": 0.99, "Mercedes": 0.91, "Ferrari": 0.90, "McLaren": 0.93,
           "Aston Martin": 0.88, "Alpine": 0.78, "Williams": 0.76, "AlphaTauri": 0.74,
           "Alfa Romeo": 0.73, "Haas": 0.72},
    2024: {"McLaren": 0.99, "Red Bull": 0.96, "Ferrari": 0.95, "Mercedes": 0.93,
           "Aston Martin": 0.79, "Racing Bulls": 0.78, "Haas": 0.75, "Williams": 0.74,
           "Sauber": 0.67, "Alpine": 0.72},
    2025: {"McLaren": 0.99, "Ferrari": 0.96, "Red Bull": 0.93, "Mercedes": 0.94,
           "Aston Martin": 0.76, "Racing Bulls": 0.77, "Haas": 0.73, "Williams": 0.72,
           "Sauber": 0.60, "Alpine": 0.68},
}

HISTORICAL_DRIVERS = {
    2018: [("Lewis Hamilton", "Mercedes"), ("Sebastian Vettel", "Ferrari"),
           ("Kimi Raikkonen", "Ferrari"), ("Valtteri Bottas", "Mercedes"),
           ("Max Verstappen", "Red Bull"), ("Daniel Ricciardo", "Red Bull"),
           ("Nico Hulkenberg", "Renault"), ("Sergio Perez", "Force India"),
           ("Fernando Alonso", "McLaren"), ("Carlos Sainz", "Renault"),
           ("Esteban Ocon", "Force India"), ("Charles Leclerc", "Sauber"),
           ("Romain Grosjean", "Haas"), ("Kevin Magnussen", "Haas"),
           ("Brendon Hartley", "Toro Rosso"), ("Pierre Gasly", "Toro Rosso"),
           ("Lance Stroll", "Williams"), ("Sergey Sirotkin", "Williams"),
           ("Marcus Ericsson", "Sauber"), ("Stoffel Vandoorne", "McLaren")],
    2019: [("Lewis Hamilton", "Mercedes"), ("Valtteri Bottas", "Mercedes"),
           ("Max Verstappen", "Red Bull"), ("Charles Leclerc", "Ferrari"),
           ("Sebastian Vettel", "Ferrari"), ("Pierre Gasly", "Red Bull"),
           ("Carlos Sainz", "McLaren"), ("Lando Norris", "McLaren"),
           ("Nico Hulkenberg", "Renault"), ("Daniel Ricciardo", "Renault"),
           ("Sergio Perez", "Racing Point"), ("Lance Stroll", "Racing Point"),
           ("Daniil Kvyat", "Toro Rosso"), ("Alexander Albon", "Toro Rosso"),
           ("Fernando Alonso", "McLaren"), ("Kimi Raikkonen", "Alfa Romeo"),
           ("Antonio Giovinazzi", "Alfa Romeo"), ("Romain Grosjean", "Haas"),
           ("Kevin Magnussen", "Haas"), ("George Russell", "Williams")],
    2020: [("Lewis Hamilton", "Mercedes"), ("Valtteri Bottas", "Mercedes"),
           ("Max Verstappen", "Red Bull"), ("Charles Leclerc", "Ferrari"),
           ("Sebastian Vettel", "Ferrari"), ("Sergio Perez", "Racing Point"),
           ("Lando Norris", "McLaren"), ("Carlos Sainz", "McLaren"),
           ("Pierre Gasly", "AlphaTauri"), ("Alexander Albon", "Red Bull"),
           ("Daniel Ricciardo", "Renault"), ("Esteban Ocon", "Renault"),
           ("Lance Stroll", "Racing Point"), ("George Russell", "Williams"),
           ("Daniil Kvyat", "AlphaTauri"), ("Kimi Raikkonen", "Alfa Romeo"),
           ("Antonio Giovinazzi", "Alfa Romeo"), ("Romain Grosjean", "Haas"),
           ("Kevin Magnussen", "Haas"), ("Nicholas Latifi", "Williams")],
    2021: [("Max Verstappen", "Red Bull"), ("Lewis Hamilton", "Mercedes"),
           ("Valtteri Bottas", "Mercedes"), ("Sergio Perez", "Red Bull"),
           ("Carlos Sainz", "Ferrari"), ("Charles Leclerc", "Ferrari"),
           ("Lando Norris", "McLaren"), ("Daniel Ricciardo", "McLaren"),
           ("Fernando Alonso", "Alpine"), ("Esteban Ocon", "Alpine"),
           ("Pierre Gasly", "AlphaTauri"), ("Yuki Tsunoda", "AlphaTauri"),
           ("Lance Stroll", "Aston Martin"), ("Sebastian Vettel", "Aston Martin"),
           ("George Russell", "Williams"), ("Nicholas Latifi", "Williams"),
           ("Kimi Raikkonen", "Alfa Romeo"), ("Antonio Giovinazzi", "Alfa Romeo"),
           ("Mick Schumacher", "Haas"), ("Nikita Mazepin", "Haas")],
    2022: [("Max Verstappen", "Red Bull"), ("Sergio Perez", "Red Bull"),
           ("Charles Leclerc", "Ferrari"), ("Carlos Sainz", "Ferrari"),
           ("George Russell", "Mercedes"), ("Lewis Hamilton", "Mercedes"),
           ("Esteban Ocon", "Alpine"), ("Lando Norris", "McLaren"),
           ("Valtteri Bottas", "Alfa Romeo"), ("Sebastian Vettel", "Aston Martin"),
           ("Fernando Alonso", "Alpine"), ("Lance Stroll", "Aston Martin"),
           ("Pierre Gasly", "AlphaTauri"), ("Yuki Tsunoda", "AlphaTauri"),
           ("Kevin Magnussen", "Haas"), ("Mick Schumacher", "Haas"),
           ("Daniel Ricciardo", "McLaren"), ("Nicholas Latifi", "Williams"),
           ("Alexander Albon", "Williams"), ("Zhou Guanyu", "Alfa Romeo")],
    2023: [("Max Verstappen", "Red Bull"), ("Sergio Perez", "Red Bull"),
           ("Fernando Alonso", "Aston Martin"), ("Lewis Hamilton", "Mercedes"),
           ("Carlos Sainz", "Ferrari"), ("George Russell", "Mercedes"),
           ("Charles Leclerc", "Ferrari"), ("Lando Norris", "McLaren"),
           ("Oscar Piastri", "McLaren"), ("Lance Stroll", "Aston Martin"),
           ("Pierre Gasly", "Alpine"), ("Esteban Ocon", "Alpine"),
           ("Alexander Albon", "Williams"), ("Logan Sargeant", "Williams"),
           ("Yuki Tsunoda", "AlphaTauri"), ("Liam Lawson", "AlphaTauri"),
           ("Kevin Magnussen", "Haas"), ("Nico Hulkenberg", "Haas"),
           ("Valtteri Bottas", "Alfa Romeo"), ("Zhou Guanyu", "Alfa Romeo")],
    2024: [("Max Verstappen", "Red Bull"), ("Lando Norris", "McLaren"),
           ("Charles Leclerc", "Ferrari"), ("Oscar Piastri", "McLaren"),
           ("Carlos Sainz", "Ferrari"), ("George Russell", "Mercedes"),
           ("Lewis Hamilton", "Mercedes"), ("Sergio Perez", "Red Bull"),
           ("Fernando Alonso", "Aston Martin"), ("Lance Stroll", "Aston Martin"),
           ("Pierre Gasly", "Alpine"), ("Esteban Ocon", "Alpine"),
           ("Alexander Albon", "Williams"), ("Franco Colapinto", "Williams"),
           ("Yuki Tsunoda", "Racing Bulls"), ("Liam Lawson", "Racing Bulls"),
           ("Kevin Magnussen", "Haas"), ("Nico Hulkenberg", "Haas"),
           ("Valtteri Bottas", "Sauber"), ("Zhou Guanyu", "Sauber")],
    2025: [("Lando Norris", "McLaren"), ("Max Verstappen", "Red Bull"),
           ("George Russell", "Mercedes"), ("Charles Leclerc", "Ferrari"),
           ("Lewis Hamilton", "Ferrari"), ("Oscar Piastri", "McLaren"),
           ("Kimi Antonelli", "Mercedes"), ("Carlos Sainz", "Williams"),
           ("Fernando Alonso", "Aston Martin"), ("Lance Stroll", "Aston Martin"),
           ("Pierre Gasly", "Alpine"), ("Jack Doohan", "Alpine"),
           ("Alexander Albon", "Williams"), ("Yuki Tsunoda", "Red Bull"),
           ("Liam Lawson", "Racing Bulls"), ("Isack Hadjar", "Racing Bulls"),
           ("Oliver Bearman", "Haas"), ("Esteban Ocon", "Haas"),
           ("Nico Hulkenberg", "Sauber"), ("Gabriel Bortoleto", "Sauber")],
}

CIRCUITS = [
    "Bahrain", "Saudi Arabia", "Australia", "Japan", "China",
    "Miami", "Emilia Romagna", "Monaco", "Canada", "Spain",
    "Austria", "Great Britain", "Hungary", "Belgium", "Netherlands",
    "Italy", "Azerbaijan", "Singapore", "USA", "Mexico",
    "Brazil", "Las Vegas", "Qatar", "Abu Dhabi"
]

DRIVER_SKILL_HIST = {
    "Lewis Hamilton": 0.97, "Sebastian Vettel": 0.94, "Max Verstappen": 0.96,
    "Charles Leclerc": 0.90, "Valtteri Bottas": 0.85, "Lando Norris": 0.92,
    "George Russell": 0.91, "Carlos Sainz": 0.88, "Oscar Piastri": 0.88,
    "Fernando Alonso": 0.90, "Kimi Raikkonen": 0.86, "Daniel Ricciardo": 0.84,
    "Sergio Perez": 0.83, "Pierre Gasly": 0.78, "Esteban Ocon": 0.76,
    "Yuki Tsunoda": 0.77, "Lance Stroll": 0.73, "Alexander Albon": 0.80,
    "Nicholas Latifi": 0.62, "Kevin Magnussen": 0.74, "Nico Hulkenberg": 0.77,
    "Mick Schumacher": 0.71, "Antonio Giovinazzi": 0.68, "Romain Grosjean": 0.72,
    "Zhou Guanyu": 0.68, "Logan Sargeant": 0.63, "Liam Lawson": 0.76,
    "Kimi Antonelli": 0.79, "Jack Doohan": 0.72, "Isack Hadjar": 0.72,
    "Oliver Bearman": 0.75, "Gabriel Bortoleto": 0.73, "Franco Colapinto": 0.72,
    "Daniil Kvyat": 0.74, "Brendon Hartley": 0.68, "Stoffel Vandoorne": 0.70,
    "Sergey Sirotkin": 0.61, "Marcus Ericsson": 0.65, "Nikita Mazepin": 0.60,
    "Valtteri Bottas": 0.85, "Robert Kubica": 0.65,
}


def _get_driver_skill(name):
    return DRIVER_SKILL_HIST.get(name, 0.68)


def _get_constructor_pace(team, year):
    hist = HIST_CONSTRUCTOR_PACE.get(year, {})
    return hist.get(team, 0.65)


def generate_historical_data(seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic F1 dataset for 2018-2025.

    Returns a DataFrame with one row per driver per race, comprising
    race metadata, performance features, and the finish position.
    """
    rng = np.random.default_rng(seed)
    records = []

    race_id = 0
    for year in range(2018, 2026):
        drivers_year = HISTORICAL_DRIVERS[year]
        n_drivers = len(drivers_year)
        circuits_year = CIRCUITS[:22] if year < 2022 else CIRCUITS

        for race_num, circuit in enumerate(circuits_year, 1):
            race_id += 1
            is_sprint = circuit in ["China", "Brazil", "Qatar", "Miami", "Belgium", "Japan"] and year >= 2021
            is_suzuka = circuit == "Japan"

            # Construct pace scores for each team this race (add small race-by-race variance)
            pace_scores = {}
            for driver_name, team in drivers_year:
                base_pace = _get_constructor_pace(team, year)
                driver_skill = _get_driver_skill(driver_name)
                pace = 0.6 * base_pace + 0.4 * driver_skill
                pace += rng.normal(0, 0.025)  # race-weekend variance
                pace_scores[driver_name] = np.clip(pace, 0.45, 1.0)

            # Simulate qualifying (faster pace = better grid position)
            sorted_drivers = sorted(pace_scores.items(), key=lambda x: x[1], reverse=True)
            qualy_noise = rng.normal(0, 1.5, n_drivers)   # swap positions by ~1-2 spots
            qualy_order = sorted(range(n_drivers), key=lambda i: -pace_scores[sorted_drivers[i][0]] + qualy_noise[i] * 0.02)

            grid_positions = {}
            for pos, idx in enumerate(qualy_order, 1):
                grid_positions[sorted_drivers[idx][0]] = pos

            # Race result: pace + luck
            race_pace = {d: pace_scores[d] + rng.normal(0, 0.04) for d, _ in drivers_year}
            dnf_mask = rng.random(n_drivers) < 0.035  # ~3.5% DNF per driver

            finish_order_raw = sorted(
                [(d, race_pace[d]) for d, _ in drivers_year],
                key=lambda x: -x[1]
            )
            finish_positions = {}
            dnf_pos = n_drivers
            dnf_count = 0
            assigned = n_drivers
            for i, (d, _) in enumerate(finish_order_raw):
                if dnf_mask[i]:
                    finish_positions[d] = assigned
                    assigned -= 1
                    dnf_count += 1
                else:
                    finish_positions[d] = i + 1 - dnf_count

            # Tire and pit stop simulation
            tire_choices = ["Soft", "Medium", "Hard"]
            tire_probs = [0.35, 0.40, 0.25] if is_sprint else [0.25, 0.45, 0.30]

            weather = rng.choice(["Dry", "Dry", "Dry", "Wet", "Wet"], p=[0.6, 0.12, 0.12, 0.08, 0.08])
            safety_car = rng.random() < 0.35

            for driver_name, team in drivers_year:
                grid = grid_positions[driver_name]
                finish = finish_positions[driver_name]
                driver_skill = _get_driver_skill(driver_name)
                constructor_pace = _get_constructor_pace(team, year)

                # Expected lap time relative to pole (higher pace score → closer to pole time)
                base_laptime = 90 + (1 - pace_scores[driver_name]) * 10 + rng.normal(0, 0.3)
                tire = rng.choice(tire_choices, p=tire_probs)
                pit_count = 0 if is_sprint else rng.choice([1, 2, 3], p=[0.55, 0.38, 0.07])
                stint_len = (19 if is_sprint else 56) / max(pit_count + 1, 1)
                tire_deg = rng.uniform(0.1, 0.4)

                records.append({
                    "race_id": race_id,
                    "season": year,
                    "circuit": circuit,
                    "race_num": race_num,
                    "is_sprint": is_sprint,
                    "is_suzuka": is_suzuka,
                    "driver": driver_name,
                    "team": team,
                    "grid_position": grid,
                    "driver_skill": driver_skill,
                    "constructor_pace": constructor_pace,
                    "lap_time": round(base_laptime, 3),
                    "tire_compound": tire,
                    "tire_degradation": round(tire_deg, 3),
                    "pit_stops": pit_count,
                    "stint_length": round(stint_len, 1),
                    "weather": weather,
                    "safety_car": int(safety_car),
                    "finish_position": finish,
                    "points": ([25,18,15,12,10,8,6,4,2,1]+[0]*15)[finish - 1] if finish <= 10 else 0,
                    "dnf": int(dnf_mask[drivers_year.index((driver_name, team))]),
                })

    return pd.DataFrame(records)


def generate_2026_grid(seed: int = 99) -> pd.DataFrame:
    """
    Generate the 2026 Japanese GP starting grid features for prediction.
    Incorporates simulated Free Practice 1 results for Suzuka.
    """
    rng = np.random.default_rng(seed)
    records = []

    try:
        import urllib.request
        import json
        import ssl
        from datetime import datetime, timezone
        
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        req = urllib.request.Request("https://api.openf1.org/v1/sessions?session_name=Practice%201")
        with urllib.request.urlopen(req, context=ctx) as response:
            sessions = json.loads(response.read().decode('utf-8'))
            
        now_iso = datetime.now(timezone.utc).isoformat()
        past_sessions = [s for s in sessions if s['date_start'] < now_iso]
        latest_session = sorted(past_sessions, key=lambda x: x['date_start'], reverse=True)[0]
        session_key = latest_session['session_key']
        
        req = urllib.request.Request(f"https://api.openf1.org/v1/laps?session_key={session_key}")
        with urllib.request.urlopen(req, context=ctx) as response:
            laps = json.loads(response.read().decode('utf-8'))
            
        req = urllib.request.Request(f"https://api.openf1.org/v1/drivers?session_key={session_key}")
        with urllib.request.urlopen(req, context=ctx) as response:
            drivers = json.loads(response.read().decode('utf-8'))
            
        driver_map = {d['driver_number']: d['full_name'] for d in drivers}
        
        best_laps = {}
        for lap in laps:
            if lap.get('lap_duration'):
                d_num = lap['driver_number']
                dur = lap['lap_duration']
                if d_num not in best_laps or dur < best_laps[d_num]:
                    best_laps[d_num] = dur
                    
        fp1_results = {}
        for d_num, dur in best_laps.items():
            if d_num in driver_map:
                name_parts = driver_map[d_num].split()
                formatted_name = " ".join([p.capitalize() for p in name_parts])
                fp1_results[formatted_name] = dur
                
    except Exception as e:
        print(f"Warning: OpenF1 API fetch failed: {e}. Falling back to defaults.")
        fp1_results = {}

    best_fp1 = min(fp1_results.values()) if fp1_results else 90.0

    # 2026 qualifying simulation for the Chinese GP, heavily weighted by FP1
    pace_scores = {}
    for d in DRIVERS:
        name = d["name"]
        team = d["team"]
        base_pace = CONSTRUCTOR_PACE_2026[team]
        skill = DRIVER_SKILL_2026[name]
        
        # Calculate pace from FP1 data (closer to best time = higher pace score)
        fp1_time = fp1_results.get(name, best_fp1 + 3.0) 
        time_delta = fp1_time - best_fp1
        
        # Convert delta to a pace boost (max 1.0 for pole)
        fp1_pace_factor = max(0.4, 1.0 - (time_delta / 8.0))
        
        # Blend base stats with real FP1 practice data (70% weight to real data)
        pace = 0.3 * (0.55 * base_pace + 0.45 * skill) + 0.7 * fp1_pace_factor
        pace += rng.normal(0, 0.01) # Small quali variance
        
        pace_scores[name] = np.clip(pace, 0.50, 1.0)

    sorted_drivers = sorted(pace_scores.items(), key=lambda x: x[1], reverse=True)
    for pos, (name, pace) in enumerate(sorted_drivers, 1):
        driver_info = next(d for d in DRIVERS if d["name"] == name)
        team = driver_info["team"]
        skill = DRIVER_SKILL_2026[name]
        con_pace = CONSTRUCTOR_PACE_2026[team]

        # Historical Suzuka track performance factor
        suzuka_factor = rng.uniform(0.85, 1.0)
        if name in ["Lewis Hamilton", "Max Verstappen", "Fernando Alonso"]:
            suzuka_factor = rng.uniform(0.93, 1.0)

        records.append({
            "driver": name,
            "code": driver_info["code"],
            "team": team,
            "grid_position": pos,
            "sprint_grid_position": pos,
            "driver_skill": skill,
            "constructor_pace": con_pace,
            "pace_score": round(pace, 4),
            "suzuka_track_factor": round(suzuka_factor, 3),
            "tire_compound": rng.choice(["Soft", "Medium"], p=[0.6, 0.4]),
            "weather": "Dry",
            "safety_car_prob": 0.42,
            "overtaking_ability": np.clip(skill + rng.normal(0, 0.03), 0.5, 1.0),
            "consistency_score": np.clip(skill - rng.uniform(0.0, 0.08), 0.5, 1.0),
            "fp1_time": round(fp1_results.get(name, 0), 3)
        })

    return pd.DataFrame(records)
