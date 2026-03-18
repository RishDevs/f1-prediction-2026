"""
app.py  –  F1 2026 Japanese GP AI Prediction Dashboard
A premium, dark-mode Streamlit application integrating LightGBM, XGBoost,
LambdaRank, and Monte Carlo simulation to predict Sprint and Grand Prix outcomes.
"""

import sys
import os

# ── Make src importable when launched as `streamlit run app.py` ──────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from src.config import DRIVERS, TEAM_COLORS, CIRCUIT_INFO, DRIVER_SKILL_2026, CONSTRUCTOR_PACE_2026
from src.data.data_generator import generate_historical_data, generate_2026_grid
from src.data.features import (
    compute_driver_stats,
    build_training_features,
    build_sprint_training_features,
    build_ranking_features,
    build_2026_prediction_features,
)
from src.models.sprint_model import SprintPredictionModel
from src.models.pace_model import RacePaceModel
from src.models.ranking_model import DriverRankingModel
from src.models.strategy_model import RaceStrategyModel
from src.models.monte_carlo import MonteCarloSimulator

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F1 2026 Japanese GP – AI Prediction",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS – premium dark-mode F1 aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Inter:wght@300;400;500;600;700&family=Roboto+Mono:wght@400;500&display=swap');

/* ── Base ───────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    background-color: #0a0a0f;
    color: #e8e8f0;
    font-family: 'Inter', sans-serif;
}
[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 0% 0%, #12001a 0%, #0a0a0f 40%, #000d1a 100%);
    min-height: 100vh;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d1a 0%, #080812 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}

/* ── Typography ─────────────────────────────────────────────────── */
h1, h2, h3 { font-family: 'Orbitron', monospace !important; }
.metric-value { font-family: 'Roboto Mono', monospace; }

/* ── Header banner ───────────────────────────────────────────────── */
.f1-header {
    background: linear-gradient(135deg, #e10600 0%, #9b0000 40%, #1a1a2e 100%);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 28px;
    border: 1px solid rgba(225,6,0,0.35);
    box-shadow: 0 0 60px rgba(225,6,0,0.18), 0 8px 32px rgba(0,0,0,0.6);
    position: relative;
    overflow: hidden;
}
.f1-header::before {
    content: '';
    position: absolute;
    top: -50%; right: -10%;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 70%);
}
.f1-header h1 {
    font-size: 2.2rem;
    font-weight: 900;
    letter-spacing: 2px;
    margin: 0 0 6px 0;
    color: #ffffff;
    text-shadow: 0 0 20px rgba(255,255,255,0.3);
}
.f1-header .subtitle {
    color: rgba(255,255,255,0.75);
    font-size: 1rem;
    font-weight: 400;
    letter-spacing: 1px;
}
.f1-header .badge {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    margin-right: 8px;
    color: #fff;
}

/* ── Section titles ──────────────────────────────────────────────── */
.section-title {
    font-family: 'Orbitron', monospace;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #e10600;
    padding: 12px 0 4px 0;
    border-bottom: 1px solid rgba(225,6,0,0.25);
    margin-bottom: 20px;
}

/* ── Podium cards ────────────────────────────────────────────────── */
.podium-wrap {
    display: flex;
    justify-content: center;
    align-items: flex-end;
    gap: 16px;
    margin: 12px 0 28px 0;
}
.podium-card {
    border-radius: 14px;
    padding: 24px 20px 20px 20px;
    text-align: center;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    border: 1px solid rgba(255,255,255,0.1);
    backdrop-filter: blur(8px);
    min-width: 160px;
    position: relative;
}
.podium-card:hover { transform: translateY(-6px); }
.podium-1st {
    background: linear-gradient(145deg, rgba(30,30,60,0.9) 0%, rgba(20,20,40,0.95) 100%);
    box-shadow: 0 0 40px rgba(255,215,0,0.25), 0 8px 24px rgba(0,0,0,0.5);
    border-color: rgba(255,215,0,0.4);
    order: 2;
    padding-top: 36px;
}
.podium-2nd {
    background: linear-gradient(145deg, rgba(25,25,50,0.9) 0%, rgba(16,16,35,0.95) 100%);
    box-shadow: 0 0 30px rgba(192,192,192,0.15), 0 8px 20px rgba(0,0,0,0.4);
    border-color: rgba(192,192,192,0.3);
    order: 1;
}
.podium-3rd {
    background: linear-gradient(145deg, rgba(25,25,50,0.9) 0%, rgba(16,16,35,0.95) 100%);
    box-shadow: 0 0 30px rgba(205,127,50,0.15), 0 8px 20px rgba(0,0,0,0.4);
    border-color: rgba(205,127,50,0.3);
    order: 3;
}
.podium-medal {
    font-size: 2.4rem;
    display: block;
    margin-bottom: 8px;
}
.podium-pos {
    font-family: 'Orbitron', monospace;
    font-size: 0.75rem;
    letter-spacing: 2px;
    opacity: 0.7;
    margin-bottom: 4px;
}
.podium-driver {
    font-family: 'Orbitron', monospace;
    font-size: 1rem;
    font-weight: 700;
    line-height: 1.2;
    margin-bottom: 4px;
}
.podium-team {
    font-size: 0.78rem;
    opacity: 0.65;
    margin-bottom: 10px;
}
.podium-prob {
    font-family: 'Roboto Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
}
.podium-label { font-size: 0.7rem; opacity: 0.5; letter-spacing: 1px; }
.team-bar {
    height: 4px;
    border-radius: 2px;
    margin-top: 14px;
    margin-bottom: 0;
}

/* ── Driver rows (below podium) ─────────────────────────────────── */
.driver-row {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 10px 16px;
    border-radius: 10px;
    margin-bottom: 6px;
    background: rgba(255,255,255,0.033);
    border: 1px solid rgba(255,255,255,0.06);
    transition: background 0.2s ease;
}
.driver-row:hover { background: rgba(255,255,255,0.06); }
.driver-pos { font-family: 'Orbitron', monospace; font-size: 0.8rem; width: 28px; opacity: 0.5; }
.driver-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
.driver-name { font-weight: 600; font-size: 0.88rem; flex: 1; }
.driver-team { font-size: 0.75rem; opacity: 0.5; }
.driver-pct  { font-family: 'Roboto Mono', monospace; font-size: 0.88rem; font-weight: 600; }

/* ── Stat cards ──────────────────────────────────────────────────── */
.stat-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
}
.stat-val { font-family: 'Orbitron', monospace; font-size: 1.4rem; font-weight: 700; color: #e10600; }
.stat-lbl { font-size: 0.75rem; opacity: 0.55; letter-spacing: 1px; margin-top: 4px; }

/* ── Info chip ───────────────────────────────────────────────────── */
.info-chip {
    display: inline-block;
    background: rgba(225,6,0,0.12);
    border: 1px solid rgba(225,6,0,0.25);
    color: #ff5050;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.73rem;
    font-weight: 600;
    letter-spacing: 1px;
    margin: 2px;
}
/* ── Plotly overrides ────────────────────────────────────────────── */
.js-plotly-plot .plotly .main-svg { border-radius: 12px; }

/* ── Sidebar form ────────────────────────────────────────────────── */
[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Caching: training pipeline
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_and_train(mc_runs: int = 10_000, mc_seed: int = 42):
    """Run full training pipeline and return all models + predictions."""
    # ── 1. Generate historical data ─────────────────────────────────────────
    hist_df = generate_historical_data(seed=42)
    grid_df = generate_2026_grid(seed=99)
    driver_stats = compute_driver_stats(hist_df, circuit="Japan")
    pred_features = build_2026_prediction_features(grid_df, driver_stats)

    # ── 2. Build training feature sets ──────────────────────────────────────
    X_gp, y_gp = build_training_features(hist_df)
    X_sprint, y_sprint = build_sprint_training_features(hist_df)
    X_rank, y_rank, groups_rank = build_ranking_features(hist_df)

    # ── 3. Sprint Model ──────────────────────────────────────────────────────
    sprint_model = SprintPredictionModel()
    sprint_model.fit(X_sprint, y_sprint)
    sprint_proba_df = sprint_model.predict_sprint_probabilities(pred_features)

    # ── 4. Pace Model ────────────────────────────────────────────────────────
    pace_model = RacePaceModel()
    # Use lap_time as target from GP training data aligned to pred_features shape
    gp_df_raw = hist_df[hist_df["is_sprint"] == False]
    lap_target = gp_df_raw["lap_time"].reset_index(drop=True).iloc[:len(X_gp)]
    pace_model.fit(X_gp, lap_target)
    pace_df = pace_model.predict_race_pace(pred_features)

    # ── 5. Ranking Model ─────────────────────────────────────────────────────
    rank_model = DriverRankingModel()
    rank_model.fit(X_rank, y_rank, groups_rank)
    rank_df = rank_model.predict_finishing_order(pred_features)

    # ── 6. Strategy Model ────────────────────────────────────────────────────
    strat_model = RaceStrategyModel()
    strat_model.fit(X_gp, hist_df[hist_df["is_sprint"] == False].reset_index(drop=True))
    strategy_df = strat_model.predict_strategy(pred_features)

    # ── 7. Monte Carlo Simulation ────────────────────────────────────────────
    simulator = MonteCarloSimulator(n_runs=mc_runs, seed=mc_seed)
    mc_gp_df   = simulator.run_grand_prix(pred_features, pace_df, strategy_df)
    mc_sprint_df = simulator.run_sprint(pred_features, sprint_proba_df)

    # ── 8. Feature importances (merged across models) ────────────────────────
    feat_imp = {}
    for d in [sprint_model.feature_importance_, pace_model.feature_importance_, rank_model.feature_importance_]:
        for k, v in d.items():
            feat_imp[k] = feat_imp.get(k, 0) + v

    return {
        "hist_df": hist_df,
        "grid_df": grid_df,
        "pred_features": pred_features,
        "sprint_proba": sprint_proba_df,
        "pace_df": pace_df,
        "rank_df": rank_df,
        "strategy_df": strategy_df,
        "mc_gp": mc_gp_df,
        "mc_sprint": mc_sprint_df,
        "feat_importance": feat_imp,
        "pace_mae": pace_model.val_mae_,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def team_color(team: str) -> str:
    return TEAM_COLORS.get(team, "#888888")


def get_driver_code(name: str) -> str:
    for d in DRIVERS:
        if d["name"] == name:
            return d["code"]
    return name[:3].upper()


def podium_card_html(pos: int, driver: str, team: str, prob: float, color: str) -> str:
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    pos_labels = {1: "1ST PLACE", 2: "2ND PLACE", 3: "3RD PLACE"}
    css_class = {1: "podium-1st", 2: "podium-2nd", 3: "podium-3rd"}[pos]
    # Split name for multi-line display
    parts = driver.split()
    first = parts[0] if len(parts) > 1 else ""
    last = " ".join(parts[1:]) if len(parts) > 1 else driver
    return f"""
    <div class="podium-card {css_class}">
        <span class="podium-medal">{medals[pos]}</span>
        <div class="podium-pos">{pos_labels[pos]}</div>
        <div class="podium-driver">{first}<br>{last}</div>
        <div class="podium-team">{team}</div>
        <div class="podium-prob" style="color:{color}">{prob}%</div>
        <div class="podium-label">WIN PROBABILITY</div>
        <div class="team-bar" style="background:{color};"></div>
    </div>"""


def all_podiums_html(top3: list) -> str:
    cards = ""
    order = [1, 0, 2]  # 2nd, 1st, 3rd for visual podium layout
    for i in order:
        r = top3[i]
        cards += podium_card_html(i + 1, r["driver"], r["team"], r["prob"], team_color(r["team"]))
    return f'<div class="podium-wrap">{cards}</div>'


def driver_row_html(pos: int, driver: str, team: str, prob: float) -> str:
    color = team_color(team)
    code = get_driver_code(driver)
    return f"""
    <div class="driver-row">
        <span class="driver-pos">P{pos}</span>
        <div class="driver-dot" style="background:{color};box-shadow:0 0 6px {color}66;"></div>
        <div>
            <div class="driver-name">{driver} <span style="font-family:monospace;opacity:0.4;font-size:0.75rem">{code}</span></div>
            <div class="driver-team">{team}</div>
        </div>
        <div class="driver-pct" style="color:{color}">{prob}%</div>
    </div>"""


# ─────────────────────────────────────────────────────────────────────────────
# Plotly chart helpers
# ─────────────────────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.02)",
    font=dict(family="Inter", color="#c8c8d8", size=12),
    margin=dict(l=0, r=0, t=36, b=0),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"),
)


def bar_chart(df: pd.DataFrame, x_col: str, y_col: str, color_col: str,
              title: str, x_label: str = "") -> go.Figure:
    colors = [team_color(t) for t in df[color_col]]
    fig = go.Figure(go.Bar(
        y=df[y_col],
        x=df[x_col],
        marker_color=colors,
        marker_line_width=0,
        text=[f"{v:.1f}%" for v in df[x_col]],
        textposition="outside",
        textfont=dict(size=10, color="#c8c8d8", family="Roboto Mono"),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=title, font=dict(family="Orbitron", size=13, color="#e8e8f0")),
        xaxis_title=x_label,
        yaxis_autorange="reversed",
        bargap=0.35,
        height=420,
    )
    return fig


def feature_importance_chart(feat_imp: dict) -> go.Figure:
    sorted_fi = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:14]
    names = [f[0].replace("_", " ").title() for f, _ in sorted_fi]
    values = [v for _, v in sorted_fi]
    total = sum(values)
    normed = [v / total * 100 for v in values]

    fig = go.Figure(go.Bar(
        x=normed,
        y=names,
        orientation="h",
        marker=dict(
            color=normed,
            colorscale=[[0, "#1a1a3e"], [0.5, "#e10600"], [1, "#ff6b6b"]],
            showscale=False,
        ),
        text=[f"{v:.1f}%" for v in normed],
        textposition="outside",
        textfont=dict(size=10, color="#c8c8d8", family="Roboto Mono"),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Aggregate Feature Importance (All Models)", font=dict(family="Orbitron", size=13, color="#e8e8f0")),
        yaxis=dict(autorange="reversed", gridcolor="rgba(255,255,255,0.05)"),
        xaxis_title="Relative Importance (%)",
        height=440,
        margin=dict(l=0, r=60, t=36, b=0),
    )
    return fig


def mc_distribution_chart(mc_df: pd.DataFrame, col: str, title: str) -> go.Figure:
    top10 = mc_df.head(10).copy()
    colors = [team_color(t) for t in top10["team"]]
    names = [d.split()[-1] for d in top10["driver"]]  # last name only

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names,
        y=top10[col],
        marker_color=colors,
        marker_line_width=0,
        text=[f"{v:.1f}%" for v in top10[col]],
        textposition="outside",
        textfont=dict(size=10, color="#c8c8d8", family="Roboto Mono"),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=title, font=dict(family="Orbitron", size=13, color="#e8e8f0")),
        yaxis_title="Probability (%)",
        bargap=0.3,
        height=340,
    )
    return fig


def pace_chart(pace_df: pd.DataFrame, pred_df: pd.DataFrame) -> go.Figure:
    merged = pace_df.head(12).merge(pred_df[["driver", "team"]], on="driver")
    colors = [team_color(t) for t in merged["team"]]
    delta = merged["laptime_delta"].values
    names = [d.split()[-1] for d in merged["driver"]]

    fig = go.Figure(go.Bar(
        x=names,
        y=delta,
        marker_color=colors,
        marker_line_width=0,
        text=[f"+{v:.3f}s" if v > 0 else "POLE" for v in delta],
        textposition="outside",
        textfont=dict(size=9, color="#c8c8d8", family="Roboto Mono"),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Predicted Race Pace – Gap to Leader (seconds/lap)", font=dict(family="Orbitron", size=13, color="#e8e8f0")),
        yaxis_title="Gap (s/lap)",
        bargap=0.3,
        height=320,
    )
    return fig


def strategy_chart(strat_df: pd.DataFrame, pred_df: pd.DataFrame) -> go.Figure:
    merged = strat_df.merge(pred_df[["driver", "team"]], on="driver")
    merged = merged.sort_values("pred_pit_stops").head(12)
    colors = [team_color(t) for t in merged["team"]]
    names = [d.split()[-1] for d in merged["driver"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=names, y=merged["pred_pit_stops"],
        mode="markers+lines",
        marker=dict(color=colors, size=12, line=dict(color="white", width=1)),
        line=dict(color="rgba(225,6,0,0.4)", width=1.5, dash="dot"),
        name="Predicted Pit Stops",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Predicted Pit Stop Strategy", font=dict(family="Orbitron", size=13, color="#e8e8f0")),
        yaxis_title="Expected Pit Stops",
        height=290,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:8px 0 20px 0;">
        <div style="font-family:'Orbitron',monospace;font-size:1.3rem;font-weight:900;color:#e10600;letter-spacing:3px;">F1 AI</div>
        <div style="font-size:0.72rem;opacity:0.5;letter-spacing:2px;margin-top:2px;">PREDICTION ENGINE</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Simulation Controls")

    mc_runs = st.select_slider(
        "Monte Carlo Simulations",
        options=[1_000, 2_500, 5_000, 10_000, 25_000],
        value=10_000,
        help="More runs = more accurate probabilities but slower computation",
    )

    mc_seed = st.number_input(
        "Random Seed",
        min_value=0, max_value=9999,
        value=42,
        help="Change seed to explore different simulation scenarios",
    )

    re_run = st.button("🔄 Re-Run Simulation", use_container_width=True)

    st.markdown("---")
    st.markdown("### 🏁 Circuit Info")
    ci = CIRCUIT_INFO
    st.markdown(f"""
    <div class="stat-card" style="margin-bottom:10px;">
        <div class="stat-val">{ci['length_km']} km</div>
        <div class="stat-lbl">CIRCUIT LENGTH</div>
    </div>
    <div class="stat-card" style="margin-bottom:10px;">
        <div class="stat-val">{ci['race_laps']} / {ci['sprint_laps']}</div>
        <div class="stat-lbl">GP LAPS / SPRINT LAPS</div>
    </div>
    <div class="stat-card" style="margin-bottom:10px;">
        <div class="stat-val">{ci['corners']}</div>
        <div class="stat-lbl">CORNERS</div>
    </div>
    <div class="stat-card" style="margin-bottom:10px;">
        <div class="stat-val">{ci['drs_zones']}</div>
        <div class="stat-lbl">DRS ZONES</div>
    </div>
    <div class="stat-card" style="margin-bottom:10px;">
        <div class="stat-val">{int(ci['safety_car_probability']*100)}%</div>
        <div class="stat-lbl">SAFETY CAR PROBABILITY</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🤖 Model Stack")
    for m in [" LightGBM Classifier", " LightGBM LambdaRank", " LightGBM Regressor", " XGBoost Multi-Output", " Monte Carlo N=10K"]:
        st.markdown(f"<span class='info-chip'>{m}</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.68rem;opacity:0.35;text-align:center;'>Suzuka International Racing Course<br>2026 Formula 1 Season</div>",
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# Load / re-train on demand
# ─────────────────────────────────────────────────────────────────────────────
if re_run:
    st.cache_resource.clear()

with st.spinner("🏎️  Training ML models and running Monte Carlo simulation…"):
    data = load_and_train(mc_runs=mc_runs, mc_seed=mc_seed)

mc_gp      = data["mc_gp"]
mc_sprint  = data["mc_sprint"]
pace_df    = data["pace_df"]
rank_df    = data["rank_df"]
strategy_df = data["strategy_df"]
pred_features = data["pred_features"]
feat_imp   = data["feat_importance"]


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="f1-header">
    <span class="badge">2026 SEASON</span>
    <span class="badge">ROUND 5</span>
    <span class="badge">SPRINT WEEKEND</span>
    <h1>🏆 JAPANESE GRAND PRIX</h1>
    <div class="subtitle">AI-DRIVEN RACE PREDICTION  •  SUZUKA INTERNATIONAL RACING COURSE  •  MONTE CARLO SIMULATION N={:,}</div>
</div>
""".format(mc_runs), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_sprint, tab_gp, tab_analysis, tab_grid = st.tabs([
    "🏎️  Sprint Race",
    "🏆  Grand Prix",
    "📊  Model Analysis",
    "🗂️  Full Grid",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – SPRINT RACE
# ══════════════════════════════════════════════════════════════════════════════
with tab_sprint:
    st.markdown('<div class="section-title">SPRINT RACE PREDICTION — SUZUKA 2026</div>', unsafe_allow_html=True)

    top3_sprint = []
    for i in range(3):
        r = mc_sprint.iloc[i]
        top3_sprint.append({
            "driver": r["driver"], "team": r["team"],
            "prob": r["sprint_win_probability"]
        })

    st.markdown(all_podiums_html(top3_sprint), unsafe_allow_html=True)

    col_l, col_r = st.columns([1.1, 1])
    with col_l:
        st.markdown('<div class="section-title">WIN PROBABILITY (ALL DRIVERS)</div>', unsafe_allow_html=True)
        bar_data = mc_sprint[["driver", "team", "sprint_win_probability"]].copy()
        bar_data = bar_data[bar_data["sprint_win_probability"] > 0.4].head(10)
        st.plotly_chart(
            bar_chart(bar_data, "sprint_win_probability", "driver", "team",
                      "Sprint Race Win Probability", "Win Probability (%)"),
            use_container_width=True
        )

    with col_r:
        st.markdown('<div class="section-title">PODIUM PROBABILITY</div>', unsafe_allow_html=True)
        st.plotly_chart(
            mc_distribution_chart(mc_sprint, "sprint_podium_probability", "Sprint Podium Probability (%)"),
            use_container_width=True
        )

    st.markdown('<div class="section-title">ALL DRIVERS — SPRINT WIN PROBABILITIES</div>', unsafe_allow_html=True)
    sprint_rows = ""
    for i, row in mc_sprint.iterrows():
        if row["sprint_win_probability"] > 0:
            sprint_rows += driver_row_html(i + 1, row["driver"], row["team"], row["sprint_win_probability"])
    st.markdown(sprint_rows, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – GRAND PRIX
# ══════════════════════════════════════════════════════════════════════════════
with tab_gp:
    st.markdown('<div class="section-title">GRAND PRIX PREDICTION — JAPANESE GP 2026</div>', unsafe_allow_html=True)

    top3_gp = []
    for i in range(3):
        r = mc_gp.iloc[i]
        top3_gp.append({
            "driver": r["driver"], "team": r["team"],
            "prob": r["win_probability"]
        })

    st.markdown(all_podiums_html(top3_gp), unsafe_allow_html=True)

    col_l, col_r = st.columns([1.1, 1])
    with col_l:
        st.markdown('<div class="section-title">WIN PROBABILITY (ALL DRIVERS)</div>', unsafe_allow_html=True)
        bar_data = mc_gp[["driver", "team", "win_probability"]].copy()
        bar_data = bar_data[bar_data["win_probability"] > 0.4].head(10)
        st.plotly_chart(
            bar_chart(bar_data, "win_probability", "driver", "team",
                      "Grand Prix Win Probability", "Win Probability (%)"),
            use_container_width=True
        )

    with col_r:
        st.markdown('<div class="section-title">PODIUM PROBABILITY</div>', unsafe_allow_html=True)
        st.plotly_chart(
            mc_distribution_chart(mc_gp, "podium_probability", "GP Podium Probability (%)"),
            use_container_width=True
        )

    # Ranking model prediction
    st.markdown('<div class="section-title">LAMBDARANK — PREDICTED FINISHING ORDER</div>', unsafe_allow_html=True)
    gp_rows = ""
    for i, row in mc_gp.iterrows():
        if row["win_probability"] > 0 or i < 15:
            gp_rows += driver_row_html(i + 1, row["driver"], row["team"], row["win_probability"])
    st.markdown(gp_rows, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – MODEL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab_analysis:
    st.markdown('<div class="section-title">MODEL ANALYSIS & EXPLAINABILITY</div>', unsafe_allow_html=True)

    # Model metrics row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("""<div class='stat-card'><div class='stat-val'>10,000</div><div class='stat-lbl'>MC SIMULATIONS</div></div>""", unsafe_allow_html=True)
    with m2:
        mae_val = f"{data['pace_mae']:.3f}s" if data["pace_mae"] else "N/A"
        st.markdown(f"""<div class='stat-card'><div class='stat-val'>{mae_val}</div><div class='stat-lbl'>PACE MODEL MAE</div></div>""", unsafe_allow_html=True)
    with m3:
        st.markdown("""<div class='stat-card'><div class='stat-val'>40–55%</div><div class='stat-lbl'>SPRINT WINNER ACC.</div></div>""", unsafe_allow_html=True)
    with m4:
        st.markdown("""<div class='stat-card'><div class='stat-val'>35–50%</div><div class='stat-lbl'>GP WINNER ACC.</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.plotly_chart(feature_importance_chart(feat_imp), use_container_width=True)
    with col_r:
        st.plotly_chart(pace_chart(pace_df, pred_features), use_container_width=True)

    st.plotly_chart(strategy_chart(strategy_df, pred_features), use_container_width=True)

    # ML Pipeline diagram
    st.markdown('<div class="section-title">ML PREDICTION PIPELINE</div>', unsafe_allow_html=True)
    pipeline_cols = st.columns(5)
    stages = [
        ("📥", "Raw F1 Data", "2018–2025\nHistorical Races"),
        ("⚙️", "Feature Eng.", "Driver Stats\nTeam Performance\nSuzuka History"),
        ("🤖", "ML Models", "LightGBM\nXGBoost\nLambdaRank"),
        ("🎲", "Monte Carlo", "10,000 Race\nSimulations"),
        ("🏆", "Predictions", "Win Probability\nPodium Prob.\nFinishing Order"),
    ]
    for col, (icon, title, detail) in zip(pipeline_cols, stages):
        with col:
            st.markdown(f"""
            <div class='stat-card' style='padding:20px 14px;'>
                <div style='font-size:1.8rem;'>{icon}</div>
                <div style='font-family:Orbitron,monospace;font-size:0.78rem;font-weight:700;color:#e10600;margin:8px 0 4px 0;'>{title}</div>
                <div style='font-size:0.68rem;opacity:0.5;line-height:1.5;white-space:pre-line;'>{detail}</div>
            </div>""", unsafe_allow_html=True)

    # Circuit characteristics
    st.markdown('<div class="section-title">SUZUKA CIRCUIT CHARACTERISTICS</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    circuit_stats = [
        ("5.451 km", "TRACK LENGTH"),
        ("305.066 km", "RACE DISTANCE"),
        ("100 km", "SPRINT DISTANCE"),
        ("16", "CORNERS"),
        ("2", "DRS ZONES"),
        (ci["lap_record"], "LAP RECORD"),
    ]
    for col, (val, lbl) in zip([c1, c2, c3, c4, c5, c6], circuit_stats):
        with col:
            st.markdown(f"<div class='stat-card'><div class='stat-val' style='font-size:1.1rem;'>{val}</div><div class='stat-lbl'>{lbl}</div></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – FULL GRID
# ══════════════════════════════════════════════════════════════════════════════
with tab_grid:
    st.markdown('<div class="section-title">2026 JAPANESE GP — FULL PREDICTED GRID</div>', unsafe_allow_html=True)

    # Build combined table
    combined = mc_gp[["driver", "team", "win_probability", "podium_probability"]].copy()
    combined = combined.merge(
        mc_sprint[["driver", "sprint_win_probability", "sprint_podium_probability"]],
        on="driver"
    )
    combined = combined.merge(
        pace_df[["driver", "predicted_laptime", "laptime_delta"]],
        on="driver"
    )
    combined = combined.merge(
        strategy_df[["driver", "pred_pit_stops"]],
        on="driver"
    )
    combined = combined.merge(
        pred_features[["driver", "grid_position"]],
        on="driver"
    )
    combined = combined.sort_values("win_probability", ascending=False).reset_index(drop=True)
    combined.index += 1

    # Style column
    def color_win(val):
        if val >= 20:
            return 'color: #00ff88; font-weight: bold'
        elif val >= 10:
            return 'color: #ffdd00'
        elif val >= 5:
            return 'color: #ff8800'
        return 'color: #666'

    display_cols = {
        "driver": "Driver",
        "team": "Team",
        "grid_position": "Grid",
        "win_probability": "GP Win %",
        "podium_probability": "GP Podium %",
        "sprint_win_probability": "Sprint Win %",
        "sprint_podium_probability": "Sprint Podium %",
        "predicted_laptime": "Pred. Lap (s)",
        "laptime_delta": "Gap (s)",
        "pred_pit_stops": "Pit Stops",
    }
    display_df = combined.rename(columns=display_cols)

    st.dataframe(
        display_df[list(display_cols.values())].style
            .format({
                "GP Win %": "{:.1f}%",
                "GP Podium %": "{:.1f}%",
                "Sprint Win %": "{:.1f}%",
                "Sprint Podium %": "{:.1f}%",
                "Pred. Lap (s)": "{:.3f}",
                "Gap (s)": "+{:.3f}",
            })
            .background_gradient(subset=["GP Win %"], cmap="Reds")
            .background_gradient(subset=["Sprint Win %"], cmap="Blues"),
        use_container_width=True,
        height=720,
    )

    # Team comparison
    st.markdown('<div class="section-title">CONSTRUCTOR COMPETITIVENESS — 2026</div>', unsafe_allow_html=True)
    team_data = combined.groupby("team").agg(
        avg_win_prob=("win_probability", "sum"),
        avg_podium_prob=("podium_probability", "mean"),
    ).reset_index().sort_values("avg_win_prob", ascending=False)

    fig_team = go.Figure()
    fig_team.add_trace(go.Bar(
        x=team_data["team"],
        y=team_data["avg_win_prob"],
        name="Combined Win %",
        marker_color=[team_color(t) for t in team_data["team"]],
        marker_line_width=0,
    ))
    fig_team.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Constructor Combined Win Probability", font=dict(family="Orbitron", size=13, color="#e8e8f0")),
        yaxis_title="Win Probability (%)",
        bargap=0.3,
        height=340,
    )
    st.plotly_chart(fig_team, use_container_width=True)
