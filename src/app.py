"""
Engine Condition Classification System - Mission-Control Streamlit Application.

Operational telemetry classification dashboard supporting the promoted LightGBM model
and the baseline Random Forest model for binary engine-condition classification.
"""

from __future__ import annotations

from datetime import datetime
import os
from typing import Any, Dict

# IMPORTANT: In 64-bit Windows Python environments, lightgbm must be imported
# prior to joblib unpickling to ensure C++ OpenMP runtime binds properly.
import lightgbm as lgb
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from inference import load_model, predict_engine_condition


# -----------------------------------------------------------------------------
# Training Dataset Reference Statistics (N = 12,502, data/processed/train_split.csv)
# Explicitly identified as dataset statistics, not manufacturer safe operating limits.
# -----------------------------------------------------------------------------
DATASET_STATS: Dict[str, Dict[str, Any]] = {
    "Engine_RPM": {
        "label": "Engine Speed",
        "subsystem": "Kinematic / Mechanical",
        "unit": "RPM",
        "min": 61.0,
        "max": 2239.0,
        "median": 748.0,
        "mean": 792.87,
        "q25": 595.0,
        "q75": 933.0,
        "std": 268.43,
    },
    "Lub_Oil_Pressure": {
        "label": "Lubrication Oil Pressure",
        "subsystem": "Lubrication Circuit",
        "unit": "bar",
        "min": 0.0,
        "max": 7.27,
        "median": 3.17,
        "mean": 3.31,
        "q25": 2.52,
        "q75": 4.06,
        "std": 1.03,
    },
    "Fuel_Pressure": {
        "label": "Fuel Injection Pressure",
        "subsystem": "Fuel Delivery",
        "unit": "bar",
        "min": 0.0,
        "max": 21.14,
        "median": 6.18,
        "mean": 6.65,
        "q25": 4.91,
        "q75": 7.70,
        "std": 2.76,
    },
    "Coolant_Pressure": {
        "label": "Coolant Pressure",
        "subsystem": "Cooling Circuit",
        "unit": "bar",
        "min": 0.0,
        "max": 7.48,
        "median": 2.17,
        "mean": 2.34,
        "q25": 1.61,
        "q75": 2.85,
        "std": 1.04,
    },
    "Lub_Oil_Temperature": {
        "label": "Lubrication Oil Temperature",
        "subsystem": "Lubrication Thermal",
        "unit": "°C",
        "min": 71.32,
        "max": 89.47,
        "median": 76.82,
        "mean": 77.65,
        "q25": 75.72,
        "q75": 78.09,
        "std": 3.11,
    },
    "Coolant_Temperature": {
        "label": "Coolant Temperature",
        "subsystem": "Cooling Thermal",
        "unit": "°C",
        "min": 63.09,
        "max": 195.53,
        "median": 78.37,
        "mean": 78.45,
        "q25": 73.92,
        "q75": 82.91,
        "std": 6.24,
    },
}

# -----------------------------------------------------------------------------
# Illustrative Demonstration Presets
# Labeled strictly as synthetic demonstration inputs, not validated physical fault signatures.
# -----------------------------------------------------------------------------
PRESETS: Dict[str, Dict[str, Any]] = {
    "Nominal Cruise": {
        "description": "Standard baseline telemetry configuration.",
        "values": {
            "Engine_RPM": 800.0,
            "Lub_Oil_Pressure": 3.0,
            "Fuel_Pressure": 10.0,
            "Coolant_Pressure": 2.0,
            "Lub_Oil_Temperature": 80.0,
            "Coolant_Temperature": 80.0,
        },
    },
    "Low Oil Pressure": {
        "description": "Attenuated lubrication oil pressure input.",
        "values": {
            "Engine_RPM": 800.0,
            "Lub_Oil_Pressure": 1.5,
            "Fuel_Pressure": 10.0,
            "Coolant_Pressure": 2.0,
            "Lub_Oil_Temperature": 80.0,
            "Coolant_Temperature": 80.0,
        },
    },
    "Elevated Temperature": {
        "description": "Elevated thermal channels across oil and coolant.",
        "values": {
            "Engine_RPM": 1200.0,
            "Lub_Oil_Pressure": 3.0,
            "Fuel_Pressure": 12.0,
            "Coolant_Pressure": 2.5,
            "Lub_Oil_Temperature": 110.0,
            "Coolant_Temperature": 105.0,
        },
    },
    "Low Fuel Pressure": {
        "description": "Reduced fuel injection pressure input.",
        "values": {
            "Engine_RPM": 1500.0,
            "Lub_Oil_Pressure": 3.2,
            "Fuel_Pressure": 3.0,
            "Coolant_Pressure": 2.0,
            "Lub_Oil_Temperature": 80.0,
            "Coolant_Temperature": 80.0,
        },
    },
}


def _is_running_in_hf_space() -> bool:
    """Check if application is running inside a Hugging Face Space."""
    return os.getenv("SPACE_ID") is not None or os.getenv("SYSTEM") == "spaces"


def _get_default_source() -> str:
    """Decide whether to load the model from HF or local based on environment."""
    if _is_running_in_hf_space():
        return "hf"
    if config.HF_TOKEN and config.HF_MODEL_REPO:
        return "hf"
    return "production"


@st.cache_resource
def get_cached_model(source: str = "production") -> object:
    """
    Cache loaded model objects across user interactions to prevent redundant disk I/O.
    """
    return load_model(source=source)


# -----------------------------------------------------------------------------
# Chart Helpers (Including Backward-Compatible Functions for Existing Test Suite)
# -----------------------------------------------------------------------------
def create_gauge_chart(value: float, title: str, color: str) -> go.Figure:
    """
    Legacy gauge chart helper preserved for backwards compatibility with tests.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": title, "font": {"size": 16, "family": "monospace", "color": "#E2E8F0"}},
        gauge={
            "axis": {"range": [None, 100], "tickcolor": "#94A3B8"},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 50], "color": "#1E2536"},
                {"range": [50, 80], "color": "#262C3D"},
            ],
            "threshold": {
                "line": {"color": "#DC2626", "width": 3},
                "thickness": 0.75,
                "value": 90,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=180,
        margin=dict(l=20, r=20, t=35, b=20),
    )
    return fig


def create_sensor_comparison_chart(sensor_data: dict) -> go.Figure:
    """
    Radar chart helper preserved for backwards compatibility with test_app_integration.py.
    """
    categories = list(sensor_data.keys())
    values = list(sensor_data.values())

    max_values = {
        "Engine_RPM": 4000,
        "Lub_Oil_Pressure": 10,
        "Fuel_Pressure": 30,
        "Coolant_Pressure": 10,
        "Lub_Oil_Temperature": 150,
        "Coolant_Temperature": 150,
    }

    normalized_values = [
        (v / max_values.get(k, 100)) * 100 for k, v in zip(categories, values)
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=normalized_values + [normalized_values[0]],
        theta=[k.replace("_", " ") for k in categories] + [categories[0].replace("_", " ")],
        fill="toself",
        name="Current Readings",
        line=dict(color="#38BDF8", width=2),
        fillcolor="rgba(56, 189, 248, 0.2)",
    ))

    fig.update_layout(
        polar=dict(
            bgcolor="#121620",
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor="#262C3D",
                tickfont=dict(color="#64748B", size=9),
            ),
            angularaxis=dict(
                gridcolor="#262C3D",
                tickfont=dict(color="#94A3B8", size=10),
            ),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        height=320,
        title=dict(text="Sensor Readings Overview", font=dict(color="#E2E8F0", size=14)),
        margin=dict(l=30, r=30, t=40, b=30),
    )
    return fig


def create_decision_score_bar(prob_faulty: float, threshold: float = 0.50) -> go.Figure:
    """
    Create a clean, industrial horizontal decision meter.
    Displays the empirical fault score along a 0% - 100% axis with a prominent
    reference marker at the operating threshold.
    """
    pct_fault = prob_faulty * 100.0
    pct_thresh = threshold * 100.0
    is_fault = prob_faulty >= threshold

    # Semantic color: Red if classified faulty, Green if normal
    bar_color = "#DC2626" if is_fault else "#059669"

    fig = go.Figure()

    # Track background (0% - 100%)
    fig.add_trace(go.Bar(
        y=["Score"],
        x=[100],
        orientation="h",
        marker=dict(color="#1A202C", line=dict(color="#2D3748", width=1)),
        hoverinfo="none",
        showlegend=False,
    ))

    # Active score fill bar
    fig.add_trace(go.Bar(
        y=["Score"],
        x=[pct_fault],
        orientation="h",
        marker=dict(color=bar_color),
        text=f" {pct_fault:.1f}%",
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(family="ui-monospace, monospace", size=13, color="#FFFFFF"),
        hoverinfo="text",
        hovertext=f"Empirical Fault Score: {pct_fault:.2f}%",
        showlegend=False,
    ))

    # Vertical threshold reference marker
    fig.add_vline(
        x=pct_thresh,
        line_width=2,
        line_dash="dash",
        line_color="#F59E0B",
        annotation_text=f"Threshold {threshold:.2f}",
        annotation_position="top",
        annotation_font=dict(family="ui-monospace, monospace", size=11, color="#F59E0B"),
    )

    fig.update_layout(
        barmode="overlay",
        xaxis=dict(
            range=[0, 100],
            showgrid=True,
            gridcolor="#262C3D",
            tickvals=[0, 20, 40, 50, 60, 80, 100],
            ticktext=["0%", "20%", "40%", "50%", "60%", "80%", "100%"],
            tickfont=dict(family="ui-monospace, monospace", size=11, color="#94A3B8"),
            zeroline=False,
        ),
        yaxis=dict(showticklabels=False, showgrid=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=85,
        margin=dict(l=5, r=5, t=25, b=20),
    )
    return fig


def render_sensor_card_html(feat_name: str, value: float) -> str:
    """
    Generate an individual sensor telemetry card displaying actual reading,
    verified unit, subsystem, and training dataset distribution context.
    """
    meta = DATASET_STATS[feat_name]
    label = meta["label"]
    unit = meta["unit"]
    subsystem = meta["subsystem"]
    s_min = meta["min"]
    s_max = meta["max"]
    s_med = meta["median"]
    q25 = meta["q25"]
    q75 = meta["q75"]

    rng = s_max - s_min if s_max > s_min else 1.0
    val_clamped = max(s_min, min(s_max, value))
    val_pct = ((val_clamped - s_min) / rng) * 100.0
    med_pct = ((s_med - s_min) / rng) * 100.0
    q25_pct = ((q25 - s_min) / rng) * 100.0
    q75_pct = ((q75 - s_min) / rng) * 100.0
    iqr_width = max(1.0, q75_pct - q25_pct)

    return (
        f'<div class="sensor-card">'
        f'<div class="sensor-header">'
        f'<span class="sensor-title">{label}</span>'
        f'<span class="sensor-subsystem">{subsystem}</span>'
        f'</div>'
        f'<div class="sensor-value-row">'
        f'<span class="sensor-value">{value:.2f}</span>'
        f'<span class="sensor-unit">{unit}</span>'
        f'</div>'
        f'<div class="sensor-track-container" title="Current: {value:.2f} {unit} | Median: {s_med:.2f} {unit} | Range: [{s_min:.1f}, {s_max:.1f}]">'
        f'<div class="sensor-range-track">'
        f'<div class="sensor-iqr-band" style="left: {q25_pct:.1f}%; width: {iqr_width:.1f}%;"></div>'
        f'<div class="sensor-median-tick" style="left: {med_pct:.1f}%;"></div>'
        f'<div class="sensor-value-marker" style="left: {val_pct:.1f}%;"></div>'
        f'</div>'
        f'</div>'
        f'<div class="sensor-stats-row">'
        f'<span>Dataset Median: {s_med:.2f} {unit}</span>'
        f'<span>IQR: [{q25:.2f}, {q75:.2f}]</span>'
        f'</div>'
        f'</div>'
    )


# -----------------------------------------------------------------------------
# Main Application Routine
# -----------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Engine Condition Classification System",
        page_icon="⚙️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # -------------------------------------------------------------------------
    # Mission-Control Industrial Theme CSS
    # -------------------------------------------------------------------------
    st.markdown("""
    <style>
        /* Base Canvas and Surfaces */
        .stApp {
            background-color: #0B0E14;
            color: #F1F5F9;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }

        /* Sidebar Customization */
        [data-testid="stSidebar"] {
            background-color: #10141D;
            border-right: 1px solid #262C3D;
        }
        [data-testid="stSidebar"] hr {
            border-color: #262C3D;
            margin: 1rem 0;
        }

        /* Header System Banner */
        .system-header-container {
            background: #111622;
            border: 1px solid #262C3D;
            border-radius: 8px;
            padding: 1rem 1.25rem;
            margin-bottom: 1.25rem;
        }
        .system-header-title {
            font-size: 1.45rem;
            font-weight: 700;
            color: #F8FAFC;
            letter-spacing: -0.02em;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }
        .system-header-subtitle {
            font-size: 0.85rem;
            color: #94A3B8;
            margin-top: 0.25rem;
            margin-bottom: 0.5rem;
        }
        .system-status-pills {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            align-items: center;
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.2rem 0.55rem;
            border-radius: 4px;
            font-size: 0.72rem;
            font-family: ui-monospace, "SF Mono", Consolas, monospace;
            font-weight: 600;
            background: #182030;
            border: 1px solid #2D3A52;
            color: #CBD5E1;
        }
        .beacon-green {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background-color: #10B981;
            box-shadow: 0 0 5px #10B981;
        }
        .beacon-blue {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background-color: #38BDF8;
        }

        /* Form & Card Containers */
        .deck-header {
            font-size: 0.92rem;
            font-weight: 600;
            color: #F1F5F9;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 0.75rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #262C3D;
            padding-bottom: 0.4rem;
        }

        /* Sensor Visual Cards */
        .sensor-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 0.65rem;
            margin-top: 0.5rem;
            margin-bottom: 0.5rem;
        }
        @media (max-width: 768px) {
            .sensor-grid {
                grid-template-columns: 1fr;
            }
        }
        .sensor-card {
            background: #131722;
            border: 1px solid #262C3D;
            border-radius: 6px;
            padding: 0.75rem 0.85rem;
        }
        .sensor-card:hover {
            border-color: #38BDF8;
        }
        .sensor-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.25rem;
        }
        .sensor-title {
            font-size: 0.72rem;
            font-weight: 600;
            color: #94A3B8;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .sensor-subsystem {
            font-size: 0.62rem;
            color: #64748B;
            background: #1A202C;
            padding: 0.1rem 0.35rem;
            border-radius: 3px;
            border: 1px solid #262C3D;
        }
        .sensor-value-row {
            display: flex;
            align-items: baseline;
            gap: 0.35rem;
            margin-bottom: 0.4rem;
        }
        .sensor-value {
            font-family: ui-monospace, "SF Mono", Consolas, monospace;
            font-size: 1.35rem;
            font-weight: 700;
            color: #F8FAFC;
        }
        .sensor-unit {
            font-family: ui-monospace, "SF Mono", Consolas, monospace;
            font-size: 0.8rem;
            color: #94A3B8;
        }
        .sensor-track-container {
            padding: 0.25rem 0;
        }
        .sensor-range-track {
            position: relative;
            height: 6px;
            background: #1E2536;
            border-radius: 3px;
            margin: 0.35rem 0;
        }
        .sensor-iqr-band {
            position: absolute;
            top: 0;
            height: 100%;
            background: #2D3748;
            border-radius: 2px;
        }
        .sensor-median-tick {
            position: absolute;
            top: -2px;
            width: 2px;
            height: 10px;
            background: #64748B;
            border-radius: 1px;
        }
        .sensor-value-marker {
            position: absolute;
            top: -3px;
            width: 8px;
            height: 12px;
            background: #38BDF8;
            border-radius: 2px;
            transform: translateX(-50%);
            box-shadow: 0 0 4px #38BDF8;
        }
        .sensor-stats-row {
            display: flex;
            justify-content: space-between;
            font-size: 0.67rem;
            color: #64748B;
            font-family: ui-monospace, "SF Mono", Consolas, monospace;
        }

        /* Telemetry Legend */
        .telemetry-legend {
            display: flex;
            gap: 1rem;
            font-size: 0.72rem;
            color: #94A3B8;
            font-family: ui-monospace, "SF Mono", Consolas, monospace;
            margin-bottom: 0.5rem;
            padding: 0.35rem 0.6rem;
            background: #111622;
            border-radius: 4px;
            border: 1px solid #1E2536;
        }
        .legend-item {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
        }
        .legend-marker {
            width: 8px;
            height: 8px;
            background: #38BDF8;
            border-radius: 2px;
        }
        .legend-median {
            width: 2px;
            height: 10px;
            background: #64748B;
        }
        .legend-iqr {
            width: 14px;
            height: 6px;
            background: #2D3748;
            border-radius: 2px;
        }

        /* Annunciator Result Boxes */
        .annunciator-box {
            border-radius: 6px;
            padding: 1.15rem 1.25rem;
            margin-top: 0.75rem;
            margin-bottom: 1rem;
            border-left: 5px solid;
        }
        .annunciator-fault {
            background: #231215;
            border: 1px solid #7F1D1D;
            border-left: 5px solid #EF4444;
        }
        .annunciator-normal {
            background: #0D221C;
            border: 1px solid #065F46;
            border-left: 5px solid #10B981;
        }
        .annunciator-title {
            font-size: 1.2rem;
            font-weight: 700;
            margin: 0 0 0.3rem 0;
            letter-spacing: 0.02em;
        }
        .annunciator-fault .annunciator-title {
            color: #FCA5A5;
        }
        .annunciator-normal .annunciator-title {
            color: #6EE7B7;
        }
        .annunciator-meta {
            font-family: ui-monospace, "SF Mono", Consolas, monospace;
            font-size: 0.85rem;
            color: #CBD5E1;
            margin: 0;
        }

        /* Buttons */
        div.stButton > button:first-child {
            background: #182234;
            border: 1px solid #3B82F6;
            color: #F8FAFC;
            font-weight: 600;
            font-size: 0.95rem;
            letter-spacing: 0.02em;
            border-radius: 6px;
            padding: 0.6rem 1rem;
            transition: background 0.15s ease, border-color 0.15s ease;
        }
        div.stButton > button:first-child:hover {
            background: #1D4ED8;
            border-color: #60A5FA;
            color: #FFFFFF;
        }

        /* Numeric Inputs & Form Overrides */
        div[data-testid="stNumberInput"] input {
            font-family: ui-monospace, "SF Mono", Consolas, monospace !important;
            font-weight: 600 !important;
            background-color: #121620 !important;
            color: #F8FAFC !important;
            border-color: #262C3D !important;
        }

        /* App Footer Styling */
        .app-footer {
            border-top: 1px solid #262C3D;
            padding: 1.5rem 0 1rem 0;
            margin-top: 2.25rem;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0.65rem;
            text-align: center;
        }
        .footer-title-group {
            display: flex;
            align-items: center;
            justify-content: center;
            flex-wrap: wrap;
            gap: 0.45rem;
        }
        .footer-icon {
            font-size: 0.85rem;
        }
        .footer-title {
            font-size: 0.82rem;
            font-weight: 700;
            color: #F8FAFC;
            letter-spacing: 0.05em;
            font-family: ui-monospace, "SF Mono", Consolas, monospace;
        }
        .footer-separator {
            color: #475569;
            font-size: 0.75rem;
        }
        .footer-subtitle {
            font-size: 0.78rem;
            color: #94A3B8;
        }
        .footer-tech-stack {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 0.4rem;
        }
        .footer-tech-badge {
            background: #131722;
            border: 1px solid #262C3D;
            color: #94A3B8;
            font-family: ui-monospace, "SF Mono", Consolas, monospace;
            font-size: 0.7rem;
            padding: 0.15rem 0.5rem;
            border-radius: 4px;
            transition: border-color 0.15s ease, color 0.15s ease;
        }
        .footer-tech-badge:hover {
            border-color: #14B8A6;
            color: #F8FAFC;
        }
        .footer-disclaimer {
            font-size: 0.72rem;
            color: #64748B;
            line-height: 1.4;
            max-width: 650px;
        }

        /* Sidebar Model Selection Cards */
        [data-testid="stSidebar"] .sidebar-section-title {
            font-size: 0.76rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #94A3B8;
            margin-bottom: 0.5rem;
            font-family: ui-monospace, "SF Mono", Consolas, monospace;
        }
        [data-testid="stSidebar"] div[data-testid="stRadio"] {
            width: 100%;
        }
        [data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] {
            display: flex;
            flex-direction: column;
            gap: 0.65rem;
            width: 100%;
        }
        [data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label {
            position: relative;
            display: flex;
            flex-direction: row;
            align-items: flex-start;
            width: 100%;
            background: #131722;
            border: 1px solid #262C3D;
            border-radius: 8px;
            padding: 0.85rem 0.95rem;
            margin: 0;
            cursor: pointer;
            transition: border-color 0.15s ease, background-color 0.15s ease, box-shadow 0.15s ease;
        }
        [data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {
            display: none !important;
        }
        [data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:last-child {
            width: 100%;
            padding-right: 4.8rem;
        }
        [data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label p {
            font-size: 0.95rem !important;
            font-weight: 700 !important;
            color: #F8FAFC !important;
            margin: 0 0 0.25rem 0 !important;
            letter-spacing: -0.01em;
        }
        [data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label [data-testid="stCaptionContainer"] {
            font-size: 0.76rem !important;
            color: #94A3B8 !important;
            line-height: 1.45 !important;
            white-space: pre-line !important;
            font-family: ui-monospace, "SF Mono", Consolas, monospace !important;
        }
        [data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label:not(:has(input:checked)) {
            border: 1px solid #262C3D !important;
            background: #131722 !important;
        }
        [data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label:nth-child(1):has(input:checked) {
            border: 1.5px solid #14B8A6 !important;
            background: rgba(20, 184, 166, 0.08) !important;
            box-shadow: 0 0 10px rgba(20, 184, 166, 0.15);
        }
        [data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label:nth-child(1):has(input:checked)::after {
            content: "ACTIVE";
            position: absolute;
            top: 0.85rem;
            right: 0.85rem;
            font-size: 0.65rem;
            font-family: ui-monospace, "SF Mono", Consolas, monospace;
            font-weight: 700;
            color: #14B8A6;
            background: rgba(20, 184, 166, 0.15);
            border: 1px solid rgba(20, 184, 166, 0.4);
            padding: 0.15rem 0.45rem;
            border-radius: 4px;
            letter-spacing: 0.05em;
        }
        [data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label:nth-child(1):not(:has(input:checked))::after {
            content: "AVAILABLE";
            position: absolute;
            top: 0.85rem;
            right: 0.85rem;
            font-size: 0.65rem;
            font-family: ui-monospace, "SF Mono", Consolas, monospace;
            font-weight: 600;
            color: #64748B;
            background: #182030;
            border: 1px solid #262C3D;
            padding: 0.15rem 0.45rem;
            border-radius: 4px;
            letter-spacing: 0.05em;
        }
        [data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label:nth-child(2):has(input:checked) {
            border: 1.5px solid #3B82F6 !important;
            background: rgba(59, 130, 246, 0.08) !important;
            box-shadow: 0 0 10px rgba(59, 130, 246, 0.15);
        }
        [data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label:nth-child(2):has(input:checked)::after {
            content: "ACTIVE";
            position: absolute;
            top: 0.85rem;
            right: 0.85rem;
            font-size: 0.65rem;
            font-family: ui-monospace, "SF Mono", Consolas, monospace;
            font-weight: 700;
            color: #3B82F6;
            background: rgba(59, 130, 246, 0.15);
            border: 1px solid rgba(59, 130, 246, 0.4);
            padding: 0.15rem 0.45rem;
            border-radius: 4px;
            letter-spacing: 0.05em;
        }
        [data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label:nth-child(2):not(:has(input:checked))::after {
            content: "AVAILABLE";
            position: absolute;
            top: 0.85rem;
            right: 0.85rem;
            font-size: 0.65rem;
            font-family: ui-monospace, "SF Mono", Consolas, monospace;
            font-weight: 600;
            color: #64748B;
            background: #182030;
            border: 1px solid #262C3D;
            padding: 0.15rem 0.45rem;
            border-radius: 4px;
            letter-spacing: 0.05em;
        }
        [data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
            border-color: #475569 !important;
            background: #182030 !important;
        }
        [data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:focus-visible) {
            outline: 2px solid #38BDF8;
            outline-offset: 2px;
        }

        /* Reduced Motion Media Query */
        @media (prefers-reduced-motion: reduce) {
            *, ::before, ::after {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
                scroll-behavior: auto !important;
            }
        }
    </style>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Session State Initialization & Synchronization
    # -------------------------------------------------------------------------
    default_telemetry = PRESETS["Nominal Cruise"]["values"]
    for feat_name, default_val in default_telemetry.items():
        if f"val_{feat_name}" not in st.session_state:
            st.session_state[f"val_{feat_name}"] = float(default_val)

    if "preset_choice" not in st.session_state:
        st.session_state["preset_choice"] = "Nominal Cruise"

    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []

    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None

    # -------------------------------------------------------------------------
    # Preset & Custom Input Synchronization Handlers
    # -------------------------------------------------------------------------
    def sync_preset_state() -> None:
        """
        Verify if current inputs match the active preset.
        If any sensor value has been modified away from the preset values,
        immediately transition preset_choice to 'Custom Input'.
        """
        current_preset = st.session_state.get("preset_choice")
        if current_preset and current_preset in PRESETS:
            preset_vals = PRESETS[current_preset]["values"]
            mismatch = any(
                abs(float(st.session_state.get(f"val_{feat}", 0.0)) - float(expected_val)) > 1e-4
                for feat, expected_val in preset_vals.items()
            )
            if mismatch:
                st.session_state["preset_choice"] = "Custom Input"

    def on_preset_selected() -> None:
        """Populate all six sensor inputs when a preset is chosen."""
        chosen_preset = st.session_state.get("preset_choice")
        if chosen_preset in PRESETS:
            preset_vals = PRESETS[chosen_preset]["values"]
            for k, v in preset_vals.items():
                st.session_state[f"val_{k}"] = float(v)

    def on_sensor_input_change() -> None:
        """Triggered whenever the user manually adjusts any sensor input."""
        sync_preset_state()

    # Synchronize state before rendering selectbox
    sync_preset_state()

    # -------------------------------------------------------------------------
    # Sidebar: Model Selection, Specifications & Architecture
    # -------------------------------------------------------------------------
    with st.sidebar:
        is_in_space = _is_running_in_hf_space()

        st.markdown('<div class="sidebar-section-title">Model Candidate Selection</div>', unsafe_allow_html=True)
        if is_in_space:
            source = "hf"
            st.caption("Deployment: Hugging Face Spaces")
        else:
            options = ["production", "baseline"]
            if config.HF_TOKEN and config.HF_MODEL_REPO:
                options.append("hf")

            model_titles = {
                "production": "🧠 LightGBM",
                "baseline": "🌲 Random Forest",
                "hf": "🤖 Hugging Face Hub",
            }
            model_captions = [
                "Promoted Production Model\nValidation F1: 0.7813",
                "Original Baseline Model\nHoldout F1: 0.7683",
            ]
            if len(options) > 2:
                model_captions.append("Cloud Repository Model\nHugging Face Hub")

            source = st.radio(
                "Model Candidate Selection",
                options=options,
                index=0,
                format_func=lambda x: model_titles.get(x, x),
                captions=model_captions,
                label_visibility="collapsed",
                help="Select between the promoted LightGBM model and original baseline",
            )

        st.markdown("---")

        # Model Specifications (Expandable to reduce permanent visual density)
        with st.expander("📊 Active Model Specifications", expanded=False):
            if source == "production":
                st.markdown(
                    "**Architecture**: LightGBM Classifier<br>"
                    "**Package**: `models/production/`<br>"
                    "**Operating Threshold**: `0.50` (Frozen)<br>"
                    "**Validation F1**: `0.7813`<br>"
                    "**Holdout Test Recall**: `94.68%`<br>"
                    "**Subsampling**: `subsample=0.7`, `subsample_freq=1`<br>"
                    "**Imputation**: `SimpleImputer(strategy='median')`",
                    unsafe_allow_html=True,
                )
            elif source == "baseline":
                st.markdown(
                    "**Architecture**: Random Forest Classifier<br>"
                    "**Package**: `models/best_model.joblib`<br>"
                    "**Operating Threshold**: `0.50` (Fixed)<br>"
                    "**Holdout Test Recall**: `89.65%`<br>"
                    "**Holdout Test F1**: `0.7683`<br>"
                    "**Imputation**: `SimpleImputer(strategy='median')`",
                    unsafe_allow_html=True,
                )
            else:
                st.caption(f"Connected to Hugging Face Hub: `{config.HF_MODEL_REPO}`")

        # Dataset Baseline Context (Expandable)
        with st.expander("📈 Training Reference Context", expanded=False):
            st.caption(
                "• Records: 12,502 training observations\n"
                "• Feature Channels: 6 physical telemetry streams\n"
                "• Preprocessing: Fold-isolated median imputation\n"
                "• Target: Binary Engine Condition (0 = Normal, 1 = Maintenance Required)"
            )

        # System Boundaries & Disclaimers (Expandable)
        with st.expander("ℹ️ Operational Scope & Boundaries", expanded=False):
            st.caption(
                "This application is an engine-condition classification system providing statistical "
                "decision support. It evaluates instantaneous telemetry snapshots and does NOT "
                "estimate Remaining Useful Life (RUL) or predict time-to-failure."
            )

    # -------------------------------------------------------------------------
    # Header Banner & System Status Beacon
    # -------------------------------------------------------------------------
    active_model_name = (
        "LightGBM (Promoted)" if source == "production"
        else ("Random Forest (Baseline)" if source == "baseline" else "Hugging Face Hub")
    )

    st.markdown(
        f'<div class="system-header-container">'
        f'<h1 class="system-header-title"><span>⚙️</span> Engine Condition Classification System</h1>'
        f'<p class="system-header-subtitle">Telemetry-Based Operational Health Classification & Diagnostic Decision Support</p>'
        f'<div class="system-status-pills">'
        f'<span class="status-pill"><span class="beacon-green"></span> SYSTEM: ONLINE</span>'
        f'<span class="status-pill"><span class="beacon-blue"></span> MODEL: {active_model_name}</span>'
        f'<span class="status-pill">THRESHOLD: {config.DEFAULT_THRESHOLD:.2f}</span>'
        f'<span class="status-pill">CHANNELS: 6 SENSORS ACTIVE</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # Main Deck: Telemetry Input (Left) & Sensor Visualization (Right)
    # -------------------------------------------------------------------------
    col_input, col_viz = st.columns([1, 1.25])

    with col_input:
        st.markdown('<div class="deck-header"><span>Telemetry Input Deck</span><span>Step 1</span></div>', unsafe_allow_html=True)

        # Preset Selector
        preset_options = ["Custom Input"] + list(PRESETS.keys())
        st.selectbox(
            "Telemetry Preset (Synthetic Demonstration Inputs):",
            options=preset_options,
            key="preset_choice",
            on_change=on_preset_selected,
            help="Synthetic demonstration inputs for interface verification; not validated physical fault signatures.",
        )
        st.caption("Note: Presets are synthetic demonstration inputs, not validated physical fault signatures.")

        # Sensor Telemetry Channel Controls
        st.markdown("##### Subsystem Telemetry Channels")
        sub_col1, sub_col2 = st.columns(2)

        with sub_col1:
            st.number_input(
                "Engine Speed (RPM)",
                min_value=0.0,
                max_value=4000.0,
                step=10.0,
                key="val_Engine_RPM",
                on_change=on_sensor_input_change,
                help="Engine rotational speed in revolutions per minute (RPM)",
            )
            st.number_input(
                "Lub Oil Pressure (bar)",
                min_value=0.0,
                max_value=10.0,
                step=0.1,
                key="val_Lub_Oil_Pressure",
                on_change=on_sensor_input_change,
                help="Lubrication oil gallery pressure in bar",
            )
            st.number_input(
                "Fuel Pressure (bar)",
                min_value=0.0,
                max_value=30.0,
                step=0.1,
                key="val_Fuel_Pressure",
                on_change=on_sensor_input_change,
                help="Fuel injection rail pressure in bar",
            )

        with sub_col2:
            st.number_input(
                "Coolant Pressure (bar)",
                min_value=0.0,
                max_value=10.0,
                step=0.1,
                key="val_Coolant_Pressure",
                on_change=on_sensor_input_change,
                help="Engine cooling jacket pressure in bar",
            )
            st.number_input(
                "Lub Oil Temperature (°C)",
                min_value=0.0,
                max_value=150.0,
                step=0.5,
                key="val_Lub_Oil_Temperature",
                on_change=on_sensor_input_change,
                help="Lubricating oil sump temperature in °C",
            )
            st.number_input(
                "Coolant Temperature (°C)",
                min_value=0.0,
                max_value=150.0,
                step=0.5,
                key="val_Coolant_Temperature",
                on_change=on_sensor_input_change,
                help="Cooling jacket coolant temperature in °C",
            )

        submitted = st.button("Classify Engine Condition", use_container_width=True, key="classify_btn")

    with col_viz:
        st.markdown('<div class="deck-header"><span>Sensor Telemetry Overview</span><span>Verified Units</span></div>', unsafe_allow_html=True)

        # Legend explaining the range track markers
        st.markdown(
            '<div class="telemetry-legend">'
            '<span class="legend-item"><span class="legend-marker"></span> Current Reading</span>'
            '<span class="legend-item"><span class="legend-median"></span> Dataset Median</span>'
            '<span class="legend-item"><span class="legend-iqr"></span> Dataset IQR (25%–75%)</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        # Build 6 individual sensor cards
        current_telemetry = {
            "Engine_RPM": float(st.session_state["val_Engine_RPM"]),
            "Lub_Oil_Pressure": float(st.session_state["val_Lub_Oil_Pressure"]),
            "Fuel_Pressure": float(st.session_state["val_Fuel_Pressure"]),
            "Coolant_Pressure": float(st.session_state["val_Coolant_Pressure"]),
            "Lub_Oil_Temperature": float(st.session_state["val_Lub_Oil_Temperature"]),
            "Coolant_Temperature": float(st.session_state["val_Coolant_Temperature"]),
        }

        cards_html = [render_sensor_card_html(k, v) for k, v in current_telemetry.items()]
        grid_html = f'<div class="sensor-grid">{"".join(cards_html)}</div>'
        st.markdown(grid_html, unsafe_allow_html=True)

        st.caption(
            "Reference Notice: Ranges and medians represent training dataset distributions (N=12,502), "
            "not manufacturer safe operating limits. Actual operational tolerances depend on engine equipment specifications."
        )

    # -------------------------------------------------------------------------
    # Inference Execution & Diagnostic Classification Deck
    # -------------------------------------------------------------------------
    # Trigger classification on form submit or on first load if no prediction exists yet
    should_run_prediction = submitted or (st.session_state.last_prediction is None)

    if should_run_prediction:
        inputs = dict(current_telemetry)

        if source == "hf" and not config.HF_TOKEN:
            st.error("HF_TOKEN is not configured for Hugging Face Hub inference.")
            st.stop()

        with st.spinner("Analyzing sensor telemetry with active model..."):
            try:
                cached_model = get_cached_model(source=source)
                result = predict_engine_condition(
                    inputs=inputs,
                    model=cached_model,
                    source=source,
                    threshold=config.DEFAULT_THRESHOLD,
                )
            except Exception as exc:
                st.error(
                    f"Prediction Failed: {exc}\n\n"
                    "Troubleshooting Checklist:\n"
                    "• Ensure all input values are finite numeric numbers.\n"
                    f"• Verify model artifact exists at `{config.PRODUCTION_MODEL_PATH}`.\n"
                    "• Ensure dependencies (`lightgbm`, `scikit-learn`) are properly installed."
                )
                return

        st.session_state.last_prediction = result

        # Record run in session prediction history (if submitted or new)
        if submitted:
            history_record = {
                "Run": len(st.session_state.prediction_history) + 1,
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Model": active_model_name,
                "RPM": f"{inputs['Engine_RPM']:.1f}",
                "Oil Press (bar)": f"{inputs['Lub_Oil_Pressure']:.2f}",
                "Fuel Press (bar)": f"{inputs['Fuel_Pressure']:.2f}",
                "Cool Press (bar)": f"{inputs['Coolant_Pressure']:.2f}",
                "Oil Temp (°C)": f"{inputs['Lub_Oil_Temperature']:.1f}",
                "Cool Temp (°C)": f"{inputs['Coolant_Temperature']:.1f}",
                "Fault Score": f"{result['probability_faulty']:.1%}",
                "Classification": "FAULT / ANOMALY" if result["prediction"] == 1 else "NORMAL",
            }
            st.session_state.prediction_history.append(history_record)
            if len(st.session_state.prediction_history) > 10:
                st.session_state.prediction_history.pop(0)

    # Display Diagnostic Classification Deck if last_prediction exists
    if st.session_state.last_prediction is not None:
        pred_res = st.session_state.last_prediction
        pred_label = pred_res["prediction"]
        prob_faulty = pred_res["probability_faulty"]
        prob_normal = pred_res["probability_normal"]
        thresh = pred_res["threshold"]

        st.markdown("---")
        st.markdown('<div class="deck-header"><span>Diagnostic Classification Deck</span><span>Model Decision Output</span></div>', unsafe_allow_html=True)

        res_col1, res_col2 = st.columns([1.3, 1])

        with res_col1:
            if pred_label == 1:
                st.markdown(
                    f'<div class="annunciator-box annunciator-fault">'
                    f'<div class="annunciator-title">🚨 CLASSIFICATION: MAINTENANCE REQUIRED</div>'
                    f'<p class="annunciator-meta">'
                    f'Engine Condition: <b>ANOMALOUS / FAULT DETECTED (CLASS 1)</b><br>'
                    f'Empirical Fault Score: <b>{prob_faulty:.1%}</b> (Operating Decision Threshold: <b>{thresh:.2f}</b>)'
                    f'</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="annunciator-box annunciator-normal">'
                    f'<div class="annunciator-title">✅ CLASSIFICATION: NORMAL OPERATION</div>'
                    f'<p class="annunciator-meta">'
                    f'Engine Condition: <b>BASELINE OPERATING PARAMETERS (CLASS 0)</b><br>'
                    f'Empirical Fault Score: <b>{prob_faulty:.1%}</b> (Operating Decision Threshold: <b>{thresh:.2f}</b>)'
                    f'</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Recommended Diagnostic Action Protocol
            with st.expander("📋 General Diagnostic Inspection Protocol", expanded=True):
                st.markdown("""
                - **Qualified Technical Assessment**: A qualified engineer must independently verify engine operational condition prior to any physical mechanical intervention or equipment shutdown.
                - **Circuit Inspection**: Conduct visual, acoustic, and fluid level checks across lubrication, cooling, and fuel delivery circuits.
                - **Transducer Calibration**: Verify calibration, electrical wiring, and signal integrity of telemetry pressure/temperature transducers.
                - **Operational Log Review**: Cross-reference current readings against operational maintenance history logs to identify trends.
                """)

            # Mandatory Disclaimers
            st.caption(
                "**Model Interpretation Disclaimer**: Probabilities represent empirical model decision scores, "
                "not calibrated Bayesian posteriors. This system classifies instantaneous operational health and does NOT predict "
                "Remaining Useful Life (RUL) or time-to-failure."
            )
            st.caption(
                "**Explainability Boundary**: Raw sensor readings and model classifications are conceptually separate. "
                "This system does not evaluate single-sensor physical causality or isolate specific failure mechanisms."
            )

        with res_col2:
            st.markdown("##### Empirical Decision Score Meter")
            meter_fig = create_decision_score_bar(prob_faulty=prob_faulty, threshold=thresh)
            st.plotly_chart(meter_fig, use_container_width=True, config={"displayModeBar": False})

            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric("Normal Score", f"{prob_normal:.1%}")
            with m_col2:
                st.metric("Fault Score", f"{prob_faulty:.1%}")
            with m_col3:
                st.metric("Threshold", f"{thresh:.2f}")

    # -------------------------------------------------------------------------
    # Session Prediction History (Interactive Runs)
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown('<div class="deck-header"><span>Session Prediction History</span><span>Interactive Comparison</span></div>', unsafe_allow_html=True)
    st.caption("Displays predictions generated during the current interactive session. Enables comparing model outputs across presets or models. Not historical engine telemetry.")

    if len(st.session_state.prediction_history) > 0:
        history_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)

        if st.button("Clear Session History", key="clear_hist_btn"):
            st.session_state.prediction_history = []
            st.rerun()
    else:
        st.caption("No interactive prediction runs recorded in this session yet. Submit telemetry inputs to record runs.")

    # -------------------------------------------------------------------------
    # Premium Industrial Footer
    # -------------------------------------------------------------------------
    st.markdown(
        '<footer class="app-footer">'
        '<div class="footer-top-row">'
        '<div class="footer-title-group">'
        '<span class="footer-icon">⚙️</span>'
        '<span class="footer-title">ENGINE CONDITION CLASSIFICATION SYSTEM</span>'
        '<span class="footer-separator">•</span>'
        '<span class="footer-subtitle">Industrial Telemetry Decision Support</span>'
        '</div>'
        '</div>'
        '<div class="footer-tech-stack">'
        '<span class="footer-tech-badge">Python</span>'
        '<span class="footer-tech-badge">Streamlit</span>'
        '<span class="footer-tech-badge">Scikit-learn</span>'
        '<span class="footer-tech-badge">LightGBM</span>'
        '<span class="footer-tech-badge">Plotly</span>'
        '</div>'
        '<div class="footer-disclaimer">'
        'Operational decision-support tool. Does not replace qualified mechanical inspection.'
        '</div>'
        '</footer>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
