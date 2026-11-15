"""
Inference utilities for the predictive maintenance model.

These functions are used both by scripts and by the Streamlit app
to load a trained model (production LightGBM, baseline Random Forest,
or Hugging Face hub) and generate validated predictions from raw sensor values.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

# IMPORTANT: In 64-bit Windows Python environments, lightgbm must be imported
# prior to joblib unpickling to ensure C++ OpenMP runtime binds properly.
import lightgbm as lgb
import joblib
import numpy as np
import pandas as pd

import config
from hf_model_utils import download_model


def load_production_model() -> object:
    """
    Load the promoted production LightGBM model from models/production/.
    """
    if not config.PRODUCTION_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Production model not found at {config.PRODUCTION_MODEL_PATH}. "
            "Ensure Phase 10 packaging has been completed."
        )
    return joblib.load(config.PRODUCTION_MODEL_PATH)


def load_baseline_model() -> object:
    """
    Load the original baseline Random Forest model from models/best_model.joblib.
    """
    if not config.BEST_MODEL_LOCAL_PATH.exists():
        raise FileNotFoundError(
            f"Baseline model not found at {config.BEST_MODEL_LOCAL_PATH}."
        )
    return joblib.load(config.BEST_MODEL_LOCAL_PATH)


def load_local_model(model_type: str = "production") -> object:
    """
    Backward-compatible loader for local models.
    """
    if model_type == "baseline":
        return load_baseline_model()
    # Default to production model if present; fall back to baseline if needed
    if config.PRODUCTION_MODEL_PATH.exists():
        return load_production_model()
    return load_baseline_model()


def load_hf_model() -> object:
    """
    Load the model directly from the Hugging Face model hub.
    """
    return download_model()


def load_model(source: str = "production") -> object:
    """
    Unified model dispatcher supporting 'production', 'baseline', and 'hf'.
    """
    if source == "production":
        return load_production_model()
    elif source == "baseline":
        return load_baseline_model()
    elif source == "local":
        return load_local_model()
    elif source == "hf":
        return load_hf_model()
    else:
        raise ValueError(
            f"Unsupported model source: '{source}'. "
            "Choose from ['production', 'baseline', 'local', 'hf']."
        )


def validate_and_build_input_dataframe(
    inputs: Dict[str, Any],
) -> pd.DataFrame:
    """
    Validate all six input features and construct a single-row DataFrame
    with columns ordered strictly according to config.FEATURE_COLUMNS.

    Validation rules:
    - All 6 required features must be present.
    - Values must be numeric (int or float).
    - NaN, null, and infinite values are strictly rejected.
    - No silent zero substitution.
    """
    if not isinstance(inputs, dict):
        raise TypeError(f"Expected inputs as dict, got {type(inputs).__name__}")

    row_data: Dict[str, float] = {}
    for col in config.FEATURE_COLUMNS:
        if col not in inputs:
            raise ValueError(
                f"Missing required feature: '{col}'. "
                f"Expected exact features: {config.FEATURE_COLUMNS}"
            )
        val = inputs[col]
        if val is None:
            raise ValueError(
                f"Feature '{col}' cannot be None. Expected a finite numeric value."
            )
        try:
            float_val = float(val)
        except (ValueError, TypeError) as err:
            raise TypeError(
                f"Feature '{col}' must be numeric. Cannot convert '{val}' of type {type(val).__name__} to float."
            ) from err

        if math.isnan(float_val) or math.isinf(float_val):
            raise ValueError(
                f"Feature '{col}' has invalid value '{val}'. Cannot be NaN or Infinite."
            )

        row_data[col] = float_val

    # Construct DataFrame with strictly preserved feature order
    return pd.DataFrame([row_data], columns=config.FEATURE_COLUMNS)


def build_input_dataframe(
    inputs: Dict[str, Any],
) -> pd.DataFrame:
    """
    Convert a dictionary of feature values into a validated single-row DataFrame.
    Preserves exact column order and enforces numeric validation.
    """
    return validate_and_build_input_dataframe(inputs)


def predict_engine_condition(
    inputs: Dict[str, Any],
    model: Optional[object] = None,
    source: str = "production",
    threshold: float = config.DEFAULT_THRESHOLD,
) -> Dict[str, Any]:
    """
    Predict whether the engine requires maintenance using the specified model.

    Parameters
    ----------
    inputs : dict
        Keys must correspond to the 6 features in config.FEATURE_COLUMNS.
    model : object, optional
        Pre-loaded sklearn Pipeline model. If None, it will be loaded from `source`.
    source : {'production', 'baseline', 'local', 'hf'}
        If model is None, determines where to load it from. Defaults to 'production'.
    threshold : float
        Decision boundary threshold for the positive class (default: 0.50).

    Returns
    -------
    dict
        Comprehensive inference results including binary prediction,
        class probabilities, applied threshold, and condition labels.
    """
    if model is None:
        model = load_model(source=source)

    df = build_input_dataframe(inputs)
    prob_array = model.predict_proba(df)[0]

    # Binary classification: index 1 is positive class (Engine Condition = 1)
    if len(prob_array) > 1:
        prob_faulty = float(prob_array[1])
        prob_normal = float(prob_array[0])
    else:
        prob_faulty = float(prob_array[0])
        prob_normal = float(1.0 - prob_faulty)

    pred = int(prob_faulty >= threshold)

    return {
        "prediction": pred,
        "probability_faulty": prob_faulty,
        "probability_normal": prob_normal,
        "threshold": float(threshold),
        "condition_label": "Faulty / Maintenance Required" if pred == 1 else "Normal / Healthy",
        "is_faulty": bool(pred == 1),
        "model_source": source,
    }


__all__ = [
    "load_production_model",
    "load_baseline_model",
    "load_local_model",
    "load_hf_model",
    "load_model",
    "validate_and_build_input_dataframe",
    "build_input_dataframe",
    "predict_engine_condition",
]
