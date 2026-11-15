"""
Phase 8: Controlled Model Optimization Experiments.

Experiments:
1. Experiment 1: Threshold Optimization (using 5-fold OOF predictions on train_split.csv)
2. Experiment 2: Corrected LightGBM Subsampling (subsample_freq=1 with subsample=[0.7, 0.85, 1.0])
3. Experiment 3: Controlled Feature Engineering (physically justified sensor interaction features)

Strict Guarantees:
- Data Isolation:
    - Training Partition: data/processed/train_split.csv (12,502 rows)
    - Validation Partition: data/processed/val.csv (3,126 rows)
    - Holdout Test Partition: data/processed/test.csv is STRICTLY UNTOUCHED AND UNACCESSED.
- All preprocessing and feature transformations are fitted strictly within cross-validation folds.
- No overwriting of existing baseline or Phase 7 artifacts.
- Hardware & resource safety: sequential execution, n_jobs=2 where appropriate, continuous memory monitoring.
"""

from __future__ import annotations

import ctypes
import gc
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import config


# -----------------------------------------------------------------------------
# Memory Monitoring
# -----------------------------------------------------------------------------
class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def get_memory_info() -> Dict[str, float]:
    """Retrieve current Windows system memory statistics via kernel32."""
    try:
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        total_gb = stat.ullTotalPhys / (1024**3)
        avail_gb = stat.ullAvailPhys / (1024**3)
        load_pct = stat.dwMemoryLoad
        return {"total_gb": total_gb, "avail_gb": avail_gb, "load_pct": float(load_pct)}
    except Exception:
        return {"total_gb": 0.0, "avail_gb": 0.0, "load_pct": 0.0}


# -----------------------------------------------------------------------------
# Metric Computation Helper
# -----------------------------------------------------------------------------
def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics including specificity.
    Note: Probabilities represent empirical voting/decision distributions, not calibrated posteriors.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "specificity": specificity,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "confusion_matrix": cm.tolist(),
    }

    if y_prob is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = np.nan
        metrics["pr_auc"] = np.nan

    return metrics


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
def load_partitions() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training and validation partitions with strict isolation.
    Guarantees test.csv is never accessed.
    """
    if not config.TRAIN_SPLIT_FILE.exists() or not config.VAL_FILE.exists():
        raise FileNotFoundError(
            f"Split files not found at {config.PROCESSED_DIR}. "
            "Run src/data_prep.py first."
        )

    train_df = pd.read_csv(config.TRAIN_SPLIT_FILE)
    val_df = pd.read_csv(config.VAL_FILE)

    assert config.TARGET_COLUMN not in config.FEATURE_COLUMNS, "Target in features!"
    return train_df, val_df


# -----------------------------------------------------------------------------
# Experiment 1: Threshold Optimization via Out-Of-Fold Probabilities
# -----------------------------------------------------------------------------
def generate_oof_probabilities(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
) -> np.ndarray:
    """
    Generate out-of-fold probability estimates strictly on the training partition.
    Preprocessing (SimpleImputer) is fitted exclusively on each fold's training slice.
    """
    oof_probs = np.zeros(len(y_train), dtype=np.float64)

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), start=1):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]

        fold_pipe = clone(pipeline)
        fold_pipe.fit(X_fold_train, y_fold_train)

        # Empirical positive class probabilities
        probs = fold_pipe.predict_proba(X_fold_val)[:, 1]
        oof_probs[val_idx] = probs

    return oof_probs


def sweep_thresholds(
    y_true: np.ndarray,
    probs: np.ndarray,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    """Evaluate classification performance across a grid of decision thresholds."""
    records = []
    for thresh in thresholds:
        y_pred = (probs >= thresh).astype(int)
        m = compute_metrics(y_true, y_pred, y_prob=None)
        records.append({
            "threshold": round(float(thresh), 4),
            "accuracy": round(m["accuracy"], 4),
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "f1": round(m["f1"], 4),
            "balanced_accuracy": round(m["balanced_accuracy"], 4),
            "specificity": round(m["specificity"], 4),
            "tn": m["tn"],
            "fp": m["fp"],
            "fn": m["fn"],
            "tp": m["tp"],
        })
    return pd.DataFrame(records)


def run_experiment_1_threshold_optimization(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cv: StratifiedKFold,
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: THRESHOLD OPTIMIZATION (STRICT OUT-OF-FOLD)")
    print("=" * 70)

    # Load existing fitted benchmark models from Phase 7
    xgb_path = config.EXPERIMENTS_DIR / "xgboost_model.joblib"
    lgb_path = config.EXPERIMENTS_DIR / "lightgbm_model.joblib"

    fitted_models = {
        "XGBoost": joblib.load(xgb_path),
        "LightGBM": joblib.load(lgb_path),
    }

    threshold_grid = np.arange(0.10, 0.92, 0.02)
    exp1_results: Dict[str, Any] = {}
    sweep_dfs: List[pd.DataFrame] = []
    val_comparison_records: List[Dict[str, Any]] = []

    for model_name, pipeline in fitted_models.items():
        print(f"\nEvaluating OOF probabilities for {model_name}...")
        t0 = time.perf_counter()
        oof_probs = generate_oof_probabilities(pipeline, X_train, y_train, cv)
        oof_time = time.perf_counter() - t0
        print(f"  OOF generation completed in {oof_time:.2f}s.")

        df_sweep = sweep_thresholds(y_train.values, oof_probs, threshold_grid)
        df_sweep["model_name"] = model_name
        sweep_dfs.append(df_sweep)

        # 1. Max F1 Threshold
        best_f1_row = df_sweep.loc[df_sweep["f1"].idxmax()]
        thresh_f1 = float(best_f1_row["threshold"])

        # 2. High Recall Safety Threshold (Recall >= 0.90 with maximum F1/Precision)
        high_rec_candidates = df_sweep[df_sweep["recall"] >= 0.90]
        if not high_rec_candidates.empty:
            best_high_rec_row = high_rec_candidates.loc[high_rec_candidates["f1"].idxmax()]
        else:
            best_high_rec_row = df_sweep.loc[df_sweep["recall"].idxmax()]
        thresh_high_rec = float(best_high_rec_row["threshold"])

        # 3. Max Balanced Accuracy Threshold (Youden's Index / Equal Sensitivity-Specificity)
        best_bal_row = df_sweep.loc[df_sweep["balanced_accuracy"].idxmax()]
        thresh_bal = float(best_bal_row["threshold"])

        selected_thresholds = {
            "Default (0.50)": 0.50,
            "Max F1": thresh_f1,
            "High Recall (>=0.90)": thresh_high_rec,
            "Max Balanced Acc": thresh_bal,
        }

        print(f"  OOF Selected Thresholds for {model_name}:")
        for obj_name, th in selected_thresholds.items():
            row = df_sweep.loc[np.abs(df_sweep["threshold"] - th).argmin()]
            print(f"    - {obj_name:22s} [T={th:.2f}]: OOF F1={row['f1']:.4f}, Rec={row['recall']:.4f}, Prec={row['precision']:.4f}, BalAcc={row['balanced_accuracy']:.4f}, Spec={row['specificity']:.4f}")

        # Apply thresholds to the validation partition
        val_probs = pipeline.predict_proba(X_val)[:, 1]
        for obj_name, th in selected_thresholds.items():
            val_preds = (val_probs >= th).astype(int)
            m_val = compute_metrics(y_val.values, val_preds, y_prob=val_probs)
            val_comparison_records.append({
                "model_name": model_name,
                "objective": obj_name,
                "threshold": th,
                "val_accuracy": round(m_val["accuracy"], 4),
                "val_precision": round(m_val["precision"], 4),
                "val_recall": round(m_val["recall"], 4),
                "val_f1": round(m_val["f1"], 4),
                "val_balanced_accuracy": round(m_val["balanced_accuracy"], 4),
                "val_specificity": round(m_val["specificity"], 4),
                "val_roc_auc": round(m_val["roc_auc"], 4),
                "val_pr_auc": round(m_val["pr_auc"], 4),
                "val_tn": m_val["tn"],
                "val_fp": m_val["fp"],
                "val_fn": m_val["fn"],
                "val_tp": m_val["tp"],
            })

    df_all_sweeps = pd.concat(sweep_dfs, ignore_index=True)
    df_val_thresholds = pd.DataFrame(val_comparison_records)

    sweeps_csv_path = config.EXPERIMENTS_DIR / "threshold_oof_sweeps.csv"
    val_csv_path = config.EXPERIMENTS_DIR / "threshold_optimization_results.csv"
    df_all_sweeps.to_csv(sweeps_csv_path, index=False)
    df_val_thresholds.to_csv(val_csv_path, index=False)

    print(f"\nSaved threshold optimization results to {val_csv_path}")
    exp1_results["oof_sweeps"] = df_all_sweeps
    exp1_results["val_results"] = df_val_thresholds
    return exp1_results


# -----------------------------------------------------------------------------
# Experiment 2: Corrected LightGBM Subsampling
# -----------------------------------------------------------------------------
def run_experiment_2_corrected_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cv: StratifiedKFold,
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: CORRECTED LIGHTGBM SUBSAMPLING")
    print("=" * 70)

    mem_before = get_memory_info()
    print(f"Pre-run Memory: {mem_before['avail_gb']:.2f} GB free ({mem_before['load_pct']:.0f}% load)")

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "clf",
            LGBMClassifier(
                n_jobs=2,
                random_state=config.RANDOM_STATE,
                verbose=-1,
                objective="binary",
            ),
        ),
    ])

    param_dist = {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [-1, 5, 10],
        "clf__num_leaves": [15, 31, 63],
        "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "clf__subsample": [0.7, 0.85, 1.0],
        "clf__subsample_freq": [1],  # Activates bagging on every iteration!
        "clf__colsample_bytree": [0.7, 0.85, 1.0],
        "clf__class_weight": [None, "balanced"],
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=15,
        cv=cv,
        scoring="f1",
        n_jobs=1,  # Safe sequential process execution
        random_state=config.RANDOM_STATE,
        verbose=0,
    )

    t_start = time.perf_counter()
    search.fit(X_train, y_train)
    t_duration = time.perf_counter() - t_start

    best_model: Pipeline = search.best_estimator_
    best_params = search.best_params_
    best_cv_f1 = float(search.best_score_)
    best_cv_std = float(search.cv_results_["std_test_score"][search.best_index_])

    # Evaluate on validation partition
    val_probs = best_model.predict_proba(X_val)[:, 1]
    val_preds = best_model.predict(X_val)
    val_metrics = compute_metrics(y_val.values, val_preds, y_prob=val_probs)

    # Save to distinct model artifact
    save_path = config.EXPERIMENTS_DIR / "lightgbm_subsampled.joblib"
    joblib.dump(best_model, save_path)
    file_size_kb = save_path.stat().st_size / 1024

    mem_after = get_memory_info()
    print(f"Completed in {t_duration:.2f}s | Saved to {save_path.name} ({file_size_kb:.1f} KB)")
    print(f"Post-run Memory: {mem_after['avail_gb']:.2f} GB free ({mem_after['load_pct']:.0f}% load)")
    print(f"  Selected Params      : {best_params}")
    print(f"  Mean 5-Fold CV F1    : {best_cv_f1:.4f} (+/- {best_cv_std:.4f})")
    print(f"  Validation F1-score  : {val_metrics['f1']:.4f}")
    print(f"  Validation Precision : {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f}")
    print(f"  Validation Accuracy  : {val_metrics['accuracy']:.4f} | Balanced Acc: {val_metrics['balanced_accuracy']:.4f}")
    print(f"  Validation Specificity: {val_metrics['specificity']:.4f}")
    print(f"  Validation ROC-AUC   : {val_metrics['roc_auc']:.4f} | PR-AUC: {val_metrics['pr_auc']:.4f}")
    print(f"  Confusion Matrix     : [TN={val_metrics['tn']}, FP={val_metrics['fp']}, FN={val_metrics['fn']}, TP={val_metrics['tp']}]")

    result = {
        "model_name": "LightGBM (Subsampling Corrected)",
        "mean_cv_f1": round(best_cv_f1, 4),
        "std_cv_f1": round(best_cv_std, 4),
        "val_accuracy": round(val_metrics["accuracy"], 4),
        "val_precision": round(val_metrics["precision"], 4),
        "val_recall": round(val_metrics["recall"], 4),
        "val_f1": round(val_metrics["f1"], 4),
        "val_balanced_accuracy": round(val_metrics["balanced_accuracy"], 4),
        "val_specificity": round(val_metrics["specificity"], 4),
        "val_roc_auc": round(val_metrics["roc_auc"], 4),
        "val_pr_auc": round(val_metrics["pr_auc"], 4),
        "val_tn": val_metrics["tn"],
        "val_fp": val_metrics["fp"],
        "val_fn": val_metrics["fn"],
        "val_tp": val_metrics["tp"],
        "train_time_sec": round(t_duration, 2),
        "artifact_path": str(save_path.relative_to(PROJECT_ROOT)),
        "best_params": str(best_params),
        "model_size_kb": round(file_size_kb, 1),
    }

    del search, best_model
    gc.collect()
    return result


# -----------------------------------------------------------------------------
# Experiment 3: Controlled Feature Engineering
# -----------------------------------------------------------------------------
class EngineFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Physically justified thermodynamic and hydrodynamic sensor features.
    
    Features computed:
    1. temp_diff: Lub_Oil_Temperature - Coolant_Temperature
       (Thermodynamic gradient between lubrication and jacket cooling)
    2. oil_fuel_ratio: Lub_Oil_Pressure / (Fuel_Pressure + 1e-4)
       (Hydraulic balance between lubrication delivery and injection pressure)
    3. oil_per_rpm: Lub_Oil_Pressure / (Engine_RPM + 1.0)
       (RPM-normalized lubrication efficiency; detects pump cavitation/bearing clearance)
    4. coolant_per_rpm: Coolant_Pressure / (Engine_RPM + 1.0)
       (RPM-normalized cooling water head pressure)
    5. press_diff: Lub_Oil_Pressure - Coolant_Pressure
       (Cross-circuit hydraulic gradient protecting against coolant ingress)
    """

    def __init__(self, include_original: bool = True):
        self.include_original = include_original

    def fit(self, X: Any, y: Any = None) -> "EngineFeatureTransformer":
        return self

    def transform(self, X: Any) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X_arr = X.values.astype(np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)

        # Expected feature order:
        # 0: Engine_RPM, 1: Lub_Oil_Pressure, 2: Fuel_Pressure,
        # 3: Coolant_Pressure, 4: Lub_Oil_Temperature, 5: Coolant_Temperature
        rpm = X_arr[:, 0]
        oil_p = X_arr[:, 1]
        fuel_p = X_arr[:, 2]
        cool_p = X_arr[:, 3]
        oil_t = X_arr[:, 4]
        cool_t = X_arr[:, 5]

        # 1. Thermal delta between oil and coolant
        temp_diff = (oil_t - cool_t).reshape(-1, 1)

        # 2. Ratio of oil pressure to fuel pressure
        oil_fuel_ratio = (oil_p / (fuel_p + 1e-4)).reshape(-1, 1)

        # 3. RPM-normalized oil pressure (direct indicator of mechanical lubrication health)
        oil_per_rpm = (oil_p / (rpm + 1.0)).reshape(-1, 1)

        # 4. RPM-normalized coolant pressure
        coolant_per_rpm = (cool_p / (rpm + 1.0)).reshape(-1, 1)

        # 5. Pressure gradient across lubrication and cooling circuits
        press_diff = (oil_p - cool_p).reshape(-1, 1)

        engineered = np.hstack([temp_diff, oil_fuel_ratio, oil_per_rpm, coolant_per_rpm, press_diff])

        if self.include_original:
            return np.hstack([X_arr, engineered])
        return engineered


def run_experiment_3_feature_engineering(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cv: StratifiedKFold,
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: CONTROLLED FEATURE ENGINEERING")
    print("=" * 70)

    # Benchmark both LightGBM (subsampled) and XGBoost with engineered features
    models_to_test = [
        {
            "name": "LightGBM + Engineered Features",
            "save_path": config.EXPERIMENTS_DIR / "lightgbm_engineered.joblib",
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("engineer", EngineFeatureTransformer(include_original=True)),
                (
                    "clf",
                    LGBMClassifier(
                        n_jobs=2,
                        random_state=config.RANDOM_STATE,
                        verbose=-1,
                        objective="binary",
                    ),
                ),
            ]),
            "param_dist": {
                "clf__n_estimators": [100, 200, 300],
                "clf__max_depth": [-1, 5, 10],
                "clf__num_leaves": [15, 31, 63],
                "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "clf__subsample": [0.7, 0.85, 1.0],
                "clf__subsample_freq": [1],
                "clf__colsample_bytree": [0.7, 0.85, 1.0],
                "clf__class_weight": [None, "balanced"],
            },
            "n_iter": 15,
            "search_n_jobs": 1,
        },
        {
            "name": "XGBoost + Engineered Features",
            "save_path": config.EXPERIMENTS_DIR / "xgboost_engineered.joblib",
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("engineer", EngineFeatureTransformer(include_original=True)),
                (
                    "clf",
                    XGBClassifier(
                        tree_method="hist",
                        n_jobs=1,
                        random_state=config.RANDOM_STATE,
                        eval_metric="logloss",
                        objective="binary:logistic",
                    ),
                ),
            ]),
            "param_dist": {
                "clf__n_estimators": [100, 200, 300],
                "clf__max_depth": [3, 5, 7],
                "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "clf__subsample": [0.7, 0.85, 1.0],
                "clf__colsample_bytree": [0.7, 0.85, 1.0],
                "clf__scale_pos_weight": [1.0, 4619 / 7883],
            },
            "n_iter": 15,
            "search_n_jobs": 2,
        },
    ]

    exp3_records: List[Dict[str, Any]] = []

    for m_cfg in models_to_test:
        name = m_cfg["name"]
        save_path = m_cfg["save_path"]
        print(f"\nBenchmarking {name}...")

        mem_before = get_memory_info()
        print(f"  Pre-run Memory: {mem_before['avail_gb']:.2f} GB free ({mem_before['load_pct']:.0f}% load)")

        search = RandomizedSearchCV(
            estimator=m_cfg["pipeline"],
            param_distributions=m_cfg["param_dist"],
            n_iter=m_cfg["n_iter"],
            cv=cv,
            scoring="f1",
            n_jobs=m_cfg["search_n_jobs"],
            random_state=config.RANDOM_STATE,
            verbose=0,
        )

        t_start = time.perf_counter()
        search.fit(X_train, y_train)
        t_duration = time.perf_counter() - t_start

        best_model: Pipeline = search.best_estimator_
        best_params = search.best_params_
        best_cv_f1 = float(search.best_score_)
        best_cv_std = float(search.cv_results_["std_test_score"][search.best_index_])

        val_probs = best_model.predict_proba(X_val)[:, 1]
        val_preds = best_model.predict(X_val)
        val_metrics = compute_metrics(y_val.values, val_preds, y_prob=val_probs)

        joblib.dump(best_model, save_path)
        file_size_kb = save_path.stat().st_size / 1024

        mem_after = get_memory_info()
        print(f"  Completed in {t_duration:.2f}s | Saved to {save_path.name} ({file_size_kb:.1f} KB)")
        print(f"  Post-run Memory: {mem_after['avail_gb']:.2f} GB free ({mem_after['load_pct']:.0f}% load)")
        print(f"  Mean 5-Fold CV F1    : {best_cv_f1:.4f} (+/- {best_cv_std:.4f})")
        print(f"  Validation F1-score  : {val_metrics['f1']:.4f}")
        print(f"  Validation Precision : {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f}")
        print(f"  Validation Accuracy  : {val_metrics['accuracy']:.4f} | Balanced Acc: {val_metrics['balanced_accuracy']:.4f}")
        print(f"  Validation Specificity: {val_metrics['specificity']:.4f}")
        print(f"  Validation ROC-AUC   : {val_metrics['roc_auc']:.4f} | PR-AUC: {val_metrics['pr_auc']:.4f}")
        print(f"  Confusion Matrix     : [TN={val_metrics['tn']}, FP={val_metrics['fp']}, FN={val_metrics['fn']}, TP={val_metrics['tp']}]")

        record = {
            "model_name": name,
            "mean_cv_f1": round(best_cv_f1, 4),
            "std_cv_f1": round(best_cv_std, 4),
            "val_accuracy": round(val_metrics["accuracy"], 4),
            "val_precision": round(val_metrics["precision"], 4),
            "val_recall": round(val_metrics["recall"], 4),
            "val_f1": round(val_metrics["f1"], 4),
            "val_balanced_accuracy": round(val_metrics["balanced_accuracy"], 4),
            "val_specificity": round(val_metrics["specificity"], 4),
            "val_roc_auc": round(val_metrics["roc_auc"], 4),
            "val_pr_auc": round(val_metrics["pr_auc"], 4),
            "val_tn": val_metrics["tn"],
            "val_fp": val_metrics["fp"],
            "val_fn": val_metrics["fn"],
            "val_tp": val_metrics["tp"],
            "train_time_sec": round(t_duration, 2),
            "artifact_path": str(save_path.relative_to(PROJECT_ROOT)),
            "best_params": str(best_params),
            "model_size_kb": round(file_size_kb, 1),
        }
        exp3_records.append(record)

        del search, best_model
        gc.collect()

    return {"records": exp3_records}


# -----------------------------------------------------------------------------
# Main Execution Runner
# -----------------------------------------------------------------------------
def run_all_phase8_experiments() -> None:
    print("=" * 70)
    print("PHASE 8: CONTROLLED MODEL OPTIMIZATION EXPERIMENTS")
    print("=" * 70)

    # 1. Load Data Partitions
    X_train_df, X_val_df = load_partitions()
    X_train = X_train_df[config.FEATURE_COLUMNS]
    y_train = X_train_df[config.TARGET_COLUMN]

    X_val = X_val_df[config.FEATURE_COLUMNS]
    y_val = X_val_df[config.TARGET_COLUMN]

    print(f"Training partition   : {X_train.shape[0]} rows, {X_train.shape[1]} features")
    print(f"Validation partition : {X_val.shape[0]} rows, {X_val.shape[1]} features")
    print("Holdout Test Set     : RESERVED (Strictly NOT loaded or accessed)")

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)

    # Experiment 1: Threshold Optimization
    exp1_out = run_experiment_1_threshold_optimization(X_train, y_train, X_val, y_val, cv_strategy)

    # Experiment 2: Corrected LightGBM Subsampling
    exp2_out = run_experiment_2_corrected_lightgbm(X_train, y_train, X_val, y_val, cv_strategy)

    # Experiment 3: Controlled Feature Engineering
    exp3_out = run_experiment_3_feature_engineering(X_train, y_train, X_val, y_val, cv_strategy)

    # Compile Overall Phase 8 Summary Table
    # Load Phase 7 benchmark results for baseline reference
    phase7_csv = config.EXPERIMENTS_DIR / "benchmark_results.csv"
    df_phase7 = pd.read_csv(phase7_csv)

    summary_records = []
    # Add Phase 7 models
    for _, row in df_phase7.iterrows():
        tn = int(row["val_tn"])
        fp = int(row["val_fp"])
        spec = round(float(tn / (tn + fp)), 4) if (tn + fp) > 0 else 0.0
        summary_records.append({
            "experiment": "Phase 7 Baseline",
            "model_name": row["model_name"],
            "mean_cv_f1": row["mean_cv_f1"],
            "std_cv_f1": row["std_cv_f1"],
            "val_f1": row["val_f1"],
            "val_precision": row["val_precision"],
            "val_recall": row["val_recall"],
            "val_balanced_accuracy": row["val_balanced_accuracy"],
            "val_specificity": spec,
            "val_roc_auc": row["val_roc_auc"],
            "val_pr_auc": row["val_pr_auc"],
            "train_time_sec": row["train_time_sec"],
            "artifact_path": row["artifact_path"],
        })

    # Add Experiment 2
    summary_records.append({
        "experiment": "Exp 2: Subsampling Corrected",
        "model_name": exp2_out["model_name"],
        "mean_cv_f1": exp2_out["mean_cv_f1"],
        "std_cv_f1": exp2_out["std_cv_f1"],
        "val_f1": exp2_out["val_f1"],
        "val_precision": exp2_out["val_precision"],
        "val_recall": exp2_out["val_recall"],
        "val_balanced_accuracy": exp2_out["val_balanced_accuracy"],
        "val_specificity": exp2_out["val_specificity"],
        "val_roc_auc": exp2_out["val_roc_auc"],
        "val_pr_auc": exp2_out["val_pr_auc"],
        "train_time_sec": exp2_out["train_time_sec"],
        "artifact_path": exp2_out["artifact_path"],
    })

    # Add Experiment 3
    for rec in exp3_out["records"]:
        summary_records.append({
            "experiment": "Exp 3: Feature Engineering",
            "model_name": rec["model_name"],
            "mean_cv_f1": rec["mean_cv_f1"],
            "std_cv_f1": rec["std_cv_f1"],
            "val_f1": rec["val_f1"],
            "val_precision": rec["val_precision"],
            "val_recall": rec["val_recall"],
            "val_balanced_accuracy": rec["val_balanced_accuracy"],
            "val_specificity": rec["val_specificity"],
            "val_roc_auc": rec["val_roc_auc"],
            "val_pr_auc": rec["val_pr_auc"],
            "train_time_sec": rec["train_time_sec"],
            "artifact_path": rec["artifact_path"],
        })

    df_summary = pd.DataFrame(summary_records)
    summary_csv = config.EXPERIMENTS_DIR / "phase8_experiments_summary.csv"
    df_summary.to_csv(summary_csv, index=False)

    print("\n" + "=" * 70)
    print(f"PHASE 8 COMPLETE: Saved overall summary to {summary_csv}")
    print("=" * 70)


if __name__ == "__main__":
    run_all_phase8_experiments()
