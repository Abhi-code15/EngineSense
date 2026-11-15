"""
Model Benchmarking Script for Engine Predictive Maintenance.

Algorithms:
1. Leakage-Free Random Forest (RandomForestClassifier)
2. XGBoost (XGBClassifier)
3. LightGBM (LGBMClassifier)

Guarantees:
- Strict Data Isolation:
    - Training Partition: data/processed/train_split.csv (12,502 rows)
    - Validation Partition: data/processed/val.csv (3,126 rows)
    - Holdout Test Set: data/processed/test.csv is NOT accessed or loaded.
- Resource-Safe Execution:
    - Run sequentially (one model at a time).
    - n_jobs=2 on RandomizedSearchCV, n_jobs=1 in estimators to prevent thread thrashing.
    - XGBoost: tree_method='hist' for fast memory-bounded CPU training.
    - LightGBM: verbose=-1, CPU execution.
    - Explicit memory monitoring and garbage collection between models.
- Artifacts:
    - models/experiments/random_forest_v2.joblib
    - models/experiments/xgboost_model.joblib
    - models/experiments/lightgbm_model.joblib
    - models/experiments/benchmark_results.csv
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
from sklearn.ensemble import RandomForestClassifier
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


def load_benchmark_partitions() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training and validation partitions with strict isolation.
    Guarantees test.csv is never accessed.
    """
    if not config.TRAIN_SPLIT_FILE.exists() or not config.VAL_FILE.exists():
        raise FileNotFoundError(
            f"Benchmark split files not found at {config.PROCESSED_DIR}. "
            "Run src/data_prep.py first to create train_split.csv and val.csv."
        )

    train_df = pd.read_csv(config.TRAIN_SPLIT_FILE)
    val_df = pd.read_csv(config.VAL_FILE)

    assert config.TARGET_COLUMN not in config.FEATURE_COLUMNS, "Target in features!"
    return train_df, val_df


def evaluate_predictions(
    model: Pipeline, X_val: pd.DataFrame, y_val: pd.Series
) -> Dict[str, Any]:
    """
    Compute comprehensive validation metrics.
    Note: Probabilities represent empirical voting/decision distributions, not calibrated posteriors.
    """
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    cm = confusion_matrix(y_val, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "precision": float(precision_score(y_val, y_pred, zero_division=0)),
        "recall": float(recall_score(y_val, y_pred, zero_division=0)),
        "f1": float(f1_score(y_val, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_val, y_pred)),
        "roc_auc": float(roc_auc_score(y_val, y_prob)),
        "pr_auc": float(average_precision_score(y_val, y_prob)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "confusion_matrix": cm.tolist(),
    }


def run_benchmark() -> pd.DataFrame:
    print("=" * 70)
    print("PHASE 7: LEAKAGE-FREE MODEL BENCHMARKING (RESOURCE-SAFE)")
    print("=" * 70)

    mem_init = get_memory_info()
    print(f"System Memory Initial: {mem_init['avail_gb']:.2f} GB free of {mem_init['total_gb']:.2f} GB ({mem_init['load_pct']:.0f}% load)")

    # 1. Load Data
    train_df, val_df = load_benchmark_partitions()
    X_train = train_df[config.FEATURE_COLUMNS]
    y_train = train_df[config.TARGET_COLUMN]

    X_val = val_df[config.FEATURE_COLUMNS]
    y_val = val_df[config.TARGET_COLUMN]

    print(f"Training partition   : {X_train.shape[0]} rows, {X_train.shape[1]} features (Class 1: {(y_train==1).sum()}, Class 0: {(y_train==0).sum()})")
    print(f"Validation partition : {X_val.shape[0]} rows, {X_val.shape[1]} features (Class 1: {(y_val==1).sum()}, Class 0: {(y_val==0).sum()})")
    print("Holdout Test Set     : RESERVED (Strictly NOT loaded or accessed)")

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    models_config = [
        {
            "name": "Random Forest",
            "save_path": config.EXPERIMENTS_DIR / "random_forest_v2.joblib",
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", RandomForestClassifier(random_state=config.RANDOM_STATE, n_jobs=1)),
            ]),
            "param_dist": {
                "clf__n_estimators": [100, 200, 300],
                "clf__max_depth": [5, 10, 15, None],
                "clf__min_samples_split": [2, 5, 10],
                "clf__min_samples_leaf": [1, 2, 4],
                "clf__max_features": ["sqrt", "log2", None],
                "clf__class_weight": [None, "balanced"],
            },
            "n_iter": 15,
            "search_n_jobs": 2,
        },
        {
            "name": "XGBoost",
            "save_path": config.EXPERIMENTS_DIR / "xgboost_model.joblib",
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
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
        {
            "name": "LightGBM",
            "save_path": config.EXPERIMENTS_DIR / "lightgbm_model.joblib",
            "pipeline": Pipeline([
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
            ]),
            "param_dist": {
                "clf__n_estimators": [100, 200, 300],
                "clf__max_depth": [-1, 5, 10],
                "clf__num_leaves": [15, 31, 63],
                "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "clf__subsample": [0.7, 0.85, 1.0],
                "clf__colsample_bytree": [0.7, 0.85, 1.0],
                "clf__class_weight": [None, "balanced"],
            },
            "n_iter": 15,
            "search_n_jobs": 1,
        },
    ]

    benchmark_records: List[Dict[str, Any]] = []

    for idx, m_cfg in enumerate(models_config, start=1):
        name = m_cfg["name"]
        save_path = m_cfg["save_path"]
        print("\n" + "-" * 70)
        print(f"[{idx}/3] Benchmarking {name}...")
        mem_before = get_memory_info()
        print(f"  Pre-run Memory: {mem_before['avail_gb']:.2f} GB free ({mem_before['load_pct']:.0f}% load)")

        search = RandomizedSearchCV(
            estimator=m_cfg["pipeline"],
            param_distributions=m_cfg["param_dist"],
            n_iter=m_cfg["n_iter"],
            cv=cv_strategy,
            scoring="f1",
            n_jobs=m_cfg.get("search_n_jobs", 2),  # Resource-safe CPU threading
            random_state=config.RANDOM_STATE,
            verbose=0,
        )

        t_start = time.perf_counter()
        search.fit(X_train, y_train)
        t_duration = time.perf_counter() - t_start

        best_model: Pipeline = search.best_estimator_
        best_params = search.best_params_
        best_cv_f1 = float(search.best_score_)

        # Calculate CV standard deviation for best candidate
        best_index = search.best_index_
        best_cv_std = float(search.cv_results_["std_test_score"][best_index])

        # Evaluate strictly on validation partition
        val_eval = evaluate_predictions(best_model, X_val, y_val)

        # Save model
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, save_path)
        file_size_kb = save_path.stat().st_size / 1024

        mem_after = get_memory_info()
        print(f"  Completed in {t_duration:.2f}s | Saved to {save_path.name} ({file_size_kb:.1f} KB)")
        print(f"  Post-run Memory: {mem_after['avail_gb']:.2f} GB free ({mem_after['load_pct']:.0f}% load)")
        print(f"  Mean 5-Fold CV F1 : {best_cv_f1:.4f} (+/- {best_cv_std:.4f})")
        print(f"  Validation F1-score: {val_eval['f1']:.4f}")
        print(f"  Validation ROC-AUC : {val_eval['roc_auc']:.4f} | PR-AUC: {val_eval['pr_auc']:.4f}")
        print(f"  Validation Accuracy: {val_eval['accuracy']:.4f} | Balanced Acc: {val_eval['balanced_accuracy']:.4f}")
        print(f"  Confusion Matrix   : [TN={val_eval['tn']}, FP={val_eval['fp']}, FN={val_eval['fn']}, TP={val_eval['tp']}]")

        record = {
            "model_name": name,
            "mean_cv_f1": round(best_cv_f1, 4),
            "std_cv_f1": round(best_cv_std, 4),
            "val_accuracy": round(val_eval["accuracy"], 4),
            "val_precision": round(val_eval["precision"], 4),
            "val_recall": round(val_eval["recall"], 4),
            "val_f1": round(val_eval["f1"], 4),
            "val_balanced_accuracy": round(val_eval["balanced_accuracy"], 4),
            "val_roc_auc": round(val_eval["roc_auc"], 4),
            "val_pr_auc": round(val_eval["pr_auc"], 4),
            "val_tn": val_eval["tn"],
            "val_fp": val_eval["fp"],
            "val_fn": val_eval["fn"],
            "val_tp": val_eval["tp"],
            "train_time_sec": round(t_duration, 2),
            "artifact_path": str(save_path.relative_to(PROJECT_ROOT)),
            "best_params": str(best_params),
        }
        benchmark_records.append(record)

        # Explicit garbage collection and memory release
        del search, best_model
        gc.collect()

    df_results = pd.DataFrame(benchmark_records)
    results_csv_path = config.EXPERIMENTS_DIR / "benchmark_results.csv"
    df_results.to_csv(results_csv_path, index=False)
    print("\n" + "=" * 70)
    print(f"BENCHMARK COMPLETE: Saved results to {results_csv_path}")
    print("=" * 70)
    return df_results


if __name__ == "__main__":
    run_benchmark()
