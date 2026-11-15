"""
Model training and experimentation tracking script for the predictive maintenance project.

Key Architectural Guarantees:
- Leakage-Free Preprocessing: SimpleImputer(strategy='median') inside sklearn Pipeline.
- Tree-Optimized: StandardScaler removed (unnecessary for Random Forest).
- Train / Validation Isolation:
    - 64% Train (hyperparameter search & 5-fold cross-validation)
    - 16% Validation (model selection & tuning checks)
- Holdout Test Set Protection:
    - 20% Holdout Test set is strictly quarantined.
    - Test data is NEVER loaded, accessed, or evaluated during ordinary training.
    - Final evaluation must be performed independently only after all modeling decisions are finalized.
- Baseline Preservation: New models saved to models/experiments/ without overwriting models/best_model.joblib.
- Resilient Error Handling: Graceful fallbacks and explicit error reporting for MLflow and Hugging Face.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline

import config
from hf_data_utils import download_dataset_file
from hf_model_utils import upload_model


def _load_training_and_validation_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training and validation splits with strict isolation.

    Guarantees:
    - Holdout test.csv is NOT loaded, accessed, or returned by this function.
    - Training partition (64%) and validation partition (16%) are returned.
    """
    if not config.TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Baseline train file not found at {config.TRAIN_FILE}. "
            "Run src/data_prep.py first to generate the splits."
        )

    if config.TRAIN_SPLIT_FILE.exists() and config.VAL_FILE.exists():
        train_df = pd.read_csv(config.TRAIN_SPLIT_FILE)
        val_df = pd.read_csv(config.VAL_FILE)
    else:
        # Partition existing train.csv into train (80%) and val (20%)
        train_full = pd.read_csv(config.TRAIN_FILE)
        X_full = train_full[config.FEATURE_COLUMNS]
        y_full = train_full[config.TARGET_COLUMN]

        X_tr, X_va, y_tr, y_va = train_test_split(
            X_full,
            y_full,
            test_size=config.VAL_SIZE,
            random_state=config.RANDOM_STATE,
            stratify=y_full,
        )

        train_df = X_tr.copy()
        train_df[config.TARGET_COLUMN] = y_tr

        val_df = X_va.copy()
        val_df[config.TARGET_COLUMN] = y_va

        # Persist splits for reproducible downstream use
        train_df.to_csv(config.TRAIN_SPLIT_FILE, index=False)
        val_df.to_csv(config.VAL_FILE, index=False)

    return train_df, val_df


def evaluate_holdout_test(model: Pipeline) -> Dict[str, float]:
    """
    Independent final evaluation workflow on the holdout test set.

    CRITICAL PROTOCOL:
    The holdout test set (data/processed/test.csv) must strictly remain isolated.
    It may ONLY be evaluated once all model architecture selection, hyperparameter
    tuning, and decision threshold decisions have been finalized.
    Do NOT invoke this function during training, model selection, or iteration.
    """
    if not config.TEST_FILE.exists():
        raise FileNotFoundError(f"Holdout test file not found at {config.TEST_FILE}")

    test_df = pd.read_csv(config.TEST_FILE)
    X_test = test_df[config.FEATURE_COLUMNS]
    y_test = test_df[config.TARGET_COLUMN]
    assert config.TARGET_COLUMN not in X_test.columns, "Target leaked into X_test!"

    return _evaluate_model(model, X_test, y_test)


def _build_model_and_search_space() -> Tuple[Pipeline, Dict]:
    """
    Build a leakage-free sklearn Pipeline and hyperparameter search space.

    Architecture:
    - Step 1: SimpleImputer(strategy='median') - computed strictly on training folds.
    - Step 2: RandomForestClassifier - tree ensemble (no scaling required).
    """
    clf = RandomForestClassifier(random_state=config.RANDOM_STATE)

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", clf),
        ]
    )

    param_distributions = {
        "clf__n_estimators": [100, 200, 300, 400],
        "clf__max_depth": [None, 5, 10, 20],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__max_features": ["sqrt", "log2", None],
        "clf__bootstrap": [True, False],
    }

    return pipeline, param_distributions


def _evaluate_model(
    model: Pipeline, X: pd.DataFrame, y: pd.Series
) -> Dict[str, float]:
    """
    Compute standard binary classification metrics including ROC-AUC.
    """
    y_pred = model.predict(X)
    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
    }
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y, y_prob))
        except Exception:
            pass
    return metrics


def main() -> None:
    """
    Execute the leakage-free training, cross-validation, and validation-selection pipeline.
    Holdout test set is strictly quarantined and NOT accessed during training.
    """
    print("=" * 60)
    print("LEAKAGE-FREE TRAINING PIPELINE EXECUTION")
    print("=" * 60)

    print("\n[Step 1/5] Loading training and validation partitions (Holdout test is protected)...")
    train_df, val_df = _load_training_and_validation_data()

    # Extract features and targets with strict exclusion guarantees
    X_train = train_df[config.FEATURE_COLUMNS]
    y_train = train_df[config.TARGET_COLUMN]

    X_val = val_df[config.FEATURE_COLUMNS]
    y_val = val_df[config.TARGET_COLUMN]

    # Verify target is never in feature matrix
    assert config.TARGET_COLUMN not in X_train.columns, "Target leaked into X_train!"
    assert config.TARGET_COLUMN not in X_val.columns, "Target leaked into X_val!"

    total_train_val = len(train_df) + len(val_df)
    print(f"  Training partition   : {X_train.shape[0]} samples (80% of training data)")
    print(f"  Validation partition : {X_val.shape[0]} samples (20% of training data)")
    print("  Holdout Test Set     : RESERVED (Quarantined; not accessed during training)")

    print("\n[Step 2/5] Building leakage-free Pipeline and hyperparameter search space...")
    pipeline, param_distributions = _build_model_and_search_space()
    print("  Pipeline steps:", [name for name, _ in pipeline.steps])

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=20,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
        random_state=config.RANDOM_STATE,
    )

    # Configure MLflow tracking with error resilience
    mlflow_enabled = False
    try:
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
        mlflow_enabled = True
        print(f"  MLflow tracking initialized: {config.MLFLOW_TRACKING_URI}")
    except Exception as e:
        print(f"  Warning: MLflow tracking initialization failed: {e}. Running without MLflow tracking.")

    print("\n[Step 3/5] Fitting RandomizedSearchCV strictly on training partition (5-fold CV)...")
    search.fit(X_train, y_train)
    best_model: Pipeline = search.best_estimator_
    best_params = search.best_params_

    print("\n[Step 4/5] Model Selection & Validation Evaluation:")
    val_metrics = _evaluate_model(best_model, X_val, y_val)
    print("  Best Hyperparameters Identified:")
    for k, v in best_params.items():
        print(f"    {k}: {v}")
    print("  Validation Metrics (Model Selection):")
    for k, v in val_metrics.items():
        print(f"    {k:12s}: {v:.4f}")

    print("\n[Step 5/5] Artifact Persistence & MLflow Logging:")
    # Persist model locally to experiments directory (NEVER overwrite baseline)
    config.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, config.LEAKAGE_FREE_MODEL_PATH)
    print(f"  Leakage-free model saved to : {config.LEAKAGE_FREE_MODEL_PATH}")
    print(f"  Original baseline preserved  : {config.BEST_MODEL_LOCAL_PATH}")

    # MLflow logging with explicit failure reporting
    if mlflow_enabled:
        try:
            with mlflow.start_run(run_name="LeakageFree_RF_RandomSearch"):
                # Log evaluated candidate configurations
                results = search.cv_results_
                for i in range(len(results["params"])):
                    param_set = results["params"][i]
                    mean_score = results["mean_test_score"][i]
                    try:
                        with mlflow.start_run(nested=True):
                            mlflow.log_params(param_set)
                            mlflow.log_metric("mean_cv_f1", float(mean_score))
                    except Exception as nested_err:
                        print(f"  Warning: Failed to log nested MLflow candidate {i}: {nested_err}")

                # Log best parameters and validation metrics
                mlflow.log_params(best_params)
                for name, val in val_metrics.items():
                    mlflow.log_metric(f"val_{name}", val)

                mlflow.log_artifact(str(config.LEAKAGE_FREE_MODEL_PATH), artifact_path="artifacts")
                mlflow.sklearn.log_model(best_model, artifact_path="leakage_free_engine_model")
                print("  MLflow tracking run logged successfully.")
        except Exception as mlflow_run_err:
            print(f"  ERROR: MLflow run logging failed: {mlflow_run_err}. Local model artifacts remain intact.")
    else:
        print("  Info: MLflow logging was disabled or unavailable. Local artifacts saved successfully.")

    # Hugging Face upload handling with clear error diagnostics
    if config.HF_TOKEN and config.HF_MODEL_REPO:
        try:
            print("\nUploading leakage-free model to Hugging Face model hub...")
            upload_model(
                local_model_path=config.LEAKAGE_FREE_MODEL_PATH,
                repo_id=config.HF_MODEL_REPO,
                repo_path="models/experiments/leakage_free_rf.joblib",
                token=config.HF_TOKEN,
            )
            print("Model upload to Hugging Face completed successfully.")
        except Exception as e:
            print(f"ERROR: Failed to upload model to Hugging Face model hub ({config.HF_MODEL_REPO}): {e}")
    else:
        print("\nInfo: HF_TOKEN not configured. Skipping Hugging Face upload. Local training and validation complete.")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE: Holdout test set was NOT evaluated.")
    print("Holdout evaluation must only be run after all modeling decisions are finalized.")
    print("=" * 60)


if __name__ == "__main__":
    main()
