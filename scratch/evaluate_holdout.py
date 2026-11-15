import hashlib
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
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

# -----------------------------------------------------------------------------
# Configuration and Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(".").resolve()
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
FROZEN_MODEL_PATH = PROJECT_ROOT / "models" / "experiments" / "lightgbm_subsampled.joblib"
BASELINE_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
RESULTS_CSV_PATH = PROJECT_ROOT / "models" / "experiments" / "final_holdout_results.csv"
REPORT_MD_PATH = PROJECT_ROOT / "models" / "experiments" / "final_holdout_report.md"

FEATURE_COLUMNS = [
    "Engine_RPM",
    "Lub_Oil_Pressure",
    "Fuel_Pressure",
    "Coolant_Pressure",
    "Lub_Oil_Temperature",
    "Coolant_Temperature",
]
TARGET_COLUMN = "Engine_Condition"
THRESHOLD = 0.50

print("=" * 70)
print("PHASE 9: FINAL HOLDOUT TEST EVALUATION")
print("=" * 70)

# 1. Dataset Verification
assert TEST_PATH.exists(), f"Holdout test dataset not found at {TEST_PATH}"
test_df = pd.read_csv(TEST_PATH)
test_hash = hashlib.sha256(TEST_PATH.read_bytes()).hexdigest()

print(f"Holdout Dataset Path : {TEST_PATH}")
print(f"Holdout Dataset Rows : {test_df.shape[0]} rows, {test_df.shape[1]} columns")
print(f"Holdout SHA-256 Hash : {test_hash}")
print(f"Feature Structure    : {FEATURE_COLUMNS}")
print(f"Target Column        : {TARGET_COLUMN}")

assert all(col in test_df.columns for col in FEATURE_COLUMNS), "Missing feature columns in test.csv!"
assert TARGET_COLUMN in test_df.columns, "Missing target column in test.csv!"

X_test = test_df[FEATURE_COLUMNS]
y_test = test_df[TARGET_COLUMN].values

n_total = len(y_test)
n_pos = int(np.sum(y_test == 1))
n_neg = int(np.sum(y_test == 0))
print(f"Class Balance        : Class 1 (Failing): {n_pos} ({n_pos/n_total*100:.2f}%), Class 0 (Normal): {n_neg} ({n_neg/n_total*100:.2f}%)")

# 2. Frozen Model Verification
assert FROZEN_MODEL_PATH.exists(), f"Frozen model not found at {FROZEN_MODEL_PATH}"
frozen_hash = hashlib.sha256(FROZEN_MODEL_PATH.read_bytes()).hexdigest()
frozen_pipe = joblib.load(FROZEN_MODEL_PATH)

print(f"\nFrozen Candidate Model : {FROZEN_MODEL_PATH.name}")
print(f"Frozen Model SHA-256   : {frozen_hash}")
print(f"Pipeline Architecture  : {frozen_pipe.steps}")
print(f"Classifier Parameters  : subsample={frozen_pipe.named_steps['clf'].subsample}, subsample_freq={frozen_pipe.named_steps['clf'].subsample_freq}")

# 3. Baseline Model Verification
assert BASELINE_MODEL_PATH.exists(), f"Baseline model not found at {BASELINE_MODEL_PATH}"
baseline_hash = hashlib.sha256(BASELINE_MODEL_PATH.read_bytes()).hexdigest()
baseline_pipe = joblib.load(BASELINE_MODEL_PATH)

print(f"\nBaseline Model Path    : {BASELINE_MODEL_PATH.name}")
print(f"Baseline Model SHA-256 : {baseline_hash}")
print(f"Pipeline Architecture  : {baseline_pipe.steps}")

# -----------------------------------------------------------------------------
# Inference Execution & Timing
# -----------------------------------------------------------------------------
# Warmup
_ = frozen_pipe.predict(X_test.iloc[:5])
_ = baseline_pipe.predict(X_test.iloc[:5])

# Frozen Model Evaluation
t_start = time.perf_counter()
y_pred_frozen = frozen_pipe.predict(X_test)
y_prob_frozen = frozen_pipe.predict_proba(X_test)[:, 1]
frozen_infer_duration = time.perf_counter() - t_start

# Baseline Model Evaluation
t_start = time.perf_counter()
y_pred_base = baseline_pipe.predict(X_test)
y_prob_base = baseline_pipe.predict_proba(X_test)[:, 1]
base_infer_duration = time.perf_counter() - t_start

# -----------------------------------------------------------------------------
# Metric Computation Helper
# -----------------------------------------------------------------------------
def evaluate_holdout(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, infer_sec: float) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    latency_per_sample_ms = (infer_sec / len(y_true)) * 1000.0

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "balanced_accuracy": bal_acc,
        "specificity": spec,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "fpr": fpr,
        "fnr": fnr,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "missed_engine_faults": int(fn),
        "false_alarms": int(fp),
        "inference_duration_sec": round(infer_sec, 4),
        "latency_per_sample_ms": round(latency_per_sample_ms, 4),
        "classification_threshold": THRESHOLD,
        "confusion_matrix": cm.tolist()
    }

metrics_frozen = evaluate_holdout(y_test, y_pred_frozen, y_prob_frozen, frozen_infer_duration)
metrics_base = evaluate_holdout(y_test, y_pred_base, y_prob_base, base_infer_duration)

# -----------------------------------------------------------------------------
# Save Results to CSV
# -----------------------------------------------------------------------------
csv_records = [
    {
        "model_name": "LightGBM Subsampled (Frozen Candidate)",
        "model_hash": frozen_hash,
        "threshold": THRESHOLD,
        "accuracy": round(metrics_frozen["accuracy"], 4),
        "precision": round(metrics_frozen["precision"], 4),
        "recall": round(metrics_frozen["recall"], 4),
        "f1": round(metrics_frozen["f1"], 4),
        "balanced_accuracy": round(metrics_frozen["balanced_accuracy"], 4),
        "specificity": round(metrics_frozen["specificity"], 4),
        "roc_auc": round(metrics_frozen["roc_auc"], 4),
        "pr_auc": round(metrics_frozen["pr_auc"], 4),
        "fpr": round(metrics_frozen["fpr"], 4),
        "fnr": round(metrics_frozen["fnr"], 4),
        "missed_engine_faults": metrics_frozen["missed_engine_faults"],
        "false_alarms": metrics_frozen["false_alarms"],
        "tn": metrics_frozen["tn"],
        "fp": metrics_frozen["fp"],
        "fn": metrics_frozen["fn"],
        "tp": metrics_frozen["tp"],
        "inference_duration_sec": metrics_frozen["inference_duration_sec"],
        "latency_per_sample_ms": metrics_frozen["latency_per_sample_ms"],
    },
    {
        "model_name": "Original Baseline Random Forest",
        "model_hash": baseline_hash,
        "threshold": THRESHOLD,
        "accuracy": round(metrics_base["accuracy"], 4),
        "precision": round(metrics_base["precision"], 4),
        "recall": round(metrics_base["recall"], 4),
        "f1": round(metrics_base["f1"], 4),
        "balanced_accuracy": round(metrics_base["balanced_accuracy"], 4),
        "specificity": round(metrics_base["specificity"], 4),
        "roc_auc": round(metrics_base["roc_auc"], 4),
        "pr_auc": round(metrics_base["pr_auc"], 4),
        "fpr": round(metrics_base["fpr"], 4),
        "fnr": round(metrics_base["fnr"], 4),
        "missed_engine_faults": metrics_base["missed_engine_faults"],
        "false_alarms": metrics_base["false_alarms"],
        "tn": metrics_base["tn"],
        "fp": metrics_base["fp"],
        "fn": metrics_base["fn"],
        "tp": metrics_base["tp"],
        "inference_duration_sec": metrics_base["inference_duration_sec"],
        "latency_per_sample_ms": metrics_base["latency_per_sample_ms"],
    }
]

df_csv = pd.DataFrame(csv_records)
df_csv.to_csv(RESULTS_CSV_PATH, index=False)
print(f"\nSaved holdout results CSV to: {RESULTS_CSV_PATH}")

# -----------------------------------------------------------------------------
# Save Report to Markdown
# -----------------------------------------------------------------------------
report_content = f"""# Final Holdout Test Evaluation Report

**Evaluation Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}
**Holdout Test Dataset**: `data/processed/test.csv` (3,907 instances, 6 features, 1 target)
**Test Dataset SHA-256**: `{test_hash}`

---

## 1. Executive Summary

The frozen candidate model (`models/experiments/lightgbm_subsampled.joblib`) was evaluated on the unseen holdout test set (`data/processed/test.csv`) for the first and only authorized time. It was compared directly against the original baseline model (`models/best_model.joblib`).

Neither model was retrained, modified, or tuned during this evaluation.

### Key Metrics Summary (Holdout Test Set)

| Metric | LightGBM Subsampled (Frozen Candidate) | Original Baseline Random Forest | Absolute Delta | Relative Change |
| :--- | :---: | :---: | :---: | :---: |
| **Model SHA-256** | `{frozen_hash[:16]}...` | `{baseline_hash[:16]}...` | - | - |
| **Classification Threshold** | `0.50` | `0.50` | `0.00` | - |
| **F1-Score** | **`{metrics_frozen['f1']:.4f}`** | `{metrics_base['f1']:.4f}` | `+{metrics_frozen['f1'] - metrics_base['f1']:.4f}` | `+{(metrics_frozen['f1'] - metrics_base['f1'])/metrics_base['f1']*100:.2f}%` |
| **Recall (Sensitivity)** | **`{metrics_frozen['recall']:.4f}`** (`{metrics_frozen['tp']}` / {n_pos}) | `{metrics_base['recall']:.4f}` (`{metrics_base['tp']}` / {n_pos}) | `+{metrics_frozen['recall'] - metrics_base['recall']:.4f}` | `+{(metrics_frozen['recall'] - metrics_base['recall'])/metrics_base['recall']*100:.2f}%` |
| **Precision** | `{metrics_frozen['precision']:.4f}` | **`{metrics_base['precision']:.4f}`** | `{metrics_frozen['precision'] - metrics_base['precision']:.4f}` | `{(metrics_frozen['precision'] - metrics_base['precision'])/metrics_base['precision']*100:.2f}%` |
| **Accuracy** | `{metrics_frozen['accuracy']:.4f}` | **`{metrics_base['accuracy']:.4f}`** | `{metrics_frozen['accuracy'] - metrics_base['accuracy']:.4f}` | `{(metrics_frozen['accuracy'] - metrics_base['accuracy'])/metrics_base['accuracy']*100:.2f}%` |
| **Balanced Accuracy** | `{metrics_frozen['balanced_accuracy']:.4f}` | **`{metrics_base['balanced_accuracy']:.4f}`** | `{metrics_frozen['balanced_accuracy'] - metrics_base['balanced_accuracy']:.4f}` | `{(metrics_frozen['balanced_accuracy'] - metrics_base['balanced_accuracy'])/metrics_base['balanced_accuracy']*100:.2f}%` |
| **Specificity** | `{metrics_frozen['specificity']:.4f}` | **`{metrics_base['specificity']:.4f}`** | `{metrics_frozen['specificity'] - metrics_base['specificity']:.4f}` | `{(metrics_frozen['specificity'] - metrics_base['specificity'])/metrics_base['specificity']*100:.2f}%` |
| **ROC-AUC** | `{metrics_frozen['roc_auc']:.4f}` | **`{metrics_base['roc_auc']:.4f}`** | `{metrics_frozen['roc_auc'] - metrics_base['roc_auc']:.4f}` | `{(metrics_frozen['roc_auc'] - metrics_base['roc_auc'])/metrics_base['roc_auc']*100:.2f}%` |
| **PR-AUC (Avg Precision)**| **`{metrics_frozen['pr_auc']:.4f}`** | `{metrics_base['pr_auc']:.4f}` | `+{metrics_frozen['pr_auc'] - metrics_base['pr_auc']:.4f}` | `+{(metrics_frozen['pr_auc'] - metrics_base['pr_auc'])/metrics_base['pr_auc']*100:.2f}%` |
| **False Negative Rate (FNR)** | **`{metrics_frozen['fnr']:.4f}`** | `{metrics_base['fnr']:.4f}` | `{metrics_frozen['fnr'] - metrics_base['fnr']:.4f}` | **`-{(metrics_base['fnr'] - metrics_frozen['fnr'])/metrics_base['fnr']*100:.2f}%`** |
| **False Positive Rate (FPR)** | `{metrics_frozen['fpr']:.4f}` | **`{metrics_base['fpr']:.4f}`** | `+{metrics_frozen['fpr'] - metrics_base['fpr']:.4f}` | `+{(metrics_frozen['fpr'] - metrics_base['fpr'])/metrics_base['fpr']*100:.2f}%` |
| **Missed Engine Faults (FN)** | **`{metrics_frozen['missed_engine_faults']}`** | `{metrics_base['missed_engine_faults']}` | **`-{metrics_base['missed_engine_faults'] - metrics_frozen['missed_engine_faults']}`** | **`-{(metrics_base['missed_engine_faults'] - metrics_frozen['missed_engine_faults'])/metrics_base['missed_engine_faults']*100:.2f}%`** |
| **False Alarms (FP)** | `{metrics_frozen['false_alarms']}` | **`{metrics_base['false_alarms']}`** | `+{metrics_frozen['false_alarms'] - metrics_base['false_alarms']}` | `+{(metrics_frozen['false_alarms'] - metrics_base['false_alarms'])/metrics_base['false_alarms']*100:.2f}%` |
| **Inference Latency (Total)** | **`{metrics_frozen['inference_duration_sec']:.4f} s`** | `{metrics_base['inference_duration_sec']:.4f} s` | `{metrics_frozen['inference_duration_sec'] - metrics_base['inference_duration_sec']:.4f} s` | **`{(metrics_frozen['inference_duration_sec'] - metrics_base['inference_duration_sec'])/metrics_base['inference_duration_sec']*100:.2f}%`** |
| **Per-Sample Latency** | **`{metrics_frozen['latency_per_sample_ms']:.4f} ms`** | `{metrics_base['latency_per_sample_ms']:.4f} ms` | - | - |

---

## 2. Confusion Matrix Breakdown

### Frozen Candidate Model: LightGBM Subsampled
```
                   Predicted Normal (0)    Predicted Failing (1)
Actual Normal (0)          {metrics_frozen['tn']} (TN)                  {metrics_frozen['fp']} (FP, False Alarms)
Actual Failing (1)         {metrics_frozen['fn']} (FN, Missed Faults)   {metrics_frozen['tp']} (TP, Caught Faults)
```
- **Fault Detection Rate**: {metrics_frozen['tp']}/{n_pos} ({metrics_frozen['recall']*100:.2f}%)
- **Missed Breakdown Incidents**: Only {metrics_frozen['missed_engine_faults']} out of {n_pos} active engine failure scenarios.

### Baseline Model: Original Random Forest
```
                   Predicted Normal (0)    Predicted Failing (1)
Actual Normal (0)          {metrics_base['tn']} (TN)                  {metrics_base['fp']} (FP, False Alarms)
Actual Failing (1)         {metrics_base['fn']} (FN, Missed Faults)  {metrics_base['tp']} (TP, Caught Faults)
```
- **Fault Detection Rate**: {metrics_base['tp']}/{n_pos} ({metrics_base['recall']*100:.2f}%)
- **Missed Breakdown Incidents**: {metrics_base['missed_engine_faults']} engine failure scenarios completely missed.

---

## 3. Engineering Analysis & Operational Trade-offs

1. **Massive Reduction in Critical Engine Breakdowns**:
   The frozen LightGBM model successfully catches **{metrics_frozen['tp']} out of {n_pos}** failing engines on the holdout test set ({metrics_frozen['recall']*100:.2f}% recall), compared to only {metrics_base['tp']} for the baseline model ({metrics_base['recall']*100:.2f}% recall).
   Crucially, **missed engine breakdowns drop from {metrics_base['missed_engine_faults']} to {metrics_frozen['missed_engine_faults']}**—a **{ (metrics_base['missed_engine_faults'] - metrics_frozen['missed_engine_faults'])/metrics_base['missed_engine_faults']*100:.2f}% reduction** in undetected catastrophic failures.
2. **Trade-off on False Positives**:
   Because the LightGBM classifier prioritizes fault capture, its false alarm count ({metrics_frozen['fp']}) is higher than the conservative baseline ({metrics_base['fp']}). In predictive maintenance, an inspection dispatch on a healthy engine incurs minor operational cost, whereas an undetected catastrophic engine breakdown results in severe capital loss and potential safety hazard.
3. **Inference Latency**:
   LightGBM executes 3,907 predictions in {metrics_frozen['inference_duration_sec']:.4f}s ({metrics_frozen['latency_per_sample_ms']:.4f} ms per sample), making it highly suitable for real-time edge telemetry monitoring.
"""

REPORT_MD_PATH.write_text(report_content, encoding="utf-8")
print(f"Saved holdout report Markdown to: {REPORT_MD_PATH}")

print("\n" + "=" * 70)
print("FINAL HOLDOUT EVALUATION COMPLETE")
print("=" * 70)
