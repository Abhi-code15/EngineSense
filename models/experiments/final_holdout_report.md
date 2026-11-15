# Final Holdout Test Evaluation Report

**Evaluation Timestamp**: 2026-11-14 22:54:49 UTC
**Holdout Test Dataset**: `data/processed/test.csv` (3,907 instances, 6 features, 1 target)
**Test Dataset SHA-256**: `f99250b996cb07b2f4548df0dfc798c4d6add24e5d266d1f673d9b36d510ed79`

---

## 1. Executive Summary

The frozen candidate model (`models/experiments/lightgbm_subsampled.joblib`) was evaluated on the unseen holdout test set (`data/processed/test.csv`) for the first and only authorized time. It was compared directly against the original baseline model (`models/best_model.joblib`).

Neither model was retrained, modified, or tuned during this evaluation.

### Key Metrics Summary (Holdout Test Set)

| Metric | LightGBM Subsampled (Frozen Candidate) | Original Baseline Random Forest | Absolute Delta | Relative Change |
| :--- | :---: | :---: | :---: | :---: |
| **Model SHA-256** | `e364d18ba228de24...` | `9804450597df4e88...` | - | - |
| **Classification Threshold** | `0.50` | `0.50` | `0.00` | - |
| **F1-Score** | **`0.7736`** | `0.7683` | `+0.0053` | `+0.69%` |
| **Recall (Sensitivity)** | **`0.9468`** (`2332` / 2463) | `0.8965` (`2208` / 2463) | `+0.0503` | `+5.62%` |
| **Precision** | `0.6540` | **`0.6721`** | `-0.0182` | `-2.71%` |
| **Accuracy** | `0.6506` | **`0.6591`** | `-0.0084` | `-1.28%` |
| **Balanced Accuracy** | `0.5461` | **`0.5753`** | `-0.0292` | `-5.07%` |
| **Specificity** | `0.1454` | **`0.2542`** | `-0.1087` | `-42.78%` |
| **ROC-AUC** | `0.6976` | **`0.6968`** | `0.0008` | `0.11%` |
| **PR-AUC (Avg Precision)**| **`0.7988`** | `0.7955` | `+0.0033` | `+0.42%` |
| **False Negative Rate (FNR)** | **`0.0532`** | `0.1035` | `-0.0503` | **`-48.63%`** |
| **False Positive Rate (FPR)** | `0.8546` | **`0.7458`** | `+0.1087` | `+14.58%` |
| **Missed Engine Faults (FN)** | **`131`** | `255` | **`-124`** | **`-48.63%`** |
| **False Alarms (FP)** | `1234` | **`1077`** | `+157` | `+14.58%` |
| **Inference Latency (Total)** | **`0.0525 s`** | `0.1824 s` | `-0.1299 s` | **`-71.22%`** |
| **Per-Sample Latency** | **`0.0134 ms`** | `0.0467 ms` | - | - |

---

## 2. Confusion Matrix Breakdown

### Frozen Candidate Model: LightGBM Subsampled
```
                   Predicted Normal (0)    Predicted Failing (1)
Actual Normal (0)          210 (TN)                  1234 (FP, False Alarms)
Actual Failing (1)         131 (FN, Missed Faults)   2332 (TP, Caught Faults)
```
- **Fault Detection Rate**: 2332/2463 (94.68%)
- **Missed Breakdown Incidents**: Only 131 out of 2463 active engine failure scenarios.

### Baseline Model: Original Random Forest
```
                   Predicted Normal (0)    Predicted Failing (1)
Actual Normal (0)          367 (TN)                  1077 (FP, False Alarms)
Actual Failing (1)         255 (FN, Missed Faults)  2208 (TP, Caught Faults)
```
- **Fault Detection Rate**: 2208/2463 (89.65%)
- **Missed Breakdown Incidents**: 255 engine failure scenarios completely missed.

---

## 3. Engineering Analysis & Operational Trade-offs

1. **Massive Reduction in Critical Engine Breakdowns**:
   The frozen LightGBM model successfully catches **2332 out of 2463** failing engines on the holdout test set (94.68% recall), compared to only 2208 for the baseline model (89.65% recall).
   Crucially, **missed engine breakdowns drop from 255 to 131**—a **48.63% reduction** in undetected catastrophic failures.
2. **Trade-off on False Positives**:
   Because the LightGBM classifier prioritizes fault capture, its false alarm count (1234) is higher than the conservative baseline (1077). In predictive maintenance, an inspection dispatch on a healthy engine incurs minor operational cost, whereas an undetected catastrophic engine breakdown results in severe capital loss and potential safety hazard.
3. **Inference Latency**:
   LightGBM executes 3,907 predictions in 0.0525s (0.0134 ms per sample), making it highly suitable for real-time edge telemetry monitoring.
