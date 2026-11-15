# Model Evaluation & Experimental Benchmarks

This document details the machine learning development lifecycle, validation protocols, bug discoveries, cross-validation benchmarks, threshold sweep analyses, and holdout evaluation results for the Engine Condition Classification system.

---

## 1. Evaluation Methodology & Data Partitions

### 1.1 Partitioning Strategy
The raw dataset (`data/engine_data.csv`, 19,535 rows) was partitioned using stratified sampling to preserve the target class distribution across all splits:

```
Total Dataset: 19,535 rows
|-- Development Set (80%): 15,628 rows
|   |-- Training Split (80% of Dev): 12,502 rows (64.0% of total)
|   `-- Validation Split (20% of Dev): 3,126 rows (16.0% of total)
`-- Holdout Test Set (20%): 3,907 rows (20.0% of total)
```

- **Holdout Isolation Protocol**: The holdout test set (`data/processed/test.csv`, SHA-256: `f99250b996cb07b2f4548df0dfc798c4d6add24e5d266d1f673d9b36d510ed79`) was independently isolated at the beginning of development and evaluated strictly once in Phase 9. It was never used for feature selection, hyperparameter tuning, model selection, or threshold calibration. The file is tracked in Git for transparency and reproducibility.
- **Cross-Validation Scheme**: Hyperparameter optimization and algorithm comparisons were performed using 5-Fold Stratified Cross-Validation (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`) strictly within the development partition.
- **Zero Data Leakage Guarantee**: Missing value imputation (`SimpleImputer(strategy='median')`) was fitted exclusively within each cross-validation fold or within `train_split.csv`. No test or validation information ever informed the preprocessing transformers.

---

## 2. Algorithm Benchmarking & Key Discoveries

### 2.1 The LightGBM Subsampling Bug Discovery
During initial gradient boosting experiments, LightGBM was configured with `subsample=0.7`. However, in LightGBM's Scikit-Learn API:
> When `subsample` (or `bagging_fraction`) is configured without setting `subsample_freq > 0`, LightGBM **silently ignores** the subsample parameter and defaults to using 100% of samples for every iteration.

This was discovered during hyperparameter audit. To activate stochastic bagging, `subsample_freq=1` was explicitly added alongside `subsample=0.7`. This stabilized tree diversity, prevented overfitting on outlier telemetry records, and produced the candidate that outperformed the baseline.

### 2.2 Controlled Feature Engineering Evaluation (Experiment 3)
In Phase 8 Experiment 3 (`src/experiments.py`), five physically motivated thermodynamic and hydrodynamic derived features were engineered via `EngineFeatureTransformer`:

1. `temp_diff`: $\text{Lub\_Oil\_Temperature} - \text{Coolant\_Temperature}$  
   *Temperature difference between lubricating oil and engine coolant.*
2. `oil_fuel_ratio`: $\frac{\text{Lub\_Oil\_Pressure}}{\text{Fuel\_Pressure} + 10^{-4}}$  
   *Ratio of lubricating oil pressure to fuel delivery pressure.*
3. `oil_per_rpm`: $\frac{\text{Lub\_Oil\_Pressure}}{\text{Engine\_RPM} + 1.0}$  
   *Lubricating oil pressure normalized by engine rotational speed.*
4. `coolant_per_rpm`: $\frac{\text{Coolant\_Pressure}}{\text{Engine\_RPM} + 1.0}$  
   *Coolant circulation pressure normalized by engine rotational speed.*
5. `press_diff`: $\text{Lub\_Oil\_Pressure} - \text{Coolant\_Pressure}$  
   *Differential pressure between lubricating oil and coolant circuits.*

These 5 features were appended to the 6 raw telemetry features (expanding from 6 to 11 channels total) and evaluated across 5-fold cross-validation and the validation partition (`models/experiments/phase8_experiments_summary.csv`):

| Model & Feature Set | Mean CV F1 | Std CV F1 | Val F1 | Val Recall | Val Precision | Train Time (s) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LightGBM (6 Raw Features, Subsampling Corrected)** | **0.7778** | **±0.0046** | **0.7813** | **0.9569** | **0.6601** | **25.80** |
| LightGBM + 5 Engineered Features (11 Total) | 0.7756 | ±0.0061 | 0.7773 | 0.9305 | 0.6674 | 31.67 |
| XGBoost (6 Raw Features, Baseline) | 0.7651 | ±0.0046 | 0.7651 | 0.8503 | 0.6954 | 36.18 |
| XGBoost + 5 Engineered Features (11 Total) | 0.7654 | ±0.0069 | 0.7606 | 0.8432 | 0.6928 | 50.67 |

- **Empirical Outcome**: For LightGBM, the engineered features degraded cross-validation F1 (from 0.7778 to 0.7756), lowered validation F1 (from 0.7813 to 0.7773), decreased validation recall (from 95.69% to 93.05%), and increased training latency by 23% (25.80s vs. 31.67s).
- **Architectural Decision**: Because feature engineering increased model training time without improving validation performance, the derived transformations were discarded. The production pipeline retains the clean 6-channel raw telemetry schema.

### 2.3 5-Fold Cross-Validation Comparison Table

| Model Architecture | Hyperparameters / Configuration | Mean CV F1 | Std CV F1 | Mean CV ROC-AUC | Mean CV Recall |
|:---|:---|:---:|:---:|:---:|:---:|
| **Random Forest (Baseline)** | `max_depth=5, n_estimators=200` | 0.7648 | ±0.0059 | 0.6988 | 0.8483 |
| **XGBoost Classifier** | `max_depth=4, lr=0.05, n_est=100` | 0.7651 | ±0.0046 | 0.7016 | 0.8503 |
| **LightGBM (Buggy Subsample)** | `subsample=0.7, subsample_freq=0` | 0.7769 | ±0.0036 | 0.6957 | 0.9508 |
| **LightGBM Subsampled (Promoted)** | `subsample=0.7, subsample_freq=1, lr=0.01` | **0.7778** | **±0.0046** | **0.6983** | **0.9569** |
| **LightGBM + Engineered Features** | 11 features (6 raw + 5 physical), `lr=0.01` | 0.7756 | ±0.0061 | 0.6969 | 0.9305 |

---

## 3. Decision Threshold Analysis

### 3.1 Frozen Operational Threshold (0.50) vs. Balanced Accuracy Maxima
The standard decision threshold of $\tau = 0.50$ is deployed as the **frozen operational threshold**. 

It is essential to maintain a clear distinction between the operational standard and thresholds that mathematically maximize specific metrics:
- **Threshold 0.50 does not maximize balanced accuracy.**
- In Phase 8 threshold optimization experiments (`models/experiments/threshold_optimization_results.csv` and `threshold_oof_sweeps.csv`):
  - **XGBoost**: The threshold that maximized validation balanced accuracy was **$\tau = 0.64$**, reaching a balanced accuracy of **0.6460** (OOF sweep: 0.6469). However, at $\tau = 0.64$, recall plummeted from 85.03% to **58.30%** (missing 41.7% of faults).
  - **LightGBM**: The threshold that maximized validation balanced accuracy was **$\tau = 0.62$**, reaching a balanced accuracy of **0.6452** (OOF sweep: 0.6407). However, at $\tau = 0.62$, recall dropped from 95.08% to **65.50%** (missing 34.5% of faults).
- At the frozen operational threshold of **$\tau = 0.50$**, the promoted LightGBM model achieves a balanced accuracy of **0.5461** on the holdout test set, while maintaining **94.68% holdout recall**.

### 3.2 Threshold Optimization Results Summary

Source: `models/experiments/threshold_optimization_results.csv` (evaluated on validation partition):

| Model Name | Optimization Objective | Threshold ($\tau$) | Val Recall | Val Precision | Val F1 | Val Balanced Acc | Val Specificity |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **XGBoost** | Default (0.50) | 0.50 | 0.8503 | 0.6954 | 0.7651 | 0.6074 | 0.3645 |
| **XGBoost** | Max F1 | 0.42 | 0.9518 | 0.6634 | 0.7818 | 0.5638 | 0.1758 |
| **XGBoost** | Max Balanced Accuracy | **0.64** | **0.5830** | 0.7737 | 0.6649 | **0.6460** | 0.7091 |
| **LightGBM** | Default (0.50) | **0.50** | **0.9508** | **0.6601** | **0.7792** | **0.5576** | **0.1645** |
| **LightGBM** | Max F1 | 0.48 | 0.9731 | 0.6517 | 0.7806 | 0.5428 | 0.1126 |
| **LightGBM** | Max Balanced Accuracy | **0.62** | **0.6550** | 0.7541 | 0.7011 | **0.6452** | 0.6355 |

**Operational Rationale for Freezing $\tau = 0.50$**:
Thresholds 0.62 and 0.64 were not selected for production because they substantially reduce recall (to 65.50% for LightGBM and 58.30% for XGBoost), conflicting with the project's primary objective of maintaining high sensitivity to developing engine anomalies. In predictive maintenance, undetected faults carry severe operational risk. Threshold 0.50 was frozen as the operational standard because it delivers **94.68% holdout recall** while avoiding ad-hoc threshold tuning. As noted in the architectural specifications, predicted decision scores represent empirical ensemble margins and are not calibrated Bayesian posterior probabilities.

---

## 4. Final Holdout Test Evaluation (Phase 9)

Following completion of all model development and freezing of pipeline artifacts, the candidate was evaluated exactly once against the isolated holdout test partition (`data/processed/test.csv`, $N=3,907$).

### 4.1 Comparative Holdout Performance Table

All metrics below are drawn from the benchmark evaluation log (`models/experiments/final_holdout_results.csv`):

| Evaluation Metric | Original Baseline (Random Forest) | Promoted Candidate (LightGBM) | Delta / Impact |
|:---|:---:|:---:|:---:|
| **Model Artifact** | `models/best_model.joblib` | `models/production/engine_condition_model.joblib` | Production package |
| **Classification Threshold** | 0.50 | 0.50 | Frozen operational standard |
| **Holdout Recall (Sensitivity)** | 89.65% | **94.68%** | **+5.03% absolute (+5.62% relative)** |
| **Missed Faults (False Negatives)** | 255 engines | **131 engines** | **-124 missed faults (-48.63% reduction)** |
| **Holdout F1 Score** | 0.7683 | **0.7736** | **+0.0053 (+0.69%)** |
| **Holdout Precision** | 67.21% | 65.40% | -1.81% (trade-off for high recall) |
| **Holdout Accuracy** | 65.91% | 65.06% | -0.85% |
| **Balanced Accuracy** | 57.53% | 54.61% | -2.92% |
| **Specificity (True Negative Rate)** | 25.42% | 14.54% | -10.88% |
| **ROC-AUC** | 0.6968 | **0.6976** | **+0.0008** |
| **PR-AUC** | 0.7955 | **0.7988** | **+0.0033** |
| **Measured Batch Time (3,907 samples)** | 0.1824 sec | **0.0525 sec** | **3.47x faster throughput** |
| **Batch-Average Latency per Sample** | 0.0467 ms | **0.0134 ms** | **~74,400 samples / sec throughput** |

### 4.2 Holdout Confusion Matrix Breakdown

#### Promoted LightGBM Model (Holdout $N = 3,907$)
```
                         Predicted Normal (0)    Predicted Fault (1)    Total
Actual Normal (0)             210 (TN)               1,234 (FP)          1,444
Actual Fault  (1)             131 (FN)               2,332 (TP)          2,463
Total                         341                    3,566               3,907
```
- **False Negative Rate (FNR)**: $\frac{131}{2463} = 5.32\%$ (only 5.3% of true faults missed).
- **False Positive Rate (FPR)**: $\frac{1234}{1444} = 85.46\%$.

#### Baseline Random Forest Model (Holdout $N = 3,907$)
```
                         Predicted Normal (0)    Predicted Fault (1)    Total
Actual Normal (0)             367 (TN)               1,077 (FP)          1,444
Actual Fault  (1)             255 (FN)               2,208 (TP)          2,463
Total                         622                    3,285               3,907
```
- **False Negative Rate (FNR)**: $\frac{255}{2463} = 10.35\%$ (10.4% of true faults missed).

---

## 5. Engineering Interpretation & Trade-Off Analysis

### 5.1 The Safety-First Maintenance Trade-Off
In an industrial monitoring setting, missing an engine fault can lead to catastrophic mechanical failure, unplanned downtime, and significant repair expenses.
- By switching from the baseline Random Forest to the promoted LightGBM model, **missed faults were reduced by 48.63%** (from 255 down to 131).
- This sensitivity increase carries a trade-off: false alarms increased from 1,077 to 1,234 (+157 false positives across 3,907 evaluations).
- In operational practice, this model functions as an automated condition triage filter: it catches nearly 95% of anomalous engine conditions, directing maintenance personnel to perform non-invasive diagnostic inspections before equipment damage compounds.

### 5.2 Inference Latency & Benchmark Profiling
Timing verified against original evaluation logs (`models/experiments/final_holdout_results.csv` and `models/production/model_metadata.json`):
- **Total Measured Batch Time**: **0.0525 seconds** across 3,907 holdout samples.
- **Approximate Batch-Average Time per Sample**: **0.0134 ms/sample** ($13.4\,\mu\text{s}$ per sample).
- **Approximate Measured Batch Throughput**: $\sim \mathbf{74,400}$ samples/second.
- **Comparison to Baseline (RF)**: Random Forest measured 0.1824 seconds across 3,907 samples (~0.0467 ms/sample, ~21,400 samples/sec), demonstrating a ~3.5x throughput advantage for LightGBM.

> [!NOTE]
> **Measurement Environment & Operational Limitations**:
> Latency was measured in vectorized batch execution mode (`pipeline.predict_proba(X_test)`) under Python 3.12.5 and 64-bit Windows during Phase 9 offline evaluation.
> This metric reflects vectorized offline batch processing capacity on a single development machine. It does **not** represent single-event network API request/response latency, nor should it be interpreted as a guaranteed production SLA or real-time hard-deadline guarantee.
