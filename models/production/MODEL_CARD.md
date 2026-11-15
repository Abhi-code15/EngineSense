# Model Card: Engine Condition Classifier (Production Candidate)

## Model Overview
- **Model Name**: Engine Predictive Maintenance Condition Classifier
- **Model Version**: `1.0.0`
- **Architecture**: Scikit-Learn `Pipeline` with `SimpleImputer(strategy="median")` and `LGBMClassifier`
- **Target Task**: Binary classification of engine health condition (`0: Normal`, `1: Warning / Fault`)
- **Promoted Artifact**: [`models/production/engine_condition_model.joblib`](file:///c:/Users/Vikash%20Tiwari/Desktop/Documents/Abhishek%20Tiwari/2028/Intern-Project/engine-predictive-maintenance/models/production/engine_condition_model.joblib)
- **Artifact SHA-256**: `e364d18ba228de24f625d92d8f28b4d3acb43ef7ad5c48274677577df60f692e`

---

## 1. Input Specifications & Feature Schema

The pipeline expects a 2D tabular array or pandas DataFrame with exactly **6 numeric features** in the following mandatory order:

| Index | Feature Name | Description | Typical Unit | Imputation Median |
| :---: | :--- | :--- | :---: | :---: |
| 0 | `Engine_RPM` | Engine rotational crankshaft speed | RPM | `748.0` |
| 1 | `Lub_Oil_Pressure` | Lubrication system oil delivery pressure | bar | `3.1664` |
| 2 | `Fuel_Pressure` | Fuel injection supply manifold pressure | bar | `6.1822` |
| 3 | `Coolant_Pressure` | Cooling jacket water circuit pressure | bar | `2.1723` |
| 4 | `Lub_Oil_Temperature` | Lubricating oil sump temperature | °C | `76.8190` |
| 5 | `Coolant_Temperature` | Cooling jacket water temperature | °C | `78.3676` |

---

## 2. Target Class Mapping & Operating Threshold

- **Target Column**: `Engine_Condition`
- **Negative Class (0)**: `Normal` (Nominal engine operation within baseline mechanical envelope)
- **Positive Class (1)**: `Warning / Fault` (Anomalous condition indicating impending or active failure)
- **Classification Threshold**: `0.50` (Standard operating point for fault detection)

---

## 3. Training & Validation Methodology

- **Dataset**: `engine_data.csv` ($19,535$ telemetry observations)
- **Partitioning**:
  - Training Partition: `data/processed/train_split.csv` ($12,502$ samples, $64\%$)
  - Validation Partition: `data/processed/val.csv` ($3,126$ samples, $16\%$)
  - Final Holdout Test Partition: `data/processed/test.csv` ($3,907$ samples, $20\%$)
- **Data Leakage Guarantees**:
  - `SimpleImputer` was fitted strictly inside cross-validation training folds.
  - Holdout test partition was strictly sealed and never accessed during preprocessing, hyperparameter search, or threshold selection.
- **Tuning Configuration**:
  - 5-Fold Stratified Cross-Validation (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`)
  - Evaluated 15 randomized candidate configurations optimizing binary F1-score.
  - Active subsampling: `subsample=0.7`, `subsample_freq=1`, `colsample_bytree=0.85`, `num_leaves=63`, `learning_rate=0.01`.

---

## 4. Evaluation Performance & Baseline Comparison

### Final Holdout Test Results (`data/processed/test.csv`, 3,907 instances)

| Metric | LightGBM Subsampled (Production Candidate) | Original Baseline Random Forest | Performance Delta |
| :--- | :---: | :---: | :---: |
| **Model SHA-256** | `e364d18ba228de24...` | `9804450597df4e88...` | Verified distinct artifacts |
| **F1-Score** | **`0.7736`** | `0.7683` | `+0.0053` (`+0.69%`) |
| **Recall (Sensitivity)** | **`0.9468`** | `0.8965` | **`+0.0503` (`+5.62%`)** |
| **Precision** | `0.6540` | **`0.6721`** | `-0.0182` (`-2.71%`) |
| **Accuracy** | `0.6506` | **`0.6591`** | `-0.0084` (`-1.28%`) |
| **Balanced Accuracy** | `0.5461` | **`0.5753`** | `-0.0292` (`-5.07%`) |
| **Specificity** | `0.1454` | **`0.2542`** | `-0.1087` (`-42.78%`) |
| **ROC-AUC** | **`0.6976`** | `0.6968` | `+0.0008` (`+0.11%`) |
| **PR-AUC** | **`0.7988`** | `0.7955` | `+0.0033` (`+0.42%`) |
| **Missed Engine Faults (FN)** | **`131`** | `255` | **`-124` missed faults (`-48.63%`)** |
| **False Alarms (FP)** | `1,234` | **`1,077`** | `+157` alarms (`+14.58%`) |
| **Inference Latency** | **`0.0525 s`** ($0.0134\text{ ms/sample}$) | `0.1824 s` ($0.0467\text{ ms/sample}$) | **$3.5\times$ faster** |

---

## 5. Deployment Guidelines & Code Example

### Python Loading Protocol (Windows / Linux)
```python
# IMPORTANT: In 64-bit Windows Python environments, import lightgbm before joblib.load
import lightgbm as lgb
import joblib
import pandas as pd

# 1. Load production pipeline
model = joblib.load("models/production/engine_condition_model.joblib")

# 2. Prepare 6-sensor input DataFrame
telemetry = pd.DataFrame([{
    "Engine_RPM": 750.0,
    "Lub_Oil_Pressure": 3.2,
    "Fuel_Pressure": 6.1,
    "Coolant_Pressure": 2.2,
    "Lub_Oil_Temperature": 78.5,
    "Coolant_Temperature": 80.0
}])

# 3. Generate inference
fault_probability = float(model.predict_proba(telemetry)[0, 1])
prediction = int(fault_probability >= 0.50)

print(f"Condition: {'FAULT / MAINTENANCE REQUIRED' if prediction == 1 else 'NORMAL'}")
print(f"Empirical Fault Score: {fault_probability:.4f}")
```

---

## 6. Known Limitations & Technical Disclaimers

1. **Uncalibrated Decision Scores**: Output probabilities reflect empirical leaf-voting frequencies and logistic scores, **not calibrated Bayesian posteriors**. They should be treated as diagnostic ranking scores rather than absolute statistical likelihoods.
2. **Instantaneous State Classifier, Not Remaining Useful Life (RUL)**: The model predicts the instantaneous binary condition of the engine at the timestamp of the telemetry reading. It does **not** estimate Remaining Useful Life (RUL), time-to-failure (TTF), or degradation trends.
3. **No Safety Certification**: This model is built for condition-based monitoring, predictive maintenance scheduling, and decision support. It is **not** certified for safety-critical automated emergency shutdowns (e.g., ISO 26262 ASIL or IEC 61508 SIL functional safety standards).
