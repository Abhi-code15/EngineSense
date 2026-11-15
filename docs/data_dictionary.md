# Telemetry Data Dictionary & Sensor Specifications

This document defines the telemetry feature channels, physical units, subsystem assignments, target variable semantics, and empirical training distribution parameters for the EngineSense dataset.

---

## 1. Dataset Overview

The dataset comprises **19,535 engine telemetry records** capturing six continuous physical sensor readings alongside an indicator of engine operational status.

- **Source File**: `data/engine_data.csv`
- **Origin & Licensing**: Sourced from the [Automotive Vehicles Engine Health Dataset on Kaggle](https://www.kaggle.com/datasets/parvmodi/automotive-vehicles-engine-health-dataset) by Parv Modi under **CC0: Public Domain** dedication; registered on Hugging Face Hub as `Abhishektripathiak/engine-maintenance-dataset`).
- **Development Partition (80%)**: `data/processed/train_split.csv` ($N = 12,502$), `data/processed/val.csv` ($N = 3,126$)
- **Holdout Partition (20%)**: `data/processed/test.csv` ($N = 3,907$)
- **Format**: Comma-Separated Values (CSV), UTF-8 encoded
- **Record Type**: Instantaneous operational snapshots (cross-sectional telemetry observations)

---

## 2. Telemetry Channels & Specifications

The system monitors six physical parameters across three core engine subsystems:

```
+--------------------------------------------------------------------------+
|                        Engine Telemetry Subsystems                       |
+--------------------------+-----------------------+-----------------------+
|  Mechanical & Powertrain |   Fluid Lubrication   |   Thermal & Cooling   |
|  - Engine_RPM (RPM)      |   - Lub_Oil_Pressure  |   - Lub_Oil_Temp      |
|  - Fuel_Pressure (bar)   |     (bar)             |     (°C)              |
|                          |                       |   - Coolant_Pressure  |
|                          |                       |     (bar)             |
|                          |                       |   - Coolant_Temp (°C) |
+--------------------------+-----------------------+-----------------------+
```

### Detailed Channel Specifications

| Channel Name | Physical Quantity | Standard Units | Engine Subsystem | Operational Description |
|:---|:---|:---:|:---|:---|
| `Engine_RPM` | Crankshaft Rotational Speed | RPM | Mechanical / Drivetrain | Speed of engine rotation in revolutions per minute. |
| `Lub_Oil_Pressure` | Lubrication Pressure | bar | Lubrication Loop | Gauge delivery pressure of the engine oil lubrication circuit. |
| `Fuel_Pressure` | Fuel Rail / Manifold Pressure | bar | Fuel Delivery System | Delivery pressure of the fuel supply feeding the injection system. |
| `Coolant_Pressure` | Jacket Water Coolant Pressure | bar | Engine Cooling System | Hydraulic circulation pressure of the jacket water cooling loop. |
| `Lub_Oil_Temperature` | Lubricating Oil Temperature | °C | Lubrication / Thermal | Bulk operating temperature of engine lubricating oil in the gallery/sump. |
| `Coolant_Temperature` | Coolant Jacket Temperature | °C | Cooling / Thermal | Bulk temperature of circulating jacket water leaving the engine head. |

---

## 3. Training Distribution Reference Statistics

The statistics below are calculated exclusively from the training partition (`data/processed/train_split.csv`, $N = 12,502$) and represent the baseline reference distributions used during model training and pipeline median imputation.

| Feature Channel | Count | Mean | Std Dev | Min | 25th % ($P_{25}$) | Median ($P_{50}$) | 75th % ($P_{75}$) | Max |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `Engine_RPM` | 12,502 | 792.87 | 268.43 | 61.00 | 595.00 | **748.00** | 933.00 | 2239.00 |
| `Lub_Oil_Pressure` (bar) | 12,502 | 3.31 | 1.03 | 0.0034 | 2.52 | **3.17** | 4.06 | 7.27 |
| `Fuel_Pressure` (bar) | 12,502 | 6.65 | 2.76 | 0.0032 | 4.91 | **6.18** | 7.70 | 21.14 |
| `Coolant_Pressure` (bar) | 12,502 | 2.34 | 1.04 | 0.0025 | 1.61 | **2.17** | 2.85 | 7.48 |
| `Lub_Oil_Temperature` (°C) | 12,502 | 77.65 | 3.11 | 71.32 | 75.72 | **76.82** | 78.09 | 89.47 |
| `Coolant_Temperature` (°C) | 12,502 | 78.45 | 6.24 | 63.09 | 73.92 | **78.37** | 82.91 | 195.53 |

> [!NOTE]
> The median values highlighted above are serialized inside `models/production/engine_condition_model.joblib` within the `SimpleImputer` stage. Any missing or unrecorded telemetry values at inference time are substituted with these exact empirical training medians.

---

## 4. Target Variable Definition

The target variable is `Engine_Condition`, representing a binary classification of engine operational state:

| Target Value | Semantic Label | Operational Meaning | Training Count | Training Proportion |
|:---:|:---|:---|:---:|:---:|
| `0` | **Normal / Healthy** | Engine is operating within baseline parameters; no immediate maintenance required. | 4,619 | 36.95% |
| `1` | **Warning / Fault** | Anomalous engine telemetry detected; mechanical inspection and diagnostic triage recommended. | 7,883 | 63.05% |

### Class Imbalance Handling
The training partition exhibits a ~63:37 fault-to-normal distribution. LightGBM handles this empirical balance effectively without artificial oversampling (`SMOTE`) or downsampling, preserving natural calibration while optimizing for F1 score and high recall.

---

## 5. Physical & Engineering Constraints

1. **Non-Time-Series Data Structure**: The observations represent cross-sectional operational states rather than sequential time series. Dynamic derivative features ($d(\text{RPM})/dt$, pressure transients) are not present in this dataset.
2. **Empirical Statistics vs. Manufacturer Envelopes**: The interquartile ranges ($P_{25} - P_{75}$) and min/max extremes documented above reflect the specific dataset distribution. They must not be conflated with original equipment manufacturer (OEM) design limits or safety shutdown trigger points.
3. **Sensor Units**: Pressure measurements are recorded in metric bar ($1\text{ bar} = 10^5\text{ Pa} \approx 14.5038\text{ PSI}$). Temperatures are recorded in degrees Celsius (°C).
