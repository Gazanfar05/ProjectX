# ProjectX — Hypoglycaemia Prediction System

An AI system for diabetics that goes beyond glucose readings — factoring in medication schedules, daily lifestyle patterns, and time-of-day trends to predict when a hypoglycaemic episode is likely and alert users before it happens.

---

## Overview

| Component | Description |
|---|---|
| **Data models** | Typed dataclasses for glucose readings, insulin doses, meals, exercise events, patient profiles |
| **Feature engineering** | 20 clinical features: rate-of-change, insulin-on-board, carbs-on-board, exercise effect, circadian trends, rolling statistics |
| **ML predictor** | Calibrated Gradient Boosting classifier trained to predict hypoglycaemia 30 minutes ahead |
| **Alert system** | Four severity levels (None / Low / Medium / High) with actionable recommendations |
| **Synthetic data generator** | Realistic 24-hour patient simulations for training and testing |

---

## Architecture

```
src/hypoglycemia_predictor/
├── __init__.py              # Public API surface
├── __main__.py              # CLI entry point / live demo
├── models/
│   └── data_models.py       # GlucoseReading, MedicationDose, MealEvent,
│                            #   ExerciseEvent, PatientProfile, PredictionResult, …
├── features.py              # Feature extraction from PredictionInput snapshots
├── predictor.py             # HypoglycaemiaPredictor (train / predict / save / load)
├── alerts.py                # Alert formatting and severity routing
└── data_generator.py        # Synthetic patient-day generator + snapshot builder
```

---

## Signals used for prediction

| Signal | Features extracted |
|---|---|
| CGM glucose trace | Current value, 15-min & 30-min lag values, rate-of-change, trend slope, 60-min rolling mean / std / min / max |
| Insulin doses | Insulin-on-board (IU, exponential decay model), minutes since last dose |
| Meals | Carbohydrates-on-board (g, linear absorption model), minutes since last meal |
| Exercise | Post-exercise insulin-sensitivity effect (intensity × duration × decay), minutes since last session |
| Circadian / time-of-day | Sine/cosine hour encoding, minutes since midnight |

---

## Requirements

- Python 3.9+
- numpy, pandas, scikit-learn, joblib

```bash
pip install -e ".[dev]"   # includes pytest for running tests
```

---

## Quick start

### Run the interactive demo

```bash
python -m hypoglycemia_predictor
```

The demo:
1. Generates synthetic training data (30 patients × 7 days)
2. Trains the gradient-boosting model and prints evaluation metrics
3. Simulates a patient day with a hypoglycaemic episode around 14:00
4. Prints a timeline of glucose readings, risk scores, and full alert messages

### Use the Python API

```python
from datetime import datetime, timedelta
from hypoglycemia_predictor import (
    AlertThresholds, GlucoseReading, HypoglycaemiaPredictor,
    MealEvent, MealSize, MedicationDose, PatientProfile, PredictionInput,
    build_alert, format_alert,
)

# 1. Define patient
patient = PatientProfile(
    patient_id="P001",
    name="Alice",
    insulin_half_life_minutes=65,
    alert_thresholds=AlertThresholds(
        hypoglycaemia_mgdl=70,
        caution_mgdl=80,
        prediction_horizon_minutes=30,
    ),
)

# 2. Train the model on synthetic data
from hypoglycemia_predictor.data_generator import generate_training_dataset
snapshots, labels = generate_training_dataset(n_patients=50, days_per_patient=10)
predictor = HypoglycaemiaPredictor()
predictor.train(snapshots, labels)

# 3. Build a real-time snapshot for the current moment
now = datetime.now()
snapshot = PredictionInput(
    timestamp=now,
    recent_glucose=[
        GlucoseReading(now - timedelta(minutes=15), 95.0),
        GlucoseReading(now - timedelta(minutes=10), 90.0),
        GlucoseReading(now - timedelta(minutes=5),  84.0),
        GlucoseReading(now,                         78.0),
    ],
    recent_medications=[
        MedicationDose(now - timedelta(hours=2), "rapid-acting insulin", 4.0),
    ],
    recent_meals=[
        MealEvent(now - timedelta(hours=2, minutes=30), MealSize.MEDIUM, 45.0),
    ],
    recent_exercises=[],
    patient=patient,
)

# 4. Predict and alert
result = predictor.predict(snapshot)
build_alert(result)
print(format_alert(result))
```

### Save and reload a trained model

```python
predictor.save_model()              # saves to trained_model.joblib
predictor.load_model()              # reloads from disk
```

---

## Prediction output

```
============================================================
  HYPOGLYCAEMIA RISK ASSESSMENT  [2024-06-15 14:30]
============================================================
  Severity      : MEDIUM
  Risk Score    : 71.4%
  Predicted BG  : 64 mg/dL (in 30 min)
  Confidence    : 57.1%
------------------------------------------------------------
  ⚠️  MEDIUM RISK: Hypoglycaemia predicted within 30 minutes.
  Predicted glucose: 64 mg/dL. Please take action soon.
------------------------------------------------------------
  Recommendations:
  1. Eat a small snack with 15 g of fast-acting carbohydrates …
  2. Avoid intense exercise until glucose stabilises.
  3. Check your glucose in 15 minutes.
  4. Consider reducing your next insulin dose if a pattern is emerging …
============================================================
```

### Alert severity levels

| Severity | Predicted glucose | Action |
|---|---|---|
| **None** | ≥ 80 mg/dL | No action required |
| **Low** | 70–80 mg/dL | Monitor closely, prepare snack |
| **Medium** | 60–70 mg/dL | Eat fast-acting carbs, check again in 15 min |
| **High** | < 60 mg/dL | Act immediately; do not drive |

---

## Running tests

```bash
pytest tests/ -v
```

All 55 tests cover data models, feature engineering, data generation, the ML predictor pipeline, and the alert system.

---

## Disclaimer

This system is for research and demonstration purposes only. It is **not a certified medical device** and must not be used as a substitute for professional medical advice, clinical-grade CGM systems, or insulin-management guidance from a qualified healthcare provider.

