"""
Synthetic data generator for training and demonstration.

Generates realistic 24-hour patient timelines that include:
- CGM glucose readings every 5 minutes
- Insulin doses (rapid-acting at mealtimes)
- Meals (breakfast, lunch, dinner, optional snacks)
- Exercise events
- Labelled hypoglycaemia episodes (ground truth for supervised learning)
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np

from hypoglycemia_predictor.features import _EXERCISE_INTENSITY_FACTOR
from hypoglycemia_predictor.models.data_models import (
    ExerciseEvent,
    ExerciseIntensity,
    GlucoseReading,
    MealEvent,
    MealSize,
    MedicationDose,
    PatientProfile,
    PredictionInput,
)


# Reproducible seed for tests; callers can override via ``seed`` parameter.
_DEFAULT_SEED = 42

# Simulation physiological constants
_INSULIN_EFFECT_SCALE = 15.0          # mg/dL drop per IU (peak effect scaling)
_INSULIN_PEAK_TIME_MINUTES = 45.0     # time to peak insulin action (minutes)
_EXERCISE_GLUCOSE_DROP_RATE = 0.15    # mg/dL per minute during active exercise
_POST_EXERCISE_SENSITIVITY_FACTOR = 0.05  # additional drop rate post-exercise
_MEAN_REVERSION_RATE = 0.01           # fraction per step towards baseline


def generate_patient_day(
    patient: PatientProfile,
    date: datetime,
    include_hypo_episode: bool = False,
    seed: int = _DEFAULT_SEED,
) -> Tuple[List[GlucoseReading], List[MedicationDose], List[MealEvent], List[ExerciseEvent]]:
    """
    Simulate a full day of diabetes-management events.

    Parameters
    ----------
    patient:
        The patient whose profile drives simulation parameters.
    date:
        The calendar day to simulate (time component is ignored; day starts
        at midnight).
    include_hypo_episode:
        If ``True``, artificially introduce a hypoglycaemic dip (glucose < 70
        mg/dL) in the afternoon.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    tuple of (glucose_readings, medication_doses, meal_events, exercise_events)
    """
    rng = np.random.default_rng(seed)
    day_start = date.replace(hour=0, minute=0, second=0, microsecond=0)

    glucose_readings: List[GlucoseReading] = []
    medications: List[MedicationDose] = []
    meals: List[MealEvent] = []
    exercises: List[ExerciseEvent] = []

    # -----------------------------------------------------------------------
    # Schedule anchor events
    # -----------------------------------------------------------------------
    breakfast_time = day_start + timedelta(hours=7, minutes=int(rng.integers(0, 30)))
    lunch_time = day_start + timedelta(hours=12, minutes=int(rng.integers(0, 30)))
    dinner_time = day_start + timedelta(hours=18, minutes=int(rng.integers(0, 30)))
    snack_time = day_start + timedelta(hours=15, minutes=int(rng.integers(0, 30)))

    meal_schedule = [
        (breakfast_time, MealSize.MEDIUM, 45),
        (lunch_time, MealSize.LARGE, 70),
        (dinner_time, MealSize.LARGE, 65),
        (snack_time, MealSize.SMALL, 15),
    ]
    for t, size, carbs in meal_schedule:
        meals.append(MealEvent(timestamp=t, size=size, carbohydrates_g=float(carbs)))
        # Rapid-acting insulin 10 min before meal
        insulin_time = t - timedelta(minutes=10)
        dose = _carbs_to_insulin(carbs, correction_factor=50.0, rng=rng)
        medications.append(
            MedicationDose(
                timestamp=insulin_time,
                medication_name="rapid-acting insulin",
                dose_amount=dose,
                dose_unit="IU",
                duration_of_action_hours=4.0,
            )
        )

    # Optional exercise event in the early afternoon
    if rng.random() > 0.4:
        ex_time = day_start + timedelta(hours=10, minutes=int(rng.integers(0, 60)))
        exercises.append(
            ExerciseEvent(
                timestamp=ex_time,
                duration_minutes=float(rng.integers(20, 60)),
                intensity=ExerciseIntensity(str(rng.choice([e.value for e in ExerciseIntensity]))),
                description="morning workout",
            )
        )

    # -----------------------------------------------------------------------
    # Simulate glucose trace (5-minute CGM readings)
    # -----------------------------------------------------------------------
    glucose_readings = _simulate_glucose(
        day_start=day_start,
        meal_schedule=meal_schedule,
        medications=medications,
        exercises=exercises,
        include_hypo_episode=include_hypo_episode,
        rng=rng,
    )

    return glucose_readings, medications, meals, exercises


def build_prediction_snapshots(
    patient: PatientProfile,
    glucose_readings: List[GlucoseReading],
    medications: List[MedicationDose],
    meals: List[MealEvent],
    exercises: List[ExerciseEvent],
    snapshot_interval_minutes: int = 5,
    lookback_minutes: int = 60,
) -> List[Tuple[PredictionInput, bool]]:
    """
    Slide a window over the day's events to build (snapshot, label) pairs.

    The label is ``True`` if the glucose drops below 70 mg/dL within the
    patient's configured prediction horizon after the snapshot timestamp.
    """
    if not glucose_readings:
        return []

    start = glucose_readings[0].timestamp
    end = glucose_readings[-1].timestamp
    horizon_min = patient.alert_thresholds.prediction_horizon_minutes
    hypo_threshold = patient.alert_thresholds.hypoglycaemia_mgdl

    snapshots: List[Tuple[PredictionInput, bool]] = []
    current = start + timedelta(minutes=lookback_minutes)

    while current <= end - timedelta(minutes=horizon_min):
        cutoff_past = current - timedelta(minutes=lookback_minutes)

        recent_glucose = [r for r in glucose_readings if cutoff_past <= r.timestamp <= current]
        recent_meds = [m for m in medications if cutoff_past <= m.timestamp <= current]
        recent_meals = [me for me in meals if cutoff_past <= me.timestamp <= current]
        recent_ex = [e for e in exercises if cutoff_past <= e.timestamp <= current]

        snapshot = PredictionInput(
            timestamp=current,
            recent_glucose=recent_glucose,
            recent_medications=recent_meds,
            recent_meals=recent_meals,
            recent_exercises=recent_ex,
            patient=patient,
        )

        # Label: does glucose drop below threshold in the next horizon window?
        future_glucose = [
            r.value_mgdl
            for r in glucose_readings
            if current < r.timestamp <= current + timedelta(minutes=horizon_min)
        ]
        label = bool(future_glucose and min(future_glucose) < hypo_threshold)

        snapshots.append((snapshot, label))
        current += timedelta(minutes=snapshot_interval_minutes)

    return snapshots


def generate_training_dataset(
    n_patients: int = 50,
    days_per_patient: int = 10,
    seed: int = _DEFAULT_SEED,
) -> Tuple[List[PredictionInput], List[bool]]:
    """
    Generate a synthetic multi-patient training dataset.

    Returns ``(snapshots, labels)`` ready to pass to the predictor's
    ``train`` method.
    """
    from hypoglycemia_predictor.models.data_models import AlertThresholds

    rng_seeds = np.random.default_rng(seed).integers(0, 99999, size=n_patients * days_per_patient)
    idx = 0
    all_snapshots: List[PredictionInput] = []
    all_labels: List[bool] = []

    base_date = datetime(2024, 1, 1)

    for patient_i in range(n_patients):
        patient = PatientProfile(
            patient_id=f"P{patient_i:03d}",
            name=f"Patient {patient_i}",
            diabetes_type="Type 1",
            insulin_half_life_minutes=float(random.randint(50, 80)),
            alert_thresholds=AlertThresholds(
                hypoglycaemia_mgdl=70.0,
                caution_mgdl=80.0,
                prediction_horizon_minutes=30,
            ),
        )
        for day_i in range(days_per_patient):
            day = base_date + timedelta(days=patient_i * days_per_patient + day_i)
            include_hypo = (day_i % 3 == 0)  # every 3rd day has a hypo episode
            glucose, meds, meal_evts, ex_evts = generate_patient_day(
                patient=patient,
                date=day,
                include_hypo_episode=include_hypo,
                seed=int(rng_seeds[idx]),
            )
            idx += 1
            pairs = build_prediction_snapshots(patient, glucose, meds, meal_evts, ex_evts)
            for snap, label in pairs:
                all_snapshots.append(snap)
                all_labels.append(label)

    return all_snapshots, all_labels


# ---------------------------------------------------------------------------
# Internal glucose simulation
# ---------------------------------------------------------------------------

def _carbs_to_insulin(carbs_g: float, correction_factor: float, rng: np.random.Generator) -> float:
    """Insulin-to-carb ratio heuristic (1 IU per 10 g), with noise."""
    base = carbs_g / 10.0
    noise = float(rng.normal(0, 0.5))
    return max(0.5, round(base + noise, 1))


def _simulate_glucose(
    day_start: datetime,
    meal_schedule: list,
    medications: List[MedicationDose],
    exercises: List[ExerciseEvent],
    include_hypo_episode: bool,
    rng: np.random.Generator,
) -> List[GlucoseReading]:
    """
    Simulate a piecewise glucose trace for 24 hours at 5-minute resolution.

    Uses a simplified physiological model:
    - Fasting baseline ~90–110 mg/dL
    - Meals cause a glucose rise peaking ~60 min after, decaying over 2 hours
    - Insulin causes a glucose drop peaking ~45 min after, decaying over 4 hours
    - Exercise causes a mild glucose drop and increases insulin sensitivity
    - Gaussian noise is added to every reading
    """
    readings: List[GlucoseReading] = []
    n_steps = 24 * 60 // 5  # 288 readings per day
    glucose = float(rng.uniform(90, 110))  # fasting start

    for step in range(n_steps):
        t = day_start + timedelta(minutes=step * 5)
        delta = 0.0

        # Meal contribution: trapezoidal rise
        for meal_t, _, carbs_g in meal_schedule:
            elapsed = (t - meal_t).total_seconds() / 60
            if 0 <= elapsed < 60:
                delta += (carbs_g / 50) * (elapsed / 60) * 4.0
            elif 60 <= elapsed < 120:
                delta += (carbs_g / 50) * (1 - (elapsed - 60) / 60) * 4.0

        # Insulin contribution: exponential decay drop
        for dose in medications:
            elapsed = (t - dose.timestamp).total_seconds() / 60
            if 0 <= elapsed < 240:
                effect = dose.dose_amount * _INSULIN_EFFECT_SCALE * math.exp(-elapsed / _INSULIN_PEAK_TIME_MINUTES) * (elapsed / _INSULIN_PEAK_TIME_MINUTES)
                delta -= effect / 60  # spread over time steps

        # Exercise contribution: mild drop + sensitivity
        for ex in exercises:
            elapsed = (t - ex.timestamp).total_seconds() / 60
            ex_dur = ex.duration_minutes
            if 0 <= elapsed < ex_dur:
                delta -= _EXERCISE_GLUCOSE_DROP_RATE * _EXERCISE_INTENSITY_FACTOR[ex.intensity]
            elif ex_dur <= elapsed < ex_dur + 120:
                # Post-exercise sensitivity amplifies any insulin drop
                delta -= _POST_EXERCISE_SENSITIVITY_FACTOR * _EXERCISE_INTENSITY_FACTOR[ex.intensity]

        # Mean-reversion towards baseline
        baseline = 100.0
        delta += (baseline - glucose) * _MEAN_REVERSION_RATE

        # Gaussian noise
        noise = float(rng.normal(0, 1.5))
        glucose = max(40.0, min(400.0, glucose + delta + noise))

        # Artificial hypo episode: override glucose to drop to ~55 mg/dL around 14:00–15:00
        if include_hypo_episode:
            hypo_peak = day_start + timedelta(hours=14, minutes=30)
            hypo_elapsed = (t - (day_start + timedelta(hours=13, minutes=30))).total_seconds() / 60
            if 0 <= hypo_elapsed < 60:
                # Gradual controlled descent to hypo territory
                fraction = hypo_elapsed / 60.0
                target = 100.0 - fraction * 50.0  # 100 → 50 mg/dL over 60 min
                glucose = glucose * (1 - 0.3) + target * 0.3
                glucose = max(48.0, glucose)
            elif 60 <= hypo_elapsed < 120:
                # Recovery
                fraction = (hypo_elapsed - 60) / 60.0
                target = 50.0 + fraction * 50.0
                glucose = glucose * (1 - 0.2) + target * 0.2

        readings.append(GlucoseReading(timestamp=t, value_mgdl=round(glucose, 1)))

    return readings
