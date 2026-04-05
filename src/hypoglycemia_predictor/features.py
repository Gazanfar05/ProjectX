"""
Feature engineering for the hypoglycaemia prediction model.

Each ``PredictionInput`` snapshot is converted into a flat numeric feature
vector that the ML model can consume.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd

from hypoglycemia_predictor.models.data_models import (
    ExerciseIntensity,
    MealSize,
    PredictionInput,
)

# Default durations used when per-dose duration is not set.
_INSULIN_DURATION_HOURS = 4.0   # rapid-acting insulin active for ~4 hours
_MEAL_ABSORPTION_HOURS = 2.0    # carb absorption window
_EXERCISE_EFFECT_HOURS = 6.0    # post-exercise insulin sensitivity increase

_MEAL_SIZE_CARBS: Dict[MealSize, float] = {
    MealSize.SMALL: 15.0,
    MealSize.MEDIUM: 40.0,
    MealSize.LARGE: 80.0,
}

_EXERCISE_INTENSITY_FACTOR: Dict[ExerciseIntensity, float] = {
    ExerciseIntensity.LOW: 0.5,
    ExerciseIntensity.MODERATE: 1.0,
    ExerciseIntensity.HIGH: 2.0,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

FEATURE_NAMES: List[str] = [
    # -- Current glucose --
    "glucose_current",
    "glucose_15min_ago",
    "glucose_30min_ago",
    # -- Trend / rate-of-change --
    "glucose_roc_15min",     # mg/dL per minute (last 15 min)
    "glucose_roc_30min",     # mg/dL per minute (last 30 min)
    "glucose_trend_slope",   # linear regression slope over available readings
    # -- Rolling statistics (last 60 min) --
    "glucose_mean_60min",
    "glucose_std_60min",
    "glucose_min_60min",
    "glucose_max_60min",
    # -- Insulin on board --
    "insulin_on_board_iu",   # estimated IU still active
    "minutes_since_last_dose",
    # -- Meal features --
    "carbs_on_board_g",      # estimated unabsorbed carbs
    "minutes_since_last_meal",
    # -- Exercise features --
    "exercise_effect",       # intensity-weighted residual effect (0–1)
    "minutes_since_last_exercise",
    # -- Time-of-day / circadian --
    "hour_sin",              # sine encoding of hour (0–23)
    "hour_cos",              # cosine encoding of hour
    "minutes_since_midnight",
    # -- Meta --
    "prediction_horizon_minutes",
]


def extract_features(snapshot: PredictionInput) -> np.ndarray:
    """
    Convert a ``PredictionInput`` snapshot into a 1-D numpy feature array.

    The order and length must stay in sync with :data:`FEATURE_NAMES`.
    """
    now = snapshot.timestamp
    horizon = snapshot.patient.alert_thresholds.prediction_horizon_minutes

    # -----------------------------------------------------------------------
    # Glucose time series features
    # -----------------------------------------------------------------------
    glucose_series = _glucose_series(snapshot.recent_glucose, now)

    current_glucose = glucose_series.get(0, float("nan"))
    glucose_15 = glucose_series.get(15, float("nan"))
    glucose_30 = glucose_series.get(30, float("nan"))

    roc_15 = _rate_of_change(current_glucose, glucose_15, 15)
    roc_30 = _rate_of_change(current_glucose, glucose_30, 30)
    trend_slope = _trend_slope(snapshot.recent_glucose, now, window_minutes=60)

    g_values_60 = _glucose_window(snapshot.recent_glucose, now, 60)
    g_mean = float(np.nanmean(g_values_60)) if g_values_60 else current_glucose
    g_std = float(np.nanstd(g_values_60)) if len(g_values_60) > 1 else 0.0
    g_min = float(np.nanmin(g_values_60)) if g_values_60 else current_glucose
    g_max = float(np.nanmax(g_values_60)) if g_values_60 else current_glucose

    # -----------------------------------------------------------------------
    # Insulin on board
    # -----------------------------------------------------------------------
    iob, min_since_dose = _insulin_on_board(
        snapshot.recent_medications,
        now,
        snapshot.patient.insulin_half_life_minutes,
    )

    # -----------------------------------------------------------------------
    # Carbohydrates on board
    # -----------------------------------------------------------------------
    cob, min_since_meal = _carbs_on_board(snapshot.recent_meals, now)

    # -----------------------------------------------------------------------
    # Exercise effect
    # -----------------------------------------------------------------------
    exercise_effect, min_since_exercise = _exercise_effect(
        snapshot.recent_exercises, now
    )

    # -----------------------------------------------------------------------
    # Circadian / time-of-day
    # -----------------------------------------------------------------------
    hour = now.hour + now.minute / 60.0
    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)
    minutes_since_midnight = now.hour * 60 + now.minute

    features = [
        _safe(current_glucose),
        _safe(glucose_15),
        _safe(glucose_30),
        _safe(roc_15),
        _safe(roc_30),
        _safe(trend_slope),
        _safe(g_mean),
        _safe(g_std),
        _safe(g_min),
        _safe(g_max),
        _safe(iob),
        _safe(min_since_dose),
        _safe(cob),
        _safe(min_since_meal),
        _safe(exercise_effect),
        _safe(min_since_exercise),
        hour_sin,
        hour_cos,
        float(minutes_since_midnight),
        float(horizon),
    ]

    return np.array(features, dtype=np.float64)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _safe(value: float, default: float = 0.0) -> float:
    """Replace NaN / Inf with *default*."""
    if value is None or not math.isfinite(value):
        return default
    return float(value)


def _glucose_series(
    readings: list, now: datetime
) -> Dict[int, float]:
    """
    Return a dict mapping {minutes_ago: glucose_value} for the readings
    nearest to 0, 15, and 30 minutes ago.
    """
    targets = {0: None, 15: None, 30: None}
    for target_min in targets:
        target_time = now - timedelta(minutes=target_min)
        best_val = None
        best_diff = float("inf")
        for r in readings:
            diff = abs((r.timestamp - target_time).total_seconds())
            if diff < best_diff:
                best_diff = diff
                best_val = r.value_mgdl
        if best_val is not None and best_diff <= 10 * 60:  # within 10 min
            targets[target_min] = best_val
    return {k: v for k, v in targets.items() if v is not None}


def _glucose_window(readings: list, now: datetime, window_minutes: int) -> List[float]:
    """Return glucose values within the last *window_minutes*."""
    cutoff = now - timedelta(minutes=window_minutes)
    return [r.value_mgdl for r in readings if r.timestamp >= cutoff]


def _rate_of_change(current: float, past: float, interval_minutes: int) -> float:
    """mg/dL per minute."""
    if math.isnan(current) or math.isnan(past) or interval_minutes == 0:
        return 0.0
    return (current - past) / interval_minutes


def _trend_slope(readings: list, now: datetime, window_minutes: int) -> float:
    """
    Linear regression slope (mg/dL per minute) over readings within window.
    Returns 0.0 if fewer than 2 data points are available.
    """
    cutoff = now - timedelta(minutes=window_minutes)
    pts = [
        ((r.timestamp - now).total_seconds() / 60, r.value_mgdl)
        for r in readings
        if r.timestamp >= cutoff
    ]
    if len(pts) < 2:
        return 0.0
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    slope = float(np.polyfit(xs, ys, 1)[0])
    return slope


def _insulin_on_board(
    medications: list,
    now: datetime,
    half_life_minutes: float,
) -> tuple[float, float]:
    """
    Estimate total insulin on board (IU) using an exponential decay model.

    Returns ``(total_iob_iu, minutes_since_last_dose)``.
    """
    total_iob = 0.0
    min_since_last = 999.0

    for dose in medications:
        # Only count insulin medications (heuristic name check).
        name = dose.medication_name.lower()
        if "insulin" not in name:
            continue

        elapsed_min = (now - dose.timestamp).total_seconds() / 60
        if elapsed_min < 0:
            continue

        if elapsed_min < min_since_last:
            min_since_last = elapsed_min

        # Exponential decay: IOB = dose * exp(-ln(2) * elapsed / half_life)
        duration_min = (
            dose.duration_of_action_hours * 60
            if dose.duration_of_action_hours
            else _INSULIN_DURATION_HOURS * 60
        )
        if elapsed_min <= duration_min:
            decay = math.exp(-math.log(2) * elapsed_min / half_life_minutes)
            total_iob += dose.dose_amount * decay

    return total_iob, min_since_last


def _carbs_on_board(meals: list, now: datetime) -> tuple[float, float]:
    """
    Estimate unabsorbed carbohydrates (g) using a linear decay over the
    absorption window.

    Returns ``(carbs_on_board_g, minutes_since_last_meal)``.
    """
    total_cob = 0.0
    min_since_last = 999.0

    for meal in meals:
        elapsed_min = (now - meal.timestamp).total_seconds() / 60
        if elapsed_min < 0:
            continue

        if elapsed_min < min_since_last:
            min_since_last = elapsed_min

        total_carbs = (
            meal.carbohydrates_g
            if meal.carbohydrates_g is not None
            else _MEAL_SIZE_CARBS[meal.size]
        )
        absorption_min = _MEAL_ABSORPTION_HOURS * 60
        if elapsed_min < absorption_min:
            fraction_remaining = 1.0 - elapsed_min / absorption_min
            total_cob += total_carbs * fraction_remaining

    return total_cob, min_since_last


def _exercise_effect(exercises: list, now: datetime) -> tuple[float, float]:
    """
    Compute a residual exercise-sensitivity multiplier (0–1 scale) that
    represents increased insulin sensitivity post-exercise.

    Returns ``(effect_score, minutes_since_last_exercise)``.
    """
    total_effect = 0.0
    min_since_last = 999.0

    for ex in exercises:
        elapsed_min = (now - ex.timestamp).total_seconds() / 60
        if elapsed_min < 0:
            continue

        if elapsed_min < min_since_last:
            min_since_last = elapsed_min

        effect_window_min = _EXERCISE_EFFECT_HOURS * 60
        if elapsed_min < effect_window_min:
            intensity_factor = _EXERCISE_INTENSITY_FACTOR[ex.intensity]
            duration_factor = min(ex.duration_minutes / 30.0, 2.0)  # cap at 2×
            decay = 1.0 - elapsed_min / effect_window_min
            total_effect += intensity_factor * duration_factor * decay

    return min(total_effect, 5.0), min_since_last  # cap at 5.0 for scaling
