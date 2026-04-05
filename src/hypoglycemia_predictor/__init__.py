"""
hypoglycemia_predictor — public API surface.
"""

from hypoglycemia_predictor.alerts import build_alert, format_alert, should_alert
from hypoglycemia_predictor.features import FEATURE_NAMES, extract_features
from hypoglycemia_predictor.models.data_models import (
    AlertSeverity,
    AlertThresholds,
    ExerciseEvent,
    ExerciseIntensity,
    GlucoseReading,
    MealEvent,
    MealSize,
    MedicationDose,
    PatientProfile,
    PredictionInput,
    PredictionResult,
)
from hypoglycemia_predictor.predictor import HypoglycaemiaPredictor

__all__ = [
    # models
    "AlertSeverity",
    "AlertThresholds",
    "ExerciseEvent",
    "ExerciseIntensity",
    "GlucoseReading",
    "MealEvent",
    "MealSize",
    "MedicationDose",
    "PatientProfile",
    "PredictionInput",
    "PredictionResult",
    # features
    "FEATURE_NAMES",
    "extract_features",
    # predictor
    "HypoglycaemiaPredictor",
    # alerts
    "build_alert",
    "format_alert",
    "should_alert",
]
