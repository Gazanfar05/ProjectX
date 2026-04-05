"""
Unit tests for feature engineering.
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

from hypoglycemia_predictor.features import FEATURE_NAMES, extract_features
from hypoglycemia_predictor.models.data_models import (
    AlertThresholds,
    ExerciseEvent,
    ExerciseIntensity,
    GlucoseReading,
    MealEvent,
    MealSize,
    MedicationDose,
    PatientProfile,
    PredictionInput,
)


def _make_patient() -> PatientProfile:
    return PatientProfile(
        patient_id="T001",
        name="Tester",
        insulin_half_life_minutes=60.0,
        alert_thresholds=AlertThresholds(prediction_horizon_minutes=30),
    )


def _make_snapshot(now: datetime, glucose_values: list[float]) -> PredictionInput:
    """Helper: build a snapshot with evenly spaced glucose readings."""
    patient = _make_patient()
    readings = [
        GlucoseReading(
            timestamp=now - timedelta(minutes=(len(glucose_values) - 1 - i) * 5),
            value_mgdl=v,
        )
        for i, v in enumerate(glucose_values)
    ]
    return PredictionInput(
        timestamp=now,
        recent_glucose=readings,
        recent_medications=[],
        recent_meals=[],
        recent_exercises=[],
        patient=patient,
    )


class TestFeatureNames:
    def test_length_matches_extract(self):
        now = datetime(2024, 6, 1, 12, 0)
        snap = _make_snapshot(now, [100.0, 98.0, 96.0, 94.0, 92.0, 90.0])
        features = extract_features(snap)
        assert len(features) == len(FEATURE_NAMES)

    def test_no_nan_in_output(self):
        now = datetime(2024, 6, 1, 12, 0)
        snap = _make_snapshot(now, [100.0, 98.0, 96.0])
        features = extract_features(snap)
        assert not np.any(np.isnan(features)), "Feature vector contains NaN"

    def test_no_inf_in_output(self):
        now = datetime(2024, 6, 1, 8, 30)
        snap = _make_snapshot(now, [110.0, 108.0, 106.0, 104.0, 102.0])
        features = extract_features(snap)
        assert np.all(np.isfinite(features)), "Feature vector contains Inf"


class TestGlucoseFeatures:
    def test_current_glucose_matches_last_reading(self):
        now = datetime(2024, 6, 1, 10, 0)
        snap = _make_snapshot(now, [90.0, 95.0, 100.0])
        features = extract_features(snap)
        idx = FEATURE_NAMES.index("glucose_current")
        assert features[idx] == pytest.approx(100.0, abs=1e-3)

    def test_declining_trend_gives_negative_roc(self):
        now = datetime(2024, 6, 1, 14, 0)
        # Glucose declining by 2 mg/dL every 5 minutes
        snap = _make_snapshot(now, [110.0, 108.0, 106.0, 104.0, 102.0, 100.0])
        features = extract_features(snap)
        roc_idx = FEATURE_NAMES.index("glucose_roc_15min")
        assert features[roc_idx] < 0, "Declining glucose should yield negative RoC"

    def test_stable_glucose_gives_near_zero_slope(self):
        now = datetime(2024, 6, 1, 11, 0)
        snap = _make_snapshot(now, [100.0, 100.0, 100.0, 100.0, 100.0])
        features = extract_features(snap)
        slope_idx = FEATURE_NAMES.index("glucose_trend_slope")
        assert abs(features[slope_idx]) < 0.1


class TestInsulinOnBoard:
    def test_iob_zero_without_insulin(self):
        now = datetime(2024, 6, 1, 12, 0)
        snap = _make_snapshot(now, [100.0])
        features = extract_features(snap)
        idx = FEATURE_NAMES.index("insulin_on_board_iu")
        assert features[idx] == 0.0

    def test_iob_nonzero_after_recent_dose(self):
        now = datetime(2024, 6, 1, 12, 0)
        patient = _make_patient()
        readings = [GlucoseReading(timestamp=now, value_mgdl=100.0)]
        dose = MedicationDose(
            timestamp=now - timedelta(minutes=30),
            medication_name="rapid-acting insulin",
            dose_amount=4.0,
            dose_unit="IU",
        )
        snap = PredictionInput(
            timestamp=now,
            recent_glucose=readings,
            recent_medications=[dose],
            recent_meals=[],
            recent_exercises=[],
            patient=patient,
        )
        features = extract_features(snap)
        idx = FEATURE_NAMES.index("insulin_on_board_iu")
        assert features[idx] > 0.0, "IOB should be > 0 after recent insulin dose"

    def test_iob_excludes_non_insulin(self):
        now = datetime(2024, 6, 1, 12, 0)
        patient = _make_patient()
        dose = MedicationDose(
            timestamp=now - timedelta(minutes=30),
            medication_name="metformin",
            dose_amount=500.0,
            dose_unit="mg",
        )
        snap = PredictionInput(
            timestamp=now,
            recent_glucose=[GlucoseReading(timestamp=now, value_mgdl=100.0)],
            recent_medications=[dose],
            recent_meals=[],
            recent_exercises=[],
            patient=patient,
        )
        features = extract_features(snap)
        idx = FEATURE_NAMES.index("insulin_on_board_iu")
        assert features[idx] == 0.0


class TestMealFeatures:
    def test_cob_nonzero_after_recent_meal(self):
        now = datetime(2024, 6, 1, 13, 30)
        patient = _make_patient()
        meal = MealEvent(
            timestamp=now - timedelta(minutes=30),
            size=MealSize.MEDIUM,
            carbohydrates_g=40.0,
        )
        snap = PredictionInput(
            timestamp=now,
            recent_glucose=[GlucoseReading(timestamp=now, value_mgdl=110.0)],
            recent_medications=[],
            recent_meals=[meal],
            recent_exercises=[],
            patient=patient,
        )
        features = extract_features(snap)
        idx = FEATURE_NAMES.index("carbs_on_board_g")
        assert features[idx] > 0.0


class TestExerciseFeatures:
    def test_exercise_effect_nonzero_after_workout(self):
        now = datetime(2024, 6, 1, 11, 0)
        patient = _make_patient()
        ex = ExerciseEvent(
            timestamp=now - timedelta(minutes=60),
            duration_minutes=30.0,
            intensity=ExerciseIntensity.MODERATE,
        )
        snap = PredictionInput(
            timestamp=now,
            recent_glucose=[GlucoseReading(timestamp=now, value_mgdl=95.0)],
            recent_medications=[],
            recent_meals=[],
            recent_exercises=[ex],
            patient=patient,
        )
        features = extract_features(snap)
        idx = FEATURE_NAMES.index("exercise_effect")
        assert features[idx] > 0.0


class TestCircadianFeatures:
    def test_midnight_features(self):
        midnight = datetime(2024, 6, 1, 0, 0)
        snap = _make_snapshot(midnight, [90.0])
        features = extract_features(snap)
        min_idx = FEATURE_NAMES.index("minutes_since_midnight")
        assert features[min_idx] == 0.0

    def test_noon_minutes_since_midnight(self):
        noon = datetime(2024, 6, 1, 12, 0)
        snap = _make_snapshot(noon, [100.0])
        features = extract_features(snap)
        min_idx = FEATURE_NAMES.index("minutes_since_midnight")
        assert features[min_idx] == pytest.approx(720.0)
