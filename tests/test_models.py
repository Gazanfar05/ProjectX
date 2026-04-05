"""
Unit tests for data models.
"""

from datetime import datetime

import pytest

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


class TestGlucoseReading:
    def test_valid(self):
        r = GlucoseReading(timestamp=datetime.now(), value_mgdl=110.0)
        assert r.value_mgdl == 110.0

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            GlucoseReading(timestamp=datetime.now(), value_mgdl=-1.0)

    def test_zero_is_valid(self):
        r = GlucoseReading(timestamp=datetime.now(), value_mgdl=0.0)
        assert r.value_mgdl == 0.0


class TestMedicationDose:
    def test_defaults(self):
        dose = MedicationDose(
            timestamp=datetime.now(),
            medication_name="rapid-acting insulin",
            dose_amount=4.0,
        )
        assert dose.dose_unit == "IU"
        assert dose.duration_of_action_hours is None


class TestPatientProfile:
    def test_default_thresholds(self):
        p = PatientProfile(patient_id="P001", name="Alice")
        assert p.alert_thresholds.hypoglycaemia_mgdl == 70.0
        assert p.alert_thresholds.caution_mgdl == 80.0
        assert p.alert_thresholds.prediction_horizon_minutes == 30


class TestAlertSeverity:
    def test_enum_values(self):
        assert AlertSeverity.NONE.value == "none"
        assert AlertSeverity.HIGH.value == "high"


class TestPredictionResult:
    def test_defaults(self):
        result = PredictionResult(
            timestamp=datetime.now(),
            risk_score=0.8,
            predicted_glucose_mgdl=62.0,
            prediction_horizon_minutes=30,
            alert_severity=AlertSeverity.HIGH,
            confidence=0.9,
        )
        assert result.recommendations == []
        assert result.alert_message == ""
