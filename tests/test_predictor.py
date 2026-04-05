"""
Integration tests for the predictor (train → predict pipeline).
"""

import pytest

from hypoglycemia_predictor.data_generator import (
    build_prediction_snapshots,
    generate_patient_day,
    generate_training_dataset,
)
from hypoglycemia_predictor.models.data_models import (
    AlertSeverity,
    AlertThresholds,
    PatientProfile,
)
from hypoglycemia_predictor.predictor import HypoglycaemiaPredictor
from datetime import datetime


def _patient() -> PatientProfile:
    return PatientProfile(
        patient_id="INT001",
        name="Integration Tester",
        alert_thresholds=AlertThresholds(
            hypoglycaemia_mgdl=70.0,
            caution_mgdl=80.0,
            prediction_horizon_minutes=30,
        ),
    )


@pytest.fixture(scope="module")
def trained_predictor():
    """Shared trained predictor for integration tests (train once per module)."""
    snapshots, labels = generate_training_dataset(
        n_patients=10, days_per_patient=5, seed=42
    )
    predictor = HypoglycaemiaPredictor()
    predictor.train(snapshots, labels, verbose=False)
    return predictor


class TestPredictorNotTrained:
    def test_predict_raises_before_train(self):
        patient = _patient()
        glucose, meds, meals, exercises = generate_patient_day(
            patient, datetime(2024, 3, 1), seed=1
        )
        snapshots = build_prediction_snapshots(patient, glucose, meds, meals, exercises)
        snap, _ = snapshots[0]
        predictor = HypoglycaemiaPredictor()
        with pytest.raises(RuntimeError, match="not been trained"):
            predictor.predict(snap)


class TestPredictorTrained:
    def test_is_trained_flag(self, trained_predictor):
        assert trained_predictor.is_trained

    def test_predict_returns_result(self, trained_predictor):
        patient = _patient()
        glucose, meds, meals, exercises = generate_patient_day(
            patient, datetime(2024, 3, 1), seed=5
        )
        snapshots = build_prediction_snapshots(patient, glucose, meds, meals, exercises)
        snap, _ = snapshots[10]
        result = trained_predictor.predict(snap)
        assert 0.0 <= result.risk_score <= 1.0
        assert result.predicted_glucose_mgdl > 0.0
        assert result.alert_severity in AlertSeverity.__members__.values()

    def test_risk_score_bounds(self, trained_predictor):
        patient = _patient()
        glucose, meds, meals, exercises = generate_patient_day(
            patient, datetime(2024, 3, 2), seed=7
        )
        snapshots = build_prediction_snapshots(patient, glucose, meds, meals, exercises)
        for snap, _ in snapshots[:20]:
            result = trained_predictor.predict(snap)
            assert 0.0 <= result.risk_score <= 1.0

    def test_hypo_day_elevates_risk(self, trained_predictor):
        """Risk scores during a simulated hypo episode should be higher
        on average than a normal day for the same patient."""
        patient = _patient()

        normal_glucose, nm, nmeal, nex = generate_patient_day(
            patient, datetime(2024, 4, 1), include_hypo_episode=False, seed=20
        )
        hypo_glucose, hm, hmeal, hex_ = generate_patient_day(
            patient, datetime(2024, 4, 2), include_hypo_episode=True, seed=21
        )

        normal_snaps = build_prediction_snapshots(patient, normal_glucose, nm, nmeal, nex)
        hypo_snaps = build_prediction_snapshots(patient, hypo_glucose, hm, hmeal, hex_)

        normal_avg = sum(
            trained_predictor.predict(s).risk_score for s, _ in normal_snaps
        ) / max(len(normal_snaps), 1)

        hypo_avg = sum(
            trained_predictor.predict(s).risk_score for s, _ in hypo_snaps
        ) / max(len(hypo_snaps), 1)

        assert hypo_avg > normal_avg, (
            f"Hypo day average risk ({hypo_avg:.3f}) should exceed "
            f"normal day average risk ({normal_avg:.3f})."
        )

    def test_batch_predict(self, trained_predictor):
        patient = _patient()
        glucose, meds, meals, exercises = generate_patient_day(
            patient, datetime(2024, 3, 3), seed=8
        )
        snapshots = build_prediction_snapshots(patient, glucose, meds, meals, exercises)
        snaps = [s for s, _ in snapshots[:5]]
        results = trained_predictor.predict_batch(snaps)
        assert len(results) == 5

    def test_feature_importances(self, trained_predictor):
        importances = trained_predictor.feature_importances()
        assert len(importances) > 0
        names, values = zip(*importances)
        # Sorted descending
        assert list(values) == sorted(values, reverse=True)
        # All non-negative
        assert all(v >= 0 for v in values)
