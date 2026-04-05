"""
Unit tests for the data generator.
"""

from datetime import datetime

import pytest

from hypoglycemia_predictor.data_generator import (
    build_prediction_snapshots,
    generate_patient_day,
    generate_training_dataset,
)
from hypoglycemia_predictor.models.data_models import AlertThresholds, PatientProfile


def _patient() -> PatientProfile:
    return PatientProfile(
        patient_id="GEN001",
        name="Generator Tester",
        alert_thresholds=AlertThresholds(
            hypoglycaemia_mgdl=70.0,
            caution_mgdl=80.0,
            prediction_horizon_minutes=30,
        ),
    )


class TestGeneratePatientDay:
    def test_returns_four_lists(self):
        patient = _patient()
        result = generate_patient_day(patient, datetime(2024, 1, 10))
        assert len(result) == 4

    def test_glucose_count(self):
        patient = _patient()
        glucose, *_ = generate_patient_day(patient, datetime(2024, 1, 10))
        # 24 hours × 12 readings/hour = 288
        assert len(glucose) == 288

    def test_glucose_values_in_range(self):
        patient = _patient()
        glucose, *_ = generate_patient_day(patient, datetime(2024, 1, 10))
        values = [r.value_mgdl for r in glucose]
        assert all(30.0 <= v <= 400.0 for v in values)

    def test_hypo_episode_included(self):
        patient = _patient()
        glucose, *_ = generate_patient_day(
            patient, datetime(2024, 1, 10), include_hypo_episode=True, seed=42
        )
        values = [r.value_mgdl for r in glucose]
        assert min(values) < 70.0, "Hypo episode should push glucose below 70 mg/dL"

    def test_no_hypo_without_flag(self):
        patient = _patient()
        glucose, *_ = generate_patient_day(
            patient, datetime(2024, 1, 10), include_hypo_episode=False, seed=42
        )
        values = [r.value_mgdl for r in glucose]
        # Without forced episode, glucose should generally stay above 70
        # (a few natural dips may occur; we check the minimum is > 60)
        assert min(values) > 60.0

    def test_reproducible_with_seed(self):
        patient = _patient()
        g1, *_ = generate_patient_day(patient, datetime(2024, 1, 1), seed=7)
        g2, *_ = generate_patient_day(patient, datetime(2024, 1, 1), seed=7)
        assert [r.value_mgdl for r in g1] == [r.value_mgdl for r in g2]

    def test_different_seeds_differ(self):
        patient = _patient()
        g1, *_ = generate_patient_day(patient, datetime(2024, 1, 1), seed=1)
        g2, *_ = generate_patient_day(patient, datetime(2024, 1, 1), seed=2)
        assert [r.value_mgdl for r in g1] != [r.value_mgdl for r in g2]


class TestBuildPredictionSnapshots:
    def test_returns_list_of_tuples(self):
        patient = _patient()
        glucose, meds, meals, exercises = generate_patient_day(
            patient, datetime(2024, 1, 5), seed=10
        )
        pairs = build_prediction_snapshots(patient, glucose, meds, meals, exercises)
        assert isinstance(pairs, list)
        assert all(isinstance(p, tuple) and len(p) == 2 for p in pairs)

    def test_labels_are_bool(self):
        patient = _patient()
        glucose, meds, meals, exercises = generate_patient_day(
            patient, datetime(2024, 1, 5), seed=10
        )
        pairs = build_prediction_snapshots(patient, glucose, meds, meals, exercises)
        labels = [label for _, label in pairs]
        assert all(isinstance(lbl, bool) for lbl in labels)

    def test_hypo_day_has_positive_labels(self):
        patient = _patient()
        glucose, meds, meals, exercises = generate_patient_day(
            patient, datetime(2024, 1, 5), include_hypo_episode=True, seed=42
        )
        pairs = build_prediction_snapshots(patient, glucose, meds, meals, exercises)
        labels = [label for _, label in pairs]
        assert any(labels), "A day with hypo episode must have at least one positive label"

    def test_empty_glucose_returns_empty(self):
        patient = _patient()
        pairs = build_prediction_snapshots(patient, [], [], [], [])
        assert pairs == []


class TestGenerateTrainingDataset:
    def test_basic_shape(self):
        snapshots, labels = generate_training_dataset(
            n_patients=3, days_per_patient=2, seed=0
        )
        assert len(snapshots) == len(labels)
        assert len(snapshots) > 0

    def test_labels_contain_both_classes(self):
        snapshots, labels = generate_training_dataset(
            n_patients=5, days_per_patient=3, seed=0
        )
        assert True in labels
        assert False in labels
