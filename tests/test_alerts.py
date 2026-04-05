"""
Unit tests for the alert system.
"""

from datetime import datetime

import pytest

from hypoglycemia_predictor.alerts import build_alert, format_alert, should_alert
from hypoglycemia_predictor.models.data_models import AlertSeverity, PredictionResult


def _make_result(severity: AlertSeverity, glucose: float = 65.0) -> PredictionResult:
    return PredictionResult(
        timestamp=datetime(2024, 6, 1, 14, 0),
        risk_score=0.85 if severity == AlertSeverity.HIGH else 0.5,
        predicted_glucose_mgdl=glucose,
        prediction_horizon_minutes=30,
        alert_severity=severity,
        confidence=0.9,
    )


class TestBuildAlert:
    def test_sets_message(self):
        result = _make_result(AlertSeverity.HIGH, glucose=58.0)
        build_alert(result)
        assert len(result.alert_message) > 0
        assert "58" in result.alert_message

    def test_sets_recommendations(self):
        result = _make_result(AlertSeverity.MEDIUM, glucose=66.0)
        build_alert(result)
        assert len(result.recommendations) > 0

    def test_none_severity_has_recommendations(self):
        result = _make_result(AlertSeverity.NONE, glucose=110.0)
        build_alert(result)
        assert result.recommendations  # even NONE has generic advice

    @pytest.mark.parametrize("severity", list(AlertSeverity))
    def test_all_severities_produce_message(self, severity):
        result = _make_result(severity)
        build_alert(result)
        assert result.alert_message != ""


class TestFormatAlert:
    def test_returns_string(self):
        result = _make_result(AlertSeverity.HIGH, glucose=55.0)
        output = format_alert(result)
        assert isinstance(output, str)
        assert len(output) > 0

    def test_contains_severity(self):
        result = _make_result(AlertSeverity.MEDIUM, glucose=68.0)
        output = format_alert(result)
        assert "MEDIUM" in output

    def test_contains_glucose(self):
        result = _make_result(AlertSeverity.LOW, glucose=76.0)
        output = format_alert(result)
        assert "76" in output

    def test_contains_recommendations(self):
        result = _make_result(AlertSeverity.HIGH, glucose=52.0)
        output = format_alert(result)
        assert "1." in output  # numbered recommendation


class TestShouldAlert:
    def test_none_severity_no_alert(self):
        result = _make_result(AlertSeverity.NONE, glucose=120.0)
        assert not should_alert(result)

    @pytest.mark.parametrize(
        "severity",
        [AlertSeverity.LOW, AlertSeverity.MEDIUM, AlertSeverity.HIGH],
    )
    def test_non_none_should_alert(self, severity):
        result = _make_result(severity)
        assert should_alert(result)
