"""
Alert system: converts a ``PredictionResult`` into human-readable messages
and actionable recommendations.
"""

from __future__ import annotations

from typing import List

from hypoglycemia_predictor.models.data_models import (
    AlertSeverity,
    PredictionResult,
)

# ---------------------------------------------------------------------------
# Alert messages
# ---------------------------------------------------------------------------

_MESSAGES = {
    AlertSeverity.HIGH: (
        "🚨 HIGH RISK: Severe hypoglycaemia predicted within {horizon} minutes. "
        "Predicted glucose: {glucose:.0f} mg/dL. Act immediately."
    ),
    AlertSeverity.MEDIUM: (
        "⚠️  MEDIUM RISK: Hypoglycaemia predicted within {horizon} minutes. "
        "Predicted glucose: {glucose:.0f} mg/dL. Please take action soon."
    ),
    AlertSeverity.LOW: (
        "ℹ️  LOW RISK: Glucose trending towards the caution zone within "
        "{horizon} minutes. Predicted glucose: {glucose:.0f} mg/dL. "
        "Monitor closely."
    ),
    AlertSeverity.NONE: (
        "✅ No hypoglycaemia risk detected. "
        "Predicted glucose: {glucose:.0f} mg/dL."
    ),
}

# ---------------------------------------------------------------------------
# Recommendations by severity
# ---------------------------------------------------------------------------

_RECOMMENDATIONS: dict = {
    AlertSeverity.HIGH: [
        "Consume 15–20 g of fast-acting carbohydrates immediately (e.g., 4 glucose tablets, 150 mL fruit juice).",
        "Do not drive or operate machinery.",
        "Check your glucose again in 15 minutes.",
        "Inform someone nearby of your situation.",
        "If unconscious or unable to swallow, seek emergency medical help (call 999 / 911).",
    ],
    AlertSeverity.MEDIUM: [
        "Eat a small snack with 15 g of fast-acting carbohydrates (e.g., glucose tablets, half a banana).",
        "Avoid intense exercise until glucose stabilises.",
        "Check your glucose in 15 minutes.",
        "Consider reducing your next insulin dose if a pattern is emerging — consult your care team.",
    ],
    AlertSeverity.LOW: [
        "Prepare a small snack in case glucose continues to drop.",
        "Monitor your glucose closely over the next 30 minutes.",
        "Avoid skipping your next meal.",
        "Consider reducing activity intensity temporarily.",
    ],
    AlertSeverity.NONE: [
        "Continue your current management plan.",
        "Stay hydrated and keep regular meal times.",
    ],
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_alert(result: PredictionResult) -> PredictionResult:
    """
    Populate the ``alert_message`` and ``recommendations`` fields of
    *result* in place, and return it.
    """
    template = _MESSAGES[result.alert_severity]
    result.alert_message = template.format(
        horizon=result.prediction_horizon_minutes,
        glucose=result.predicted_glucose_mgdl,
    )
    result.recommendations = list(_RECOMMENDATIONS[result.alert_severity])
    return result


def format_alert(result: PredictionResult) -> str:
    """
    Return a multi-line string suitable for printing to the console or
    sending as a push notification.
    """
    if not result.alert_message:
        build_alert(result)

    lines: List[str] = [
        "=" * 60,
        f"  HYPOGLYCAEMIA RISK ASSESSMENT  [{result.timestamp:%Y-%m-%d %H:%M}]",
        "=" * 60,
        f"  Severity      : {result.alert_severity.value.upper()}",
        f"  Risk Score    : {result.risk_score:.1%}",
        f"  Predicted BG  : {result.predicted_glucose_mgdl:.0f} mg/dL "
        f"(in {result.prediction_horizon_minutes} min)",
        f"  Confidence    : {result.confidence:.1%}",
        "-" * 60,
        f"  {result.alert_message}",
        "-" * 60,
        "  Recommendations:",
    ]
    for i, rec in enumerate(result.recommendations, 1):
        lines.append(f"  {i}. {rec}")
    lines.append("=" * 60)
    return "\n".join(lines)


def should_alert(result: PredictionResult) -> bool:
    """Return ``True`` if the result warrants notifying the user."""
    return result.alert_severity != AlertSeverity.NONE
