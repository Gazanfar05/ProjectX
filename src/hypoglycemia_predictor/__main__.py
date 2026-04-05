"""
CLI entry point: trains the model on synthetic data, then runs a live
demonstration by simulating a patient day that includes a hypoglycaemic
episode and printing alerts as they would appear in real time.

Usage::

    python -m hypoglycemia_predictor          # demo mode
    python -m hypoglycemia_predictor --train  # train + save model then demo
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime

from hypoglycemia_predictor.alerts import build_alert, format_alert, should_alert
from hypoglycemia_predictor.data_generator import (
    build_prediction_snapshots,
    generate_patient_day,
    generate_training_dataset,
)
from hypoglycemia_predictor.models.data_models import AlertThresholds, PatientProfile
from hypoglycemia_predictor.predictor import HypoglycaemiaPredictor


def _build_demo_patient() -> PatientProfile:
    return PatientProfile(
        patient_id="DEMO-001",
        name="Alex Johnson",
        diabetes_type="Type 1",
        insulin_half_life_minutes=65.0,
        alert_thresholds=AlertThresholds(
            hypoglycaemia_mgdl=70.0,
            caution_mgdl=80.0,
            prediction_horizon_minutes=30,
        ),
    )


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Hypoglycaemia Prediction System — AI-powered early warning for diabetics."
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model on fresh synthetic data before running the demo.",
    )
    parser.add_argument(
        "--n-patients",
        type=int,
        default=30,
        metavar="N",
        help="Number of synthetic patients to generate for training (default: 30).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        metavar="D",
        help="Days of data per patient (default: 7).",
    )
    args = parser.parse_args(argv)

    predictor = HypoglycaemiaPredictor()

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    print("\n🩺  Hypoglycaemia Prediction System")
    print("=" * 60)

    print(f"\n📊 Generating synthetic training data "
          f"({args.n_patients} patients × {args.days} days)…")
    snapshots, labels = generate_training_dataset(
        n_patients=args.n_patients,
        days_per_patient=args.days,
    )
    print(f"   → {len(snapshots):,} training snapshots "
          f"({sum(labels):,} positive / {len(labels) - sum(labels):,} negative)")

    print("\n🤖 Training gradient-boosting model…")
    metrics = predictor.train(snapshots, labels, verbose=True)

    print("\n📈 Top 10 most important features:")
    for name, imp in predictor.feature_importances()[:10]:
        bar = "█" * int(imp * 500)
        print(f"   {name:<35} {imp:.4f}  {bar}")

    # -----------------------------------------------------------------------
    # Real-time demo: simulate a patient day with a hypo episode
    # -----------------------------------------------------------------------
    patient = _build_demo_patient()
    demo_date = datetime(2024, 6, 15)

    print(f"\n\n🕐  Live Demo — {patient.name}  ({demo_date:%A, %d %B %Y})")
    print("=" * 60)
    print("   Simulating a day with a hypoglycaemic episode around 14:00…\n")

    glucose, meds, meals, exercises = generate_patient_day(
        patient=patient,
        date=demo_date,
        include_hypo_episode=True,
        seed=99,
    )

    demo_snapshots = build_prediction_snapshots(
        patient, glucose, meds, meals, exercises
    )

    alerts_fired = 0
    print(f"   {'Time':>6}  {'BG (mg/dL)':>10}  {'Risk':>6}  {'Severity'}")
    print("   " + "-" * 50)

    for snap, true_label in demo_snapshots:
        result = predictor.predict(snap)
        build_alert(result)

        current_bg = snap.recent_glucose[-1].value_mgdl if snap.recent_glucose else 0

        severity_icon = {
            "none": "  ✅",
            "low": "  ℹ️ ",
            "medium": "  ⚠️ ",
            "high": "  🚨",
        }[result.alert_severity.value]

        t_str = snap.timestamp.strftime("%H:%M")
        print(
            f"   {t_str:>6}  {current_bg:>10.1f}  "
            f"{result.risk_score:>5.1%}  {severity_icon} {result.alert_severity.value.upper()}"
        )

        if should_alert(result) and result.alert_severity.value in ("medium", "high"):
            alerts_fired += 1
            if alerts_fired <= 3:  # show first 3 full alerts
                print()
                print(format_alert(result))
                print()

    print("\n" + "=" * 60)
    print(f"   Demo complete. {alerts_fired} alert(s) generated.")
    print("=" * 60 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
