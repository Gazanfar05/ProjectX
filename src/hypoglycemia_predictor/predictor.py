"""
Hypoglycaemia prediction engine.

Wraps a scikit-learn GradientBoosting classifier with a full training /
inference pipeline including feature extraction, probability calibration,
and model persistence.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import joblib
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False

from hypoglycemia_predictor.features import FEATURE_NAMES, extract_features
from hypoglycemia_predictor.models.data_models import (
    AlertSeverity,
    AlertThresholds,
    PredictionInput,
    PredictionResult,
)

_DEFAULT_MODEL_PATH = Path(__file__).parent / "trained_model.joblib"


class HypoglycaemiaPredictor:
    """
    Trains, evaluates, and serves a hypoglycaemia risk model.

    Usage
    -----
    >>> predictor = HypoglycaemiaPredictor()
    >>> predictor.train(snapshots, labels)
    >>> result = predictor.predict(snapshot)
    >>> print(result.alert_severity, result.risk_score)
    """

    def __init__(self, model_path: Optional[Path] = None) -> None:
        self._pipeline: Optional[Pipeline] = None
        self._model_path = model_path or _DEFAULT_MODEL_PATH

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        snapshots: List[PredictionInput],
        labels: List[bool],
        test_size: float = 0.2,
        random_state: int = 42,
        verbose: bool = True,
    ) -> dict:
        """
        Build feature matrix and fit the ML pipeline.

        Parameters
        ----------
        snapshots:
            List of ``PredictionInput`` objects (one per training example).
        labels:
            Corresponding boolean labels — ``True`` if a hypoglycaemic
            episode occurred within the prediction horizon.
        test_size:
            Fraction of data to hold out for evaluation.
        random_state:
            Seed for reproducibility.
        verbose:
            Print evaluation metrics after training.

        Returns
        -------
        dict with evaluation metrics on the held-out test set.
        """
        X = np.array([extract_features(s) for s in snapshots])
        y = np.array(labels, dtype=int)

        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            raise ValueError(
                f"Training requires at least 2 classes but only found: {unique_classes.tolist()}. "
                "Ensure the training dataset includes both positive (hypo) and negative examples."
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        base_clf = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=random_state,
        )
        # Calibrate probabilities using isotonic regression so that the
        # predicted probability is well-aligned with the true risk.
        calibrated = CalibratedClassifierCV(base_clf, method="isotonic", cv=3)

        self._pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", calibrated),
        ])

        self._pipeline.fit(X_train, y_train)

        # ---- evaluation ----
        y_pred = self._pipeline.predict(X_test)
        y_prob = self._pipeline.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        report = classification_report(y_test, y_pred, target_names=["No Hypo", "Hypo"])

        metrics = {
            "roc_auc": round(auc, 4),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_positive": int(y.sum()),
            "feature_count": X.shape[1],
        }

        if verbose:
            print("\n=== Hypoglycaemia Predictor — Training Complete ===")
            print(f"Training samples : {metrics['n_train']}")
            print(f"Test samples     : {metrics['n_test']}")
            print(f"Positive labels  : {metrics['n_positive']}")
            print(f"ROC-AUC          : {metrics['roc_auc']}")
            print("\nClassification Report (test set):")
            print(report)

        return metrics

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, snapshot: PredictionInput) -> PredictionResult:
        """
        Return a ``PredictionResult`` for the given snapshot.

        The predicted glucose is estimated from a weighted combination of
        the current glucose and the trend, corrected for the model's risk
        score so that high-risk predictions map to plausible glucose values.
        """
        if self._pipeline is None:
            raise RuntimeError(
                "Model has not been trained. Call train() or load_model() first."
            )

        features = extract_features(snapshot).reshape(1, -1)
        risk_score = float(self._pipeline.predict_proba(features)[0, 1])

        # Estimate predicted glucose from trend extrapolation
        thresholds: AlertThresholds = snapshot.patient.alert_thresholds
        horizon = thresholds.prediction_horizon_minutes
        predicted_glucose = self._estimate_future_glucose(snapshot, horizon, risk_score)

        severity = self._severity(predicted_glucose, thresholds)
        confidence = 1.0 - abs(risk_score - 0.5) * 2  # highest near 0 or 1

        return PredictionResult(
            timestamp=snapshot.timestamp,
            risk_score=round(risk_score, 4),
            predicted_glucose_mgdl=round(predicted_glucose, 1),
            prediction_horizon_minutes=horizon,
            alert_severity=severity,
            confidence=round(confidence, 4),
        )

    def predict_batch(
        self, snapshots: List[PredictionInput]
    ) -> List[PredictionResult]:
        """Batch version of :meth:`predict`."""
        return [self.predict(s) for s in snapshots]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, path: Optional[Path] = None) -> Path:
        """Persist the trained pipeline to disk."""
        if not _JOBLIB_AVAILABLE:
            raise ImportError("joblib is required to save/load models.")
        if self._pipeline is None:
            raise RuntimeError("No trained model to save.")
        save_path = path or self._model_path
        joblib.dump(self._pipeline, save_path)
        return save_path

    def load_model(self, path: Optional[Path] = None) -> None:
        """Load a previously saved pipeline from disk."""
        if not _JOBLIB_AVAILABLE:
            raise ImportError("joblib is required to save/load models.")
        load_path = path or self._model_path
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model file not found: {load_path}")
        self._pipeline = joblib.load(load_path)

    @property
    def is_trained(self) -> bool:
        """Return ``True`` if the model is ready for inference."""
        return self._pipeline is not None

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importances(self) -> List[Tuple[str, float]]:
        """
        Return feature name / importance pairs sorted descending.

        Only available for tree-based models.
        """
        if self._pipeline is None:
            raise RuntimeError("Model is not trained.")

        # Navigate through CalibratedClassifierCV → GradientBoostingClassifier
        clf_step = self._pipeline.named_steps["clf"]
        estimators = getattr(clf_step, "calibrated_classifiers_", None)
        if estimators:
            importances = np.mean(
                [e.estimator.feature_importances_ for e in estimators], axis=0
            )
        else:
            importances = clf_step.feature_importances_

        pairs = sorted(
            zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True
        )
        return [(name, round(float(imp), 6)) for name, imp in pairs]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_future_glucose(
        snapshot: PredictionInput,
        horizon_minutes: int,
        risk_score: float,
    ) -> float:
        """
        Heuristic estimate of glucose in *horizon_minutes*.

        Combines linear trend extrapolation with a risk-score-weighted
        downward correction to keep the predicted value consistent with
        the model's risk output.
        """
        if not snapshot.recent_glucose:
            return 100.0

        current = snapshot.recent_glucose[-1].value_mgdl

        # Calculate short-term trend
        readings_15 = [
            r for r in snapshot.recent_glucose
            if (snapshot.timestamp - r.timestamp).total_seconds() / 60 <= 15
        ]
        if len(readings_15) >= 2:
            deltas = [
                readings_15[i + 1].value_mgdl - readings_15[i].value_mgdl
                for i in range(len(readings_15) - 1)
            ]
            avg_delta_per_5min = np.mean(deltas)
        else:
            avg_delta_per_5min = 0.0

        trend_steps = horizon_minutes / 5
        trend_prediction = current + avg_delta_per_5min * trend_steps

        # Blend trend with a risk-weighted downward correction:
        # risk_score=0 → no correction; risk_score=1 → large downward correction
        correction = risk_score * 30.0  # up to 30 mg/dL reduction
        blended = trend_prediction - correction

        return max(30.0, min(400.0, blended))

    @staticmethod
    def _severity(predicted_glucose: float, thresholds: AlertThresholds) -> AlertSeverity:
        """Map predicted glucose to an alert severity level."""
        if predicted_glucose < thresholds.hypoglycaemia_mgdl - 10:
            return AlertSeverity.HIGH
        if predicted_glucose < thresholds.hypoglycaemia_mgdl:
            return AlertSeverity.MEDIUM
        if predicted_glucose < thresholds.caution_mgdl:
            return AlertSeverity.LOW
        return AlertSeverity.NONE
