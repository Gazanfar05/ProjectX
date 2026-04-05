"""
Core data models for the hypoglycaemia prediction system.

Units
-----
- Glucose values are in mg/dL throughout the system.
- Insulin doses are in International Units (IU).
- Exercise intensity is 0–10 (subjective perceived exertion scale).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional


class ExerciseIntensity(str, Enum):
    """Subjective exercise intensity level."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class MealSize(str, Enum):
    """Approximate meal carbohydrate load."""

    SMALL = "small"      # <20 g carbs
    MEDIUM = "medium"    # 20–60 g carbs
    LARGE = "large"      # >60 g carbs


class AlertSeverity(str, Enum):
    """Severity level of a hypoglycaemia risk alert."""

    NONE = "none"
    LOW = "low"          # predicted glucose 70–80 mg/dL → caution
    MEDIUM = "medium"    # predicted glucose 60–70 mg/dL → act soon
    HIGH = "high"        # predicted glucose <60 mg/dL → act immediately


@dataclass
class GlucoseReading:
    """A single continuous-glucose-monitor (CGM) or fingerstick reading."""

    timestamp: datetime
    value_mgdl: float  # mg/dL

    def __post_init__(self) -> None:
        if self.value_mgdl < 0:
            raise ValueError("Glucose value cannot be negative.")


@dataclass
class MedicationDose:
    """A recorded insulin or oral-medication administration."""

    timestamp: datetime
    medication_name: str  # e.g. "rapid-acting insulin", "metformin"
    dose_amount: float
    dose_unit: str = "IU"  # IU for insulin, mg for tablets
    # Duration of action in hours; None means system default is used.
    duration_of_action_hours: Optional[float] = None


@dataclass
class MealEvent:
    """A meal or snack that affects blood glucose."""

    timestamp: datetime
    size: MealSize
    carbohydrates_g: Optional[float] = None  # grams; optional if size is used
    description: str = ""


@dataclass
class ExerciseEvent:
    """A physical-activity event."""

    timestamp: datetime
    duration_minutes: float
    intensity: ExerciseIntensity
    description: str = ""


@dataclass
class AlertThresholds:
    """Per-patient glucose thresholds used to trigger alerts."""

    hypoglycaemia_mgdl: float = 70.0   # clinical hypoglycaemia
    caution_mgdl: float = 80.0         # early-warning zone
    prediction_horizon_minutes: int = 30  # how far ahead to predict


@dataclass
class PatientProfile:
    """Static and configuration information about the patient."""

    patient_id: str
    name: str
    diabetes_type: str = "Type 1"
    # Default insulin-on-board half-life (minutes); used if not set on a dose.
    insulin_half_life_minutes: float = 60.0
    alert_thresholds: AlertThresholds = field(default_factory=AlertThresholds)


@dataclass
class PredictionInput:
    """
    A snapshot of all signals available at a given moment, used as input to
    the prediction engine.
    """

    timestamp: datetime
    recent_glucose: List[GlucoseReading]       # last N readings (oldest first)
    recent_medications: List[MedicationDose]   # last N doses
    recent_meals: List[MealEvent]              # last N meals
    recent_exercises: List[ExerciseEvent]      # last N exercise events
    patient: PatientProfile


@dataclass
class PredictionResult:
    """Output of the prediction engine for a single evaluation."""

    timestamp: datetime
    risk_score: float           # 0.0 (no risk) – 1.0 (certain hypoglycaemia)
    predicted_glucose_mgdl: float
    prediction_horizon_minutes: int
    alert_severity: AlertSeverity
    confidence: float           # model confidence (0–1)
    alert_message: str = ""
    recommendations: List[str] = field(default_factory=list)
