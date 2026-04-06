# src/data_models.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List

@dataclass
class GlucoseReading:
    timestamp: datetime
    glucose_level: float  # mg/dL
    user_id: str

@dataclass
class Medication:
    timestamp: datetime
    medication_name: str
    dosage: float
    medication_type: str  # 'insulin', 'oral', etc.
    user_id: str

@dataclass
class LifestyleEvent:
    timestamp: datetime
    event_type: str  # 'meal', 'exercise', 'sleep'
    duration_minutes: Optional[int]
    intensity: Optional[str]  # 'low', 'medium', 'high'
    carbs: Optional[float]  # for meals
    user_id: str

@dataclass
class PredictionResult:
    timestamp: datetime
    risk_score: float  # 0-1 probability
    predicted_glucose: float
    time_to_hypo_minutes: Optional[int]
    alert_level: str  # 'none', 'warning', 'critical'