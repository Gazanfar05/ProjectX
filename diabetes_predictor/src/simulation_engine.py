"""Glucose simulation engine - Core autonomous logic"""
import random
from datetime import datetime
from enum import Enum
from typing import Tuple, List
from dataclasses import dataclass


class ActivityType(Enum):
    SLEEPING = "sleeping"
    WALKING = "walking"
    SITTING = "sitting"
    ACTIVE = "active"
    RESTING = "resting"


class Trend(Enum):
    RISING = "rising"
    STABLE = "stable"
    DROPPING = "dropping"


@dataclass
class GlucoseReading:
    """Single glucose measurement"""
    timestamp: datetime
    glucose: float
    activity: ActivityType
    activity_confidence: float
    trend: Trend
    rate_of_change: float


class ActivityDetector:
    """3-Layer Hybrid Activity Detection"""

    def __init__(self):
        self.learned_patterns = {
            7: ActivityType.WALKING,
            8: ActivityType.SITTING,
            9: ActivityType.SITTING,
            12: ActivityType.WALKING,
            13: ActivityType.SITTING,
            17: ActivityType.WALKING,
            18: ActivityType.ACTIVE,
            19: ActivityType.ACTIVE,
            20: ActivityType.RESTING,
            21: ActivityType.RESTING,
            22: ActivityType.SLEEPING,
        }

    def get_time_based_activity(self, hour: int) -> Tuple[ActivityType, float]:
        """Layer 1: Time-based inference"""
        if 0 <= hour < 6:
            return (ActivityType.SLEEPING, 0.95)
        elif 7 <= hour < 9:
            return (ActivityType.WALKING, 0.70)
        elif 10 <= hour < 17:
            return (ActivityType.SITTING, 0.75)
        elif 18 <= hour < 20:
            return (ActivityType.ACTIVE, 0.70)
        else:
            return (ActivityType.RESTING, 0.60)

    def get_pattern_activity(self, hour: int) -> Tuple[ActivityType, float]:
        """Layer 2: Pattern-based learning"""
        if hour in self.learned_patterns:
            return (self.learned_patterns[hour], 0.92)
        return (None, 0)

    def get_sensor_activity(self) -> Tuple[ActivityType, float]:
        """Layer 3: Sensor simulation"""
        motion_intensity = max(0, min(1, random.gauss(0.5, 0.2)))

        if motion_intensity < 0.25:
            return (ActivityType.RESTING, motion_intensity / 0.25)
        elif motion_intensity < 0.6:
            return (ActivityType.WALKING, motion_intensity / 0.6)
        else:
            return (ActivityType.ACTIVE, motion_intensity)

    def detect(self, hour: int) -> Tuple[ActivityType, float]:
        """Priority: sensor > pattern > time"""
        sensor_activity, sensor_conf = self.get_sensor_activity()
        
        if sensor_conf > 0.6:
            return (sensor_activity, sensor_conf)
        
        pattern_activity, pattern_conf = self.get_pattern_activity(hour)
        if pattern_activity:
            return (pattern_activity, pattern_conf)
        
        return self.get_time_based_activity(hour)


class GlucoseSimulator:
    """Realistic glucose simulation"""

    def __init__(self, initial_glucose: float = 110.0):
        self.glucose = initial_glucose
        self.history: List[float] = [initial_glucose]
        self.max_history = 20

    def update(self, activity: ActivityType, hour: int) -> float:
        """Update glucose based on activity and time"""
        activity_impact = self._get_activity_impact(activity)
        time_impact = self._get_time_impact(hour)
        noise = random.gauss(0, 1.5)

        glucose_change = activity_impact + time_impact + noise
        self.glucose += glucose_change
        self.glucose = max(60, min(180, self.glucose))

        self.history.append(self.glucose)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        return self.glucose

    def _get_activity_impact(self, activity: ActivityType) -> float:
        impacts = {
            ActivityType.SLEEPING: 0.5,
            ActivityType.RESTING: 0.2,
            ActivityType.SITTING: -1.0,
            ActivityType.WALKING: -3.0,
            ActivityType.ACTIVE: -6.0,
        }
        return impacts.get(activity, 0)

    def _get_time_impact(self, hour: int) -> float:
        if 11 <= hour <= 13:
            return -2.0
        elif 14 <= hour <= 17:
            return -1.5
        elif 6 <= hour <= 8:
            return 1.0
        return 0

    def get_trend(self) -> Trend:
        """Determine trend from history"""
        if len(self.history) < 3:
            return Trend.STABLE

        recent = self.history[-3:]
        avg_recent = sum(recent) / len(recent)
        avg_previous = sum(self.history[-6:-3]) / 3 if len(self.history) >= 6 else avg_recent

        diff = avg_recent - avg_previous

        if diff > 2:
            return Trend.RISING
        elif diff < -2:
            return Trend.DROPPING
        else:
            return Trend.STABLE

    def get_rate_of_change(self) -> float:
        """Rate of change per reading"""
        if len(self.history) < 2:
            return 0
        return self.history[-1] - self.history[-2]


class SimulationEngine:
    """Main orchestrator"""

    def __init__(self, initial_glucose: float = 110.0):
        self.activity_detector = ActivityDetector()
        self.glucose_simulator = GlucoseSimulator(initial_glucose)
        self.start_time = datetime.now()

    def tick(self) -> GlucoseReading:
        """Run one simulation cycle"""
        current_time = datetime.now()
        hour = current_time.hour

        # Detect activity
        activity, confidence = self.activity_detector.detect(hour)

        # Update glucose
        self.glucose_simulator.update(activity, hour)

        # Analyze trend
        trend = self.glucose_simulator.get_trend()
        rate_of_change = self.glucose_simulator.get_rate_of_change()

        return GlucoseReading(
            timestamp=current_time,
            glucose=self.glucose_simulator.glucose,
            activity=activity,
            activity_confidence=confidence,
            trend=trend,
            rate_of_change=rate_of_change,
        )

    def get_history(self) -> List[float]:
        """Get glucose history"""
        return self.glucose_simulator.history.copy()
