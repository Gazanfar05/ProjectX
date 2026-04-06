"""Bridge between simulation engine and existing system"""
from src.simulation_engine import SimulationEngine, GlucoseReading
from src.alert_system import AlertSystem
from datetime import datetime


class AutonomousMonitor:
    """Autonomous monitoring without user input"""

    def __init__(self, user_id: str = "sim_user_001"):
        self.engine = SimulationEngine(initial_glucose=115.0)
        self.alert_system = AlertSystem()
        self.user_id = user_id
        self.readings_log = []

    def update(self) -> dict:
        """Get simulated reading and generate alert"""
        reading = self.engine.tick()

        # Convert to format compatible with your existing system
        self.readings_log.append({
            'timestamp': reading.timestamp,
            'glucose_level': reading.glucose,
            'activity': reading.activity.value,
            'trend': reading.trend.value,
            'rate_of_change': reading.rate_of_change,
            'user_id': self.user_id
        })

        # Generate alert
        risk_score = self._calculate_risk(reading.glucose, reading.trend)
        alert = self.alert_system.evaluate_risk(risk_score, reading.glucose)

        return {
            'reading': reading,
            'alert': alert,
            'risk_score': risk_score,
            'message': self.alert_system.generate_alert_message(alert)
        }

    def _calculate_risk(self, glucose: float, trend) -> float:
        """Simple risk calculation"""
        # Risk based on glucose level
        if glucose < 80:
            base_risk = 0.8
        elif glucose < 100:
            base_risk = 0.5
        else:
            base_risk = 0.2

        # Adjust for trend
        if trend.value == 'dropping':
            base_risk += 0.15
        elif trend.value == 'rising':
            base_risk -= 0.1

        return min(1.0, max(0, base_risk))

    def get_dashboard_data(self) -> dict:
        """Get data for dashboard"""
        if not self.readings_log:
            return {}

        latest = self.readings_log[-1]
        history = [r['glucose_level'] for r in self.readings_log]

        return {
            'current_glucose': latest['glucose_level'],
            'activity': latest['activity'],
            'trend': latest['trend'],
            'glucose_history': history,
            'min_glucose': min(history),
            'max_glucose': max(history),
            'avg_glucose': sum(history) / len(history),
            'readings_count': len(self.readings_log),
        }
