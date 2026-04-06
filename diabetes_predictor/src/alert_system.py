from datetime import datetime
from src.data_models import PredictionResult


class AlertSystem:
    def __init__(self, 
                 warning_threshold: float = 0.4,
                 critical_threshold: float = 0.7):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        
    def evaluate_risk(self, risk_score: float, current_glucose: float) -> dict:
        """Evaluate risk and create alert"""
        alert_level = self._determine_alert_level(risk_score)
        time_to_hypo = self._estimate_time_to_hypo(risk_score, current_glucose)
        
        return {
            'timestamp': datetime.now(),
            'risk_score': risk_score,
            'predicted_glucose': current_glucose,
            'time_to_hypo_minutes': time_to_hypo,
            'alert_level': alert_level
        }
    
    def _determine_alert_level(self, risk_score: float) -> str:
        """Determine alert level based on risk score"""
        if risk_score >= self.critical_threshold:
            return 'critical'
        elif risk_score >= self.warning_threshold:
            return 'warning'
        return 'safe'
    
    def _estimate_time_to_hypo(self, risk_score: float, current_glucose: float) -> int:
        """Estimate minutes until hypoglycemia"""
        if risk_score < self.warning_threshold:
            return None
        
        estimated_minutes = int((1 - risk_score) * 60)
        return max(5, min(estimated_minutes, 120))
    
    def generate_alert_message(self, prediction: dict) -> str:
        """Generate user-friendly alert message"""
        alert_level = prediction['alert_level']
        
        if alert_level == 'critical':
            return f"🔴 CRITICAL: High risk of hypoglycemia in ~{prediction['time_to_hypo_minutes']} minutes. Check glucose and prepare fast-acting carbs."
        elif alert_level == 'warning':
            return f"⚡ WARNING: Possible hypoglycemia in ~{prediction['time_to_hypo_minutes']} minutes. Monitor closely."
        return "✓ Glucose levels stable."
