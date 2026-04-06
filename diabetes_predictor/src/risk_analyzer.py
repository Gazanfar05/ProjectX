import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class GlucoseRiskAnalyzer:
    """Analyze glucose risk using: Past Glucose + Time + Activity → ML Model → Future Glucose → Risk Alert"""
    
    def __init__(self, model):
        self.model = model
        self.risk_thresholds = {
            'low': 0.3,
            'warning': 0.6,
            'critical': 0.8
        }
    
    def analyze_risk(self, user_data: dict) -> dict:
        """
        Analyze glucose risk
        
        Input:
        {
            'past_glucose': [100, 102, 101, 99, 98, 95, 92, 90, 88, 87, 86, 85],
            'current_time': datetime,
            'current_glucose': 85,
            'time_of_day': 'afternoon',
            'activity': 'exercise',
            'hours_since_insulin': 2.5,
            'hours_since_meal': 1.5,
            'last_carbs': 45
        }
        """
        
        print("\n" + "="*60)
        print("🔍 GLUCOSE RISK ANALYSIS")
        print("="*60)
        
        # 1. PAST GLUCOSE ANALYSIS
        past_glucose = np.array(user_data['past_glucose'])
        print(f"\n📊 PAST GLUCOSE (Last 60 minutes)")
        print(f"   Current: {user_data['current_glucose']} mg/dL")
        print(f"   Range: {past_glucose.min():.0f} - {past_glucose.max():.0f} mg/dL")
        print(f"   Trend: {self._analyze_trend(past_glucose)}")
        
        # 2. TIME & ACTIVITY ANALYSIS
        print(f"\n⏰ TIME & ACTIVITY CONTEXT")
        print(f"   Time of Day: {user_data['time_of_day']}")
        print(f"   Activity: {user_data['activity']}")
        print(f"   Hours Since Insulin: {user_data['hours_since_insulin']:.1f}h")
        print(f"   Hours Since Meal: {user_data['hours_since_meal']:.1f}h")
        
        # 3. ML MODEL PREDICTION
        print(f"\n🧠 ML MODEL ANALYSIS")
        risk_score = self._predict_risk(user_data)
        print(f"   Risk Score: {risk_score:.2%}")
        
        # 4. FUTURE GLUCOSE PREDICTION
        print(f"\n🔮 PREDICTED GLUCOSE (Next 30 minutes)")
        future_glucose = self._predict_future_glucose(user_data, risk_score)
        print(f"   Predicted Glucose: {future_glucose:.0f} mg/dL")
        print(f"   Trend: {self._get_future_trend(user_data, future_glucose)}")
        
        # 5. RISK ALERT
        print(f"\n⚠️  RISK ALERT")
        alert = self._generate_alert(risk_score, future_glucose, user_data)
        print(f"   Alert Level: {alert['level']}")
        print(f"   Message: {alert['message']}")
        print(f"   Recommendation: {alert['recommendation']}")
        
        print("\n" + "="*60)
        
        return {
            'risk_score': risk_score,
            'predicted_glucose': future_glucose,
            'alert': alert,
            'past_glucose_trend': self._analyze_trend(past_glucose),
            'analysis_timestamp': datetime.now()
        }
    
    def _analyze_trend(self, glucose_values):
        """Analyze glucose trend from past readings"""
        if len(glucose_values) < 2:
            return "Insufficient data"
        
        recent = glucose_values[-3:]
        previous = glucose_values[-6:-3] if len(glucose_values) >= 6 else glucose_values[:3]
        
        avg_recent = np.mean(recent)
        avg_previous = np.mean(previous)
        
        rate = (glucose_values[-1] - glucose_values[0]) / len(glucose_values)
        
        if rate < -2:
            return f"📉 Rapidly Declining ({rate:.1f} mg/dL per reading)"
        elif rate < -0.5:
            return f"📉 Slowly Declining ({rate:.1f} mg/dL per reading)"
        elif rate > 2:
            return f"📈 Rapidly Rising ({rate:.1f} mg/dL per reading)"
        elif rate > 0.5:
            return f"📈 Slowly Rising ({rate:.1f} mg/dL per reading)"
        else:
            return f"➡️  Stable ({rate:.1f} mg/dL per reading)"
    
    def _predict_risk(self, user_data):
        """Use ML model to predict hypoglycemia risk"""
        # Prepare sequence from user data
        sequence = self._prepare_sequence(user_data)
        
        # Get model prediction
        risk_score = self.model.predict_risk(sequence)
        
        return float(risk_score)
    
    def _prepare_sequence(self, user_data):
        """Prepare 12-step sequence for ML model"""
        past_glucose = np.array(user_data['past_glucose'])
        
        # Ensure we have exactly 12 readings
        if len(past_glucose) < 12:
            # Pad with first value if not enough readings
            past_glucose = np.pad(past_glucose, (12 - len(past_glucose), 0), mode='edge')
        elif len(past_glucose) > 12:
            # Take last 12 if too many
            past_glucose = past_glucose[-12:]
        
        # Calculate features for each timestep
        sequence = []
        
        for i in range(len(past_glucose)):
            glucose_diff = past_glucose[i] - past_glucose[i-1] if i > 0 else 0
            glucose_rate = glucose_diff  # Rate of change
            
            hour = datetime.now().hour
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            
            is_weekend = float(datetime.now().weekday() >= 5)
            
            features = [
                past_glucose[i],  # glucose level
                glucose_diff,     # glucose change
                glucose_rate,     # glucose rate
                hour_sin,         # time context
                hour_cos,
                is_weekend,
                user_data['hours_since_insulin'],
                user_data['hours_since_meal'],
                user_data['last_carbs']
            ]
            
            sequence.append(features)
        
        sequence = np.array(sequence).reshape(1, 12, 9)
        return sequence
    
    def _predict_future_glucose(self, user_data, risk_score):
        """Predict glucose in 30 minutes"""
        current_glucose = user_data['current_glucose']
        activity = user_data['activity']
        hours_since_insulin = user_data['hours_since_insulin']
        hours_since_meal = user_data['hours_since_meal']
        
        # Base trend from past
        past_glucose = np.array(user_data['past_glucose'])
        trend = (past_glucose[-1] - past_glucose[0]) / len(past_glucose)
        
        # Activity impact (30 min ahead)
        activity_impact = {
            'exercise': -15,
            'meal': +20,
            'work': -5,
            'rest': -2,
            'sleep': -8
        }
        
        impact = activity_impact.get(activity, 0)
        
        # Insulin on board (IOB) impact
        iob_impact = -5 if hours_since_insulin < 4 else 0
        
        # Project future glucose
        future_glucose = current_glucose + trend + impact + iob_impact
        
        # Apply risk as modifier
        if risk_score > 0.7:
            future_glucose -= 5
        
        return max(40, min(400, future_glucose))
    
    def _get_future_trend(self, user_data, future_glucose):
        """Describe future trend"""
        current = user_data['current_glucose']
        diff = future_glucose - current
        
        if diff < -20:
            return f"🔴 Sharp Decline (→ {future_glucose:.0f} mg/dL)"
        elif diff < -5:
            return f"🟡 Moderate Decline (→ {future_glucose:.0f} mg/dL)"
        elif diff > 20:
            return f"🟢 Sharp Rise (→ {future_glucose:.0f} mg/dL)"
        elif diff > 5:
            return f"🟢 Moderate Rise (→ {future_glucose:.0f} mg/dL)"
        else:
            return f"🟢 Stable (→ {future_glucose:.0f} mg/dL)"
    
    def _generate_alert(self, risk_score, predicted_glucose, user_data):
        """Generate risk alert based on analysis"""
        
        alert = {
            'level': 'normal',
            'message': '',
            'recommendation': '',
            'urgent': False
        }
        
        # Determine alert level
        if predicted_glucose < 54:
            alert['level'] = '🔴 CRITICAL'
            alert['message'] = f"SEVERE HYPOGLYCEMIA IMMINENT! Predicted glucose: {predicted_glucose:.0f} mg/dL"
            alert['recommendation'] = "🚨 TAKE 20g FAST CARBS IMMEDIATELY! (juice, glucose tablet, honey)"
            alert['urgent'] = True
        
        elif predicted_glucose < 70:
            alert['level'] = '🟡 WARNING'
            alert['message'] = f"HIGH RISK OF HYPOGLYCEMIA. Predicted glucose: {predicted_glucose:.0f} mg/dL"
            alert['recommendation'] = "⚡ Take 15g carbs now (4oz juice or 4 glucose tablets)"
            alert['urgent'] = True
        
        elif risk_score > 0.6 and user_data['current_glucose'] < 100:
            alert['level'] = '🟡 CAUTION'
            alert['message'] = f"Elevated risk with low glucose. Current: {user_data['current_glucose']:.0f} mg/dL"
            alert['recommendation'] = "⏱️ Monitor closely. Have carbs ready. Recheck in 15 min"
            alert['urgent'] = False
        
        elif risk_score > 0.4:
            alert['level'] = '🟢 MONITOR'
            alert['message'] = f"Moderate risk detected. Keep an eye on glucose."
            alert['recommendation'] = "📊 Continue normal activity. Check glucose in 30 min"
            alert['urgent'] = False
        
        else:
            alert['level'] = '✅ SAFE'
            alert['message'] = f"Glucose levels stable and safe."
            alert['recommendation'] = "👍 Continue normal routine"
            alert['urgent'] = False
        
        return alert
