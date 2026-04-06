from src.risk_analyzer import GlucoseRiskAnalyzer
from src.predictor import HypoglycemiaPredictorModel
from datetime import datetime
import numpy as np

# Initialize model
model = HypoglycemiaPredictorModel()
model.build_model()

analyzer = GlucoseRiskAnalyzer(model)

# SCENARIO 1: Declining glucose during exercise
print("\n\n🏃 SCENARIO 1: Exercise with Declining Glucose")
result1 = analyzer.analyze_risk({
    'past_glucose': [120, 118, 115, 110, 105, 100, 95, 92, 88, 85, 82, 80],
    'current_glucose': 80,
    'time_of_day': 'afternoon',
    'activity': 'exercise',
    'hours_since_insulin': 1.5,
    'hours_since_meal': 3.0,
    'last_carbs': 30
})

# SCENARIO 2: Stable glucose after meal
print("\n\n🍽️ SCENARIO 2: Stable Glucose After Meal")
result2 = analyzer.analyze_risk({
    'past_glucose': [95, 105, 115, 125, 135, 145, 150, 148, 145, 140, 135, 130],
    'current_glucose': 130,
    'time_of_day': 'lunch',
    'activity': 'meal',
    'hours_since_insulin': 0.5,
    'hours_since_meal': 0.25,
    'last_carbs': 60
})

# SCENARIO 3: Rapid decline (HIGH RISK)
print("\n\n⚠️ SCENARIO 3: Rapid Glucose Decline (HIGH RISK)")
result3 = analyzer.analyze_risk({
    'past_glucose': [130, 125, 120, 110, 100, 90, 80, 75, 70, 68, 65, 62],
    'current_glucose': 62,
    'time_of_day': 'evening',
    'activity': 'work',
    'hours_since_insulin': 3.5,
    'hours_since_meal': 5.0,
    'last_carbs': 45
})

# SCENARIO 4: Stable overnight
print("\n\n�� SCENARIO 4: Stable Glucose During Sleep")
result4 = analyzer.analyze_risk({
    'past_glucose': [110, 110, 111, 110, 109, 110, 109, 111, 110, 111, 110, 109],
    'current_glucose': 110,
    'time_of_day': 'night',
    'activity': 'sleep',
    'hours_since_insulin': 6.0,
    'hours_since_meal': 7.0,
    'last_carbs': 0
})

print("\n\n" + "="*60)
print("Analysis Complete!")
print("="*60)
