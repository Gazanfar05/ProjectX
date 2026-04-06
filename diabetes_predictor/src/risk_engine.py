"""Risk scoring engine - tuned thresholds"""
from typing import Dict, List, Any, Tuple

class RiskEngine:
    def __init__(self):
        self.glucose_weight = 0.50
        self.physiological_weight = 0.25
        self.circadian_weight = 0.10
        self.medication_weight = 0.10
        self.lifestyle_weight = 0.05
    
    def calculate_risk(self, data: Dict[str, Any], glucose_history: List[float], hour_of_day: int) -> Tuple[int, str, Dict[str, Any]]:
        factors = {}
        
        glucose = data.get('glucose', 100)
        rate_mgdl_per_min = data.get('rate_mgdl_per_min', 0)
        hrv_ms = data.get('hrv_ms', 45)
        skin_temp_delta = data.get('skin_temp_c', 36.5) - 36.5
        spo2 = data.get('spo2', 98)
        insulin_on_board = data.get('insulin_on_board', 0)
        activity = data.get('activity_type', 'resting')
        
        # ========== GLUCOSE SCORE (50%) - MAIN DRIVER ==========
        glucose_score = 0
        glucose_breakdown = {}
        
        # Absolute glucose level - AGGRESSIVE SCORING
        if glucose < 50:
            glucose_score += 70
            glucose_breakdown['severe_hypo'] = 70
        elif glucose < 60:
            glucose_score += 60
            glucose_breakdown['critical_hypo'] = 60
        elif glucose < 70:
            glucose_score += 50
            glucose_breakdown['hypo'] = 50
        elif glucose < 80:
            glucose_score += 35
            glucose_breakdown['low'] = 35
        elif glucose < 90:
            glucose_score += 20
            glucose_breakdown['approaching_low'] = 20
        elif glucose < 100:
            glucose_score += 10
            glucose_breakdown['slightly_low'] = 10
        else:
            glucose_score += 2
            glucose_breakdown['normal'] = 2
        
        # Rate of drop - CRITICAL INDICATOR
        if rate_mgdl_per_min < -3:
            glucose_score += 40
            glucose_breakdown['rapid_drop'] = 40
        elif rate_mgdl_per_min < -2:
            glucose_score += 30
            glucose_breakdown['fast_drop'] = 30
        elif rate_mgdl_per_min < -1:
            glucose_score += 20
            glucose_breakdown['moderate_drop'] = 20
        elif rate_mgdl_per_min < 0:
            glucose_score += 10
            glucose_breakdown['slow_drop'] = 10
        else:
            glucose_score += 0
            glucose_breakdown['rising'] = 0
        
        factors['glucose'] = {
            'score': min(100, glucose_score),
            'weight': self.glucose_weight,
            'breakdown': glucose_breakdown,
            'contribution': min(100, glucose_score) * self.glucose_weight
        }
        
        # ========== PHYSIOLOGICAL (25%) ==========
        phys_score = 0
        phys_breakdown = {}
        
        # HRV stress
        if hrv_ms < 15:
            phys_score += 40
            phys_breakdown['severe_stress'] = 40
        elif hrv_ms < 20:
            phys_score += 30
            phys_breakdown['high_stress'] = 30
        elif hrv_ms < 30:
            phys_score += 20
            phys_breakdown['moderate_stress'] = 20
        elif hrv_ms < 40:
            phys_score += 10
            phys_breakdown['mild_stress'] = 10
        else:
            phys_score += 2
            phys_breakdown['normal_stress'] = 2
        
        # Skin temperature
        if skin_temp_delta < -0.5:
            phys_score += 25
            phys_breakdown['severe_cold'] = 25
        elif skin_temp_delta < -0.2:
            phys_score += 15
            phys_breakdown['cold_response'] = 15
        else:
            phys_score += 0
        
        # SpO2
        if spo2 < 94:
            phys_score += 15
            phys_breakdown['hypoxia'] = 15
        elif spo2 < 96:
            phys_score += 8
            phys_breakdown['low_o2'] = 8
        else:
            phys_score += 0
        
        factors['physiological'] = {
            'score': min(100, phys_score),
            'weight': self.physiological_weight,
            'breakdown': phys_breakdown,
            'contribution': min(100, phys_score) * self.physiological_weight
        }
        
        # ========== CIRCADIAN (10%) ==========
        circadian_score = 0
        circadian_breakdown = {}
        
        if 0 <= hour_of_day <= 5:
            circadian_score = 30
            circadian_breakdown['nocturnal_high_risk'] = 30
        elif 5 < hour_of_day <= 8:
            circadian_score = 15
            circadian_breakdown['dawn_phenomenon'] = 15
        else:
            circadian_score = 5
            circadian_breakdown['daytime'] = 5
        
        factors['circadian'] = {
            'score': circadian_score,
            'weight': self.circadian_weight,
            'breakdown': circadian_breakdown,
            'contribution': circadian_score * self.circadian_weight
        }
        
        # ========== MEDICATION (10%) ==========
        med_score = 0
        med_breakdown = {}
        
        if insulin_on_board > 2.5:
            med_score = 30
            med_breakdown['high_iob'] = 30
        elif insulin_on_board > 1.5:
            med_score = 20
            med_breakdown['moderate_high_iob'] = 20
        elif insulin_on_board > 0.8:
            med_score = 12
            med_breakdown['moderate_iob'] = 12
        elif insulin_on_board > 0:
            med_score = 6
            med_breakdown['low_iob'] = 6
        else:
            med_score = 0
            med_breakdown['no_iob'] = 0
        
        factors['medication'] = {
            'score': med_score,
            'weight': self.medication_weight,
            'breakdown': med_breakdown,
            'contribution': med_score * self.medication_weight
        }
        
        # ========== LIFESTYLE (5%) ==========
        lifestyle_score = 0
        lifestyle_breakdown = {}
        
        if activity == 'active':
            lifestyle_score = 25
            lifestyle_breakdown['exercise_effect'] = 25
        elif activity == 'eating':
            lifestyle_score = 5
            lifestyle_breakdown['meal'] = 5
        else:
            lifestyle_score = 3
            lifestyle_breakdown['normal'] = 3
        
        factors['lifestyle'] = {
            'score': lifestyle_score,
            'weight': self.lifestyle_weight,
            'breakdown': lifestyle_breakdown,
            'contribution': lifestyle_score * self.lifestyle_weight
        }
        
        # ========== CALCULATE FINAL SCORE ==========
        total_score = (
            factors['glucose']['contribution'] +
            factors['physiological']['contribution'] +
            factors['circadian']['contribution'] +
            factors['medication']['contribution'] +
            factors['lifestyle']['contribution']
        )
        
        total_score = min(100, int(total_score))
        
        # Determine level - LOWERED THRESHOLDS
        if total_score >= 70:  # Was 85
            level = 'critical'
        elif total_score >= 55:  # Was 70
            level = 'high'
        elif total_score >= 35:  # Was 40
            level = 'elevated'
        else:
            level = 'safe'
        
        return total_score, level, factors
    
    def _calculate_cov(self, values: List[float]) -> float:
        """Calculate coefficient of variation"""
        if len(values) < 2:
            return 0
        mean = sum(values) / len(values)
        if mean == 0:
            return 0
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std_dev = variance ** 0.5
        return (std_dev / mean) * 100
