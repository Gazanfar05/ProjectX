"""Realistic daily diabetes simulation with guaranteed critical episodes"""
import random
from datetime import datetime
from typing import Dict, Any
from src.simulation_engine import SimulationEngine

class DataSimulator:
    def __init__(self, user_id: str = "sim_user_001"):
        self.user_id = user_id
        self.engine = SimulationEngine(initial_glucose=120.0)
        self.heart_rate = 70
        self.hrv_ms = 45
        self.spo2 = 98
        self.skin_temp_c = 36.5
        self.steps = 0
        self.calories = 0
        self.insulin_on_board = 0
        self.cycle = 0
        self.previous_glucose = 120.0
        self.scenario = self._create_scenario()
        print(f"\n🎭 Scenario loaded - Critical dip cycles 15-23\n")
    
    def _create_scenario(self):
        """Create scenario with GUARANTEED critical dip"""
        return {
            'breakfast': {'start': 1, 'duration': 3, 'glucose_rise': 40},
            'morning': {'start': 4, 'duration': 5, 'glucose_change': -2},
            'lunch': {'start': 9, 'duration': 3, 'glucose_rise': 38},
            'afternoon_stable': {'start': 12, 'duration': 3, 'glucose_change': -5},
            'afternoon_CRITICAL_DIP': {'start': 15, 'duration': 8, 'glucose_change': -25},  # AGGRESSIVE
            'dinner': {'start': 23, 'duration': 3, 'glucose_rise': 40},
            'evening': {'start': 26, 'duration': 24, 'glucose_change': -1},
        }
    
    def _get_phase_impact(self):
        activity = 'resting'
        impact = 0
        
        for phase_name, phase_data in self.scenario.items():
            if phase_data['start'] <= self.cycle < phase_data['start'] + phase_data['duration']:
                if 'glucose_rise' in phase_data:
                    impact = random.uniform(phase_data['glucose_rise'] * 0.75, phase_data['glucose_rise'])
                    activity = 'eating'
                elif 'glucose_change' in phase_data:
                    impact = phase_data['glucose_change'] + random.uniform(-4, 4)
                    if 'CRITICAL' in phase_name:
                        activity = 'active'
        
        return impact, activity
    
    def tick(self) -> Dict[str, Any]:
        self.cycle += 1
        current_time = datetime.now()
        
        try:
            glucose = self.previous_glucose
            phase_impact, activity = self._get_phase_impact()
            
            glucose += phase_impact
            
            # Strong insulin effect during critical dip
            if 15 <= self.cycle <= 23:
                glucose -= random.uniform(1.5, 3.5) * max(self.insulin_on_board, 1.0)
            elif self.insulin_on_board > 0:
                glucose -= random.uniform(0.8, 2.0) * self.insulin_on_board
            
            glucose += random.gauss(0, 1)
            glucose = max(35, min(280, glucose))
            
            # Insulin injections
            if self.cycle in [1, 9, 23]:
                self.insulin_on_board = random.uniform(3.0, 4.5)
                print(f"  💉 Insulin: {self.insulin_on_board:.1f}U (Cycle {self.cycle})")
            
            if self.insulin_on_board > 0:
                self.insulin_on_board *= 0.85
            
            # Heart rate
            if glucose < 55:
                self.heart_rate = 115 + random.randint(20, 45)
            elif glucose < 75:
                self.heart_rate = 95 + random.randint(10, 30)
            elif glucose < 90:
                self.heart_rate = 82 + random.randint(5, 20)
            elif activity == 'active':
                self.heart_rate = 120 + random.randint(10, 35)
            else:
                self.heart_rate = 70 + random.randint(-10, 15)
            
            self.heart_rate = max(50, min(160, self.heart_rate))
            
            # HRV
            if glucose < 60:
                hrv_change = -random.uniform(3, 7)
            elif glucose > 170:
                hrv_change = -random.uniform(1, 3)
            elif activity == 'active':
                hrv_change = -random.uniform(2, 5)
            else:
                hrv_change = random.uniform(-1, 2)
            
            self.hrv_ms = max(8, min(70, self.hrv_ms + hrv_change))
            
            # SpO2
            if glucose < 60:
                self.spo2 = 93 + random.gauss(0, 2.5)
            elif glucose > 180:
                self.spo2 = 96 + random.gauss(0, 1.2)
            else:
                self.spo2 = 97 + random.gauss(0, 0.8)
            
            self.spo2 = max(91, min(100, self.spo2))
            
            # Skin temp
            if glucose < 65:
                temp_change = -random.uniform(0.2, 0.5)
            elif activity == 'active':
                temp_change = random.uniform(0.3, 0.7)
            else:
                temp_change = random.uniform(-0.1, 0.2)
            
            self.skin_temp_c += temp_change
            self.skin_temp_c = max(34.0, min(39.0, self.skin_temp_c))
            
            # Steps
            if activity == 'active':
                self.steps += random.randint(150, 300)
            elif activity == 'eating':
                self.steps += random.randint(20, 60)
            else:
                self.steps += random.randint(20, 70)
            
            self.calories += (self.steps % 80) * 0.05
            
            # Trend
            diff = glucose - self.previous_glucose
            if diff < -2.5:
                trend = 'dropping'
            elif diff > 2.5:
                trend = 'rising'
            else:
                trend = 'stable'
            
            rate_of_change = diff / 5
            self.previous_glucose = glucose
            
            return {
                'glucose': glucose,
                'trend': trend,
                'rate_mgdl_per_min': rate_of_change,
                'heart_rate': int(self.heart_rate),
                'hrv_ms': max(8, self.hrv_ms),
                'spo2': self.spo2,
                'skin_temp_c': self.skin_temp_c,
                'steps_last_hour': self.steps % 3000,
                'calories_burned': int(self.calories),
                'activity_type': activity,
                'insulin_on_board': self.insulin_on_board,
                'timestamp': current_time
            }
        except Exception as e:
            print(f"Simulator error: {e}")
            raise
