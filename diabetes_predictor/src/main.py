# src/main.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
from data_models import GlucoseReading, Medication, LifestyleEvent, PredictionResult
from preprocessor import DiabetesDataPreprocessor
from predictor import HypoglycemiaPredictorModel
from alert_system import AlertSystem

class DiabetesMonitoringSystem:
    def __init__(self):
        self.preprocessor = DiabetesDataPreprocessor(lookback_hours=6)
        self.predictor = HypoglycemiaPredictorModel(sequence_length=12, n_features=9)
        self.alert_system = AlertSystem()
        self.user_data = {}  # Store user data in memory (use DB in production)
        
    def initialize_model(self, model_path: str = None):
        """Initialize or load the prediction model"""
        if model_path:
            self.predictor.load_model(model_path)
        else:
            self.predictor.build_model()
    
    def add_glucose_reading(self, reading: GlucoseReading):
        """Add a new glucose reading"""
        if reading.user_id not in self.user_data:
            self.user_data[reading.user_id] = {
                'glucose': [],
                'medications': [],
                'lifestyle': []
            }
        self.user_data[reading.user_id]['glucose'].append(reading)
    
    def add_medication(self, medication: Medication):
        """Add a medication record"""
        if medication.user_id not in self.user_data:
            self.user_data[medication.user_id] = {
                'glucose': [],
                'medications': [],
                'lifestyle': []
            }
        self.user_data[medication.user_id]['medications'].append(medication)
    
    def add_lifestyle_event(self, event: LifestyleEvent):
        """Add a lifestyle event"""
        if event.user_id not in self.user_data:
            self.user_data[event.user_id] = {
                'glucose': [],
                'medications': [],
                'lifestyle': []
            }
        self.user_data[event.user_id]['lifestyle'].append(event)
    
    def _prepare_user_dataframe(self, user_id: str) -> pd.DataFrame:
        """Prepare user data as a DataFrame"""
        if user_id not in self.user_data:
            return pd.DataFrame()
        
        glucose_data = self.user_data[user_id]['glucose']
        medications = self.user_data[user_id]['medications']
        lifestyle = self.user_data[user_id]['lifestyle']
        
        # Create base DataFrame from glucose readings
        df = pd.DataFrame([
            {
                'timestamp': g.timestamp,
                'glucose_level': g.glucose_level
            }
            for g in glucose_data
        ])
        
        if df.empty:
            return df
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Add medication information
        for med in medications:
            mask = df['timestamp'] >= med.timestamp
            if mask.any():
                idx = df[mask].index[0]
                df.loc[idx:, 'last_medication'] = med.medication_name
                df.loc[idx:, 'hours_since_insulin'] = (
                    df.loc[idx:, 'timestamp'] - med.timestamp
                ).dt.total_seconds() / 3600
        
        # Add lifestyle information
        for event in lifestyle:
            mask = df['timestamp'] >= event.timestamp
            if mask.any() and event.event_type == 'meal':
                idx = df[mask].index[0]
                df.loc[idx:, 'hours_since_meal'] = (
                    df.loc[idx:, 'timestamp'] - event.timestamp
                ).dt.total_seconds() / 3600
                df.loc[idx:, 'last_carbs'] = event.carbs or 0
        
        # Fill NaN values
        df['hours_since_insulin'] = df.get('hours_since_insulin', 0).fillna(999)
        df['hours_since_meal'] = df.get('hours_since_meal', 0).fillna(999)
        df['last_carbs'] = df.get('last_carbs', 0).fillna(0)
        
        return df
    
    def predict_hypoglycemia(self, user_id: str) -> PredictionResult:
        """Predict hypoglycemia risk for a user"""
        df = self._prepare_user_dataframe(user_id)
        
        if df.empty or len(df) < 12:
            return PredictionResult(
                timestamp=datetime.now(),
                risk_score=0.0,
                predicted_glucose=0.0,
                time_to_hypo_minutes=None,
                alert_level='none'
            )
        
        # Create features
        df = self.preprocessor.create_features(df)
        
        # Get the last sequence
        feature_columns = [
            'glucose_level', 'glucose_diff', 'glucose_rate',
            'hour_sin', 'hour_cos', 'is_weekend',
            'hours_since_insulin', 'hours_since_meal', 'last_carbs'
        ]
        
        # Take last 12 readings
        sequence = df[feature_columns].tail(12).values
        
        # Normalize
        sequence_normalized = self.preprocessor.scaler.fit_transform(sequence)
        
        # Predict
        risk_score = self.predictor.predict_risk(sequence_normalized)
        current_glucose = df['glucose_level'].iloc[-1]
        
        # Generate alert
        prediction = self.alert_system.evaluate_risk(risk_score, current_glucose)
        
        return prediction
    
    def get_user_status(self, user_id: str) -> Dict:
        """Get comprehensive user status"""
        df = self._prepare_user_dataframe(user_id)
        
        if df.empty:
            return {'status': 'no_data'}
        
        prediction = self.predict_hypoglycemia(user_id)
        latest_glucose = df['glucose_level'].iloc[-1]
        
        return {
            'user_id': user_id,
            'latest_glucose': latest_glucose,
            'timestamp': df['timestamp'].iloc[-1].isoformat(),
            'risk_score': float(prediction.risk_score),
            'alert_level': prediction.alert_level,
            'alert_message': self.alert_system.generate_alert_message(prediction),
            'readings_count': len(df)
        }