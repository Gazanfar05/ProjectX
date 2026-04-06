import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple

class DiabetesDataPreprocessor:
    def __init__(self, lookback_hours: int = 6):
        self.lookback_hours = lookback_hours
        self.scaler = StandardScaler()
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based and rolling features"""
        df = df.copy()
        
        # Time features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Cyclical encoding for hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Glucose trend features
        df['glucose_diff'] = df['glucose_level'].diff()
        df['glucose_rate'] = df['glucose_diff'] / (df['timestamp'].diff().dt.total_seconds() / 60)
        df['glucose_rolling_mean_30min'] = df['glucose_level'].rolling(window=6, min_periods=1).mean()
        df['glucose_rolling_std_30min'] = df['glucose_level'].rolling(window=6, min_periods=1).std()
        
        # Time since last medication (in hours)
        df['hours_since_insulin'] = df['hours_since_insulin'].fillna(999.0)
        df['hours_since_meal'] = df['hours_since_meal'].fillna(999.0)
        df['last_carbs'] = df['last_carbs'].fillna(0.0)
        
        # Handle NaN and inf values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        return df
    
    def prepare_sequences(self, df: pd.DataFrame, sequence_length: int = 12) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM model"""
        feature_columns = [
            'glucose_level', 'glucose_diff', 'glucose_rate',
            'hour_sin', 'hour_cos', 'is_weekend',
            'hours_since_insulin', 'hours_since_meal', 'last_carbs'
        ]
        
        # Ensure all features exist and handle NaN
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
            df[col] = df[col].fillna(0)
            df[col] = df[col].replace([np.inf, -np.inf], 0)
        
        # Get feature data
        feature_data = df[feature_columns].values
        
        # Normalize features
        scaled_data = self.scaler.fit_transform(feature_data)
        
        # Ensure no NaN or inf after scaling
        scaled_data = np.nan_to_num(scaled_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            # Target: 1 if hypoglycemia in next 30 mins, 0 otherwise
            if 'label' in df.columns:
                y.append(df.iloc[i]['label'])
            else:
                future_glucose = df.iloc[i]['glucose_level']
                y.append(1 if future_glucose < 70 else 0)
        
        return np.array(X), np.array(y)
