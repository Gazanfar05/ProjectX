import pandas as pd
import numpy as np
from datetime import timedelta

class DiabetesDataLoader:
    def __init__(self):
        pass
    
    def merge_data(self, df_glucose, df_meds, df_lifestyle):
        """Merge glucose, medication, and lifestyle data"""
        print("\n🔗 Merging datasets...")
        
        if df_glucose.empty:
            print("❌ No glucose data available")
            return pd.DataFrame()
        
        df = df_glucose.copy()
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        df['hours_since_insulin'] = 999.0
        df['hours_since_meal'] = 999.0
        df['last_carbs'] = 0.0
        
        # Merge medication data
        if not df_meds.empty:
            df_meds = df_meds.sort_values('timestamp')
            
            for idx, row in df.iterrows():
                recent_insulin = df_meds[
                    (df_meds['timestamp'] <= row['timestamp']) & 
                    (df_meds['medication_type'].str.contains('insulin', case=False))
                ]
                
                if not recent_insulin.empty:
                    last_insulin_time = recent_insulin.iloc[-1]['timestamp']
                    hours_diff = (row['timestamp'] - last_insulin_time).total_seconds() / 3600
                    df.at[idx, 'hours_since_insulin'] = min(hours_diff, 24)
        
        # Merge lifestyle data
        if not df_lifestyle.empty:
            df_lifestyle = df_lifestyle.sort_values('timestamp')
            
            for idx, row in df.iterrows():
                recent_meals = df_lifestyle[
                    (df_lifestyle['timestamp'] <= row['timestamp']) & 
                    (df_lifestyle['event_type'] == 'meal')
                ]
                
                if not recent_meals.empty:
                    last_meal = recent_meals.iloc[-1]
                    last_meal_time = last_meal['timestamp']
                    hours_diff = (row['timestamp'] - last_meal_time).total_seconds() / 3600
                    df.at[idx, 'hours_since_meal'] = min(hours_diff, 24)
                    df.at[idx, 'last_carbs'] = last_meal.get('carbs', 0)
        
        print(f"   ✓ Merged {len(df)} records")
        return df
    
    def create_labels(self, df, lookahead_minutes=30, threshold=70):
        """Create labels for hypoglycemia prediction"""
        print(f"\n🏷️  Creating labels (lookahead: {lookahead_minutes} min, threshold: {threshold} mg/dL)")
        
        labels = []
        
        for idx in range(len(df) - 1):
            current_time = df.iloc[idx]['timestamp']
            future_time = current_time + timedelta(minutes=lookahead_minutes)
            
            future_readings = df[
                (df['timestamp'] > current_time) & 
                (df['timestamp'] <= future_time)
            ]['glucose_level']
            
            if not future_readings.empty and future_readings.min() < threshold:
                labels.append(1)
            else:
                labels.append(0)
        
        labels.append(0)
        df['label'] = labels
        
        hypo_count = sum(labels)
        print(f"   ✓ Hypoglycemia events: {hypo_count} ({hypo_count/len(labels)*100:.1f}%)")
        
        return df
