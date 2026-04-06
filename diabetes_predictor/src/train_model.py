# src/train_model.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from preprocessor import DiabetesDataPreprocessor
from predictor import HypoglycemiaPredictorModel
from sklearn.model_selection import train_test_split

def generate_sample_data(n_days: int = 30, readings_per_day: int = 24):
    """Generate synthetic data for testing"""
    np.random.seed(42)
    
    data = []
    start_date = datetime.now() - timedelta(days=n_days)
    
    base_glucose = 120  # Starting glucose level
    
    for day in range(n_days):
        for hour in range(readings_per_day):
            timestamp = start_date + timedelta(days=day, hours=hour)
            
            # Simulate glucose variations
            time_factor = np.sin(2 * np.pi * hour / 24) * 20  # Daily rhythm
            random_factor = np.random.normal(0, 10)
            
            # Simulate hypoglycemia events (10% chance)
            if np.random.random() < 0.1:
                glucose = np.random.uniform(50, 70)  # Hypoglycemia
            else:
                glucose = base_glucose + time_factor + random_factor
                glucose = np.clip(glucose, 70, 200)
            
            # Simulate medication events
            hours_since_insulin = np.random.uniform(0, 8) if hour in [7, 12, 18] else 999
            hours_since_meal = np.random.uniform(0, 4) if hour in [8, 13, 19] else 999
            last_carbs = np.random.uniform(30, 80) if hours_since_meal < 4 else 0
            
            data.append({
                'timestamp': timestamp,
                'glucose_level': glucose,
                'hours_since_insulin': hours_since_insulin,
                'hours_since_meal': hours_since_meal,
                'last_carbs': last_carbs
            })
    
    return pd.DataFrame(data)

def main():
    print("🏥 Diabetes Predictor - Model Training")
    print("=" * 50)
    
    # Generate sample data
    print("\n📊 Generating sample training data...")
    df = generate_sample_data(n_days=60, readings_per_day=24)
    print(f"Generated {len(df)} data points")
    
    # Initialize preprocessor
    print("\n🔧 Preprocessing data...")
    preprocessor = DiabetesDataPreprocessor(lookback_hours=6)
    df = preprocessor.create_features(df)
    
    # Create sequences
    print("\n📈 Creating sequences...")
    X, y = preprocessor.prepare_sequences(df, sequence_length=12)
    print(f"Sequences shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Hypoglycemia events: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
    
    # Split data
    print("\n✂️  Splitting into train/validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    
    # Build and train model
    print("\n🧠 Building model...")
    model = HypoglycemiaPredictorModel(sequence_length=12, n_features=9)
    model.build_model()
    
    print("\n🎯 Training model...")
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=50,
        batch_size=32
    )
    
    print("\n✅ Training complete!")
    print(f"Best validation loss: {history['best_val_loss']:.4f}")
    print(f"Model saved to: models/best_model.pth")
    
    # Test prediction
    print("\n🔮 Testing prediction on sample data...")
    sample_sequence = X_val[0]
    risk_score = model.predict_risk(sample_sequence)
    actual_label = y_val[0]
    print(f"Risk score: {risk_score:.2%}")
    print(f"Actual label: {'Hypoglycemia' if actual_label == 1 else 'Normal'}")

if __name__ == "__main__":
    main()