import os
import sys
import pandas as pd
import numpy as np
from xml_parser import parse_multiple_xml_files
from data_loader import DiabetesDataLoader
from preprocessor import DiabetesDataPreprocessor
from predictor import HypoglycemiaPredictorModel
from sklearn.model_selection import train_test_split

def main(data_path):
    print("🏥 Diabetes Predictor - Training with Kaggle Data")
    print("=" * 60)
    
    if not os.path.exists(data_path):
        print(f"❌ Path not found: {data_path}")
        return
    
    print(f"\n📂 Loading XML files from: {data_path}")
    df_glucose, df_meds, df_lifestyle = parse_multiple_xml_files(data_path)
    
    if df_glucose.empty:
        print("❌ No glucose data loaded")
        return
    
    print(f"\n�� Data summary:")
    print(f"   Glucose readings: {len(df_glucose)}")
    print(f"   Medication events: {len(df_meds)}")
    print(f"   Lifestyle events: {len(df_lifestyle)}")
    
    # Merge data
    loader = DiabetesDataLoader()
    df = loader.merge_data(df_glucose, df_meds, df_lifestyle)
    
    # Create labels
    df = loader.create_labels(df, lookahead_minutes=30, threshold=70)
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/merged_data.csv', index=False)
    print(f"\n💾 Saved processed data to: data/processed/merged_data.csv")
    
    # Preprocess
    print("\n🔧 Preprocessing data...")
    preprocessor = DiabetesDataPreprocessor(lookback_hours=6)
    df = preprocessor.create_features(df)
    
    # Create sequences
    print("\n📈 Creating sequences...")
    X, y = preprocessor.prepare_sequences(df, sequence_length=12)
    print(f"   Sequences shape: {X.shape}")
    print(f"   Labels shape: {y.shape}")
    
    if len(X) == 0:
        print("❌ Not enough data to create sequences")
        return
    
    hypo_ratio = y.sum() / len(y)
    print(f"   Hypoglycemia ratio: {hypo_ratio:.2%}")
    
    # Split data
    print("\n✂️  Splitting into train/validation sets...")
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    print(f"   Training set: {len(X_train)} samples")
    print(f"   Validation set: {len(X_val)} samples")
    
    # Build and train model
    print("\n🧠 Building model...")
    model = HypoglycemiaPredictorModel(sequence_length=12, n_features=9)
    model.build_model()
    
    print("\n🎯 Training model...")
    print("   (This may take several minutes...)\n")
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=100,
        batch_size=32
    )
    
    print("\n✅ Training complete!")
    print(f"   Best validation loss: {history['best_val_loss']:.4f}")
    print(f"   Model saved to: models/best_model.pth")
    
    # Test predictions
    print("\n🔮 Testing predictions on validation set...")
    for i in range(min(5, len(X_val))):
        sample_sequence = X_val[i]
        risk_score = model.predict_risk(sample_sequence)
        actual_label = y_val[i]
        actual_text = '🔴 Hypo' if actual_label == 1 else '🟢 Normal'
        print(f"   Sample {i+1}: Risk={risk_score:.2%}, Actual={actual_text}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/train_with_xml.py <path_to_xml_directory>")
        print("\nExample:")
        print("  python src/train_with_xml.py data/raw/")
        sys.exit(1)
    
    main(sys.argv[1])
