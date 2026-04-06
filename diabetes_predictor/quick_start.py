# quick_start.py
import os
import sys

def setup_project():
    """Setup project structure"""
    print("🚀 Setting up Diabetes Predictor...")
    
    # Create directories
    dirs = ['models', 'data', 'logs']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"✓ Created directory: {d}")
    
    print("\n📦 Project structure ready!")
    print("\nNext steps:")
    print("1. Train the model: python src/train_model.py")
    print("2. Start the API: python src/api.py")
    print("3. Open templates/index.html in your browser")

if __name__ == "__main__":
    setup_project()