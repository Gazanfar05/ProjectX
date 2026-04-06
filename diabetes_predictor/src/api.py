# src/api.py (updated)
from flask import Blueprint, request, jsonify
from datetime import datetime
from main import DiabetesMonitoringSystem
from data_models import GlucoseReading, Medication, LifestyleEvent
import os

# Create blueprint instead of app
api = Blueprint('api', __name__, url_prefix='/api')

# Initialize the monitoring system
monitoring_system = DiabetesMonitoringSystem()

# Load model if it exists
model_path = 'models/best_model.pth'
if os.path.exists(model_path):
    monitoring_system.initialize_model(model_path)
    print(f"✓ Model loaded from {model_path}")
else:
    monitoring_system.initialize_model()
    print("⚠️  No pre-trained model found. Please train the model first.")

@api.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@api.route('/glucose', methods=['POST'])
def add_glucose_reading():
    """Add a glucose reading"""
    data = request.json
    
    reading = GlucoseReading(
        timestamp=datetime.fromisoformat(data['timestamp']),
        glucose_level=float(data['glucose_level']),
        user_id=data['user_id']
    )
    
    monitoring_system.add_glucose_reading(reading)
    
    # Get prediction after adding reading
    prediction = monitoring_system.predict_hypoglycemia(data['user_id'])
    
    return jsonify({
        'success': True,
        'risk_score': float(prediction.risk_score),
        'alert_level': prediction.alert_level,
        'message': monitoring_system.alert_system.generate_alert_message(prediction)
    })

@api.route('/medication', methods=['POST'])
def add_medication():
    """Add a medication record"""
    data = request.json
    
    medication = Medication(
        timestamp=datetime.fromisoformat(data['timestamp']),
        medication_name=data['medication_name'],
        dosage=float(data['dosage']),
        medication_type=data['medication_type'],
        user_id=data['user_id']
    )
    
    monitoring_system.add_medication(medication)
    
    return jsonify({'success': True, 'message': 'Medication recorded'})

@api.route('/lifestyle', methods=['POST'])
def add_lifestyle_event():
    """Add a lifestyle event"""
    data = request.json
    
    event = LifestyleEvent(
        timestamp=datetime.fromisoformat(data['timestamp']),
        event_type=data['event_type'],
        duration_minutes=data.get('duration_minutes'),
        intensity=data.get('intensity'),
        carbs=data.get('carbs'),
        user_id=data['user_id']
    )
    
    monitoring_system.add_lifestyle_event(event)
    
    return jsonify({'success': True, 'message': 'Lifestyle event recorded'})

@api.route('/status/<user_id>', methods=['GET'])
def get_user_status(user_id):
    """Get user status and prediction"""
    status = monitoring_system.get_user_status(user_id)
    return jsonify(status)

@api.route('/predict/<user_id>', methods=['GET'])
def predict_hypoglycemia(user_id):
    """Get hypoglycemia prediction"""
    prediction = monitoring_system.predict_hypoglycemia(user_id)
    
    return jsonify({
        'user_id': user_id,
        'timestamp': prediction.timestamp.isoformat(),
        'risk_score': float(prediction.risk_score),
        'alert_level': prediction.alert_level,
        'time_to_hypo_minutes': prediction.time_to_hypo_minutes,
        'message': monitoring_system.alert_system.generate_alert_message(prediction)
    })

# Export the blueprint
app = api