"""Flask API with error handling"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import json
import traceback
from src.database import Database
from src.data_simulator import DataSimulator
from src.risk_engine import RiskEngine

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

db = Database()
risk_engine = RiskEngine()
user_monitor = {}

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/api/simulate/tick', methods=['POST', 'OPTIONS'])
def simulate_tick():
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        user_id = 'sim_user_001'
        if request.json and 'user_id' in request.json:
            user_id = request.json['user_id']
        
        if user_id not in user_monitor:
            user_monitor[user_id] = {'simulator': DataSimulator(user_id), 'glucose_history': []}
        
        monitor = user_monitor[user_id]
        sim_data = monitor['simulator'].tick()
        
        monitor['glucose_history'].append(sim_data['glucose'])
        if len(monitor['glucose_history']) > 100:
            monitor['glucose_history'].pop(0)
        
        hour = datetime.now().hour
        risk_score, risk_level, factors = risk_engine.calculate_risk(sim_data, monitor['glucose_history'], hour)
        
        db.add_glucose_reading(user_id, sim_data['glucose'], sim_data['trend'], sim_data['rate_mgdl_per_min'])
        db.add_biometric(user_id, {
            'heart_rate': sim_data['heart_rate'], 
            'hrv_ms': sim_data['hrv_ms'], 
            'spo2': sim_data['spo2'], 
            'skin_temp_c': sim_data['skin_temp_c'], 
            'skin_temp_delta': sim_data['skin_temp_c'] - 36.5, 
            'steps_last_hour': sim_data['steps_last_hour'], 
            'calories_burned': sim_data['calories_burned'], 
            'activity_type': sim_data['activity_type']
        })
        db.add_risk_score(user_id, risk_score, risk_level, risk_score, min(100, risk_score + 15), json.dumps(factors))
        
        return jsonify({
            'success': True, 
            'glucose': sim_data['glucose'], 
            'trend': sim_data['trend'], 
            'heart_rate': sim_data['heart_rate'], 
            'hrv_ms': sim_data['hrv_ms'], 
            'spo2': sim_data['spo2'], 
            'risk_score': risk_score, 
            'risk_level': risk_level, 
            'factors': factors, 
            'timestamp': sim_data['timestamp'].isoformat()
        })
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/dashboard/<user_id>', methods=['GET'])
def get_dashboard(user_id):
    try:
        data = db.get_dashboard_data(user_id)
        if not data['glucose_readings']:
            return jsonify({'error': 'No data for user'}), 404
        
        glucose_values = [r['value_mgdl'] for r in data['glucose_readings']]
        
        return jsonify({
            'glucose_history': glucose_values[-20:], 
            'latest_glucose': glucose_values[0], 
            'min_glucose': min(glucose_values), 
            'max_glucose': max(glucose_values), 
            'avg_glucose': sum(glucose_values) / len(glucose_values), 
            'latest_biometric': data['latest_biometric'], 
            'latest_risk': data['latest_risk'], 
            'readings_count': len(data['glucose_readings'])
        })
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8000, host='0.0.0.0')
