"""SQLite database with thread-safe connection handling"""
import sqlite3
import threading
from datetime import datetime
from typing import List, Dict, Any

class Database:
    def __init__(self, db_path: str = "diabetes_monitor.db"):
        self.db_path = db_path
        self.local = threading.local()
        self.init_db()
    
    def get_conn(self):
        """Get thread-local database connection"""
        if not hasattr(self.local, 'conn') or self.local.conn is None:
            self.local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.local.conn.row_factory = sqlite3.Row
        return self.local.conn
    
    def init_db(self):
        """Initialize database with schema"""
        conn = self.get_conn()
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS glucose_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                value_mgdl REAL NOT NULL,
                trend TEXT,
                rate_mgdl_per_min REAL,
                source TEXT DEFAULT 'simulated',
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS biometrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                heart_rate INTEGER,
                hrv_ms REAL,
                spo2 REAL,
                skin_temp_c REAL,
                skin_temp_delta REAL,
                steps_last_hour INTEGER,
                calories_burned INTEGER,
                activity_type TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS risk_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                score INTEGER,
                level TEXT,
                predicted_score_30min INTEGER,
                confidence INTEGER,
                factors_json TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        print("✅ Database initialized")
    
    def add_glucose_reading(self, user_id: str, value_mgdl: float, trend: str, rate: float):
        conn = self.get_conn()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO glucose_readings (user_id, value_mgdl, trend, rate_mgdl_per_min)
            VALUES (?, ?, ?, ?)
        ''', (user_id, value_mgdl, trend, rate))
        conn.commit()
    
    def add_biometric(self, user_id: str, data: Dict[str, Any]):
        conn = self.get_conn()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO biometrics 
            (user_id, heart_rate, hrv_ms, spo2, skin_temp_c, skin_temp_delta, steps_last_hour, calories_burned, activity_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, data.get('heart_rate'), data.get('hrv_ms'), data.get('spo2'), data.get('skin_temp_c'), data.get('skin_temp_delta'), data.get('steps_last_hour'), data.get('calories_burned'), data.get('activity_type')))
        conn.commit()
    
    def add_risk_score(self, user_id: str, score: int, level: str, predicted_score: int, confidence: int, factors_json: str):
        conn = self.get_conn()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO risk_scores (user_id, score, level, predicted_score_30min, confidence, factors_json)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, score, level, predicted_score, confidence, factors_json))
        conn.commit()
    
    def get_latest_glucose(self, user_id: str, limit: int = 20) -> List[Dict]:
        conn = self.get_conn()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM glucose_readings WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?', (user_id, limit))
        rows = cursor.fetchall()
        return [dict(row) for row in rows] if rows else []
    
    def get_latest_biometric(self, user_id: str) -> Dict:
        conn = self.get_conn()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM biometrics WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1', (user_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def get_dashboard_data(self, user_id: str) -> Dict:
        glucose_readings = self.get_latest_glucose(user_id, 20)
        latest_biometric = self.get_latest_biometric(user_id)
        
        conn = self.get_conn()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM risk_scores WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1', (user_id,))
        row = cursor.fetchone()
        latest_risk = dict(row) if row else None
        
        return {'glucose_readings': glucose_readings, 'latest_biometric': latest_biometric, 'latest_risk': latest_risk}
    
    def close(self):
        if hasattr(self.local, 'conn') and self.local.conn:
            self.local.conn.close()
