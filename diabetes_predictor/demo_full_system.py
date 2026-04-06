#!/usr/bin/env python3
import time
import requests
import json

def run_demo():
    print("\n" + "="*80)
    print("🏥 GlucoSense AI - Full System Demo (50 Readings)")
    print("="*80 + "\n")
    
    user_id = "demo_patient_001"
    critical_alerts = []
    high_alerts = []
    
    for cycle in range(50):
        try:
            response = requests.post('http://localhost:8000/api/simulate/tick', json={'user_id': user_id}, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                glucose = data['glucose']
                risk = data['risk_score']
                level = data['risk_level']
                
                if level == 'critical':
                    emoji = "🔴"
                    critical_alerts.append(f"[{cycle+1}] Glucose: {glucose:.1f}, Risk: {risk}%")
                    print(f"[{cycle+1:2d}] {emoji} Glucose: {glucose:6.1f} | Risk: {risk:3d}% ({level:9s}) | HR: {data['heart_rate']:3d} | HRV: {data['hrv_ms']:5.1f}ms ⚠️ CRITICAL")
                elif level == 'high':
                    emoji = "🟡"
                    high_alerts.append(f"[{cycle+1}] Glucose: {glucose:.1f}, Risk: {risk}%")
                    print(f"[{cycle+1:2d}] {emoji} Glucose: {glucose:6.1f} | Risk: {risk:3d}% ({level:9s}) | HR: {data['heart_rate']:3d} | HRV: {data['hrv_ms']:5.1f}ms")
                elif level == 'elevated':
                    emoji = "🟠"
                    print(f"[{cycle+1:2d}] {emoji} Glucose: {glucose:6.1f} | Risk: {risk:3d}% ({level:9s}) | HR: {data['heart_rate']:3d} | HRV: {data['hrv_ms']:5.1f}ms")
                else:
                    emoji = "🟢"
                    print(f"[{cycle+1:2d}] {emoji} Glucose: {glucose:6.1f} | Risk: {risk:3d}% ({level:9s}) | HR: {data['heart_rate']:3d} | HRV: {data['hrv_ms']:5.1f}ms")
            else:
                print(f"Error: {response.status_code}")
                break
        except Exception as e:
            print(f"Connection error: {e}")
            break
        
        time.sleep(0.15)
    
    print("\n" + "="*80)
    print("📊 Dashboard Summary")
    print("="*80 + "\n")
    
    try:
        response = requests.get(f'http://localhost:8000/api/dashboard/{user_id}', timeout=5)
        if response.status_code == 200:
            dashboard = response.json()
            print(f"Total Readings: {dashboard['readings_count']}")
            print(f"Glucose Range: {dashboard['min_glucose']:.1f} - {dashboard['max_glucose']:.1f} mg/dL")
            print(f"Average Glucose: {dashboard['avg_glucose']:.1f} mg/dL")
            
            if dashboard['latest_risk']:
                risk_data = dashboard['latest_risk']
                print(f"\n🎯 Latest Risk Score: {risk_data['score']}/100")
                print(f"Risk Level: {risk_data['level'].upper()}")
                
                if risk_data['factors_json']:
                    factors = json.loads(risk_data['factors_json'])
                    print(f"\n📊 Factor Breakdown:")
                    total_contribution = 0
                    for factor_name, factor_data in factors.items():
                        contribution = factor_data['contribution']
                        total_contribution += contribution
                        score = factor_data['score']
                        weight_pct = factor_data['weight'] * 100
                        print(f"  {factor_name.capitalize():15s}: {contribution:6.1f}% (score: {score:3d}/100, weight: {weight_pct:3.0f}%)")
                    print(f"  {'─'*50}")
                    print(f"  {'TOTAL':15s}: {total_contribution:6.1f}%")
    except Exception as e:
        print(f"Error: {e}")
    
    # Show alerts summary
    print("\n" + "="*80)
    print("🚨 ALERT SUMMARY")
    print("="*80)
    print(f"\n🔴 CRITICAL Episodes: {len(critical_alerts)}")
    if critical_alerts:
        for alert in critical_alerts:
            print(f"   {alert}")
    
    print(f"\n🟡 HIGH Risk Episodes: {len(high_alerts)}")
    if high_alerts:
        for alert in high_alerts[:5]:  # Show first 5
            print(f"   {alert}")
        if len(high_alerts) > 5:
            print(f"   ... and {len(high_alerts) - 5} more")
    
    if not critical_alerts and not high_alerts:
        print("\n✅ No critical or high-risk episodes detected (stable day)")
    else:
        print(f"\n⚠️ Total risk events: {len(critical_alerts) + len(high_alerts)}")
    
    print("\n✅ Demo complete!")

if __name__ == "__main__":
    run_demo()
