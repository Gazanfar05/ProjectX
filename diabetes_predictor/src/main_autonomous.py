"""Main entry point with autonomous simulation"""
from autonomous_monitor import AutonomousMonitor
import time
import json


def main():
    """Run autonomous monitoring"""
    monitor = AutonomousMonitor(user_id="sim_patient_001")

    print("\n🏥 Autonomous Hypoglycemia Monitoring System")
    print("=" * 60)

    try:
        cycle = 0
        while True:
            cycle += 1

            # Get simulated reading and alert
            result = monitor.update()
            reading = result['reading']
            alert = result['alert']

            # Print status
            print(f"\n[{reading.timestamp.strftime('%H:%M:%S')}] Cycle {cycle}")
            print(f"  Glucose: {reading.glucose:.1f} mg/dL")
            print(f"  Activity: {reading.activity.value}")
            print(f"  Trend: {reading.trend.value}")
            print(f"  Risk: {alert['alert_level']}")
            print(f"  Message: {result['message']}")

            # Show dashboard data every 10 cycles
            if cycle % 10 == 0:
                data = monitor.get_dashboard_data()
                print(f"\n  📊 Dashboard Data:")
                print(f"     History: {len(data['glucose_history'])} readings")
                print(f"     Range: {data['min_glucose']:.1f} - {data['max_glucose']:.1f}")
                print(f"     Average: {data['avg_glucose']:.1f}")

            # Wait before next update
            time.sleep(3)

    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped")
        print(f"Total readings: {len(monitor.readings_log)}")


if __name__ == "__main__":
    main()
