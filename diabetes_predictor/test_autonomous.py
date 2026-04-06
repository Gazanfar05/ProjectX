#!/usr/bin/env python3
"""Test autonomous system"""
from src.autonomous_monitor import AutonomousMonitor
import time


def test_basic():
    """Test basic functionality"""
    print("🧪 Testing Autonomous Monitor...")
    print("=" * 80)
    
    monitor = AutonomousMonitor()
    
    for i in range(20):
        result = monitor.update()
        reading = result['reading']
        
        print(f"{i+1:2d}. Glucose: {reading.glucose:6.1f} | "
              f"Activity: {reading.activity.value:8} | "
              f"Trend: {reading.trend.value:7} | "
              f"Risk: {result['alert']['alert_level']:8} | "
              f"Conf: {reading.activity_confidence*100:5.1f}%")
        
        time.sleep(0.5)
    
    print("\n" + "=" * 80)
    print("✅ Test complete!")
    
    # Show summary
    data = monitor.get_dashboard_data()
    print(f"\n📊 Summary:")
    print(f"   Total readings: {data['readings_count']}")
    print(f"   Glucose range: {data['min_glucose']:.1f} - {data['max_glucose']:.1f}")
    print(f"   Average glucose: {data['avg_glucose']:.1f}")


if __name__ == "__main__":
    test_basic()
