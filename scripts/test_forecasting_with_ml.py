#!/usr/bin/env python3
"""
Test script to verify forecasting service uses existing ML model
"""

import os
import requests
import json
import time
import sys
from pathlib import Path

# Add scripts directory to path for imports
scripts_dir = Path(__file__).parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

try:
    from config_helper import get_service_url
    ML_SERVICE_URL = get_service_url("ML_SERVICE", "8001")
except ImportError:
    ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://localhost:8001")

def test_forecast_endpoint():
    """Test the forecasting endpoint with ML model integration"""
    print("🧪 Testing Forecasting Endpoint with ML Model Integration\n")
    
    # Test case 1: Normal pod (low risk)
    print("Test 1: Normal Pod (Low Risk)")
    print("-" * 50)
    request1 = {
        "pod_name": "test-pod-normal",
        "namespace": "default",
        "metrics": {
            "cpu_utilization": [30, 32, 35, 33, 34, 36, 35, 37, 36, 38],
            "memory_utilization": [40, 42, 45, 43, 44, 46, 45, 47, 46, 48]
        },
        "horizon_seconds": 900,
        "metrics_to_forecast": ["cpu_utilization", "memory_utilization"]
    }
    
    start = time.time()
    try:
        response = requests.post(
            f"{ML_SERVICE_URL}/v1/forecast",
            json=request1,
            timeout=10
        )
        elapsed = (time.time() - start) * 1000
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Time: {elapsed:.2f}ms")
            print(f"   Risk Score: {result.get('risk_score', 0):.1f}")
            print(f"   Severity: {result.get('severity', 'Unknown')}")
            print(f"   Confidence: {result.get('confidence', 0):.2f}")
            print(f"   Anomaly Probabilities: {result.get('anomaly_probabilities', {})}")
            print(f"   Recommended Actions: {result.get('recommended_actions', [])}")
        else:
            print(f"❌ Failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n")
    
    # Test case 2: High-risk pod (approaching threshold)
    print("Test 2: High-Risk Pod (Approaching Threshold)")
    print("-" * 50)
    request2 = {
        "pod_name": "test-pod-high-risk",
        "namespace": "default",
        "metrics": {
            "cpu_utilization": [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
            "memory_utilization": [60, 65, 70, 75, 80, 85, 90, 95, 100, 100]
        },
        "horizon_seconds": 900,
        "metrics_to_forecast": ["cpu_utilization", "memory_utilization"]
    }
    
    start = time.time()
    try:
        response = requests.post(
            f"{ML_SERVICE_URL}/v1/forecast",
            json=request2,
            timeout=10
        )
        elapsed = (time.time() - start) * 1000
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Time: {elapsed:.2f}ms")
            print(f"   Risk Score: {result.get('risk_score', 0):.1f}")
            print(f"   Severity: {result.get('severity', 'Unknown')}")
            print(f"   Confidence: {result.get('confidence', 0):.2f}")
            print(f"   Time to Anomaly: {result.get('time_to_anomaly', 'N/A')} seconds")
            print(f"   Anomaly Probabilities: {result.get('anomaly_probabilities', {})}")
            print(f"   Recommended Actions: {result.get('recommended_actions', [])}")
            
            # Verify ML model was used
            if result.get('confidence', 0) > 0.5:
                print(f"   ✅ ML Model Used: Confidence = {result.get('confidence', 0):.2f}")
            else:
                print(f"   ⚠️  Low confidence - may be using fallback")
        else:
            print(f"❌ Failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n")
    
    # Test case 3: Performance test (multiple requests)
    print("Test 3: Performance Test (10 requests)")
    print("-" * 50)
    times = []
    for i in range(10):
        start = time.time()
        try:
            response = requests.post(
                f"{ML_SERVICE_URL}/v1/forecast",
                json=request1,
                timeout=10
            )
            elapsed = (time.time() - start) * 1000
            if response.status_code == 200:
                times.append(elapsed)
        except:
            pass
    
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        print(f"✅ Performance Results:")
        print(f"   Average: {avg_time:.2f}ms")
        print(f"   Min: {min_time:.2f}ms")
        print(f"   Max: {max_time:.2f}ms")
        print(f"   Requests: {len(times)}/10 successful")
        
        if avg_time < 300:
            print(f"   ✅ Performance is good (<300ms target)")
        else:
            print(f"   ⚠️  Performance is slower than target")
    
    print("\n" + "=" * 50)
    print("Testing Complete!")
    print("=" * 50)

if __name__ == "__main__":
    print("🚀 Forecasting Service ML Model Integration Test\n")
    print(f"ML Service URL: {ML_SERVICE_URL}\n")
    
    # Check if service is running
    try:
        response = requests.get(f"{ML_SERVICE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ ML Service is running\n")
        else:
            print("❌ ML Service is not healthy")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Cannot connect to ML Service: {e}")
        print(f"   Make sure the service is running: python3 aura-cli.py start")
        sys.exit(1)
    
    test_forecast_endpoint()

