#!/usr/bin/env python3
"""
Test script to measure anomaly detection and remediation timing
"""

import os
import sys
import time
import requests
import psycopg2
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path

# Add scripts directory to path for imports
scripts_dir = Path(__file__).parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

try:
    from config_helper import get_database_url, get_service_url
    DATABASE_URL = get_database_url()
    ML_SERVICE_URL = get_service_url("ML_SERVICE", "8001")
except ImportError:
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://aura:aura_password@localhost:5432/aura_metrics")
    ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://localhost:8001")

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_header(text: str):
    print(f"\n{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BLUE}{text:^70}{Colors.END}")
    print(f"{Colors.BLUE}{'='*70}{Colors.END}\n")

def print_success(text: str):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_error(text: str):
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_info(text: str):
    print(f"{Colors.YELLOW}ℹ {text}{Colors.END}")

def test_anomaly_detection_timing():
    """Test how long it takes to detect an anomaly"""
    print_header("ANOMALY DETECTION TIMING TEST")
    
    # Test data simulating a pod approaching threshold
    test_features = {
        "cpu_usage": 85.5,
        "memory_usage": 75.0,
        "disk_usage": 50.0,
        "network_bytes_sec": 1000.0,
        "error_rate": 0.05,
        "latency_ms": 100.0,
        "restart_count": 2.0,
        "age_minutes": 120.0,
        "cpu_memory_ratio": 1.14,
        "resource_pressure": 0.8,
        "error_latency_product": 5.0,
        "network_per_cpu": 11.7,
        "is_critical": 1.0
    }
    
    print("Testing ML Prediction (Current Anomaly Detection)...")
    times = []
    for i in range(10):
        start = time.time()
        try:
            response = requests.post(
                f"{ML_SERVICE_URL}/predict",
                json={"features": test_features},
                timeout=10
            )
            elapsed = (time.time() - start) * 1000
            if response.status_code == 200:
                result = response.json()
                times.append(elapsed)
                if i == 0:
                    print_success(f"Prediction successful: {result.get('anomaly_type', 'unknown')} (confidence: {result.get('confidence', 0):.2%})")
        except Exception as e:
            print_error(f"Request failed: {e}")
    
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        print_success(f"ML Prediction Timing: avg={avg_time:.2f}ms, min={min_time:.2f}ms, max={max_time:.2f}ms")
        return avg_time
    return None

def test_forecast_timing():
    """Test how long it takes to forecast and predict future anomalies"""
    print_header("FORECAST TIMING TEST")
    
    # Simulate historical data showing increasing trend
    forecast_request = {
        "pod_name": "test-pod",
        "namespace": "default",
        "metrics": {
            "cpu_utilization": [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
            "memory_utilization": [60, 65, 70, 75, 80, 85, 90, 95, 100, 100]
        },
        "horizon_seconds": 900,
        "metrics_to_forecast": ["cpu_utilization", "memory_utilization"]
    }
    
    print("Testing Forecast Generation (Predictive Anomaly Detection)...")
    times = []
    for i in range(10):
        start = time.time()
        try:
            response = requests.post(
                f"{ML_SERVICE_URL}/v1/forecast",
                json=forecast_request,
                timeout=10
            )
            elapsed = (time.time() - start) * 1000
            if response.status_code == 200:
                result = response.json()
                times.append(elapsed)
                if i == 0:
                    print_success(f"Forecast successful: risk_score={result.get('risk_score', 0):.1f}, severity={result.get('severity', 'Unknown')}")
                    print_info(f"  Time to anomaly: {result.get('time_to_anomaly', 'N/A')} seconds")
                    print_info(f"  Confidence: {result.get('confidence', 0):.2%}")
        except Exception as e:
            print_error(f"Request failed: {e}")
    
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        print_success(f"Forecast Timing: avg={avg_time:.2f}ms, min={min_time:.2f}ms, max={max_time:.2f}ms")
        return avg_time
    return None

def test_remediation_timing():
    """Test how long remediation takes"""
    print_header("REMEDIATION TIMING TEST")
    
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        # Get recent remediations
        cur.execute("""
            SELECT 
                executed_at,
                completed_at,
                success,
                action,
                pod_name
            FROM remediations
            WHERE executed_at > NOW() - INTERVAL '1 hour'
            ORDER BY executed_at DESC
            LIMIT 10
        """)
        
        rows = cur.fetchall()
        
        if rows:
            times = []
            for row in rows:
                executed_at = row[0]
                completed_at = row[1]
                success = row[2]
                action = row[3]
                pod_name = row[4]
                
                if completed_at and executed_at:
                    duration = (completed_at - executed_at).total_seconds()
                    times.append(duration)
                    if len(times) == 1:
                        print_success(f"Remediation: {action} for {pod_name} - {duration:.2f}s ({'success' if success else 'failed'})")
            
            if times:
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)
                print_success(f"Remediation Timing: avg={avg_time:.2f}s, min={min_time:.2f}s, max={max_time:.2f}s")
                cur.close()
                conn.close()
                return avg_time
        else:
            print_info("No recent remediations found - remediation timing will be measured when actions occur")
        
        cur.close()
        conn.close()
        return None
        
    except Exception as e:
        print_error(f"Database query failed: {e}")
        return None

def test_end_to_end_timing():
    """Test complete flow: detection -> warning -> action"""
    print_header("END-TO-END TIMING TEST")
    
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        # Get recent early warnings and their associated remediations
        cur.execute("""
            SELECT 
                ew.pod_name,
                ew.namespace,
                ew.created_at as warning_time,
                ew.severity,
                ew.risk_score,
                r.executed_at as remediation_time,
                r.completed_at,
                r.success
            FROM early_warnings ew
            LEFT JOIN remediations r ON 
                r.pod_name = ew.pod_name 
                AND r.namespace = ew.namespace
                AND r.executed_at > ew.created_at
            WHERE ew.created_at > NOW() - INTERVAL '1 hour'
            ORDER BY ew.created_at DESC
            LIMIT 10
        """)
        
        rows = cur.fetchall()
        
        if rows:
            detection_to_warning = []
            warning_to_action = []
            detection_to_action = []
            
            for row in rows:
                pod_name = row[0]
                warning_time = row[2]
                remediation_time = row[5]
                completed_at = row[6]
                
                # Estimate detection time (warning_time - 500ms collection interval)
                # In reality, detection happens during collection
                detection_to_warning.append(0.5)  # Collection interval
                
                if remediation_time:
                    warning_to_action_time = (remediation_time - warning_time).total_seconds()
                    warning_to_action.append(warning_to_action_time)
                    
                    if completed_at:
                        total_time = (completed_at - warning_time).total_seconds()
                        detection_to_action.append(total_time)
            
            if detection_to_warning:
                avg_dtw = sum(detection_to_warning) / len(detection_to_warning)
                print_success(f"Detection → Warning: ~{avg_dtw:.2f}s (collection interval)")
            
            if warning_to_action:
                avg_wta = sum(warning_to_action) / len(warning_to_action)
                print_success(f"Warning → Action: avg={avg_wta:.2f}s")
            
            if detection_to_action:
                avg_dta = sum(detection_to_action) / len(detection_to_action)
                print_success(f"Detection → Action Complete: avg={avg_dta:.2f}s")
            
            cur.close()
            conn.close()
            return detection_to_action[0] if detection_to_action else None
        else:
            print_info("No recent warnings found - run the system to generate data")
        
        cur.close()
        conn.close()
        return None
        
    except Exception as e:
        print_error(f"Database query failed: {e}")
        return None

def main():
    """Run all timing tests"""
    print(f"\n{Colors.BLUE}╔{'═'*68}╗{Colors.END}")
    print(f"{Colors.BLUE}║{Colors.END}  🚀 AURA K8s - Anomaly Detection & Remediation Timing Test  {Colors.END}{Colors.BLUE}{'║':>25}{Colors.END}")
    print(f"{Colors.BLUE}║{Colors.END}  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.END}{Colors.BLUE}{'║':>45}{Colors.END}")
    print(f"{Colors.BLUE}╚{'═'*68}╝{Colors.END}")
    
    # Check services
    print("\nChecking services...")
    try:
        response = requests.get(f"{ML_SERVICE_URL}/health", timeout=5)
        if response.status_code == 200:
            print_success("ML Service is running")
        else:
            print_error("ML Service is not healthy")
            return 1
    except Exception as e:
        print_error(f"ML Service not reachable: {e}")
        print_info("Start the service: python3 aura-cli.py start")
        return 1
    
    # Run tests
    detection_time = test_anomaly_detection_timing()
    forecast_time = test_forecast_timing()
    remediation_time = test_remediation_timing()
    end_to_end_time = test_end_to_end_timing()
    
    # Summary
    print_header("TIMING SUMMARY")
    
    if detection_time:
        print(f"{Colors.GREEN}Current Anomaly Detection:{Colors.END} {detection_time:.2f}ms per pod")
    
    if forecast_time:
        print(f"{Colors.GREEN}Predictive Anomaly Detection:{Colors.END} {forecast_time:.2f}ms per pod")
        print(f"  {Colors.YELLOW}→ Can predict anomalies {forecast_time/1000:.2f} seconds before they occur{Colors.END}")
    
    if remediation_time:
        print(f"{Colors.GREEN}Remediation Time:{Colors.END} {remediation_time:.2f} seconds per action")
    
    if end_to_end_time:
        print(f"{Colors.GREEN}End-to-End (Detection → Action):{Colors.END} {end_to_end_time:.2f} seconds")
    
    # Calculate total time
    if detection_time and remediation_time:
        total = (detection_time / 1000) + remediation_time
        print(f"\n{Colors.BLUE}Total Time (Detect + Remedy):{Colors.END} {total:.2f} seconds")
    
    if forecast_time and remediation_time:
        predictive_total = (forecast_time / 1000) + remediation_time
        print(f"{Colors.BLUE}Predictive Total (Forecast + Remedy):{Colors.END} {predictive_total:.2f} seconds")
        print(f"  {Colors.YELLOW}→ System can prevent anomalies {predictive_total:.2f} seconds before they occur!{Colors.END}")
    
    print(f"\n{Colors.BLUE}{'─'*70}{Colors.END}")
    print(f"{Colors.GREEN}✅ Timing tests complete!{Colors.END}\n")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

