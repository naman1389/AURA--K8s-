#!/usr/bin/env python3
"""
Comprehensive Grafana Verification Script
Tests all Grafana connections, datasources, dashboards, and queries
"""

import os
import sys
import requests
import psycopg2
import json
import time
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

# Add scripts directory to path for imports
scripts_dir = Path(__file__).parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

# Configuration
GRAFANA_URL = os.getenv("GRAFANA_URL", "http://localhost:3000")
GRAFANA_USER = os.getenv("GRAFANA_USER", "admin")
GRAFANA_PASSWORD = os.getenv("GRAFANA_PASSWORD", "admin")
try:
    from config_helper import get_database_url
    DATABASE_URL = get_database_url()
except ImportError:
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://aura:aura_password@localhost:5432/aura_metrics")

# ANSI colors
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

def print_warning(text: str):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

def test_grafana_connection() -> bool:
    """Test if Grafana is accessible"""
    print_header("GRAFANA CONNECTION TEST")
    try:
        response = requests.get(f"{GRAFANA_URL}/api/health", timeout=5)
        if response.status_code == 200:
            print_success(f"Grafana is accessible at {GRAFANA_URL}")
            return True
        else:
            print_error(f"Grafana returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print_error(f"Grafana not reachable at {GRAFANA_URL}")
        print_warning("Make sure Grafana is running: docker-compose up -d grafana")
        return False
    except Exception as e:
        print_error(f"Failed to connect to Grafana: {e}")
        return False

def test_grafana_login() -> Optional[str]:
    """Test Grafana login and return session"""
    print_header("GRAFANA AUTHENTICATION TEST")
    try:
        response = requests.post(
            f"{GRAFANA_URL}/api/login",
            json={"user": GRAFANA_USER, "password": GRAFANA_PASSWORD},
            timeout=5
        )
        if response.status_code == 200:
            print_success("Grafana login successful")
            # Extract session cookie
            cookies = response.cookies
            return cookies.get('grafana_session', None)
        else:
            print_error(f"Login failed: {response.status_code}")
            return None
    except Exception as e:
        print_error(f"Login error: {e}")
        return None

def test_datasource(session: Optional[str] = None) -> bool:
    """Test TimescaleDB datasource"""
    print_header("DATASOURCE TEST")
    try:
        headers = {}
        if session:
            headers['Cookie'] = f'grafana_session={session}'
        
        response = requests.get(
            f"{GRAFANA_URL}/api/datasources",
            auth=(GRAFANA_USER, GRAFANA_PASSWORD),
            headers=headers,
            timeout=5
        )
        
        if response.status_code == 200:
            datasources = response.json()
            if not datasources:
                print_warning("No datasources configured")
                return False
            
            print_success(f"Found {len(datasources)} datasource(s)")
            timescaledb_found = False
            
            for ds in datasources:
                print_success(f"  - {ds['name']} ({ds['type']})")
                if ds['name'] == 'TimescaleDB' or 'timescaledb' in ds.get('uid', '').lower():
                    timescaledb_found = True
                    
                    # Test datasource connection
                    test_response = requests.post(
                        f"{GRAFANA_URL}/api/datasources/{ds['id']}/health",
                        auth=(GRAFANA_USER, GRAFANA_PASSWORD),
                        headers=headers,
                        timeout=10
                    )
                    if test_response.status_code == 200:
                        print_success(f"  ✓ Datasource '{ds['name']}' is healthy")
                    else:
                        print_error(f"  ✗ Datasource '{ds['name']}' health check failed")
            
            if not timescaledb_found:
                print_warning("TimescaleDB datasource not found")
                return False
            
            return True
        else:
            print_error(f"Failed to fetch datasources: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Datasource test failed: {e}")
        return False

def test_database_connection() -> bool:
    """Test database connection"""
    print_header("DATABASE CONNECTION TEST")
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        # Test basic query
        cur.execute("SELECT 1")
        result = cur.fetchone()
        
        if result and result[0] == 1:
            print_success("Database connection successful")
            
            # Test TimescaleDB
            cur.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'timescaledb')")
            has_timescale = cur.fetchone()[0]
            if has_timescale:
                print_success("TimescaleDB extension is active")
            else:
                print_warning("TimescaleDB extension not found (using PostgreSQL mode)")
            
            # Test views
            cur.execute("""
                SELECT viewname FROM pg_views 
                WHERE schemaname = 'public' 
                AND viewname IN ('metrics', 'predictions')
            """)
            views = [row[0] for row in cur.fetchall()]
            if 'metrics' in views:
                print_success("View 'metrics' exists")
            if 'predictions' in views:
                print_success("View 'predictions' exists")
            
            cur.close()
            conn.close()
            return True
        else:
            print_error("Database query failed")
            return False
    except Exception as e:
        print_error(f"Database connection failed: {e}")
        return False

def test_dashboard_queries() -> bool:
    """Test queries used by dashboards"""
    print_header("DASHBOARD QUERY TEST")
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        test_queries = [
            ("Health Score Query", """
                SELECT
                  NOW() as time,
                  CASE
                    WHEN COUNT(*) = 0 THEN 100.0
                    ELSE GREATEST(0.0, 100.0 - (COUNT(*) FILTER (WHERE is_anomaly = 1) * 100.0 / NULLIF(COUNT(*), 0)))
                  END as health_score
                FROM ml_predictions
                WHERE timestamp > NOW() - INTERVAL '1 hour'
            """),
            ("CPU/Memory Usage Query", """
                SELECT
                  time_bucket('5 minutes', timestamp) as time,
                  AVG(COALESCE(cpu_utilization, 0)) as cpu_usage,
                  AVG(COALESCE(memory_utilization, 0)) as memory_usage
                FROM pod_metrics
                WHERE timestamp > NOW() - INTERVAL '1 hour'
                GROUP BY time
                ORDER BY time
                LIMIT 10
            """),
            ("Anomaly Count Query", """
                SELECT
                  COUNT(*) as anomaly_count
                FROM ml_predictions
                WHERE timestamp > NOW() - INTERVAL '24 hours'
                  AND is_anomaly = 1
            """),
            ("Model Confidence Query", """
                SELECT
                  time_bucket('1 minute', timestamp) as time,
                  AVG(confidence) as model_confidence
                FROM ml_predictions
                WHERE timestamp > NOW() - INTERVAL '1 hour'
                GROUP BY time
                ORDER BY time
                LIMIT 10
            """),
            ("Anomaly Types Query", """
                SELECT
                  COALESCE(anomaly_type, 'unknown') as anomaly_type,
                  COUNT(*) as count
                FROM ml_predictions
                WHERE timestamp > NOW() - INTERVAL '24 hours'
                  AND is_anomaly = 1
                GROUP BY anomaly_type
                ORDER BY count DESC
                LIMIT 10
            """),
            ("Remediation Success Rate Query", """
                SELECT
                  time_bucket('1 minute', executed_at) as time,
                  CASE
                    WHEN COUNT(*) = 0 THEN NULL
                    ELSE (COUNT(*) FILTER (WHERE success = true) * 100.0 / NULLIF(COUNT(*), 0))
                  END as success_rate
                FROM remediations
                WHERE executed_at > NOW() - INTERVAL '24 hours'
                GROUP BY time
                ORDER BY time
                LIMIT 10
            """),
            ("Recent Anomalies Query", """
                SELECT
                  mp.timestamp,
                  mp.pod_name,
                  mp.namespace,
                  COALESCE(mp.anomaly_type, 'unknown') as anomaly_type,
                  mp.confidence
                FROM ml_predictions mp
                WHERE mp.timestamp > NOW() - INTERVAL '24 hours'
                  AND mp.is_anomaly = 1
                ORDER BY mp.timestamp DESC
                LIMIT 20
            """),
        ]
        
        passed = 0
        failed = 0
        
        for name, query in test_queries:
            try:
                cur.execute(query)
                result = cur.fetchall()
                print_success(f"{name}: OK ({len(result)} rows)")
                passed += 1
            except psycopg2.Error as e:
                print_error(f"{name}: FAILED - {str(e)[:80]}")
                failed += 1
        
        cur.close()
        conn.close()
        
        print(f"\nQuery Test Results: {passed} passed, {failed} failed")
        return failed == 0
        
    except Exception as e:
        print_error(f"Dashboard query test failed: {e}")
        return False

def test_dashboards(session: Optional[str] = None) -> bool:
    """Test dashboard availability"""
    print_header("DASHBOARD AVAILABILITY TEST")
    try:
        headers = {}
        if session:
            headers['Cookie'] = f'grafana_session={session}'
        
        response = requests.get(
            f"{GRAFANA_URL}/api/search?type=dash-db",
            auth=(GRAFANA_USER, GRAFANA_PASSWORD),
            headers=headers,
            timeout=5
        )
        
        if response.status_code == 200:
            dashboards = response.json()
            print_success(f"Found {len(dashboards)} dashboard(s)")
            
            expected_dashboards = [
                "AURA K8s - Main Overview",
                "AURA K8s - AI Predictions & ML Analytics",
                "AURA K8s - Cost Optimization",
                "AURA K8s - Remediation Tracking",
                "AURA K8s - Resource Analysis",
            ]
            
            found_dashboards = [d['title'] for d in dashboards]
            
            for expected in expected_dashboards:
                if any(expected in title for title in found_dashboards):
                    print_success(f"  ✓ Dashboard '{expected}' found")
                else:
                    print_warning(f"  ⚠ Dashboard '{expected}' not found")
            
            return True
        else:
            print_error(f"Failed to fetch dashboards: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Dashboard test failed: {e}")
        return False

def test_real_time_updates() -> bool:
    """Test that data is being updated in real-time"""
    print_header("REAL-TIME DATA UPDATE TEST")
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        # Get current timestamp
        cur.execute("SELECT NOW()")
        current_time = cur.fetchone()[0]
        
        # Check for recent data (last 5 minutes)
        cur.execute("""
            SELECT COUNT(*) 
            FROM pod_metrics 
            WHERE timestamp > NOW() - INTERVAL '5 minutes'
        """)
        recent_metrics = cur.fetchone()[0]
        
        cur.execute("""
            SELECT COUNT(*) 
            FROM ml_predictions 
            WHERE timestamp > NOW() - INTERVAL '5 minutes'
        """)
        recent_predictions = cur.fetchone()[0]
        
        if recent_metrics > 0:
            print_success(f"Recent metrics found: {recent_metrics} in last 5 minutes")
        else:
            print_warning("No recent metrics found - system may not be collecting data")
        
        if recent_predictions > 0:
            print_success(f"Recent predictions found: {recent_predictions} in last 5 minutes")
        else:
            print_warning("No recent predictions found - orchestrator may not be running")
        
        cur.close()
        conn.close()
        
        return recent_metrics > 0 or recent_predictions > 0
        
    except Exception as e:
        print_error(f"Real-time update test failed: {e}")
        return False

def test_all_panels() -> bool:
    """Test all dashboard panel queries"""
    print_header("DASHBOARD PANEL QUERY TEST")
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        # Test queries from all 5 dashboards
        panel_queries = [
            # Main Overview Dashboard
            ("Cluster Health Score", """
                SELECT
                  time_bucket('1 minute', timestamp) as time,
                  CASE
                    WHEN COUNT(*) = 0 THEN 100.0
                    ELSE GREATEST(0.0, 100.0 - (COUNT(*) FILTER (WHERE is_anomaly = 1) * 100.0 / NULLIF(COUNT(*), 0)))
                  END as health_score
                FROM ml_predictions
                WHERE timestamp > NOW() - INTERVAL '1 hour'
                GROUP BY time
                ORDER BY time
            """),
            ("Active Issues Distribution", """
                SELECT
                  COALESCE(anomaly_type, 'unknown') as anomaly_type,
                  COUNT(*) as count
                FROM ml_predictions
                WHERE timestamp > NOW() - INTERVAL '24 hours'
                  AND is_anomaly = 1
                GROUP BY anomaly_type
                ORDER BY count DESC
                LIMIT 10
            """),
            ("Resource Utilization Trends", """
                SELECT
                  time_bucket('5 minutes', timestamp) as time,
                  AVG(COALESCE(cpu_utilization, 0)) as cpu_usage,
                  AVG(COALESCE(memory_utilization, 0)) as memory_usage
                FROM pod_metrics
                WHERE timestamp > NOW() - INTERVAL '1 hour'
                GROUP BY time
                ORDER BY time
            """),
            ("Recent Anomalies Table", """
                SELECT
                  mp.timestamp,
                  mp.pod_name,
                  mp.namespace,
                  COALESCE(mp.anomaly_type, 'unknown') as anomaly_type,
                  mp.confidence
                FROM ml_predictions mp
                WHERE mp.timestamp > NOW() - INTERVAL '24 hours'
                  AND mp.is_anomaly = 1
                ORDER BY mp.timestamp DESC
                LIMIT 20
            """),
            ("Remediation Success Rate", """
                SELECT
                  time_bucket('1 minute', executed_at) as time,
                  CASE
                    WHEN COUNT(*) = 0 THEN NULL
                    ELSE (COUNT(*) FILTER (WHERE success = true) * 100.0 / NULLIF(COUNT(*), 0))
                  END as success_rate
                FROM remediations
                WHERE executed_at > NOW() - INTERVAL '24 hours'
                GROUP BY time
                ORDER BY time
            """),
            ("Anomaly Detection Rate", """
                SELECT
                  time_bucket('10 minutes', timestamp) as time,
                  COUNT(*) FILTER (WHERE is_anomaly = 1) as anomalies_detected
                FROM ml_predictions
                WHERE timestamp > NOW() - INTERVAL '24 hours'
                GROUP BY time
                ORDER BY time
            """),
            
            # AI Predictions Dashboard
            ("Model Confidence", """
                SELECT
                  time_bucket('1 minute', timestamp) as time,
                  AVG(confidence) as model_confidence
                FROM ml_predictions
                WHERE timestamp > NOW() - INTERVAL '1 hour'
                GROUP BY time
                ORDER BY time
            """),
            ("Model Accuracy", """
                SELECT
                  time_bucket('1 hour', timestamp) as time,
                  CASE
                    WHEN COUNT(*) > 0 THEN
                      ROUND(
                        CAST(
                          COUNT(*) FILTER (WHERE is_anomaly = 1 AND confidence > 0.5) * 100.0 / NULLIF(COUNT(*), 0)
                        AS NUMERIC), 3
                      ) / 100.0
                    ELSE 0.0
                  END as accuracy
                FROM ml_predictions
                WHERE timestamp > NOW() - INTERVAL '24 hours'
                GROUP BY time
                ORDER BY time
            """),
            ("Predictions Count", """
                SELECT
                  time_bucket('1 minute', timestamp) as time,
                  COUNT(*) as ml_predictions_count
                FROM ml_predictions
                WHERE timestamp > NOW() - INTERVAL '1 hour'
                GROUP BY time
                ORDER BY time
            """),
            ("Detections by Model", """
                SELECT
                  CASE
                    WHEN model_version IS NULL OR model_version = '' THEN 'Unknown'
                    WHEN LOWER(model_version) LIKE '%ensemble%' THEN 'Ensemble'
                    WHEN LOWER(model_version) LIKE '%xgboost%' THEN 'XGBoost'
                    WHEN LOWER(model_version) LIKE '%random%forest%' THEN 'Random Forest'
                    ELSE COALESCE(model_version, 'Other')
                  END as model,
                  COUNT(*) as detections
                FROM ml_predictions
                WHERE timestamp > NOW() - INTERVAL '24 hours'
                  AND is_anomaly = 1
                GROUP BY model
                ORDER BY detections DESC
            """),
            ("Prediction Confidence Over Time", """
                SELECT
                  time_bucket('5 minutes', timestamp) as time,
                  AVG(confidence) as avg_confidence
                FROM ml_predictions
                WHERE timestamp > NOW() - INTERVAL '24 hours'
                  AND is_anomaly = 1
                GROUP BY time
                ORDER BY time
            """),
            ("Anomaly Types Over Time", """
                SELECT
                  time_bucket('10 minutes', timestamp) as time,
                  anomaly_type,
                  COUNT(*) as count
                FROM ml_predictions
                WHERE timestamp > NOW() - INTERVAL '24 hours'
                  AND is_anomaly = 1
                GROUP BY time, anomaly_type
                ORDER BY time
            """),
            ("Detailed Prediction Log", """
                SELECT
                  mp.timestamp,
                  mp.pod_name,
                  mp.namespace,
                  COALESCE(mp.anomaly_type, 'unknown') as anomaly_type,
                  mp.confidence,
                  COALESCE(mp.model_version, 'ensemble') as model_version
                FROM ml_predictions mp
                WHERE mp.timestamp > NOW() - INTERVAL '24 hours'
                ORDER BY mp.timestamp DESC
                LIMIT 100
            """),
            ("Real-time Anomaly Detection Rate", """
                SELECT
                  time_bucket('5 minutes', timestamp) as time,
                  AVG(CASE WHEN is_anomaly = 1 THEN 1.0 ELSE 0.0 END) as anomaly_rate
                FROM ml_predictions
                WHERE timestamp > NOW() - INTERVAL '24 hours'
                GROUP BY time
                ORDER BY time
            """),
        ]
        
        passed = 0
        failed = 0
        
        for name, query in panel_queries:
            try:
                cur.execute(query)
                result = cur.fetchall()
                print_success(f"{name}: OK ({len(result)} rows)")
                passed += 1
            except psycopg2.Error as e:
                print_error(f"{name}: FAILED - {str(e)[:80]}")
                failed += 1
        
        cur.close()
        conn.close()
        
        print(f"\nPanel Query Test Results: {passed} passed, {failed} failed")
        return failed == 0
        
    except Exception as e:
        print_error(f"Panel query test failed: {e}")
        return False

def main():
    """Run all Grafana verification tests"""
    print(f"\n{Colors.BLUE}╔{'═'*68}╗{Colors.END}")
    print(f"{Colors.BLUE}║{Colors.END}  🚀 AURA K8s - Comprehensive Grafana Verification  {Colors.END}{Colors.BLUE}{'║':>25}{Colors.END}")
    print(f"{Colors.BLUE}║{Colors.END}  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.END}{Colors.BLUE}{'║':>45}{Colors.END}")
    print(f"{Colors.BLUE}╚{'═'*68}╝{Colors.END}")
    
    results = {}
    
    # Run all tests
    results['grafana_connection'] = test_grafana_connection()
    time.sleep(1)
    
    session = test_grafana_login()
    results['grafana_login'] = session is not None
    time.sleep(1)
    
    results['datasource'] = test_datasource(session)
    time.sleep(1)
    
    results['database'] = test_database_connection()
    time.sleep(1)
    
    results['dashboard_queries'] = test_dashboard_queries()
    time.sleep(1)
    
    results['dashboards'] = test_dashboards(session)
    time.sleep(1)
    
    results['real_time'] = test_real_time_updates()
    time.sleep(1)
    
    results['all_panels'] = test_all_panels()
    
    # Summary
    print_header("VERIFICATION SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test, result in results.items():
        status = f"{Colors.GREEN}PASS{Colors.END}" if result else f"{Colors.RED}FAIL{Colors.END}"
        test_name = test.replace('_', ' ').title()
        print(f"  {test_name:.<50} {status}")
    
    print(f"\n{Colors.BLUE}{'─'*70}{Colors.END}")
    
    if passed == total:
        print(f"{Colors.GREEN}✓ All tests passed! ({passed}/{total}){Colors.END}")
        print(f"\n{Colors.GREEN}🎉 Grafana is fully configured and working!{Colors.END}")
        print(f"\n{Colors.BLUE}Access Grafana:{Colors.END} {GRAFANA_URL}")
        print(f"{Colors.BLUE}Login:{Colors.END} {GRAFANA_USER} / {GRAFANA_PASSWORD}")
        print(f"\n{Colors.BLUE}Dashboard Location:{Colors.END} Dashboards → AURA K8s folder")
        return 0
    else:
        print(f"{Colors.RED}✗ Some tests failed ({passed}/{total} passed){Colors.END}")
        print(f"\n{Colors.YELLOW}Fix the issues above and run again{Colors.END}")
        return 1

if __name__ == '__main__':
    sys.exit(main())

