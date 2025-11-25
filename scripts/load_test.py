#!/usr/bin/env python3
"""
Load Testing Script for AURA K8s
Tests system performance under various loads
"""

import os
import sys
import time
import asyncio
import httpx
import psycopg2
import psycopg2.pool
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict
import statistics
from datetime import datetime
from pathlib import Path

# Add scripts directory to path for imports
scripts_dir = Path(__file__).parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

# Configuration
try:
    from config_helper import get_database_url, get_service_url
    DATABASE_URL = get_database_url()
    ML_SERVICE_URL = get_service_url("ML_SERVICE", "8001")
except ImportError:
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://aura:aura_password@localhost:5432/aura_metrics")
    ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://localhost:8001")
COLLECTOR_URL = os.getenv("COLLECTOR_URL", "http://localhost:9090")

class LoadTester:
    """Load testing for AURA K8s components"""
    
    def __init__(self):
        self.results = {
            'collection': [],
            'prediction': [],
            'forecast': [],
            'database': [],
        }
    
    async def test_collection_performance(self, num_pods: int = 100):
        """Test metrics collection performance"""
        print(f"\n📊 Testing collection performance with {num_pods} pods...")
        
        # Simulate collection by querying database
        start = time.time()
        try:
            conn = psycopg2.connect(DATABASE_URL)
            cur = conn.cursor()
            
            # Query recent metrics (simulating collection)
            cur.execute("""
                SELECT COUNT(*) 
                FROM pod_metrics 
                WHERE timestamp > NOW() - INTERVAL '1 minute'
            """)
            count = cur.fetchone()[0]
            
            elapsed = time.time() - start
            self.results['collection'].append(elapsed)
            print(f"  ✓ Collection query: {elapsed:.3f}s ({count} metrics)")
            
            cur.close()
            conn.close()
        except Exception as e:
            print(f"  ✗ Collection test failed: {e}")
    
    async def test_prediction_performance(self, num_requests: int = 100):
        """Test ML prediction performance"""
        print(f"\n🤖 Testing prediction performance with {num_requests} requests...")
        
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
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            tasks = []
            for i in range(num_requests):
                task = self._make_prediction_request(client, test_features)
                tasks.append(task)
            
            start = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start
            
            successful = sum(1 for r in results if not isinstance(r, Exception))
            avg_time = total_time / num_requests if num_requests > 0 else 0
            
            self.results['prediction'].append(avg_time)
            print(f"  ✓ Predictions: {successful}/{num_requests} successful, avg: {avg_time:.3f}s")
    
    async def _make_prediction_request(self, client: httpx.AsyncClient, features: Dict):
        """Make a single prediction request"""
        start = time.time()
        try:
            response = await client.post(
                f"{ML_SERVICE_URL}/predict",
                json={"features": features}
            )
            if response.status_code == 200:
                response.json()
                return time.time() - start
            else:
                raise Exception(f"HTTP {response.status_code}")
        except Exception as e:
            raise e
    
    async def test_forecast_performance(self, num_requests: int = 50):
        """Test forecasting performance"""
        print(f"\n🔮 Testing forecast performance with {num_requests} requests...")
        
        # Generate test historical data
        historical_cpu = [50.0 + i * 0.5 for i in range(100)]
        historical_memory = [60.0 + i * 0.3 for i in range(100)]
        
        forecast_request = {
            "pod_name": "test-pod",
            "namespace": "default",
            "metrics": {
                "cpu_utilization": historical_cpu,
                "memory_utilization": historical_memory,
            },
            "horizon_seconds": 900,
            "metrics_to_forecast": ["cpu_utilization", "memory_utilization"],
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            tasks = []
            for i in range(num_requests):
                task = self._make_forecast_request(client, forecast_request)
                tasks.append(task)
            
            start = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start
            
            successful = sum(1 for r in results if not isinstance(r, Exception))
            avg_time = total_time / num_requests if num_requests > 0 else 0
            
            self.results['forecast'].append(avg_time)
            print(f"  ✓ Forecasts: {successful}/{num_requests} successful, avg: {avg_time:.3f}s")
    
    async def _make_forecast_request(self, client: httpx.AsyncClient, request: Dict):
        """Make a single forecast request"""
        start = time.time()
        try:
            response = await client.post(
                f"{ML_SERVICE_URL}/v1/forecast",
                json=request
            )
            if response.status_code == 200:
                response.json()
                return time.time() - start
            else:
                raise Exception(f"HTTP {response.status_code}")
        except Exception as e:
            raise e
    
    async def test_database_performance(self, num_queries: int = 100):
        """Test database query performance"""
        print(f"\n💾 Testing database performance with {num_queries} queries...")
        
        queries = [
            "SELECT COUNT(*) FROM pod_metrics WHERE timestamp > NOW() - INTERVAL '1 hour'",
            "SELECT AVG(cpu_utilization) FROM pod_metrics WHERE timestamp > NOW() - INTERVAL '1 hour'",
            "SELECT COUNT(*) FROM ml_predictions WHERE timestamp > NOW() - INTERVAL '1 hour'",
        ]
        
        times = []
        try:
            conn = psycopg2.connect(DATABASE_URL)
            cur = conn.cursor()
            
            for i in range(num_queries):
                query = queries[i % len(queries)]
                start = time.time()
                cur.execute(query)
                cur.fetchone()
                elapsed = time.time() - start
                times.append(elapsed)
            
            avg_time = statistics.mean(times)
            p95_time = statistics.quantiles(times, n=20)[18] if len(times) > 20 else max(times)
            
            self.results['database'].append(avg_time)
            print(f"  ✓ Database queries: avg: {avg_time:.3f}s, p95: {p95_time:.3f}s")
            
            cur.close()
            conn.close()
        except Exception as e:
            print(f"  ✗ Database test failed: {e}")
    
    def print_summary(self):
        """Print load test summary"""
        print(f"\n{'='*70}")
        print("LOAD TEST SUMMARY")
        print(f"{'='*70}\n")
        
        if self.results['collection']:
            avg = statistics.mean(self.results['collection'])
            print(f"Collection:     avg {avg:.3f}s")
        
        if self.results['prediction']:
            avg = statistics.mean(self.results['prediction'])
            print(f"Prediction:     avg {avg:.3f}s per request")
        
        if self.results['forecast']:
            avg = statistics.mean(self.results['forecast'])
            print(f"Forecast:       avg {avg:.3f}s per request")
        
        if self.results['database']:
            avg = statistics.mean(self.results['database'])
            print(f"Database:       avg {avg:.3f}s per query")


async def main():
    """Run load tests"""
    print(f"\n🚀 AURA K8s Load Testing")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    tester = LoadTester()
    
    # Test with different loads
    for load in [10, 50, 100]:
        print(f"\n{'='*70}")
        print(f"Testing with {load} concurrent operations")
        print(f"{'='*70}")
        
        await tester.test_collection_performance(load)
        await tester.test_prediction_performance(load)
        await tester.test_forecast_performance(load // 2)
        await tester.test_database_performance(load)
        
        time.sleep(2)  # Brief pause between test rounds
    
    tester.print_summary()


if __name__ == '__main__':
    asyncio.run(main())

