#!/usr/bin/env python3
"""
AURA K8s - Comprehensive System Status Check
Validates all components and displays current system state
"""

import psycopg2
import requests
import sys
import os
from datetime import datetime, timedelta

# Colors
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{text:^70}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'='*70}{Colors.END}\n")

def check_service(name, url, timeout=3):
    """Check if a service is responding"""
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            print(f"{Colors.GREEN}✓{Colors.END} {name:30} {Colors.GREEN}ONLINE{Colors.END}")
            return True
        else:
            print(f"{Colors.RED}✗{Colors.END} {name:30} {Colors.RED}ERROR (HTTP {resp.status_code}){Colors.END}")
            return False
    except Exception as e:
        print(f"{Colors.RED}✗{Colors.END} {name:30} {Colors.RED}OFFLINE{Colors.END}")
        return False

def get_db_stats():
    """Get database statistics"""
    try:
        conn = psycopg2.connect("postgresql://aura:aura_password@localhost:5432/aura_metrics")
        cur = conn.cursor()
        
        # Get counts
        stats = {}
        
        cur.execute("SELECT COUNT(*) FROM pod_metrics WHERE timestamp > NOW() - INTERVAL '1 hour'")
        stats['metrics_1h'] = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM ml_predictions WHERE timestamp > NOW() - INTERVAL '1 hour'")
        stats['predictions_1h'] = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM issues WHERE status != 'Resolved'")
        stats['open_issues'] = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM remediations WHERE success = true")
        stats['successful_remediations'] = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM remediations WHERE success = false")
        stats['failed_remediations'] = cur.fetchone()[0]
        
        cur.execute("SELECT MAX(timestamp) FROM pod_metrics")
        stats['last_metric'] = cur.fetchone()[0]
        
        cur.execute("SELECT MAX(timestamp) FROM ml_predictions")
        stats['last_prediction'] = cur.fetchone()[0]
        
        # Get anomaly rate
        cur.execute("""
            SELECT 
                COUNT(*) FILTER (WHERE is_anomaly = 1) * 100.0 / NULLIF(COUNT(*), 0) as anomaly_rate
            FROM ml_predictions 
            WHERE timestamp > NOW() - INTERVAL '1 hour'
        """)
        stats['anomaly_rate'] = cur.fetchone()[0] or 0
        
        # Get top issues
        cur.execute("""
            SELECT issue_type, COUNT(*) as count
            FROM issues
            WHERE status != 'Resolved'
            GROUP BY issue_type
            ORDER BY count DESC
            LIMIT 5
        """)
        stats['top_issues'] = cur.fetchall()
        
        conn.close()
        return stats
    except Exception as e:
        print(f"{Colors.RED}Database Error: {e}{Colors.END}")
        return None

def main():
    print(f"\n{Colors.CYAN}{Colors.BOLD}")
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                  AURA K8s System Status                      ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    print(f"{Colors.END}")
    
    # Check services
    print_header("SERVICE STATUS")
    services = {
        "ML Service": "http://localhost:8001/health",
        "MCP Server": "http://localhost:8000/health",
        "Grafana": "http://localhost:3000/api/health",
        "Collector Metrics": "http://localhost:9090/metrics",
        "Remediator Metrics": "http://localhost:9091/metrics",
    }
    
    services_up = 0
    for name, url in services.items():
        if check_service(name, url):
            services_up += 1
    
    print(f"\n{Colors.CYAN}Services Running: {services_up}/{len(services)}{Colors.END}")
    
    # Database statistics
    print_header("DATABASE STATISTICS")
    stats = get_db_stats()
    
    if stats:
        print(f"{Colors.BOLD}Recent Activity (Last Hour):{Colors.END}")
        print(f"  Pod Metrics Collected:     {Colors.GREEN}{stats['metrics_1h']:,}{Colors.END}")
        print(f"  ML Predictions Generated:  {Colors.GREEN}{stats['predictions_1h']:,}{Colors.END}")
        print(f"  Anomaly Detection Rate:    {Colors.YELLOW}{stats['anomaly_rate']:.1f}%{Colors.END}")
        
        print(f"\n{Colors.BOLD}Issues & Remediations:{Colors.END}")
        print(f"  Open Issues:               {Colors.YELLOW}{stats['open_issues']}{Colors.END}")
        print(f"  Successful Remediations:   {Colors.GREEN}{stats['successful_remediations']}{Colors.END}")
        print(f"  Failed Remediations:       {Colors.RED}{stats['failed_remediations']}{Colors.END}")
        
        print(f"\n{Colors.BOLD}Last Activity:{Colors.END}")
        if stats['last_metric']:
            time_diff = datetime.now(stats['last_metric'].tzinfo) - stats['last_metric']
            print(f"  Last Metric:               {Colors.GREEN}{time_diff.seconds}s ago{Colors.END}")
        if stats['last_prediction']:
            time_diff = datetime.now(stats['last_prediction'].tzinfo) - stats['last_prediction']
            print(f"  Last Prediction:           {Colors.GREEN}{time_diff.seconds}s ago{Colors.END}")
        
        if stats['top_issues']:
            print(f"\n{Colors.BOLD}Top Issues:{Colors.END}")
            for issue_type, count in stats['top_issues']:
                print(f"  {issue_type:25} {Colors.YELLOW}{count}{Colors.END}")
    else:
        print(f"{Colors.RED}Unable to retrieve database statistics{Colors.END}")
    
    # Access points
    print_header("ACCESS POINTS")
    print(f"{Colors.CYAN}📊 Grafana:{Colors.END}          http://localhost:3000 (admin/admin)")
    
    # Check Grafana dashboards
    try:
        resp = requests.get("http://admin:admin@localhost:3000/api/search?type=dash-db", timeout=3)
        if resp.status_code == 200:
            dashboards = resp.json()
            unique_dashboards = {}
            for d in dashboards:
                unique_dashboards[d['title']] = d
            print(f"{Colors.CYAN}   Dashboards:{Colors.END}       {Colors.GREEN}{len(unique_dashboards)} available{Colors.END}")
            for title in sorted(unique_dashboards.keys()):
                print(f"{Colors.CYAN}     •{Colors.END} {title}")
    except:
        pass
    
    print(f"{Colors.CYAN}🤖 ML Service API:{Colors.END}   http://localhost:8001/docs")
    print(f"{Colors.CYAN}🧠 MCP Server:{Colors.END}       http://localhost:8000/health")
    print(f"{Colors.CYAN}💾 Database:{Colors.END}         localhost:5432 (aura/aura_password)")
    print(f"{Colors.CYAN}📈 Metrics:{Colors.END}          http://localhost:9090/metrics")
    
    print(f"\n{Colors.BLUE}{'='*70}{Colors.END}\n")
    
    # Health summary
    if services_up == len(services) and stats and stats['metrics_1h'] > 0:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ System is HEALTHY and OPERATIONAL{Colors.END}\n")
        return 0
    elif services_up >= len(services) - 1:
        print(f"{Colors.YELLOW}{Colors.BOLD}⚠ System is PARTIALLY OPERATIONAL{Colors.END}\n")
        return 1
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ System has CRITICAL ISSUES{Colors.END}\n")
        return 2

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Interrupted by user{Colors.END}\n")
        sys.exit(130)
