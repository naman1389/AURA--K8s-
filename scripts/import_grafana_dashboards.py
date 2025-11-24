#!/usr/bin/env python3
"""
Import Grafana dashboards from JSON files
Handles dashboard provisioning and updates
"""

import os
import sys
import json
import requests
import time
from pathlib import Path
from typing import Dict, List, Optional

# Add scripts directory to path for imports
scripts_dir = Path(__file__).parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

# Configuration - use config helper for environment-aware URLs
try:
    from config_helper import get_service_url
    GRAFANA_URL = get_service_url("GRAFANA", "3000")
except ImportError:
    GRAFANA_URL = os.getenv("GRAFANA_URL", "http://localhost:3000")

GRAFANA_USER = os.getenv("GRAFANA_USER", "admin")
GRAFANA_PASSWORD = os.getenv("GRAFANA_PASSWORD", "admin")
DASHBOARD_DIR = Path(__file__).parent.parent / "grafana" / "dashboards"

def get_grafana_session() -> Optional[str]:
    """Get Grafana session cookie"""
    try:
        response = requests.post(
            f"{GRAFANA_URL}/api/login",
            json={"user": GRAFANA_USER, "password": GRAFANA_PASSWORD},
            timeout=5
        )
        if response.status_code == 200:
            return response.cookies.get('grafana_session', None)
    except Exception as e:
        print(f"Failed to login to Grafana: {e}")
    return None

def import_dashboard(dashboard_path: Path, session: Optional[str] = None, folder_id: Optional[int] = None) -> bool:
    """Import a single dashboard"""
    try:
        # Validate JSON before loading
        with open(dashboard_path, 'r') as f:
            try:
                dashboard_data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"  ✗ Invalid JSON in {dashboard_path.name}: {e}")
                return False
        
        # Validate required fields
        if not isinstance(dashboard_data, dict):
            print(f"  ✗ Invalid dashboard format in {dashboard_path.name}: not a JSON object")
            return False
        
        if "title" not in dashboard_data:
            print(f"  ✗ Missing 'title' field in {dashboard_path.name}")
            return False
        
        # Prepare dashboard payload
        payload = {
            "dashboard": dashboard_data,
            "overwrite": True,  # Update if exists
            "folderId": folder_id,  # Use provided folder or None for root
        }
        
        headers = {
            "Content-Type": "application/json",
        }
        if session:
            headers["Cookie"] = f"grafana_session={session}"
        
        response = requests.post(
            f"{GRAFANA_URL}/api/dashboards/db",
            json=payload,
            auth=(GRAFANA_USER, GRAFANA_PASSWORD),
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            dashboard_title = dashboard_data.get("title", dashboard_path.name)
            print(f"  ✓ Imported: {dashboard_title}")
            return True
        else:
            print(f"  ✗ Failed to import {dashboard_path.name}: {response.status_code} - {response.text[:100]}")
            return False
            
    except Exception as e:
        print(f"  ✗ Error importing {dashboard_path.name}: {e}")
        return False

def main():
    """Import all dashboards"""
    print("Importing Grafana dashboards...")
    
    # Check if Grafana is accessible
    try:
        response = requests.get(f"{GRAFANA_URL}/api/health", timeout=5)
        if response.status_code != 200:
            print(f"Grafana health check failed: {response.status_code}")
            return 1
    except Exception as e:
        print(f"Grafana not accessible: {e}")
        return 1
    
    # Get session
    session = get_grafana_session()
    if not session:
        print("Warning: Could not get Grafana session, trying without session")
    
    # Find all JSON dashboard files (exclude YAML config files)
    dashboard_files = [f for f in DASHBOARD_DIR.glob("*.json") if f.is_file()]
    
    if not dashboard_files:
        print(f"No dashboard files found in {DASHBOARD_DIR}")
        return 1
    
    print(f"Found {len(dashboard_files)} dashboard(s) to import")
    
    # Use None for root folder (dashboards will be in folder specified by Grafana provisioning)
    folder_id = None
    
    imported = 0
    failed = 0
    
    for dashboard_file in dashboard_files:
        if import_dashboard(dashboard_file, session, folder_id):
            imported += 1
        else:
            failed += 1
        time.sleep(0.5)  # Small delay between imports
    
    print(f"\nImport complete: {imported} succeeded, {failed} failed")
    
    if failed > 0:
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())


