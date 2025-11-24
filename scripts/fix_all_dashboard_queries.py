#!/usr/bin/env python3
"""Fix all dashboard queries to use 24-hour window and proper null handling"""
import json
import glob
import re
import sys
from pathlib import Path

def fix_all_queries(sql):
    """Fix all query issues"""
    if not sql or not isinstance(sql, str):
        return sql
    
    # Fix all time intervals to 24 hours
    sql = re.sub(r"NOW\(\) - INTERVAL '15 minutes'", "NOW() - INTERVAL '24 hours'", sql)
    sql = re.sub(r"NOW\(\) - INTERVAL '1 hour'", "NOW() - INTERVAL '24 hours'", sql)
    sql = re.sub(r"NOW\(\) - INTERVAL '10 minutes'", "NOW() - INTERVAL '24 hours'", sql)
    sql = re.sub(r"NOW\(\) - INTERVAL '5 minutes'", "NOW() - INTERVAL '24 hours'", sql)
    sql = re.sub(r"NOW\(\) - INTERVAL '30 minutes'", "NOW() - INTERVAL '24 hours'", sql)
    
    # Fix $__timeFilter
    sql = sql.replace('$__timeFilter(executed_at)', "executed_at >= NOW() - INTERVAL '24 hours'")
    sql = sql.replace('$__timeFilter(timestamp)', "timestamp >= NOW() - INTERVAL '24 hours'")
    sql = sql.replace('$__timeFilter(created_at)', "created_at >= NOW() - INTERVAL '24 hours'")
    
    # Add COALESCE for COUNT
    if 'COUNT(*)' in sql and 'COALESCE' not in sql and 'FILTER' not in sql:
        sql = sql.replace('COUNT(*) as value', 'COALESCE(COUNT(*), 0) as value')
        sql = sql.replace('COUNT(*) as', 'COALESCE(COUNT(*), 0) as')
    
    # Add COALESCE for AVG
    if 'AVG(' in sql and 'COALESCE(AVG' not in sql and 'FILTER' not in sql:
        sql = re.sub(r'AVG\(([^)]+)\) as value', r'COALESCE(AVG(\1), 0.0) as value', sql)
        sql = re.sub(r'AVG\(([^)]+)\) as', r'COALESCE(AVG(\1), 0.0) as', sql)
    
    # Add COALESCE for MAX
    if 'MAX(' in sql and 'COALESCE(MAX' not in sql and 'FILTER' not in sql:
        sql = re.sub(r'MAX\(([^)]+)\) as value', r'COALESCE(MAX(\1), 0) as value', sql)
    
    # Fix time_bucket
    sql = re.sub(r"time_bucket\('5 minutes'", "time_bucket('1 minute'", sql)
    sql = re.sub(r"time_bucket\('10 minutes'", "time_bucket('1 minute'", sql)
    sql = re.sub(r"time_bucket\('15 minutes'", "time_bucket('1 minute'", sql)
    sql = re.sub(r"time_bucket\('30 minutes'", "time_bucket('1 minute'", sql)
    sql = re.sub(r"time_bucket\('1 hour'", "time_bucket('1 minute'", sql)
    sql = re.sub(r"time_bucket\('1 day'", "time_bucket('1 minute'", sql)
    
    return sql

def main():
    dashboard_dir = Path(__file__).parent.parent / "grafana" / "dashboards"
    dashboards = list(dashboard_dir.glob("*.json"))
    
    total_fixed = 0
    
    for dashboard_file in dashboards:
        try:
            with open(dashboard_file, 'r') as f:
                dashboard = json.load(f)
            
            file_changed = False
            
            def fix_queries_in_obj(obj):
                nonlocal file_changed
                if isinstance(obj, dict):
                    if 'rawSql' in obj:
                        old_sql = obj['rawSql']
                        new_sql = fix_all_queries(old_sql)
                        if new_sql != old_sql:
                            obj['rawSql'] = new_sql
                            file_changed = True
                    for value in obj.values():
                        fix_queries_in_obj(value)
                elif isinstance(obj, list):
                    for item in obj:
                        fix_queries_in_obj(item)
            
            fix_queries_in_obj(dashboard)
            
            if file_changed:
                with open(dashboard_file, 'w') as f:
                    json.dump(dashboard, f, indent=2)
                print(f"✅ Fixed: {dashboard_file.name}")
                total_fixed += 1
        except Exception as e:
            print(f"⚠️  Error: {dashboard_file.name}: {e}")
    
    print(f"\n✅ Fixed {total_fixed} dashboard files")
    return 0

if __name__ == "__main__":
    sys.exit(main())

