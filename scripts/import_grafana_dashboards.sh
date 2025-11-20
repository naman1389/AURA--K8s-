#!/bin/bash
# Import Grafana Dashboards

GRAFANA_URL="http://localhost:3000"
GRAFANA_USER="admin"
GRAFANA_PASS="admin"

echo "Importing Grafana dashboards..."

for dashboard_file in /Users/namansharma/AURA--K8s--1/grafana/dashboards/*.json; do
    dashboard_name=$(basename "$dashboard_file")
    
    # Create proper import payload
    payload=$(jq -n --slurpfile dashboard "$dashboard_file" '{
        dashboard: $dashboard[0],
        overwrite: true,
        message: "Updated dashboard"
    }')
    
    response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -u "$GRAFANA_USER:$GRAFANA_PASS" \
        -d "$payload" \
        "$GRAFANA_URL/api/dashboards/db")
    
    if echo "$response" | grep -q '"status":"success"'; then
        echo "✓ Successfully imported $dashboard_name"
    else
        echo "✗ Failed to import $dashboard_name"
        echo "  Response: $response"
    fi
done

echo "Dashboard import complete!"
