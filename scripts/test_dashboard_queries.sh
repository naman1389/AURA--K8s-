#!/bin/bash
# Test all dashboard queries to verify they return data

echo "🔍 Testing AURA K8s Dashboard Queries..."
echo ""

DB_CMD="psql -h localhost -U aura -d aura_metrics -t -c"

# Test 1: Health Score (Main Overview)
echo "1. Testing Health Score Query..."
RESULT=$($DB_CMD "SELECT COUNT(*) FROM (SELECT time_bucket('1 minute'::interval, timestamp) as time, 100.0 - (COUNT(*) FILTER (WHERE is_anomaly = 1) * 100.0 / NULLIF(COUNT(*), 0)) as health_score FROM ml_predictions WHERE timestamp >= NOW() - INTERVAL '1 hour' GROUP BY time ORDER BY time DESC LIMIT 10) AS q;")
echo "   Results: $RESULT rows"

# Test 2: CPU/Memory Utilization (Main Overview)
echo "2. Testing Resource Utilization..."
RESULT=$($DB_CMD "SELECT COUNT(*) FROM (SELECT time_bucket('5 minutes'::interval, timestamp) as time, AVG(COALESCE(cpu_utilization, 0)) as cpu, AVG(COALESCE(memory_utilization, 0)) as mem FROM pod_metrics WHERE timestamp >= NOW() - INTERVAL '1 hour' GROUP BY time ORDER BY time DESC) AS q;")
echo "   Results: $RESULT rows"

# Test 3: ML Predictions (AI Dashboard)
echo "3. Testing ML Predictions..."
RESULT=$($DB_CMD "SELECT COUNT(*) FROM (SELECT time_bucket('1 minute'::interval, timestamp) as time, AVG(confidence) as model_confidence FROM ml_predictions WHERE timestamp >= NOW() - INTERVAL '1 hour' AND is_anomaly = 1 GROUP BY time ORDER BY time DESC) AS q;")
echo "   Results: $RESULT rows"

# Test 4: Active Issues
echo "4. Testing Active Issues..."
RESULT=$($DB_CMD "SELECT COUNT(*) FROM (SELECT predicted_issue, COUNT(*) as count FROM ml_predictions WHERE timestamp >= NOW() - INTERVAL '24 hours' AND is_anomaly = 1 GROUP BY predicted_issue) AS q;")
echo "   Results: $RESULT issue types"

# Test 5: Pod Metrics Existence
echo "5. Testing Pod Metrics..."
RESULT=$($DB_CMD "SELECT COUNT(DISTINCT pod_name) FROM pod_metrics WHERE timestamp >= NOW() - INTERVAL '1 hour';")
echo "   Unique pods: $RESULT"

# Test 6: Node Metrics
echo "6. Testing Node Metrics..."
RESULT=$($DB_CMD "SELECT COUNT(*) FROM node_metrics WHERE timestamp >= NOW() - INTERVAL '1 hour';")
echo "   Node metrics: $RESULT rows"

# Test 7: Time Range Check
echo ""
echo "📅 Data Time Range:"
$DB_CMD "SELECT 'Oldest data:' as label, MIN(timestamp)::text as value FROM pod_metrics UNION ALL SELECT 'Newest data:', MAX(timestamp)::text FROM pod_metrics UNION ALL SELECT 'Age:', (NOW() - MAX(timestamp))::text FROM pod_metrics;" | column -t -s'|'

echo ""
echo "✅ Query Test Complete!"
echo ""
echo "💡 If dashboards show 'No data':"
echo "   1. In Grafana, click the time range selector (top right)"
echo "   2. Select 'Last 1 hour' or 'Last 30 minutes'"
echo "   3. Click the refresh button"
echo "   4. Make sure you're logged in as admin/admin"
