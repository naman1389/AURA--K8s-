#!/bin/bash
# AURA K8s - Single Unified Startup Script
# Starts all services and verifies everything works

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}AURA K8s - Starting System${NC}"
echo -e "${GREEN}========================================${NC}"

# Load environment
if [ ! -f .env.local ]; then
    cat > .env.local << 'EOF'
POSTGRES_USER=aura
POSTGRES_PASSWORD=aura_password
POSTGRES_DB=aura_metrics
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
DATABASE_URL=postgresql://aura:aura_password@localhost:5432/aura_metrics?sslmode=disable
ML_SERVICE_URL=http://localhost:8001
MCP_SERVER_URL=http://localhost:8000
GRAFANA_PORT=3000
GRAFANA_USER=admin
GRAFANA_PASSWORD=admin
COLLECTION_INTERVAL=500ms
USE_PARALLEL_COLLECTION=true
REMEDIATION_INTERVAL=30s
PREVENTIVE_REMEDIATION_INTERVAL=10s
ENABLE_PREVENTIVE_REMEDIATION=true
DRY_RUN=false
ENVIRONMENT=development
EOF
fi
export $(cat .env.local | grep -v '^#' | xargs)

# Prerequisites
echo -e "${BLUE}Checking prerequisites...${NC}"
for cmd in docker go python3 kind kubectl; do
    if ! command -v $cmd >/dev/null 2>&1; then
        echo -e "${RED}✗ $cmd not found${NC}"
        exit 1
    fi
done
echo -e "${GREEN}✓ Prerequisites OK${NC}"

# Kubernetes cluster
echo -e "${BLUE}Setting up Kubernetes cluster...${NC}"
if ! kubectl cluster-info >/dev/null 2>&1; then
    if ! kind get clusters | grep -q aura-k8s-local; then
        kind create cluster --name aura-k8s-local --config configs/kind-cluster-simple.yaml
    fi
fi
kubectl config use-context kind-aura-k8s-local 2>/dev/null || true

# Install metrics-server if not present
if ! kubectl get deployment metrics-server -n kube-system >/dev/null 2>&1; then
    echo -e "${BLUE}Installing metrics-server...${NC}"
    kubectl apply -f - <<EOF
apiVersion: v1
kind: ServiceAccount
metadata:
  name: metrics-server
  namespace: kube-system
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: metrics-server
  namespace: kube-system
spec:
  selector:
    matchLabels:
      k8s-app: metrics-server
  template:
    metadata:
      labels:
        k8s-app: metrics-server
    spec:
      serviceAccountName: metrics-server
      containers:
      - name: metrics-server
        image: registry.k8s.io/metrics-server/metrics-server:v0.7.0
        args:
          - --cert-dir=/tmp
          - --secure-port=4443
          - --kubelet-preferred-address-types=InternalIP,ExternalIP,Hostname
          - --kubelet-use-node-status-port
          - --metric-resolution=15s
          - --kubelet-insecure-tls
        ports:
        - name: https
          containerPort: 4443
          protocol: TCP
EOF
    kubectl rollout status deployment/metrics-server -n kube-system --timeout=60s
    sleep 10
    for i in {1..20}; do
        if kubectl top nodes >/dev/null 2>&1; then
            echo -e "${GREEN}✓ Metrics-server ready${NC}"
            break
        fi
        sleep 2
    done
fi

# Note: User should deploy their own pods for metrics collection

# Docker services
echo -e "${BLUE}Starting TimescaleDB...${NC}"
docker-compose up -d timescaledb
for i in {1..30}; do
    if docker exec aura-timescaledb pg_isready -U aura >/dev/null 2>&1; then
        echo -e "${GREEN}✓ TimescaleDB ready${NC}"
        break
    fi
    sleep 1
done

echo -e "${BLUE}Starting Grafana...${NC}"
if lsof -ti:3000 >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Grafana already running on port 3000${NC}"
else
    docker-compose up -d grafana
    sleep 5
    for i in {1..10}; do
        if curl -s http://localhost:3000/api/health >/dev/null 2>&1; then
            echo -e "${GREEN}✓ Grafana ready${NC}"
            break
        fi
        sleep 2
    done
fi

# Database schema
echo -e "${BLUE}Initializing database schema...${NC}"
docker exec aura-timescaledb psql -U aura -d aura_metrics -f /docker-entrypoint-initdb.d/init.sql >/dev/null 2>&1 || true

# Python environment
if [ ! -d venv ]; then
    echo -e "${BLUE}Creating Python environment...${NC}"
    python3 -m venv venv
fi
source venv/bin/activate
pip install -q -r ml/train/requirements.txt -r mcp/requirements.txt -r ml/serve/requirements.txt

# Train models if needed
if [ ! -f ml/train/models/random_forest_model.joblib ]; then
    echo -e "${BLUE}Training ML models...${NC}"
    cd ml/train && python simple_train.py && cd ../..
fi

# Import Grafana dashboards
echo -e "${BLUE}Importing Grafana dashboards...${NC}"
for i in {1..10}; do
    if curl -s http://localhost:3000/api/health >/dev/null 2>&1; then
        python3 scripts/import_grafana_dashboards.py >/dev/null 2>&1 && break
    fi
    sleep 2
done

# Create directories
mkdir -p logs .pids

# Start services
echo -e "${BLUE}Starting services...${NC}"

start_service() {
    local name=$1
    local cmd=$2
    local pid_file=$3
    local port=$4
    
    if [ -f .pids/$pid_file ]; then
        kill $(cat .pids/$pid_file) 2>/dev/null || true
        rm .pids/$pid_file 2>/dev/null || true
    fi
    
    # Convert name to lowercase for log file
    local log_name=$(echo "$name" | tr '[:upper:]' '[:lower:]' | tr ' ' '-')
    nohup bash -c "$cmd" > logs/${log_name}.log 2>&1 &
    local pid=$!
    echo $pid > .pids/$pid_file
    
    if [ -n "$port" ]; then
        for i in {1..15}; do
            if lsof -ti:$port >/dev/null 2>&1; then
                echo -e "${GREEN}✓ $name started${NC}"
                return 0
            fi
            sleep 1
        done
        echo -e "${YELLOW}⚠ $name may not have started (check logs/${log_name}.log)${NC}"
    else
        sleep 3
        if ps -p $pid >/dev/null 2>&1; then
            echo -e "${GREEN}✓ $name started${NC}"
        else
            echo -e "${YELLOW}⚠ $name may have failed (check logs/${log_name}.log)${NC}"
        fi
    fi
}

export PYTHONPATH=$(pwd)
cd $(dirname $0)

# Ensure kubeconfig is set correctly
if [ -f "$HOME/.kube/config" ]; then
    export KUBECONFIG="$HOME/.kube/config"
elif [ -f "$(kind get kubeconfig --name aura-k8s-local 2>/dev/null)" ]; then
    export KUBECONFIG=$(kind get kubeconfig --name aura-k8s-local)
else
    kind get kubeconfig --name aura-k8s-local > "$HOME/.kube/config" 2>/dev/null
    export KUBECONFIG="$HOME/.kube/config"
fi

# Build Go binaries first to catch compilation errors early
echo -e "${BLUE}Building Go services...${NC}"
if go build -o bin/collector ./cmd/collector 2>&1 | tee logs/build-collector.log; then
    echo -e "${GREEN}✓ Collector built${NC}"
else
    echo -e "${RED}✗ Collector build failed${NC}"
    exit 1
fi

if go build -o bin/remediator ./cmd/remediator 2>&1 | tee logs/build-remediator.log; then
    echo -e "${GREEN}✓ Remediator built${NC}"
else
    echo -e "${RED}✗ Remediator build failed${NC}"
    exit 1
fi

start_service "ML Service" "source venv/bin/activate && python ml/serve/predictor.py" "ml-service.pid" "8001"
start_service "MCP Server" "source venv/bin/activate && python mcp/server_ollama.py" "mcp-server.pid" "8000"
start_service "Collector" "KUBECONFIG=$KUBECONFIG ./bin/collector" "collector.pid" "9090"
start_service "Remediator" "KUBECONFIG=$KUBECONFIG ./bin/remediator" "remediator.pid" "9091"
start_service "Orchestrator" "source venv/bin/activate && python scripts/orchestrator.py" "orchestrator.pid" ""
start_service "Predictive Orchestrator" "source venv/bin/activate && python scripts/predictive_orchestrator.py" "predictive-orchestrator.pid" ""

# Wait for services to initialize
echo -e "${BLUE}Waiting for services to initialize...${NC}"
sleep 15

# Verify services
echo -e "${BLUE}Verifying services...${NC}"
if curl -sf http://localhost:8001/health >/dev/null 2>&1; then
    echo -e "${GREEN}✓ ML Service healthy${NC}"
else
    echo -e "${YELLOW}⚠ ML Service starting...${NC}"
fi

if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
    echo -e "${GREEN}✓ MCP Server healthy${NC}"
else
    echo -e "${YELLOW}⚠ MCP Server starting...${NC}"
fi

if curl -sf http://localhost:9090/health >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Collector healthy${NC}"
else
    echo -e "${YELLOW}⚠ Collector starting...${NC}"
fi

if curl -sf http://localhost:9091/health >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Remediator healthy${NC}"
else
    echo -e "${YELLOW}⚠ Remediator starting...${NC}"
fi

# Verify metrics collection from Kubernetes pods
echo -e "${BLUE}Verifying pod metrics collection...${NC}"
sleep 30
RECENT_METRICS=$(docker exec aura-timescaledb psql -U aura -d aura_metrics -t -c "SELECT COUNT(*) FROM pod_metrics WHERE timestamp > NOW() - INTERVAL '2 minutes';" 2>/dev/null | tr -d ' ')
if [ -n "$RECENT_METRICS" ] && [ "$RECENT_METRICS" -gt 0 ]; then
    echo -e "${GREEN}✓ Collecting Kubernetes pod metrics: $RECENT_METRICS recent records${NC}"
    
    # Show sample of pod data
    SAMPLE=$(docker exec aura-timescaledb psql -U aura -d aura_metrics -t -c "SELECT pod_name, namespace, cpu_utilization, memory_utilization FROM pod_metrics ORDER BY timestamp DESC LIMIT 1;" 2>/dev/null)
    if [ -n "$SAMPLE" ]; then
        echo -e "${GREEN}  Sample: $SAMPLE${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Waiting for metrics collection. Ensure you have pods running in your cluster.${NC}"
fi

# Verify predictions
PREDICTIONS_COUNT=$(docker exec aura-timescaledb psql -U aura -d aura_metrics -t -c "SELECT COUNT(*) FROM ml_predictions WHERE timestamp > NOW() - INTERVAL '5 minutes';" 2>/dev/null | tr -d ' ')
if [ -n "$PREDICTIONS_COUNT" ] && [ "$PREDICTIONS_COUNT" -gt 0 ]; then
    echo -e "${GREEN}✓ ML predictions active: $PREDICTIONS_COUNT recent predictions${NC}"
    
    # Get model accuracy from training metadata
    if [ -f ml/train/models/training_metadata.json ]; then
        ACCURACY=$(python3 -c "import json; data=json.load(open('ml/train/models/training_metadata.json')); perf=data.get('model_performance', {}); ens=perf.get('ensemble', {}); print('{:.2%}'.format(ens.get('accuracy', 0)))" 2>/dev/null || echo "N/A")
        if [ "$ACCURACY" != "N/A" ]; then
            echo -e "${GREEN}✓ Model accuracy: ${ACCURACY}${NC}"
        fi
    fi
else
    echo -e "${YELLOW}⚠ No predictions yet (orchestrator will process metrics once collected)${NC}"
fi

# Verify pod metrics collection (from any namespace)
ALL_PODS=$(kubectl get pods -A --no-headers 2>/dev/null | grep -v kube-system | grep Running | wc -l | tr -d ' ')
if [ "$ALL_PODS" -gt 0 ]; then
    echo -e "${GREEN}✓ Kubernetes pods detected: $ALL_PODS pods${NC}"
    
    # Check if we're collecting metrics from any pods
    ANY_METRICS=$(docker exec aura-timescaledb psql -U aura -d aura_metrics -t -c "SELECT COUNT(DISTINCT pod_name) FROM pod_metrics WHERE timestamp > NOW() - INTERVAL '5 minutes';" 2>/dev/null | tr -d ' ')
    if [ -n "$ANY_METRICS" ] && [ "$ANY_METRICS" -gt 0 ]; then
        echo -e "${GREEN}✓ Collecting metrics from Kubernetes pods: $ANY_METRICS pods${NC}"
    else
        echo -e "${YELLOW}⚠ Waiting for first metrics collection cycle...${NC}"
    fi
else
    echo -e "${YELLOW}⚠ No pods detected. Deploy your applications to start collecting metrics.${NC}"
fi

# Verify remediation
REMEDIATIONS_COUNT=$(docker exec aura-timescaledb psql -U aura -d aura_metrics -t -c "SELECT COUNT(*) FROM remediations WHERE success=true AND executed_at > NOW() - INTERVAL '5 minutes';" 2>/dev/null | tr -d ' ')
if [ -n "$REMEDIATIONS_COUNT" ] && [ "$REMEDIATIONS_COUNT" -gt 0 ]; then
    echo -e "${GREEN}✓ Remediations active: $REMEDIATIONS_COUNT recent successful${NC}"
fi

# Verify Grafana
if curl -sf http://localhost:3000/api/health >/dev/null 2>&1; then
    DASHBOARD_COUNT=$(curl -s -u admin:admin http://localhost:3000/api/search?type=dash-db 2>/dev/null | python3 -c "import sys, json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "0")
    if [ "$DASHBOARD_COUNT" -gt 0 ]; then
        echo -e "${GREEN}✓ Grafana dashboards loaded: $DASHBOARD_COUNT dashboards${NC}"
    else
        echo -e "${YELLOW}⚠ Dashboards may not be loaded yet${NC}"
    fi
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}System Started Successfully${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Access Points:"
echo "  Grafana:     http://localhost:3000 (admin/admin)"
echo "  Collector:   http://localhost:9090"
echo "  Remediator:  http://localhost:9091"
echo "  ML Service:  http://localhost:8001"
echo "  MCP Server:  http://localhost:8000"
echo ""
echo "Logs: ./logs/"
echo "Stop: ./stop.sh"
echo ""
