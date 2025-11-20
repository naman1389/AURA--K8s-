#!/bin/bash
# AURA K8s - Complete Local Setup with Kind Kubernetes
# This script sets up everything: Kind cluster + Local services

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CLUSTER_NAME="aura-k8s-local"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║    🚀 AURA K8s - Complete Local Environment Setup 🚀      ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 1. Check Prerequisites
echo -e "\n${BLUE}[1/8] Checking Prerequisites...${NC}"

# Check Docker
if ! command_exists docker; then
    echo -e "${RED}❌ Docker is not installed${NC}"
    exit 1
fi
if ! docker info >/dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker is running${NC}"

# Check/Install Kind
if ! command_exists kind; then
    echo -e "${YELLOW}Installing kind...${NC}"
    brew install kind || {
        echo -e "${RED}Failed to install kind. Install manually: brew install kind${NC}"
        exit 1
    }
fi
echo -e "${GREEN}✓ Kind installed: $(kind version)${NC}"

# Check/Install kubectl
if ! command_exists kubectl; then
    echo -e "${YELLOW}Installing kubectl...${NC}"
    brew install kubectl || {
        echo -e "${RED}Failed to install kubectl. Install manually: brew install kubectl${NC}"
        exit 1
    }
fi
echo -e "${GREEN}✓ kubectl installed: $(kubectl version --client --short 2>/dev/null || kubectl version --client)${NC}"

# Check Go
if ! command_exists go; then
    echo -e "${RED}❌ Go is not installed${NC}"
    echo -e "${YELLOW}Install with: brew install go${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Go installed: $(go version)${NC}"

# Check Python
if ! command_exists python3; then
    echo -e "${RED}❌ Python 3 is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python installed: $(python3 --version)${NC}"

# Check PostgreSQL
if ! command_exists psql; then
    echo -e "${YELLOW}Installing PostgreSQL...${NC}"
    brew install postgresql@15 || {
        echo -e "${RED}Failed to install PostgreSQL${NC}"
        exit 1
    }
fi
echo -e "${GREEN}✓ PostgreSQL installed${NC}"

# Check Ollama (optional but recommended)
if ! command_exists ollama; then
    echo -e "${YELLOW}⚠️  Ollama not found. Install with: brew install ollama${NC}"
    echo -e "${YELLOW}   MCP server will use fallback mode without AI recommendations${NC}"
else
    echo -e "${GREEN}✓ Ollama installed${NC}"
fi

# 2. Create/Reset Kind Cluster
echo -e "\n${BLUE}[2/8] Setting up Kind Kubernetes Cluster...${NC}"

# Check if cluster already exists
if kind get clusters 2>/dev/null | grep -q "^${CLUSTER_NAME}$"; then
    echo -e "${YELLOW}Cluster '${CLUSTER_NAME}' already exists. Deleting...${NC}"
    kind delete cluster --name "${CLUSTER_NAME}"
fi

# Create Kind cluster
echo -e "${YELLOW}Creating Kind cluster...${NC}"
cat <<EOF | kind create cluster --name "${CLUSTER_NAME}" --config=-
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
    kubeadmConfigPatches:
      - |
        kind: InitConfiguration
        nodeRegistration:
          kubeletExtraArgs:
            node-labels: "workload-type=all"
EOF

# Wait for cluster to be ready
echo -e "${YELLOW}Waiting for cluster to be ready...${NC}"
kubectl wait --for=condition=Ready nodes --all --timeout=120s

echo -e "${GREEN}✓ Kind cluster created and ready${NC}"

# 3. Install Metrics Server
echo -e "\n${BLUE}[3/8] Installing Metrics Server...${NC}"

# Install metrics-server with insecure TLS (required for Kind)
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# Patch metrics-server for Kind compatibility
kubectl patch deployment metrics-server -n kube-system --type='json' -p='[
  {
    "op": "add",
    "path": "/spec/template/spec/containers/0/args/-",
    "value": "--kubelet-insecure-tls"
  },
  {
    "op": "add",
    "path": "/spec/template/spec/containers/0/args/-",
    "value": "--kubelet-preferred-address-types=InternalIP"
  }
]'

# Wait for metrics-server
echo -e "${YELLOW}Waiting for metrics-server to be ready...${NC}"
kubectl wait --for=condition=Available deployment/metrics-server -n kube-system --timeout=120s

echo -e "${GREEN}✓ Metrics server installed${NC}"

# 4. Deploy Test Workloads
echo -e "\n${BLUE}[4/8] Deploying test workloads...${NC}"

kubectl apply -f - <<EOF
apiVersion: v1
kind: Namespace
metadata:
  name: demo
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-demo
  namespace: demo
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:latest
        resources:
          requests:
            memory: "64Mi"
            cpu: "100m"
          limits:
            memory: "128Mi"
            cpu: "200m"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis-demo
  namespace: demo
spec:
  replicas: 2
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:alpine
        resources:
          requests:
            memory: "32Mi"
            cpu: "50m"
          limits:
            memory: "64Mi"
            cpu: "100m"
EOF

echo -e "${GREEN}✓ Test workloads deployed${NC}"

# 5. Setup PostgreSQL Database
echo -e "\n${BLUE}[5/8] Setting up PostgreSQL Database...${NC}"

# Start PostgreSQL
brew services start postgresql@15 2>/dev/null || true
sleep 3

# Create database and user
echo -e "${YELLOW}Creating database...${NC}"
# Drop database if exists (must disconnect users first)
psql postgres -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = 'aura_metrics';" 2>/dev/null || true
psql postgres -c "DROP DATABASE IF EXISTS aura_metrics;" 2>/dev/null || true
psql postgres -c "DROP USER IF EXISTS aura;" 2>/dev/null || true
# Create user (ignore if exists)
psql postgres -c "CREATE USER aura WITH PASSWORD 'aura_password';" 2>/dev/null || echo -e "${YELLOW}User aura already exists, continuing...${NC}"
# Create database
psql postgres -c "CREATE DATABASE aura_metrics OWNER aura;" 2>/dev/null || echo -e "${YELLOW}Database aura_metrics already exists, continuing...${NC}"
psql postgres -c "GRANT ALL PRIVILEGES ON DATABASE aura_metrics TO aura;" 2>/dev/null || true

# Initialize schema (note: local uses regular Postgres, not TimescaleDB)
echo -e "${YELLOW}Initializing database schema...${NC}"
PGPASSWORD=aura_password psql -U aura -d aura_metrics -f "$SCRIPT_DIR/init-db-local.sql"

echo -e "${GREEN}✓ Database created and initialized${NC}"

# 6. Build Go Services
echo -e "\n${BLUE}[6/8] Building Go Services...${NC}"

cd "$PROJECT_ROOT"

# Clean old binaries
rm -f bin/collector bin/remediator

# Build collector
echo -e "${YELLOW}Building collector...${NC}"
go build -o bin/collector ./cmd/collector
echo -e "${GREEN}✓ Collector built${NC}"

# Build remediator
echo -e "${YELLOW}Building remediator...${NC}"
go build -o bin/remediator ./cmd/remediator
echo -e "${GREEN}✓ Remediator built${NC}"

# 7. Setup Python Environment
echo -e "\n${BLUE}[7/8] Setting up Python Environment...${NC}"

VENV_DIR="$PROJECT_ROOT/venv"

if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo -e "${YELLOW}Installing Python dependencies...${NC}"
pip install -q --upgrade pip setuptools wheel
pip install -q -r "$PROJECT_ROOT/mcp/requirements.txt" 2>/dev/null || pip install -q fastapi uvicorn httpx pydantic psycopg2-binary
pip install -q -r "$PROJECT_ROOT/ml/train/requirements.txt" 2>/dev/null || pip install -q numpy scikit-learn joblib

echo -e "${GREEN}✓ Python environment ready${NC}"

# 8. Start Ollama (if installed)
echo -e "\n${BLUE}[8/8] Starting Ollama...${NC}"

if command_exists ollama; then
    if ! pgrep -x ollama >/dev/null; then
        echo -e "${YELLOW}Starting Ollama server...${NC}"
        nohup ollama serve > "$PROJECT_ROOT/logs/ollama.log" 2>&1 &
        sleep 3
    fi
    
    # Pull model if not already pulled
    echo -e "${YELLOW}Checking Ollama model...${NC}"
    if ! ollama list | grep -q "llama3.2"; then
        echo -e "${YELLOW}Pulling llama3.2 model (this may take a few minutes)...${NC}"
        ollama pull llama3.2
    fi
    echo -e "${GREEN}✓ Ollama ready with llama3.2 model${NC}"
else
    echo -e "${YELLOW}⚠️  Ollama not installed - MCP will use fallback mode${NC}"
fi

# Summary
echo -e "\n${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║            ✅ Local Environment Setup Complete! ✅          ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"

echo -e "\n${BLUE}📊 Environment Status:${NC}"
echo -e "  ${GREEN}✓${NC} Kind Cluster:     ${CLUSTER_NAME}"
echo -e "  ${GREEN}✓${NC} Metrics Server:   Installed"
echo -e "  ${GREEN}✓${NC} Test Pods:        Deployed in 'demo' namespace"
echo -e "  ${GREEN}✓${NC} PostgreSQL:       localhost:5432/aura_metrics"
echo -e "  ${GREEN}✓${NC} Go Binaries:      Built in bin/"
echo -e "  ${GREEN}✓${NC} Python Env:       Virtual environment ready"
echo -e "  ${GREEN}✓${NC} ML Models:        Available in ml/train/models/"

echo -e "\n${BLUE}📝 Next Steps:${NC}"
echo -e "  ${YELLOW}1.${NC} Export kubeconfig: ${YELLOW}export KUBECONFIG=\$(kind get kubeconfig --name ${CLUSTER_NAME})${NC}"
echo -e "  ${YELLOW}2.${NC} Start services:    ${YELLOW}./scripts/start_local.sh${NC}"
echo -e "  ${YELLOW}3.${NC} Check cluster:     ${YELLOW}kubectl get pods -A${NC}"

echo -e "\n${BLUE}🔍 Useful Commands:${NC}"
echo -e "  ${YELLOW}→${NC} View pods:            kubectl get pods -A"
echo -e "  ${YELLOW}→${NC} View node metrics:    kubectl top nodes"
echo -e "  ${YELLOW}→${NC} View pod metrics:     kubectl top pods -A"
echo -e "  ${YELLOW}→${NC} Delete cluster:       kind delete cluster --name ${CLUSTER_NAME}"

# Export kubeconfig for current session
export KUBECONFIG="$(kind get kubeconfig --name ${CLUSTER_NAME})"
echo -e "\n${GREEN}✓ KUBECONFIG set for current shell${NC}"

echo -e "\n${BLUE}Ready to start services! Run:${NC} ${YELLOW}./scripts/start_local.sh${NC}\n"
