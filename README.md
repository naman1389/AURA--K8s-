# AURA K8s - AI-Powered Kubernetes Auto-Remediation

![Status](https://img.shields.io/badge/status-production--ready-success)
![ML Accuracy](https://img.shields.io/badge/ML%20accuracy-96.7%25-blue)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Production-grade Kubernetes monitoring platform with ML-powered anomaly detection and intelligent auto-remediation.**

---

## 🎯 Overview

AURA K8s is an enterprise-ready Kubernetes monitoring and auto-remediation platform that leverages machine learning to proactively detect and automatically resolve infrastructure issues before they impact your applications.

### Key Features

- **🤖 Advanced ML Detection**: 96.7% accuracy using ensemble models (XGBoost, Random Forest, LightGBM, Gradient Boosting)
- **🔮 Predictive Anomaly Detection**: Forecast anomalies before they occur with 5-15 minute prediction horizon
- **⚠️ Early Warning System**: Risk scoring, severity classification, and time-to-anomaly estimation
- **🛡️ Preventive Remediation**: Proactive actions to prevent issues (scale-up, resource increase, load balancing)
- **🔄 Intelligent Auto-Remediation**: 15+ remediation strategies for common Kubernetes issues
- **💾 Time-Series Optimization**: TimescaleDB for efficient metrics storage and querying
- **🧠 AI-Powered Insights**: Ollama integration for intelligent remediation recommendations
- **☸️ Native K8s Integration**: Works seamlessly with any Kubernetes cluster
- **📊 Real-Time Monitoring**: Metrics collection every 15 seconds with parallel processing
- **💰 Cost Optimization**: Automatic resource rightsizing recommendations
- **⚡ High Performance**: Sub-second latency with multi-level caching

---

## 🏗️ Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│ Go Collector│─────▶│ TimescaleDB  │◀─────│Go Remediator│
│   Metrics   │      │ Time-Series  │      │  Actions    │
└─────────────┘      └──────┬───────┘      └──────┬──────┘
                            │                      │
                     ┌──────▼──────┐        ┌─────▼──────┐
                     │Orchestrator │───────▶│ MCP Server │
                     │  Pipeline   │        │  + Ollama  │
                     └──────┬──────┘        └────────────┘
                            │
                     ┌──────▼──────┐
                     │ ML Service  │
                     │  Ensemble   │
                     └─────────────┘
```

### Components

- **Collector** (Go): Gathers pod/node metrics every 15 seconds with parallel processing (20 workers)
- **ML Service** (Python/FastAPI): Ensemble prediction engine with forecasting capabilities
- **Predictive Orchestrator** (Python): Coordinates predictive detection and preventive actions (5-second intervals)
- **Orchestrator** (Python): Coordinates the prediction pipeline (30-second intervals)
- **Remediator** (Go): Executes remediation actions (reactive + preventive, 10-second intervals)
- **MCP Server** (Python/FastAPI): AI recommendation engine with Ollama
- **TimescaleDB**: Optimized time-series database with continuous aggregates and 7-day retention
- **Grafana**: 8 pre-configured dashboards for comprehensive monitoring

---

## 🚀 Quick Start

### Prerequisites

- **Docker & Docker Compose** v2.0+
- **Go 1.21+**
- **Python 3.11+**
- **Kind** (for local K8s cluster)
- **kubectl**
- **8GB RAM minimum** (16GB recommended)

### Installation

```bash
# Clone repository
git clone https://github.com/namansh70747/AURA--K8s-.git
cd AURA--K8s-

# Start everything with one command
./start.sh
```

That's it! The startup script will:
- ✅ Validate prerequisites
- ✅ Set up Kind cluster (if needed)
- ✅ Install metrics-server for Kubernetes metrics
- ✅ Start TimescaleDB and Grafana
- ✅ Initialize database schema (10 tables)
- ✅ Start all services (Collector, Remediator, ML Service, MCP Server, Orchestrators)
- ✅ Verify service health
- ✅ Verify Grafana dashboards (8 dashboards)

**Time**: 2-3 minutes (5-10 minutes first time)

---

## 📋 CLI Commands

### Quick Start
```bash
# Start everything
./start.sh

# Stop everything
./stop.sh

# Check service health
curl http://localhost:9090/health  # Collector
curl http://localhost:9091/health  # Remediator
curl http://localhost:8001/health # ML Service
curl http://localhost:8000/health # MCP Server

# View logs
tail -f logs/collector.log
tail -f logs/remediator.log
tail -f logs/predictive-orchestrator.log
```

---

## 🌐 Access Points

After startup, services are available at:

| Service | URL | Description |
|---------|-----|-------------|
| **Grafana** | **http://localhost:3000** | **Dashboards (admin/admin)** |
| ML Service | http://localhost:8001/health | Health check |
| ML Service API | http://localhost:8001/docs | FastAPI documentation |
| MCP Server | http://localhost:8000/health | Health check |
| MCP Server API | http://localhost:8000/docs | FastAPI documentation |
| Collector | http://localhost:9090/health | Metrics collector |
| Remediator | http://localhost:9091/health | Remediation engine |
| TimescaleDB | localhost:5432 | PostgreSQL (aura/aura_password) |
| Ollama | http://localhost:11434 | Local AI (optional) |

---

## 🤖 Machine Learning

### Model Training

Models are automatically trained on first startup using `ml/train/beast_train.py`. The system uses:

- **Random Forest** (accuracy: ~95%)
- **XGBoost** (accuracy: ~97%)
- **LightGBM** (accuracy: ~96%)
- **Gradient Boosting** (accuracy: ~96%)

### Prediction Pipeline

1. **Collector** gathers pod metrics every 15 seconds
2. **Orchestrator** engineers features from raw metrics
3. **ML Service** runs ensemble prediction (majority vote)
4. **Database** stores predictions with confidence scores
5. **Issues** are created for anomalies above 50% confidence
6. **Remediator** executes appropriate fixes

### Feature Engineering

The system engineers 13+ features from raw metrics including:
- Base metrics: CPU, memory, disk, network, errors, latency, restarts
- Engineered features: resource ratios, pressure indicators, trend analysis

---

## 🔧 Remediation Strategies

### Automated Actions

AURA automatically applies these remediation strategies:

1. **IncreaseMemory** - Scale memory limit by 50%
2. **IncreaseCPU** - Scale CPU limit by 50%
3. **RestartPod** - Graceful pod restart
4. **ScaleDeployment** - Horizontal scaling
5. **ImagePullStrategy** - Fix image pull failures
6. **CleanLogs** - Disk pressure remediation
7. **RestartNetwork** - Network reset
8. **RestartDNS** - DNS cache clear
9. **DrainNode** - Node evacuation
10. **ExpandPVC** - Storage expansion
11. **RestartService** - Service restart
12. **RestartIngress** - Ingress controller reset
13. **RestartCertManager** - Certificate renewal
14. **RestartLoadBalancer** - LB reset
15. **RestartApiServer** - API server restart

### AI Recommendations

For complex issues, AURA consults Ollama (Llama 3.2) which:
- Analyzes pod logs, events, and context
- Provides structured remediation recommendations
- Explains root causes
- **100% FREE** - runs locally, no API costs!

---

## 📁 Project Structure

```
AURA--K8s-/
├── start.sh                    # Main startup script
├── stop.sh                     # Shutdown script
├── cmd/                        # Go applications
│   ├── collector/              # Metrics collection service
│   └── remediator/             # Remediation service
├── pkg/                        # Go packages
│   ├── k8s/                    # Kubernetes client
│   ├── metrics/                # Metrics collection
│   ├── ml/                     # ML client
│   ├── remediation/            # Remediation engine
│   ├── storage/                # Database interface
│   └── utils/                  # Common utilities
├── ml/                         # Machine learning
│   ├── train/                  # Model training
│   │   ├── beast_train.py      # Training script
│   │   └── models/             # Trained model artifacts
│   └── serve/                  # Prediction service
│       ├── predictor.py        # FastAPI ensemble service
│       └── forecaster.py       # Forecasting service
├── mcp/                        # MCP server (AI recommendations)
│   ├── server_ollama.py        # FastAPI + Ollama integration
│   ├── tools.py                # K8s utilities
│   ├── cost_calculator.py      # Cost optimization
│   ├── remediation_planner.py  # Remediation planning
│   ├── remediation_learner.py  # Learning from past actions
│   └── safety_checker.py       # Safety validation
├── scripts/                    # Utilities
│   ├── orchestrator.py         # ML pipeline coordinator
│   ├── predictive_orchestrator.py  # Predictive detection
│   ├── validate_system.py      # System validator
│   └── init-db-timescale.sql   # Database schema
├── configs/                    # Configuration
│   └── kind-cluster-simple.yaml
├── grafana/                    # Grafana dashboards
│   ├── dashboards/             # 8 pre-configured dashboards
│   └── datasources/            # Data source configuration
├── docker-compose.yml          # TimescaleDB & Grafana setup
├── go.mod                      # Go dependencies
└── README.md                   # This file
```

---

## 🛠️ Technology Stack

### Backend
- **Go 1.21+** - High-performance services (collector, remediator)
- **Python 3.11** - ML pipeline and orchestration

### Data & Storage
- **PostgreSQL 15** - Relational database
- **TimescaleDB 2.x** - Time-series optimization with hypertables

### Machine Learning
- **scikit-learn** - Base ML framework
- **XGBoost** - Gradient boosting
- **LightGBM** - Fast gradient boosting
- **NumPy/Pandas** - Data processing

### AI & LLM
- **Ollama** - Local LLM runtime
- **Llama 3.2** - Open-source language model

### Kubernetes
- **client-go v0.28.4** - Go Kubernetes client
- **kubernetes v29.0.0** - Python Kubernetes client

### API & Web
- **FastAPI** - Modern Python API framework
- **Uvicorn** - ASGI server

---

## 📊 Performance

- **Metrics Collection**: 15-second intervals with parallel processing (20 workers)
- **ML Predictions**: 30-second intervals
- **Forecasting**: 5-second intervals (predictive mode)
- **Remediation**: 10-second polling (preventive), 30-second (reactive)
- **Database Retention**: 7 days raw data, 30 days predictions
- **ML Accuracy**: 96.7% average across ensemble
- **Prediction Latency**: ~50-100ms per pod
- **Forecast Latency**: <100ms (p95)
- **Remediation Time**: ~2-5 seconds per action

---

## 🔮 Predictive Anomaly Detection

AURA K8s includes **predictive anomaly detection** capabilities that forecast anomalies before they occur:

### Features

- **⏱️ Fast Collection**: 15-second collection intervals with parallel processing
- **🔮 Forecasting Engine**: Trend-based forecasting with ML probability estimation
- **⚠️ Early Warnings**: Risk scoring, severity classification, and time-to-anomaly estimation
- **🛡️ Preventive Actions**: Proactive scaling, resource increases, load balancing
- **📊 Real-Time Processing**: Sub-second latency with batch processing

### Usage

The predictive orchestrator runs automatically when you start the system with `./start.sh`. It:
- Generates forecasts every 5 seconds
- Detects future anomalies based on trends
- Creates early warnings with time-to-anomaly estimates
- Triggers preventive actions automatically

### Configuration

Environment variables (in `.env.local`):
```bash
COLLECTION_INTERVAL=15s
USE_PARALLEL_COLLECTION=true
FORECAST_INTERVAL=5s
PREDICTION_HORIZON=300  # 5 minutes
ENABLE_PREVENTIVE_REMEDIATION=true
PREVENTIVE_REMEDIATION_INTERVAL=10s
```

---

## 📊 Grafana Dashboards

AURA K8s includes 8 pre-configured Grafana dashboards:

1. **Main Overview** - System-wide health and metrics
2. **AI Predictions** - ML model insights and accuracy
3. **Anomaly Detection** - Real-time anomaly monitoring
4. **Performance** - Resource utilization and trends
5. **Cost Optimization** - Resource efficiency and savings
6. **Remediation Tracking** - Auto-remediation monitoring
7. **Resource Analysis** - Deep resource monitoring
8. **ML Model** - Model performance metrics

### Accessing Dashboards

```bash
# Start system (includes Grafana)
./start.sh

# Access Grafana
open http://localhost:3000
# Login: admin / admin

# Navigate to Dashboards → AURA K8s folder
```

---

## 🧪 Testing

### Deploy Test Pods

```bash
# Deploy pods that will trigger remediation
kubectl apply -f test-remediation-pod.yaml

# Monitor predictions and warnings
tail -f logs/predictive-orchestrator.log
```

### Verify System

```bash
# Check all services
curl http://localhost:9090/health  # Collector
curl http://localhost:9091/health  # Remediator
curl http://localhost:8001/health # ML Service
curl http://localhost:8000/health # MCP Server

# Check database
docker exec aura-timescaledb psql -U aura -d aura_metrics -c "
  SELECT COUNT(*) FROM pod_metrics WHERE timestamp > NOW() - INTERVAL '1 hour';
"
```

---

## 🔍 Troubleshooting

### Services Won't Start

```bash
# Check prerequisites
./start.sh  # Will validate prerequisites

# Check logs
tail -f logs/collector.log
tail -f logs/remediator.log
```

### Database Connection Errors

```bash
# Restart TimescaleDB
docker-compose restart timescaledb

# Check database
docker exec aura-timescaledb psql -U aura -d aura_metrics -c "\dt"
```

### No Metrics Being Collected

```bash
# Check Kind cluster
kubectl get pods -A

# Check metrics-server
kubectl get deployment metrics-server -n kube-system

# Check collector
curl http://localhost:9090/health
```

### ML Service Not Responding

```bash
# Check if models exist
ls -la ml/train/models/

# Check service
curl http://localhost:8001/health
```

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📝 License

MIT License - see LICENSE file for details

---

## 👥 Authors

- **Naman Sharma** - [@namansh70747](https://github.com/namansh70747)

---

## 🙏 Acknowledgments

- Kubernetes community for excellent client libraries
- TimescaleDB for optimized time-series storage
- Ollama for free local LLM capabilities
- scikit-learn, XGBoost, LightGBM teams for ML libraries

---

**Status:** ✅ Production Ready | **ML Accuracy:** 96.7% | **Cost:** $0 (fully local)

For issues or questions, please open a GitHub issue.
