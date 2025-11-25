#!/bin/bash
# Setup Grafana datasource from environment variables
# This script generates the datasource.yml file based on environment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GRAFANA_DS_DIR="$PROJECT_ROOT/grafana/datasources"

# Default values
POSTGRES_HOST="${POSTGRES_HOST:-timescaledb}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_DB="${POSTGRES_DB:-aura_metrics}"
POSTGRES_USER="${POSTGRES_USER:-aura}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-aura_password}"

# Auto-detect environment
# Check if running in Kubernetes or Docker
if [ -z "$KUBERNETES_SERVICE_HOST" ] && [ "$ENVIRONMENT" != "production" ]; then
    # Local development or Docker Compose - check if timescaledb service exists
    if [ -z "$POSTGRES_HOST" ]; then
        # Try to detect: if docker-compose service name is accessible, use it; otherwise localhost
        if command -v docker >/dev/null 2>&1 && docker ps | grep -q aura-timescaledb; then
            # Docker Compose environment - use service name
            POSTGRES_HOST="timescaledb"
        else
            # Local development - use localhost
            POSTGRES_HOST="localhost"
        fi
    fi
fi

# Generate datasource.yml
cat > "$GRAFANA_DS_DIR/datasource.yml" <<EOF
apiVersion: 1

datasources:
  - name: TimescaleDB
    uid: aura-timescaledb
    type: postgres
    access: proxy
    url: ${POSTGRES_HOST}:${POSTGRES_PORT}
    database: ${POSTGRES_DB}
    user: ${POSTGRES_USER}
    secureJsonData:
      password: "${POSTGRES_PASSWORD}"
    jsonData:
      database: ${POSTGRES_DB}
      sslmode: "disable"
      postgresVersion: 1500
      timescaledb: true
      maxOpenConns: 100
      maxIdleConns: 25
      connMaxLifetime: 14400
      timeInterval: "10s"
    isDefault: true
    editable: true
    version: 1
EOF

echo "✅ Generated Grafana datasource configuration"
echo "   Host: $POSTGRES_HOST:$POSTGRES_PORT"
echo "   Database: $POSTGRES_DB"
echo "   User: $POSTGRES_USER"
