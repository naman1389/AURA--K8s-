#!/bin/bash
# Setup Grafana for local development

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}Setting up Grafana for local development...${NC}"

# Create provisioning directories
GRAFANA_PROVISIONING="/opt/homebrew/etc/grafana/provisioning"
mkdir -p "$GRAFANA_PROVISIONING/datasources" 2>/dev/null || true
mkdir -p "$GRAFANA_PROVISIONING/dashboards" 2>/dev/null || true

echo -e "${YELLOW}Copying datasource configuration...${NC}"
cp "$PROJECT_ROOT/grafana/datasources/datasource-local.yml" "$GRAFANA_PROVISIONING/datasources/aura.yml" 2>/dev/null || \
  echo -e "${YELLOW}Note: Could not copy datasource config. You may need to configure manually.${NC}"

echo -e "${YELLOW}Copying dashboard configuration...${NC}"
cp "$PROJECT_ROOT/grafana/dashboards/dashboard.yml" "$GRAFANA_PROVISIONING/dashboards/aura.yml" 2>/dev/null || \
  echo -e "${YELLOW}Note: Could not copy dashboard config. You may need to configure manually.${NC}"

# Update dashboard.yml to point to correct path
if [ -f "$GRAFANA_PROVISIONING/dashboards/aura.yml" ]; then
  sed -i '' "s|path: /etc/grafana/dashboards|path: $PROJECT_ROOT/grafana/dashboards|g" "$GRAFANA_PROVISIONING/dashboards/aura.yml" 2>/dev/null || true
fi

echo -e "${GREEN}✓ Grafana provisioning configured${NC}"
echo -e "${YELLOW}Restarting Grafana...${NC}"

# Restart Grafana to pick up new configuration
brew services restart grafana

sleep 2
echo -e "${GREEN}✓ Grafana restarted${NC}"
echo -e "${BLUE}Grafana should be available at http://localhost:3000${NC}"
echo -e "${BLUE}Default credentials: admin/admin${NC}"
