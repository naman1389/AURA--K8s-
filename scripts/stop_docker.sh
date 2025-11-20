#!/bin/bash
# AURA K8s - Stop Docker Environment

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         🐳 Stopping AURA K8s Docker Services 🐳            ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}\n"

cd "$PROJECT_ROOT"

# Stop all containers
echo -e "${YELLOW}Stopping Docker containers...${NC}"
if command -v docker-compose &> /dev/null; then
    docker-compose down
else
    docker compose down
fi

echo -e "\n${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║         ✅ Docker Services Stopped! ✅                      ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"

echo -e "\n${BLUE}💡 Tip: To remove volumes (database data):${NC}"
echo -e "${YELLOW}   docker-compose down -v${NC}"
echo -e "\n${BLUE}To restart Docker environment:${NC}"
echo -e "${YELLOW}   ./scripts/start_docker.sh${NC}\n"
