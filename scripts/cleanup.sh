#!/bin/bash
# AURA K8s - Complete Project Cleanup Script
# Removes all temporary files, logs, and resets the project to clean state

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         🧹 AURA K8s - Project Cleanup 🧹                   ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}\n"

echo -e "${YELLOW}This will clean:${NC}"
echo -e "  - All log files"
echo -e "  - PID files"
echo -e "  - Python cache (__pycache__)"
echo -e "  - Compiled Python files (*.pyc)"
echo -e "  - Virtual environment cache"
echo -e "  - Go build artifacts"
echo -e ""
read -p "Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Cleanup cancelled${NC}"
    exit 0
fi

cd "$PROJECT_ROOT"

# Stop all services first
echo -e "\n${BLUE}[1/7] Stopping all services...${NC}"
./scripts/stop_local.sh > /dev/null 2>&1 || true
pkill -f "python3.*orchestrator" || true
pkill -f "python3.*generate_test_data" || true
echo -e "${GREEN}✓ All services stopped${NC}"

# Clean logs
echo -e "\n${BLUE}[2/7] Cleaning log files...${NC}"
rm -f logs/*.log 2>/dev/null || true
echo -e "${GREEN}✓ Log files cleaned${NC}"

# Clean PID files
echo -e "\n${BLUE}[3/7] Cleaning PID files...${NC}"
rm -f .pids/*.pid 2>/dev/null || true
echo -e "${GREEN}✓ PID files cleaned${NC}"

# Clean Python cache
echo -e "\n${BLUE}[4/7] Cleaning Python cache...${NC}"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
echo -e "${GREEN}✓ Python cache cleaned${NC}"

# Clean Go build cache
echo -e "\n${BLUE}[5/7] Cleaning Go build artifacts...${NC}"
go clean -cache 2>/dev/null || true
echo -e "${GREEN}✓ Go cache cleaned${NC}"

# Clean virtual environment cache
echo -e "\n${BLUE}[6/7] Cleaning venv cache...${NC}"
if [ -d "venv" ]; then
    find venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
fi
echo -e "${GREEN}✓ Venv cache cleaned${NC}"

# Optional: Clean database
echo -e "\n${BLUE}[7/7] Database cleanup...${NC}"
read -p "Reset database tables? This will delete all data (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    PGPASSWORD=aura_password psql -U aura -d aura_metrics -f scripts/init-db-local.sql > /dev/null 2>&1 && \
        echo -e "${GREEN}✓ Database reset${NC}" || \
        echo -e "${RED}✗ Database reset failed${NC}"
else
    echo -e "${YELLOW}⊘ Database cleanup skipped${NC}"
fi

echo -e "\n${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║         ✅ Project Cleanup Complete! ✅                     ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}\n"

echo -e "${BLUE}Project is clean and ready!${NC}"
echo -e "${BLUE}To start services: ./RUN.sh${NC}\n"
