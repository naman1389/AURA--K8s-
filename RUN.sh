#!/bin/bash
# AURA K8s - Single Command Startup Script
# Sets up everything and starts all services

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

clear

echo -e "${BLUE}"
cat << "EOF"
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║        █████╗ ██╗   ██╗██████╗  █████╗     ██╗  ██╗███████╗ ║
║       ██╔══██╗██║   ██║██╔══██╗██╔══██╗    ██║ ██╔╝██╔════╝ ║
║       ███████║██║   ██║██████╔╝███████║    █████╔╝ ███████╗ ║
║       ██╔══██║██║   ██║██╔══██╗██╔══██║    ██╔═██╗ ╚════██║ ║
║       ██║  ██║╚██████╔╝██║  ██║██║  ██║    ██║  ██╗███████║ ║
║       ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝    ╚═╝  ╚═╝╚══════╝ ║
║                                                              ║
║           AI-Powered Kubernetes Auto-Remediation            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# Make scripts executable
chmod +x "$SCRIPT_DIR"/scripts/*.sh 2>/dev/null || true

# Check mode - Local or Docker
echo -e "\n${BLUE}Select Mode:${NC}"
echo -e "${YELLOW}1)${NC} Local Mode (Kind Kubernetes + Local Services)"
echo -e "${YELLOW}2)${NC} Docker Mode (Full Docker Compose)"
echo -e "${YELLOW}3)${NC} Stop All Services"
echo -e "${YELLOW}4)${NC} Validate System"
echo -e ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${BLUE}Starting LOCAL Mode...${NC}"
        echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
        
        # Setup environment if needed
        if [ ! -d "$SCRIPT_DIR/.kind-cluster" ] || ! kind get clusters 2>/dev/null | grep -q "aura-k8s-local"; then
            echo -e "${YELLOW}Setting up environment (first time only)...${NC}"
            "$SCRIPT_DIR/scripts/setup_complete_local.sh"
        fi
        
        # Start services
        "$SCRIPT_DIR/scripts/start_local.sh"
        ;;
        
    2)
        echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${BLUE}Starting DOCKER Mode...${NC}"
        echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
        "$SCRIPT_DIR/scripts/start_docker.sh"
        ;;
        
    3)
        echo -e "\n${YELLOW}Stopping all services...${NC}\n"
        "$SCRIPT_DIR/scripts/stop_local.sh" 2>/dev/null || true
        "$SCRIPT_DIR/scripts/stop_docker.sh" 2>/dev/null || true
        echo -e "\n${GREEN}✓ All services stopped${NC}\n"
        ;;
        
    4)
        echo -e "\n${BLUE}Running system validation...${NC}\n"
        if [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
            source "$SCRIPT_DIR/venv/bin/activate"
        fi
        python3 "$SCRIPT_DIR/scripts/validate_system.py"
        ;;
        
    *)
        echo -e "\n${RED}Invalid choice. Exiting.${NC}\n"
        exit 1
        ;;
esac
