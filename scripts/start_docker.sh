#!/bin/bash
# AURA K8s - Docker Environment Startup Script

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
echo -e "${BLUE}║         🐳 AURA K8s - Docker Environment Startup 🐳        ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed${NC}"
    echo -e "${YELLOW}Please install Docker Desktop from https://www.docker.com/products/docker-desktop${NC}"
    exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running${NC}"
    echo -e "${YELLOW}Please start Docker Desktop${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Docker is installed and running${NC}"

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null 2>&1; then
    echo -e "${RED}❌ Docker Compose is not available${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Docker Compose is available${NC}"

# Copy .env.docker to .env for docker-compose
echo -e "\n${YELLOW}Setting up environment variables...${NC}"
if [ -f "$PROJECT_ROOT/.env.docker" ]; then
    cp "$PROJECT_ROOT/.env.docker" "$PROJECT_ROOT/.env"
    echo -e "${GREEN}✓ Using .env.docker configuration${NC}"
else
    echo -e "${RED}❌ .env.docker not found${NC}"
    exit 1
fi

# Stop any existing containers
echo -e "\n${YELLOW}Stopping existing containers...${NC}"
cd "$PROJECT_ROOT"
docker-compose down 2>/dev/null || docker compose down 2>/dev/null || true

# Build and start services
echo -e "\n${BLUE}Building and starting Docker containers...${NC}"
echo -e "${YELLOW}This may take several minutes on first run...${NC}\n"

if command -v docker-compose &> /dev/null; then
    docker-compose up -d --build
else
    docker compose up -d --build
fi

# Wait for services to be healthy
echo -e "\n${YELLOW}Waiting for services to be healthy...${NC}"
sleep 10

# Check service health
echo -e "\n${BLUE}Checking service health...${NC}\n"

# Check TimescaleDB
if docker ps | grep -q aura-timescaledb; then
    echo -e "${GREEN}✓ TimescaleDB is running${NC}"
else
    echo -e "${RED}✗ TimescaleDB is not running${NC}"
fi

# Check ML Service
for i in {1..30}; do
    if curl -s http://localhost:8001/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ ML Service is healthy${NC}"
        break
    fi
    sleep 2
done

# Check MCP Server
if docker ps | grep -q aura-mcp-server; then
    echo -e "${GREEN}✓ MCP Server is running${NC}"
else
    echo -e "${YELLOW}⚠️  MCP Server may not be running${NC}"
fi

# Check Ollama
if docker ps | grep -q aura-ollama; then
    echo -e "${GREEN}✓ Ollama is running${NC}"
else
    echo -e "${YELLOW}⚠️  Ollama may not be running${NC}"
fi

# Summary
echo -e "\n${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║         ✅ AURA K8s Docker Environment Started! ✅          ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"

echo -e "\n${BLUE}📊 Docker Services:${NC}"
echo -e "  ${GREEN}✓${NC} TimescaleDB:     localhost:5432"
echo -e "  ${GREEN}✓${NC} ML Service:      http://localhost:8001/health"
echo -e "  ${GREEN}✓${NC} MCP Server:      http://localhost:8000/docs"
echo -e "  ${GREEN}✓${NC} Ollama:          http://localhost:11434"

echo -e "\n${BLUE}🎯 Quick Commands:${NC}"
echo -e "  ${YELLOW}→${NC} View logs:           docker-compose logs -f"
echo -e "  ${YELLOW}→${NC} Stop all:            docker-compose down"
echo -e "  ${YELLOW}→${NC} Restart service:     docker-compose restart <service>"
echo -e "  ${YELLOW}→${NC} View containers:     docker-compose ps"

echo -e "\n${GREEN}✅ Docker environment is ready!${NC}\n"
