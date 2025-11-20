#!/bin/bash
# Reset and reinitialize the local PostgreSQL database

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}Resetting AURA PostgreSQL Database...${NC}"

# Database connection parameters
DB_NAME="aura_metrics"
DB_USER="aura"
DB_PASS="aura_password"

# Check if PostgreSQL is running
if ! pg_isready -h localhost -p 5432 > /dev/null 2>&1; then
    echo -e "${YELLOW}PostgreSQL is not running. Starting...${NC}"
    brew services start postgresql@14
    sleep 3
fi

echo -e "${YELLOW}Dropping existing database (if exists)...${NC}"
dropdb -h localhost -U $USER --if-exists $DB_NAME 2>/dev/null || true

echo -e "${YELLOW}Creating database...${NC}"
createdb -h localhost -U $USER $DB_NAME

echo -e "${YELLOW}Creating user...${NC}"
psql -h localhost -U $USER -d postgres -c "DROP USER IF EXISTS $DB_USER;" 2>/dev/null || true
psql -h localhost -U $USER -d postgres -c "CREATE USER $DB_USER WITH PASSWORD '$DB_PASS';"
psql -h localhost -U $USER -d postgres -c "GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;"

echo -e "${YELLOW}Initializing schema...${NC}"
psql -h localhost -U $USER -d $DB_NAME -f "$SCRIPT_DIR/init-db-local.sql"

echo -e "${GREEN}✓ Database initialized successfully${NC}"
echo -e "${BLUE}Connection string: postgres://$DB_USER:$DB_PASS@localhost:5432/$DB_NAME?sslmode=disable${NC}"
