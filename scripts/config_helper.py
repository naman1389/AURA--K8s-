#!/usr/bin/env python3
"""
Configuration helper for AURA K8s services
Provides environment-aware service discovery and database URL resolution
"""

import os
from typing import Optional


def get_service_url(service_name: str, default_port: str) -> str:
    """
    Get service URL with proper host resolution.
    Supports both local development and containerized environments.
    
    Args:
        service_name: Service name (e.g., "ML_SERVICE", "MCP_SERVER")
        default_port: Default port number as string
        
    Returns:
        Service URL (e.g., "http://localhost:8001" or "http://ml-service:8001")
    """
    # Check for explicit URL first (highest priority)
    env_key = f"{service_name}_URL"
    url = os.getenv(env_key)
    if url:
        return url
    
    # Build from components
    host_key = f"{service_name}_HOST"
    port_key = f"{service_name}_PORT"
    
    host = os.getenv(host_key)
    port = os.getenv(port_key, default_port)
    
    if not host:
        # Auto-detect environment
        env = os.getenv("ENVIRONMENT", "development")
        kubernetes_host = os.getenv("KUBERNETES_SERVICE_HOST")
        
        if env == "production" or kubernetes_host:
            # In K8s/Docker, use service name
            # Convert SERVICE_NAME to service-name format
            host = service_name.lower().replace("_", "-")
        else:
            # Local development
            host = "localhost"
    
    return f"http://{host}:{port}"


def get_database_url() -> str:
    """
    Get database connection string with proper host resolution.
    Supports both local development and containerized environments.
    
    Returns:
        Database connection string
    """
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        return db_url
    
    # Build from components
    env = os.getenv("ENVIRONMENT", "development")
    
    user = os.getenv("POSTGRES_USER", "aura")
    password = os.getenv("POSTGRES_PASSWORD", "aura_password")
    host = os.getenv("POSTGRES_HOST")
    port = os.getenv("POSTGRES_PORT", "5432")
    database = os.getenv("POSTGRES_DB", "aura_metrics")
    
    if not host:
        # Auto-detect environment
        kubernetes_host = os.getenv("KUBERNETES_SERVICE_HOST")
        if env == "production" or kubernetes_host:
            host = "timescaledb"  # Docker service name
        else:
            host = "localhost"
    
    return f"postgresql://{user}:{password}@{host}:{port}/{database}?sslmode=disable"


def validate_database_url(db_url: Optional[str] = None) -> bool:
    """
    Validate database URL format.
    
    Args:
        db_url: Database URL to validate (uses get_database_url() if None)
        
    Returns:
        True if valid, False otherwise
    """
    if not db_url:
        db_url = get_database_url()
    
    return db_url.startswith(("postgresql://", "postgres://"))


def validate_service_url(url: str) -> bool:
    """
    Validate service URL format.
    
    Args:
        url: Service URL to validate
        
    Returns:
        True if valid, False otherwise
    """
    return url.startswith(("http://", "https://"))

