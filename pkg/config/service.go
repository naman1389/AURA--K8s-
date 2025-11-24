package config

import (
	"fmt"
	"os"
	"strings"
)

// GetServiceURL returns service URL with proper host resolution
// Supports both local development and containerized environments
func GetServiceURL(serviceName, defaultPort string) string {
	// Check for explicit URL first (highest priority)
	envKey := fmt.Sprintf("%s_URL", serviceName)
	if url := os.Getenv(envKey); url != "" {
		return url
	}

	// Build from components
	hostKey := fmt.Sprintf("%s_HOST", serviceName)
	portKey := fmt.Sprintf("%s_PORT", serviceName)
	
	host := os.Getenv(hostKey)
	port := os.Getenv(portKey)

	if host == "" {
		// Auto-detect environment
		env := os.Getenv("ENVIRONMENT")
		kubernetesHost := os.Getenv("KUBERNETES_SERVICE_HOST")
		
		if env == "production" || kubernetesHost != "" {
			// In K8s/Docker, use service name
			// Convert SERVICE_NAME to service-name format
			host = convertToServiceName(serviceName)
		} else {
			// Local development
			host = "localhost"
		}
	}

	if port == "" {
		port = defaultPort
	}

	return fmt.Sprintf("http://%s:%s", host, port)
}

// GetDatabaseURL returns database connection string with proper host resolution
func GetDatabaseURL() string {
	dbURL := os.Getenv("DATABASE_URL")
	if dbURL != "" {
		return dbURL
	}

	// Build from components
	env := os.Getenv("ENVIRONMENT")
	if env == "production" {
		// In production, DATABASE_URL should be set
		// But we'll still try to build it
	}

	user := os.Getenv("POSTGRES_USER")
	password := os.Getenv("POSTGRES_PASSWORD")
	host := os.Getenv("POSTGRES_HOST")
	port := os.Getenv("POSTGRES_PORT")
	database := os.Getenv("POSTGRES_DB")

	// Set defaults
	if user == "" {
		user = "aura"
	}
	if password == "" {
		password = "aura_password"
	}
	if host == "" {
		// Auto-detect environment
		kubernetesHost := os.Getenv("KUBERNETES_SERVICE_HOST")
		if env == "production" || kubernetesHost != "" {
			host = "timescaledb" // Docker service name
		} else {
			host = "localhost"
		}
	}
	if port == "" {
		port = "5432"
	}
	if database == "" {
		database = "aura_metrics"
	}

	return fmt.Sprintf("postgresql://%s:%s@%s:%s/%s?sslmode=disable", user, password, host, port, database)
}

// convertToServiceName converts SERVICE_NAME to service-name format
// Examples: ML_SERVICE -> ml-service, MCP_SERVER -> mcp-server
func convertToServiceName(name string) string {
	// Convert to lowercase and replace underscores with hyphens
	result := strings.ToLower(name)
	result = strings.ReplaceAll(result, "_", "-")
	return result
}

