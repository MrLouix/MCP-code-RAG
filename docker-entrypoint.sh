#!/bin/bash
set -e

# Configuration
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
HEALTH_CHECK_TIMEOUT=60
HEALTH_CHECK_INTERVAL=2
ELAPSED=0

echo "Waiting for Ollama service to be ready at $OLLAMA_URL..."

# Health check loop for Ollama
while [ $ELAPSED -lt $HEALTH_CHECK_TIMEOUT ]; do
    if curl -sf "$OLLAMA_URL/api/tags" > /dev/null 2>&1; then
        echo "✓ Ollama service is ready!"
        break
    fi

    ELAPSED=$((ELAPSED + HEALTH_CHECK_INTERVAL))
    echo "Ollama not ready yet... ($ELAPSED/$HEALTH_CHECK_TIMEOUT seconds)"
    sleep $HEALTH_CHECK_INTERVAL
done

# Check if timeout was reached
if [ $ELAPSED -ge $HEALTH_CHECK_TIMEOUT ]; then
    echo "✗ Ollama service did not respond within ${HEALTH_CHECK_TIMEOUT} seconds"
    exit 1
fi

echo "Starting MCP Code RAG server..."
exec python -m mcp_code_rag.server
