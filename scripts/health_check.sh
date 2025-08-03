#!/bin/bash
# Health check script

HEALTH_URL="http://localhost:8000/health"

# Check if service is running
if ! systemctl is-active --quiet car-depreciation; then
    echo "Service is not running. Attempting restart..."
    sudo systemctl restart car-depreciation
    sleep 5
fi

# Check health endpoint
if curl -f $HEALTH_URL > /dev/null 2>&1; then
    echo "Health check passed"
    exit 0
else
    echo "Health check failed"
    # Try to restart service
    sudo systemctl restart car-depreciation
    sleep 5
    
    # Check again
    if curl -f $HEALTH_URL > /dev/null 2>&1; then
        echo "Service recovered after restart"
        exit 0
    else
        echo "Service failed to recover"
        exit 1
    fi
fi