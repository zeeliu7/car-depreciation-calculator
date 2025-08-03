#!/bin/bash
# Deployment script for car depreciation system

set -e

echo "Starting deployment..."

# Navigate to application directory
cd /opt/car-depreciation

# Pull latest changes
echo "Pulling latest changes from GitHub..."
git pull origin main

# Install/update dependencies
echo "Installing dependencies..."
source venv/bin/activate
pip install -r requirements.txt

# Restart services
echo "Restarting services..."
sudo systemctl restart car-depreciation

# Wait for service to start
sleep 3

# Check service status
echo "Checking service status..."
sudo systemctl status car-depreciation --no-pager

# Test health endpoint
echo "Testing health endpoint..."
curl -f http://localhost:8000/health || echo "Health check failed"

echo "Deployment completed!"