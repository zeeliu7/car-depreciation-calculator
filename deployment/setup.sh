#!/bin/bash
# Initial server setup script

set -e

echo "Setting up Car Depreciation System server..."

# Update system
echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install dependencies
echo "Installing system dependencies..."
sudo apt install python3 python3-pip python3-venv git nginx certbot python3-certbot-nginx htop curl -y

# Install AWS CLI
echo "Installing AWS CLI..."
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
rm -rf aws awscliv2.zip

# Create application directory
echo "Setting up application directory..."
sudo mkdir -p /opt/car-depreciation
sudo chown ubuntu:ubuntu /opt/car-depreciation

echo "Server setup completed!"
echo "Next steps:"
echo "1. Clone your repository to /opt/car-depreciation"
echo "2. Create .env file with your configuration"
echo "3. Run the deployment script"