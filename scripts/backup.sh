#!/bin/bash
# Backup script for car depreciation system

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/opt/car-depreciation/backups"
S3_BUCKET=$(grep S3_BUCKET_NAME /opt/car-depreciation/.env | cut -d '=' -f2)

mkdir -p $BACKUP_DIR

echo "Starting backup process..."

# Backup S3 data
echo "Backing up S3 data..."
aws s3 sync s3://$S3_BUCKET $BACKUP_DIR/s3_backup_$DATE/

# Backup application files
echo "Backing up application files..."
tar -czf $BACKUP_DIR/app_backup_$DATE.tar.gz /opt/car-depreciation/src/ /opt/car-depreciation/.env

# Clean old backups (keep last 7 days)
echo "Cleaning old backups..."
find $BACKUP_DIR -type f -mtime +7 -delete
find $BACKUP_DIR -type d -empty -delete

echo "Backup completed: $BACKUP_DIR/app_backup_$DATE.tar.gz"