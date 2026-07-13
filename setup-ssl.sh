#!/bin/bash
# Setup Nginx + SSL for RAG Pipeline
# Usage: ./setup-ssl.sh

set -e

DOMAIN="wikivn.top"
REPO_RAW="https://raw.githubusercontent.com/giangkh1908/RAG-Pipeline-WikiVN/main"

echo "=== Setting up Nginx + SSL for $DOMAIN ==="

# Install Nginx + Certbot
apt update
apt install -y nginx certbot python3-certbot-nginx curl

# Download the latest nginx config from the repo
wget -q -O /etc/nginx/sites-available/rag "${REPO_RAW}/nginx.conf"

# Enable site
ln -sf /etc/nginx/sites-available/rag /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

# Test Nginx config
nginx -t

# Restart Nginx
systemctl restart nginx

# Get SSL certificate
certbot --nginx -d "$DOMAIN" -d "www.$DOMAIN" --non-interactive --agree-tos --email "admin@$DOMAIN"

# Auto-renew SSL (avoid duplicate cron entries)
CRON_LINE="0 0,12 * * * root certbot renew --quiet"
if ! grep -qF "$CRON_LINE" /etc/crontab; then
    echo "$CRON_LINE" >> /etc/crontab
fi

echo "=== Setup complete! ==="
echo "Your site is now available at: https://$DOMAIN"
