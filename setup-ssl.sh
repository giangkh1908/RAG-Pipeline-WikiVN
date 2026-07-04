#!/bin/bash
# Setup Nginx + SSL for RAG Pipeline
# Usage: ./setup-ssl.sh

set -e

DOMAIN="wikivn.top"

echo "=== Setting up Nginx + SSL for $DOMAIN ==="

# Install Nginx + Certbot
apt update
apt install -y nginx certbot python3-certbot-nginx

# Create Nginx config
cat > /etc/nginx/sites-available/rag << EOF
server {
    listen 80;
    server_name $DOMAIN www.$DOMAIN;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;

        # SSE streaming
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
    }
}
EOF

# Enable site
ln -sf /etc/nginx/sites-available/rag /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

# Test Nginx config
nginx -t

# Restart Nginx
systemctl restart nginx

# Get SSL certificate
certbot --nginx -d $DOMAIN -d www.$DOMAIN --non-interactive --agree-tos --email admin@$DOMAIN

# Auto-renew SSL
echo "0 0,12 * * * root certbot renew --quiet" >> /etc/crontab

echo "=== Setup complete! ==="
echo "Your site is now available at: https://$DOMAIN"
