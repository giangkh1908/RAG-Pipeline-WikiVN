# Deployment Guide

Hướng dẫn deploy RAG Pipeline lên VPS với Docker, domain, và SSL.

## Tổng quan kiến trúc

```
User → Cloudflare (DNS + SSL) → Nginx (reverse proxy) → Docker (FastAPI + Qdrant)
```

```
┌──────────────────────────────────────────────────────────────┐
│                          VPS                                 │
│                                                              │
│  ┌─────────────┐    ┌─────────────────────────────────────┐  │
│  │    Nginx     │    │          Docker Compose             │  │
│  │  (port 443)  │    │  ┌──────────────┐  ┌────────────┐  │  │
│  │              │───▶│  │   API + UI    │  │   Qdrant   │  │  │
│  │  wikivn.top  │    │  │  (port 8000)  │  │  (6333)    │  │  │
│  └─────────────┘    │  └──────────────┘  └────────────┘  │  │
│                      └─────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

## Yêu cầu

- VPS: Ubuntu 22.04+, 2GB+ RAM, 20GB+ SSD
- Domain: đã trỏ nameservers về Cloudflare
- GitHub account (cho GHCR)
- API keys: OpenRouter, Cohere, LangSmith (tùy chọn)

---

## Bước 1: Chuẩn bị VPS

### 1.1 SSH vào VPS

```bash
ssh root@<vps-ip>
```

### 1.2 Cài Docker

```bash
curl -fsSL https://get.docker.com | sh
systemctl enable docker
systemctl start docker
```

### 1.3 Tạo thư mục làm việc

```bash
mkdir -p /opt/rag
cd /opt/rag
```

---

## Bước 2: Docker Compose

### 2.1 Tải docker-compose.yml

```bash
wget https://raw.githubusercontent.com/<owner>/<repo>/main/docker-compose.yml
```

Hoặc tạo thủ công:

```yaml
services:
  qdrant:
    image: qdrant/qdrant:v1.13.6
    container_name: rag-qdrant
    ports:
      - "6333:6333"
    volumes:
      - qdrant_storage:/qdrant/storage
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334
    restart: unless-stopped

  api:
    image: ghcr.io/<owner>/rag-pipeline-wikivn:latest
    container_name: rag-api
    ports:
      - "8000:8000"
    env_file:
      - .env
    environment:
      - QDRANT_URL=http://qdrant:6333
    depends_on:
      qdrant:
        condition: service_started
    restart: unless-stopped

volumes:
  qdrant_storage:
```

### 2.2 Tạo file .env

```bash
nano /opt/rag/.env
```

Nội dung:

```env
# Bắt buộc
OPENROUTER_API_KEY=<your-key>

# Qdrant (trong Docker dùng tên service)
QDRANT_URL=http://qdrant:6333

# Cohere re-ranking
COHERE_API_KEY=<your-key>

# LangSmith (tùy chọn)
LANGSMITH_TRACING_V2=true
LANGSMITH_API_KEY=<your-key>
LANGSMITH_PROJECT=rag-pipeline
LANGSMITH_ENDPOINT=https://apac.api.smith.langchain.com
```

---

## Bước 3: Upload Qdrant Snapshot

Snapshot chứa 433,500 vectors (1.1M chunks đã được embed). File ~4GB.

### 3.1 Upload từ máy local

```bash
# Từ máy local (PowerShell)
scp wikipedia_vi.snapshot root@<vps-ip>:/opt/rag/
```

Hoặc dùng `rsync` (hỗ trợ resume khi mất kết nối):

```bash
rsync -avz --progress wikipedia_vi.snapshot root@<vps-ip>:/opt/rag/
```

### 3.2 Restore snapshot

```bash
# SSH vào VPS
ssh root@<vps-ip>

# Start Qdrant trước
cd /opt/rag
docker-compose up -d qdrant

# Chờ Qdrant khởi động (~10s)
sleep 10

# Tạo collection (phải match với snapshot)
curl -X PUT http://localhost:6333/collections/wikipedia_vi_chunks \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": {
      "dense": {
        "size": 2048,
        "distance": "Cosine"
      }
    }
  }'

# Restore snapshot (dùng multipart form, KHÔNG dùng --data-binary)
curl -X POST http://localhost:6333/collections/wikipedia_vi_chunks/snapshots/upload \
  -F "snapshot=@/opt/rag/wikipedia_vi.snapshot"
```

### 3.3 Kiểm tra

```bash
# Kiểm tra collection
curl http://localhost:6333/collections/wikipedia_vi_chunks | python3 -m json.tool

# Expected: "vectors_count": 433500
```

---

## Bước 4: Start Services

```bash
cd /opt/rag

# Login GHCR (cần GitHub Personal Access Token)
echo "<github-token>" | docker login ghcr.io -u <username> --password-stdin

# Pull image và start
docker-compose pull
docker-compose up -d

# Kiểm tra logs
docker-compose logs -f api
```

### Kiểm tra

```bash
# Health check
curl http://localhost:8000/api/health

# Test chat
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Python là gì?"}'
```

---

## Bước 5: Domain + Cloudflare

### 5.1 Cấu hình Cloudflare

1. Đăng nhập [Cloudflare](https://dash.cloudflare.com)
2. Thêm domain `wikivn.top`
3. Cập nhật nameservers tại registrar:
   - `aria.ns.cloudflare.com`
   - `chad.ns.cloudflare.com`
4. Chờ propagate (~5 phút đến 24h)

### 5.2 DNS Records

Thêm trong Cloudflare → DNS → Records:

| Type | Name | Content | Proxy |
|------|------|---------|-------|
| A | @ | `<vps-ip>` | ✅ Proxied |
| A | www | `<vps-ip>` | ✅ Proxied |

### 5.3 SSL/TLS Settings

Vào **SSL/TLS → Overview**:

- **Encryption mode**: Full (không phải Full Strict)
- **Always Use HTTPS**: ON
- **TLS 1.3**: ON
- **Automatic HTTPS Rewrites**: ON
- **Minimum TLS Version**: TLS 1.2

### 5.4 Edge Certificates

Cloudflare tự tạo certificate cho `*.wikivn.top` và `wikivn.top`. Kiểm tra trong **SSL/TLS → Edge Certificates** — trạng thái phải là **Active**.

---

## Bước 6: Nginx + SSL trên VPS

### 6.1 Cài Nginx và Certbot

```bash
apt update
apt install -y nginx certbot python3-certbot-nginx
```

### 6.2 Tạo Nginx config

```bash
nano /etc/nginx/sites-available/wikivn.top
```

Nội dung:

```nginx
server {
    listen 80;
    server_name wikivn.top www.wikivn.top;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # SSE streaming support
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;

        # WebSocket support (nếu cần sau này)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### 6.3 Kích hoạt config

```bash
ln -sf /etc/nginx/sites-available/wikivn.top /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

# Test config
nginx -t

# Reload
systemctl reload nginx
```

### 6.4 Lấy SSL certificate

```bash
certbot --nginx -d wikivn.top -d www.wikivn.top --non-interactive --agree-tos -m your-email@example.com
```

Certbot tự động:
- Lấy certificate từ Let's Encrypt
- Cập nhật Nginx config (thêm SSL)
- Tạo cron job tự renew

### 6.5 Kiểm tra SSL

```bash
# Test HTTPS
curl -I https://wikivn.top

# Kiểm tra certificate
certbot certificates
```

---

## Bước 7: GitHub Actions CD

### 7.1 Secrets

Thêm vào GitHub repo → Settings → Secrets → Actions:

| Secret | Giá trị |
|--------|---------|
| `VPS_HOST` | `<vps-ip>` |
| `VPS_USER` | `root` |
| `VPS_PASSWORD` | Mật khẩu SSH |

### 7.2 Workflow

File `.github/workflows/deploy.yml` đã có sẵn. Mỗi lần push lên `main`:

1. Build Docker image (multi-stage)
2. Push lên GHCR (`ghcr.io/<owner>/rag-pipeline-wikivn:latest`)
3. SSH vào VPS → pull image mới → restart containers

### 7.3 Trigger deploy

```bash
# Từ máy local
git add .
git commit -m "update something"
git push origin main

# GitHub Actions sẽ tự động deploy
```

Theo dõi progress tại GitHub Actions tab trong repo.

---

## Bảo trì

### Xem logs

```bash
ssh root@<vps-ip>
cd /opt/rag

# Logs API
docker-compose logs -f api

# Logs Qdrant
docker-compose logs -f qdrant

# Tất cả
docker-compose logs -f
```

### Restart services

```bash
cd /opt/rag

# Restart tất cả
docker-compose restart

# Restart chỉ API
docker-compose restart api

# Restart chỉ Qdrant
docker-compose restart qdrant
```

### Update image mới

```bash
cd /opt/rag
docker-compose pull
docker-compose up -d
```

### Backup Qdrant

```bash
# Tạo snapshot
curl -X POST http://localhost:6333/collections/wikipedia_vi_chunks/snapshots \
  -o /opt/rag/backup_$(date +%Y%m%d).snapshot

# Download về máy local
scp root@<vps-ip>:/opt/rag/backup_*.snapshot .
```

### Renew SSL

Certbot tự động renew. Kiểm tra:

```bash
certbot renew --dry-run
```

### Disk space

```bash
# Kiểm tra disk usage
df -h

# Docker cleanup
docker system prune -f
docker volume prune -f
```

---

## Troubleshooting

### API không kết nối được Qdrant

```bash
# Kiểm tra Qdrant đang chạy
docker-compose ps

# Kiểm tra Qdrant health
curl http://localhost:6333/healthz

# Kiểm tra network
docker network ls
docker network inspect rag_default
```

### SSL handshake failure

1. Kiểm tra Cloudflare SSL mode = **Full** (không phải Full Strict)
2. Kiểm tra certificate trên VPS:
   ```bash
   ls -la /etc/letsencrypt/live/wikivn.top/
   ```
3. Kiểm tra Nginx config:
   ```bash
   nginx -t
   cat /etc/nginx/sites-enabled/wikivn.top
   ```

### Snapshot restore lỗi

```bash
# Lỗi 415 Unsupported Media Type
# → Phải dùng multipart form:
curl -X POST http://localhost:6333/collections/wikipedia_vi_chunks/snapshots/upload \
  -F "snapshot=@/opt/rag/wikipedia_vi.snapshot"

# KHÔNG dùng:
# curl --data-binary @file  (sẽ lỗi OOM với file 4GB)
```

### Docker build chậm

Dockerfile đã tối ưu layer caching:
- `pyproject.toml` + `pip install` ở layer riêng (ít thay đổi)
- `src/` copy sau (hay thay đổi)
- Frontend build trong stage riêng

### Frontend không load

```bash
# Kiểm tra frontend đã build chưa
docker exec rag-api ls /app/frontend/dist

# Kiểm tra API serve static
curl http://localhost:8000/
```

---

## Chi phí

| Item | Chi phí |
|------|---------|
| VPS (Ubuntu 22.04, 2GB RAM) | ~$5/tháng |
| Domain (.top) | ~$3/năm |
| Cloudflare | Free |
| SSL (Let's Encrypt) | Free |
| OpenRouter API | Free tier |
| Cohere API | Free tier (100 search units/tháng) |
| LangSmith | Free tier |

**Tổng: ~$5/tháng**

---

## URLs

| URL | Mô tả |
|-----|--------|
| https://wikivn.top | Frontend (production) |
| http://<vps-ip>:8000 | API (direct) |
| http://<vps-ip>:6333 | Qdrant dashboard |
| GitHub repo | Source code |
| https://smith.langchain.com | LangSmith dashboard |
