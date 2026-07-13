# Hướng dẫn Deploy

Tài liệu này mô tả toàn bộ quá trình deploy hệ thống RAG lên VPS và cách GitHub Actions tự động hóa việc đó.

---

## Kiến trúc triển khai

```text
GitHub (push main)
    │
    ▼
GitHub Actions
    ├── Build Docker image
    └── SSH vào VPS
            │
            ▼
        git pull
        docker compose up -d --build
            │
            ▼
        VPS: Nginx → API:8000
                Qdrant:6333
```

---

## Yêu cầu

### VPS

- OS: Ubuntu 22.04/24.04 (hoặc bất kỳ distro nào chạy được Docker)
- Docker + Docker Compose v2 đã cài
- Port mở: `80`, `443`, `8000` (nếu test trực tiếp)
- Tối thiểu **2 GB RAM**, khuyến nghị **4 GB**
- Disk tối thiểu **20 GB**

### Domain

- Domain trỏ về IP VPS (ví dụ: `wikivn.top` → `<VPS_IP>`)
- DNS đã propagate

### GitHub Secrets

Vào repo → **Settings → Secrets and variables → Actions**, thêm:

| Secret | Mô tả |
|--------|-------|
| `VPS_HOST` | IP hoặc domain VPS |
| `VPS_USER` | User SSH, thường là `root` |
| `SSH_PRIVATE_KEY` | Private key SSH có quyền đăng nhập VPS |

---

## Cài đặt lần đầu trên VPS

### 1. Clone repo

```bash
mkdir -p /opt
rm -rf /opt/rag
git clone https://github.com/giangkh1908/RAG-Pipeline-WikiVN.git /opt/rag
cd /opt/rag
```

### 2. Tạo file `.env`

```bash
cat > /opt/rag/.env <<EOF
OPENROUTER_API_KEY=sk-or-v1-...
QDRANT_URL=http://qdrant:6333
EOF
```

> **Lưu ý:** `OPENROUTER_API_KEY` bắt buộc. Không có key, indexer và API sẽ fail.

### 3. Cài Nginx + SSL

Dùng script có sẵn:

```bash
bash setup-ssl.sh wikivn.top
```

Hoặc cấu hình thủ công:

```bash
apt update
apt install -y nginx certbot python3-certbot-nginx
```

Tạo file `/etc/nginx/sites-available/rag`:

```nginx
server {
    listen 80;
    server_name wikivn.top www.wikivn.top;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }
}
```

Kích hoạt:

```bash
ln -sf /etc/nginx/sites-available/rag /etc/nginx/sites-enabled/
nginx -t
systemctl reload nginx
certbot --nginx -d wikivn.top -d www.wikivn.top
```

### 4. Deploy lần đầu

```bash
cd /opt/rag
docker compose up -d --build
```

Lần đầu sẽ mất **5–15 phút** vì indexer phải:
- Chunk 1386 documents
- Embed 2801 chunks qua OpenRouter
- Index vào Qdrant

Theo dõi tiến trình:

```bash
docker logs rag-indexer -f
```

Khi thấy:

```text
Full ingest complete: 1386 documents, 2801 chunks
Ingested and indexed 2801 chunks into Qdrant.
```

là xong.

---

## Deploy tự động qua GitHub Actions

Sau lần deploy thủ công đầu tiên, mỗi khi push lên `main`, workflow `.github/workflows/deploy.yml` sẽ tự động:

1. Build image trên GitHub Actions
2. SSH vào VPS
3. `git pull`
4. `docker compose up -d --build`
5. Chờ API healthy

### Trigger thủ công

GitHub repo → **Actions → Build and Deploy → Run workflow**

### Kiểm tra deploy thành công

```bash
# Trên VPS
curl http://localhost:8000/api/health
```

Kết quả mong đợi:

```json
{"status": "ok", "qdrant": "connected", "version": "0.2.0"}
```

Mở trình duyệt: `https://wikivn.top`

---

## Các lệnh thường dùng

### Xem log

```bash
docker logs rag-api -f
docker logs rag-indexer -f
docker logs rag-qdrant -f
```

### Restart toàn bộ

```bash
cd /opt/rag
docker compose down
docker compose up -d --build
```

### Xóa dữ liệu và build lại từ đầu

```bash
cd /opt/rag
docker compose down
docker volume rm rag_rag_data rag_qdrant_storage
docker compose up -d --build
```

> Cẩn thận: lệnh này xóa SQLite + Qdrant vectors, phải ingest lại từ đầu.

### Kiểm tra container

```bash
docker ps -a
docker compose ps
```

---

## Troubleshooting

### 1. API không healthy

```bash
curl http://localhost:8000/api/health
docker logs rag-api --tail 50
```

### 2. Qdrant không start

```bash
docker logs rag-qdrant --tail 50
```

Thường do port `6333` bị chiếm hoặc volume permission.

### 3. Indexer fail do SQLite

Lỗi `unable to open database file` thường do volume `rag_data` bị tạo với permission sai. Fix:

```bash
docker compose down
docker volume rm rag_rag_data
docker compose up -d --build
```

### 4. Lỗi rate limit OpenRouter

Nếu thấy log indexer toàn retry/rate limit, chỉ có cách:
- Đợi cho xong
- Hoặc nâng cấp lên gói trả phí OpenRouter

---

## Lưu ý bảo mật

- Không commit `.env` lên GitHub
- Giữ `OPENROUTER_API_KEY` trong GitHub Secrets hoặc trên VPS
- Giới hạn IP SSH nếu có thể
- Đặt firewall chỉ mở `80`, `443`, `22`
