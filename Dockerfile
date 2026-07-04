# ============================================
# Stage 1: Build frontend
# ============================================
FROM node:22-alpine AS frontend-builder

WORKDIR /app/frontend

COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci

COPY frontend/ .
RUN npm run build

# ============================================
# Stage 2: Python backend + serve frontend
# ============================================
FROM python:3.12-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY pyproject.toml .
RUN pip install --no-cache-dir \
    "python-dotenv>=1.0.0" \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.34.0" \
    "httpx>=0.27.0" \
    "qdrant-client>=1.13.0,<1.14.0" \
    "langsmith>=0.9.0" \
    "cohere>=5.13.0" \
    "datasets>=2.20.0"

# Copy source code
COPY src/ src/

# Copy frontend build
COPY --from=frontend-builder /app/frontend/dist frontend/dist

# Set Python path
ENV PYTHONPATH=src

EXPOSE 8000

CMD ["uvicorn", "rag_pipeline.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
