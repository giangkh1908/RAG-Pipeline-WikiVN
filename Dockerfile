# Multi-stage Dockerfile: build frontend then run Python backend + serve static files.

# ============================================
# Stage 1: Build frontend
# ============================================
FROM node:22-slim AS frontend-builder

WORKDIR /app/frontend

# Cache npm install layer
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm config set fetch-retries 5 && \
    npm config set fetch-retry-mintimeout 20000 && \
    npm config set fetch-retry-maxtimeout 120000 && \
    npm ci --ignore-scripts

# Only rebuild when source changes
COPY frontend/ .
RUN npm run build

# ============================================
# Stage 2: Python backend + serve frontend
# ============================================
FROM python:3.12-slim AS runtime

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

# Install Python deps matching pyproject.toml
COPY pyproject.toml .
RUN pip install --no-cache-dir --no-compile \
    "python-dotenv>=1.0.0" \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.34.0" \
    "httpx>=0.27.0" \
    "qdrant-client>=1.13.0,<1.14.0" \
    "pydantic>=2.0.0"

# Copy source code
COPY src/ src/

# Copy scripts and raw dataset for deployment initialization.
# The SQLite DB and BM25 vocab are generated on first deploy by the indexer.
COPY scripts/ scripts/
COPY documents/vietnam_tourism_v2.json documents/

# Copy frontend build from the previous stage
COPY --from=frontend-builder /app/frontend/dist frontend/dist

# Set ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Set Python path
ENV PYTHONPATH=src

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "rag_pipeline.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
