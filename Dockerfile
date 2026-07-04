# ============================================
# Stage 1: Build frontend (cached separately)
# ============================================
FROM node:22-alpine AS frontend-builder

WORKDIR /app/frontend

# Cache npm install layer
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --ignore-scripts

# Only rebuild when source changes
COPY frontend/ .
RUN npm run build

# ============================================
# Stage 2: Python backend + serve frontend
# ============================================
FROM python:3.12-slim AS runtime

WORKDIR /app

# Install system deps (rarely changes)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (rarely changes)
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

# Install Python deps (cached layer - only changes when pyproject.toml changes)
COPY pyproject.toml .
RUN pip install --no-cache-dir --no-compile \
    "python-dotenv>=1.0.0" \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.34.0" \
    "httpx>=0.27.0" \
    "qdrant-client>=1.13.0,<1.14.0" \
    "langsmith>=0.9.0"

# Copy source code (changes most often)
COPY src/ src/

# Copy frontend build (from cached stage)
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
