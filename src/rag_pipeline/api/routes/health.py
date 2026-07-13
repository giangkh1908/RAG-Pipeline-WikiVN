"""Health check endpoint."""

from __future__ import annotations

import os

import httpx
from fastapi import APIRouter

from rag_pipeline.api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check API and Qdrant connectivity."""
    qdrant_status = "disconnected"
    try:
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{qdrant_url}/collections")
            if response.status_code == 200:
                qdrant_status = "connected"
    except Exception:
        pass

    status = "ok" if qdrant_status == "connected" else "degraded"
    return HealthResponse(status=status, qdrant=qdrant_status)
