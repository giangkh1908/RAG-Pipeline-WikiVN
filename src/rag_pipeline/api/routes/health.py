"""Health check endpoint."""

from __future__ import annotations

import os

from fastapi import APIRouter

from rag_pipeline.api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check API and dependencies status.

    Returns:
        HealthResponse with status of Qdrant and LangSmith
    """
    # Check Qdrant
    qdrant_status = "disconnected"
    try:
        import httpx

        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{qdrant_url}/collections")
            if response.status_code == 200:
                qdrant_status = "connected"
    except Exception:
        pass

    # Check LangSmith
    langsmith_status = "enabled" if os.getenv("LANGSMITH_TRACING_V2") == "true" else "disabled"

    # Overall status
    status = "ok" if qdrant_status == "connected" else "degraded"

    return HealthResponse(
        status=status,
        qdrant=qdrant_status,
        langsmith=langsmith_status,
    )
