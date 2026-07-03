"""FastAPI application — RAG Pipeline API.

Run with:
    python -m rag_pipeline.api.app
    uvicorn rag_pipeline.api.app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

# Load .env before anything else
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent.parent / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag_pipeline.api.routes import chat, eval, health


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    print("[STARTUP] RAG Pipeline API starting...")
    print("          Docs: http://localhost:8000/docs")
    yield
    # Shutdown
    print("[SHUTDOWN] RAG Pipeline API shutting down...")


app = FastAPI(
    title="RAG Pipeline API",
    description="Vietnamese Wikipedia RAG pipeline — ask questions, get answers with citations.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow frontend to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routes
app.include_router(health.router)
app.include_router(chat.router)
app.include_router(eval.router)


@app.get("/")
async def root():
    """API root — redirect to docs."""
    return {
        "name": "RAG Pipeline API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/api/health",
    }


if __name__ == "__main__":
    import uvicorn

    # Fix Windows encoding
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    uvicorn.run(
        "rag_pipeline.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
