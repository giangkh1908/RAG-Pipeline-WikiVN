"""FastAPI application serving the RAG API and frontend."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from rag_pipeline.api.dependencies import PipelineStore
from rag_pipeline.api.routes import chat, health

# Load environment variables from project root .env
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent.parent / ".env")

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent.parent / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    """Manage the shared RAG pipeline lifecycle."""
    print("[STARTUP] RAG Pipeline API starting...")
    try:
        yield
    finally:
        print("[SHUTDOWN] RAG Pipeline API shutting down...")
        PipelineStore.close()


app = FastAPI(
    title="Vietnam Tourism RAG API",
    version="0.3.0",
    description="API cho RAG pipeline hỏi đáp du lịch Việt Nam",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(chat.router, prefix="/api", tags=["chat"])


@app.get("/api/info")
async def api_info() -> dict[str, str]:
    """Basic API metadata."""
    return {
        "name": "Vietnam Tourism RAG API",
        "version": "0.2.0",
        "docs": "/docs",
    }


# Serve frontend static files when built
if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="static-assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str) -> FileResponse:
        """Catch-all: serve frontend files or index.html for SPA routing."""
        file_path = FRONTEND_DIR / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(FRONTEND_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("rag_pipeline.api.app:app", host="0.0.0.0", port=8000, reload=True)
