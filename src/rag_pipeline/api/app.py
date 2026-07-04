from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.routing import Mount

from rag_pipeline.api.routes import chat, health, eval

# Path to frontend build output
FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent.parent / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[STARTUP] RAG Pipeline API starting...")
    yield
    print("[SHUTDOWN] RAG Pipeline API shutting down.")


app = FastAPI(
    title="Vietnamese Wikipedia RAG API",
    version="1.0.0",
    description="API cho RAG pipeline hỏi đáp Wikipedia tiếng Việt",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://103.82.25.191",
        "https://103.82.25.191",
        "https://wikivn.top",
        "https://www.wikivn.top",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(eval.router, prefix="/api", tags=["eval"])


@app.get("/api/info")
async def api_info():
    return {
        "name": "Vietnamese Wikipedia RAG API",
        "version": "1.0.0",
        "docs": "/docs",
    }


# Serve frontend static files
if FRONTEND_DIR.exists():
    # Mount assets directory
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="static-assets")

    # SPA catch-all: must use add_route AFTER all API routes
    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        """Catch-all: serve frontend files or index.html for SPA routing."""
        file_path = FRONTEND_DIR / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(FRONTEND_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("rag_pipeline.api.app:app", host="0.0.0.0", port=8000, reload=True)
