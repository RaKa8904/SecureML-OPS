"""FastAPI entrypoint for SecureML Ops backend."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers.attacks import router as attacks_router
from backend.routers.defenses import router as defenses_router
from backend.routers.models import router as models_router
from backend.routers.reports import router as reports_router


app = FastAPI(
    title="SecureML Ops API",
    version="0.1.0",
    description="Adversarial robustness testing platform backend.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(models_router, prefix="/api/models", tags=["models"])
app.include_router(attacks_router, prefix="/api/attacks", tags=["attacks"])
app.include_router(defenses_router, prefix="/api/defenses", tags=["defenses"])
app.include_router(reports_router, prefix="/api/reports", tags=["reports"])


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}
