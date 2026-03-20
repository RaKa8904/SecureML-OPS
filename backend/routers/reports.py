"""Reports router placeholders for Phase 3."""

from __future__ import annotations

from fastapi import APIRouter


router = APIRouter()


@router.get("")
async def list_reports() -> dict[str, list[dict]]:
    """Return available reports (empty until Phase 4/5 wiring)."""
    return {"reports": []}
