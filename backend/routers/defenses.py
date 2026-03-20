"""Defenses router placeholders for Phase 3."""

from __future__ import annotations

from fastapi import APIRouter


router = APIRouter()


@router.get("")
async def list_defenses() -> dict[str, list[str]]:
    """List available defense modules."""
    return {
        "defenses": [
            "adversarial_training",
            "preprocessing",
            "randomized_smoothing",
        ]
    }
