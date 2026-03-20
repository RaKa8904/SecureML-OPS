"""Model management routes."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile


router = APIRouter()

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "storage" / "models"
INDEX_PATH = MODELS_DIR / "index.json"
ALLOWED_EXTENSIONS = {".pt", ".pth", ".onnx", ".h5"}


def _load_index() -> list[dict]:
    if not INDEX_PATH.exists():
        return []
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))


def _save_index(items: list[dict]) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(items, indent=2), encoding="utf-8")


@router.post("/upload")
async def upload_model(file: UploadFile = File(...)) -> dict:
    """Upload a model artifact and register it in storage index."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported model format. Use .pt/.pth first; .onnx/.h5 are supported as secondary formats."
            ),
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_id = str(uuid.uuid4())
    target_name = f"{model_id}_{Path(file.filename or 'model').name}"
    target_path = MODELS_DIR / target_name

    content = await file.read()
    target_path.write_bytes(content)

    item = {
        "model_id": model_id,
        "filename": Path(file.filename or "model").name,
        "path": str(target_path),
        "format": suffix,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }

    index = _load_index()
    index.append(item)
    _save_index(index)

    return item


@router.get("")
async def list_models() -> dict[str, list[dict]]:
    """List uploaded models."""
    return {"models": _load_index()}


@router.get("/{model_id}")
async def get_model(model_id: str) -> dict:
    """Get a model metadata record by ID."""
    for item in _load_index():
        if item["model_id"] == model_id:
            return item
    raise HTTPException(status_code=404, detail="Model not found")
