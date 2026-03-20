"""Attack execution routes backed by Celery async jobs."""

# pyright: reportFunctionMemberAccess=false

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from celery.result import AsyncResult
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.routers.models import _load_index
from backend.worker import celery_app


router = APIRouter()


class RunAttackRequest(BaseModel):
    model_id: str
    attacks: list[str] = Field(default_factory=lambda: ["FGSM", "PGD", "C&W"])


@router.post("/run")
async def run_attacks(payload: RunAttackRequest) -> dict:
    """Queue attack execution and return Celery job ID immediately."""
    model_meta = next((m for m in _load_index() if m["model_id"] == payload.model_id), None)
    if model_meta is None:
        raise HTTPException(status_code=404, detail="Model not found")

    model_path = model_meta.get("path", "")
    if not Path(model_path).exists():
        raise HTTPException(status_code=404, detail="Stored model file is missing")

    if not payload.attacks:
        raise HTTPException(status_code=400, detail="At least one attack must be selected")

    celery_runtime = cast(Any, celery_app)
    job = celery_runtime.send_task(
        "run_attack_job",
        args=[payload.model_id, model_path, payload.attacks],
    )
    return {"job_id": job.id, "status": "QUEUED"}


@router.get("/status/{job_id}")
async def get_attack_status(job_id: str) -> dict:
    """Return current attack-job status and result when complete."""
    result = AsyncResult(job_id, app=celery_app)

    if result.state in {"PENDING", "RECEIVED", "STARTED", "ATTACKING"}:
        info = result.info if isinstance(result.info, dict) else {}
        return {
            "status": result.state,
            "progress": int(info.get("progress", 0)),
            "current_attack": info.get("current_attack"),
        }

    if result.state == "FAILURE":
        return {"status": "FAILURE", "error": str(result.result)}

    return {
        "status": "SUCCESS",
        "progress": 100,
        "result": result.result,
    }
