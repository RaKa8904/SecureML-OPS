"""Phase 3 API smoke tests (models + attacks async endpoints)."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_health() -> None:
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_upload_and_list_models(tmp_path, monkeypatch) -> None:
    from backend.routers import models

    models_dir = tmp_path / "models"
    index_path = models_dir / "index.json"
    monkeypatch.setattr(models, "MODELS_DIR", models_dir)
    monkeypatch.setattr(models, "INDEX_PATH", index_path)

    payload = b"dummy-model-bytes"
    files = {"file": ("toy_model.pt", payload, "application/octet-stream")}
    up = client.post("/api/models/upload", files=files)

    assert up.status_code == 200
    uploaded = up.json()
    assert uploaded["format"] == ".pt"
    assert Path(uploaded["path"]).exists()

    ls = client.get("/api/models")
    assert ls.status_code == 200
    assert len(ls.json()["models"]) == 1


def test_run_attacks_returns_job_id(monkeypatch, tmp_path) -> None:
    from backend.routers import attacks

    model_file = tmp_path / "model.pt"
    model_file.write_bytes(b"x")

    monkeypatch.setattr(
        attacks,
        "_load_index",
        lambda: [{"model_id": "m1", "path": str(model_file)}],
    )

    class _FakeJob:
        id = "job-123"

    monkeypatch.setattr(attacks.celery_app, "send_task", lambda *a, **k: _FakeJob())

    res = client.post("/api/attacks/run", json={"model_id": "m1", "attacks": ["FGSM"]})
    assert res.status_code == 200
    assert res.json() == {"job_id": "job-123", "status": "QUEUED"}


def test_attack_status_success(monkeypatch) -> None:
    from backend.routers import attacks

    class _FakeAsyncResult:
        def __init__(self, *_args, **_kwargs) -> None:
            self.state = "SUCCESS"
            self.result = {"score": 42.0}
            self.info = None

    monkeypatch.setattr(attacks, "AsyncResult", _FakeAsyncResult)

    res = client.get("/api/attacks/status/job-123")
    assert res.status_code == 200
    assert res.json()["status"] == "SUCCESS"
    assert res.json()["result"]["score"] == 42.0


def test_attack_status_in_progress(monkeypatch) -> None:
    from backend.routers import attacks

    class _FakeAsyncResult:
        def __init__(self, *_args, **_kwargs) -> None:
            self.state = "ATTACKING"
            self.result = None
            self.info = {"progress": 55, "current_attack": "PGD"}

    monkeypatch.setattr(attacks, "AsyncResult", _FakeAsyncResult)

    res = client.get("/api/attacks/status/job-123")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ATTACKING"
    assert body["progress"] == 55
    assert body["current_attack"] == "PGD"
