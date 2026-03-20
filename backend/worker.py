"""Celery worker and async attack execution pipeline."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from celery import Celery
from torchvision import datasets, transforms

from backend.attacks.cw import CW
from backend.attacks.fgsm import FGSM
from backend.attacks.hopskipjump import HopSkipJump
from backend.attacks.pgd import PGD
from backend.attacks.square import Square
from backend.attacks.transfer import Transfer
from backend.utils.defense_advisor import recommend_defenses
from backend.utils.scorer import compute_robustness_score
from backend.utils.tracker import log_attack_run


BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", BROKER_URL)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
os.environ.setdefault("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)

celery_app = Celery("secureml_ops", broker=BROKER_URL, backend=RESULT_BACKEND)


ATTACK_REGISTRY = {
    "FGSM": FGSM,
    "PGD": PGD,
    "C&W": CW,
    "Transfer": Transfer,
    "HopSkipJump": HopSkipJump,
    "Square": Square,
}


def _load_model(model_path: str) -> nn.Module:
    """Load a PyTorch model artifact from disk."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        model = torch.jit.load(str(path), map_location="cpu")
        model.eval()
        return model
    except Exception:
        pass

    loaded = torch.load(str(path), map_location="cpu")

    if isinstance(loaded, nn.Module):
        loaded.eval()
        return loaded

    if isinstance(loaded, dict) and isinstance(loaded.get("model"), nn.Module):
        model = loaded["model"]
        model.eval()
        return model

    raise ValueError(
        "Unsupported .pt/.pth payload. Upload TorchScript (.pt) or full nn.Module checkpoint."
    )


def _load_eval_data(batch_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Load MNIST batch for Phase 1/3 attack evaluation."""
    ds = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
    images, labels = next(iter(loader))
    return images.numpy(), labels.numpy()


@celery_app.task(bind=True, name="run_attack_job")
def run_attack_job(self, model_id: str, model_path: str, attack_names: list[str]) -> dict:
    """Run selected attacks, compute robustness score, and emit recommendations."""
    self.update_state(state="ATTACKING", meta={"progress": 10, "current_attack": "loading"})

    model = _load_model(model_path)
    x, y = _load_eval_data()

    results: dict[str, dict] = {}
    total = max(len(attack_names), 1)

    for i, attack_name in enumerate(attack_names, start=1):
        if attack_name not in ATTACK_REGISTRY:
            continue

        self.update_state(
            state="ATTACKING",
            meta={
                "progress": min(90, 10 + int((i - 1) / total * 80)),
                "current_attack": attack_name,
            },
        )

        attack = ATTACK_REGISTRY[attack_name]()
        attack_result = attack.run(model, (x, y))
        results[attack_name] = attack_result

    clean_accuracy = float(np.mean([r["clean_accuracy"] for r in results.values()])) if results else 0.0
    score = compute_robustness_score(results, clean_accuracy)
    recommendations = recommend_defenses(results)

    for attack_name, result in results.items():
        log_attack_run(
            model_id=model_id,
            model_name=Path(model_path).name,
            attack_result=result,
            robustness_score=score["score"],
            sample_original=x[0],
        )

    self.update_state(state="SUCCESS", meta={"progress": 100, "current_attack": "done"})

    return {
        "score": score["score"],
        "severity": score["severity"],
        "breakdown": results,
        "recommendations": recommendations,
    }
