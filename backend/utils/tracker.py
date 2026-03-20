"""MLflow tracking helpers for attack/defense runs."""

from __future__ import annotations

import tempfile
from pathlib import Path

import mlflow
import numpy as np

from backend.utils.visualizer import plot_perturbation_comparison


def log_attack_run(
    model_id: str,
    model_name: str,
    attack_result: dict,
    robustness_score: float,
    sample_original: np.ndarray | None = None,
) -> None:
    """Log one attack run to MLflow with optional perturbation artifact."""
    attack_name = attack_result.get("attack", "unknown")

    with mlflow.start_run(run_name=f"{model_name}-{attack_name}", nested=True):
        mlflow.log_params(
            {
                "model_id": model_id,
                "model_name": model_name,
                "attack_type": attack_name,
                "epsilon": attack_result.get("epsilon", 0.0),
            }
        )
        mlflow.log_metrics(
            {
                "clean_accuracy": float(attack_result.get("clean_accuracy", 0.0)),
                "adv_accuracy": float(attack_result.get("adv_accuracy", 0.0)),
                "robustness_score": float(robustness_score),
            }
        )

        if sample_original is not None:
            x_adv = attack_result.get("x_adv")
            if isinstance(x_adv, np.ndarray) and len(x_adv) > 0:
                png = plot_perturbation_comparison(
                    original=sample_original,
                    adversarial=x_adv[0],
                    attack_name=attack_name,
                )
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp.write(png)
                    tmp_path = Path(tmp.name)
                try:
                    mlflow.log_artifact(str(tmp_path), artifact_path="perturbations")
                finally:
                    if tmp_path.exists():
                        tmp_path.unlink()
