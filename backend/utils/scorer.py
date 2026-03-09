"""Robustness scoring engine for SecureML Ops."""

from __future__ import annotations

import numpy as np

ATTACK_WEIGHTS: dict[str, float] = {
    "FGSM": 0.10,
    "PGD": 0.20,
    "C&W": 0.20,
    "Transfer": 0.20,
    "HopSkipJump": 0.15,
    "Square": 0.15,
}


def compute_robustness_score(
    results: dict[str, dict],
    clean_accuracy: float,
) -> dict:
    """Compute a weighted robustness score from attack results.

    Parameters
    ----------
    results : dict[str, dict]
        Mapping of attack name → attack result dict.  Each value must
        contain at least ``"adv_accuracy"`` (float).
    clean_accuracy : float
        Overall clean accuracy of the model (0–1).

    Returns
    -------
    dict
        ``{"score": float, "severity": str, "breakdown": dict}``
    """
    if clean_accuracy <= 0:
        return {"score": 0.0, "severity": "CRITICAL", "breakdown": results}

    weighted_sum = sum(
        (results[attack]["adv_accuracy"] / clean_accuracy) * weight
        for attack, weight in ATTACK_WEIGHTS.items()
        if attack in results
    )

    # Normalise: if only a subset of attacks ran, scale to full weight
    total_weight = sum(w for a, w in ATTACK_WEIGHTS.items() if a in results)
    if total_weight > 0:
        weighted_sum = weighted_sum / total_weight

    score = round(weighted_sum * 100, 1)

    if score < 30:
        severity = "CRITICAL"
    elif score < 50:
        severity = "HIGH"
    elif score < 70:
        severity = "MODERATE"
    elif score < 85:
        severity = "STRONG"
    else:
        severity = "EXCELLENT"

    return {"score": score, "severity": severity, "breakdown": results}
