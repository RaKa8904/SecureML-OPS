"""Perturbation visualizer — generates comparison images for reports."""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def plot_perturbation_comparison(
    original: np.ndarray,
    adversarial: np.ndarray,
    attack_name: str,
    save_path: str | Path | None = None,
) -> bytes:
    """Generate a side-by-side comparison: Original | Perturbation | Adversarial.

    Parameters
    ----------
    original : np.ndarray
        Single image, shape ``(C, H, W)`` or ``(H, W)`` in [0, 1].
    adversarial : np.ndarray
        Corresponding adversarial image, same shape.
    attack_name : str
        Name shown in the figure title.
    save_path : str | Path | None
        If provided, save the figure to this path as PNG.

    Returns
    -------
    bytes
        PNG image as bytes (useful for MLflow artifacts / API responses).
    """
    orig_disp = _to_display(original)
    adv_disp = _to_display(adversarial)
    diff = np.abs(adv_disp - orig_disp)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    cmap = "gray" if orig_disp.ndim == 2 else None

    axes[0].imshow(orig_disp, cmap=cmap, vmin=0, vmax=1)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(diff, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title("Perturbation (amplified)")
    axes[1].axis("off")

    axes[2].imshow(adv_disp, cmap=cmap, vmin=0, vmax=1)
    axes[2].set_title("Adversarial")
    axes[2].axis("off")

    fig.suptitle(f"Attack: {attack_name}", fontsize=13, fontweight="bold")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    png_bytes = buf.read()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(save_path).write_bytes(png_bytes)

    return png_bytes


def plot_score_breakdown(
    breakdown: dict[str, dict],
    score: float,
    severity: str,
    save_path: str | Path | None = None,
) -> bytes:
    """Bar chart showing per-attack accuracy drop + overall score.

    Parameters
    ----------
    breakdown : dict[str, dict]
        Attack name → result dict (must have ``clean_accuracy``, ``adv_accuracy``).
    score : float
        Overall robustness score (0–100).
    severity : str
        Severity label (CRITICAL / HIGH / MODERATE / STRONG / EXCELLENT).
    save_path : str | Path | None
        If provided, save figure to this path.

    Returns
    -------
    bytes
        PNG image as bytes.
    """
    attacks = list(breakdown.keys())
    clean_accs = [breakdown[a]["clean_accuracy"] * 100 for a in attacks]
    adv_accs = [breakdown[a]["adv_accuracy"] * 100 for a in attacks]

    x = np.arange(len(attacks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, clean_accs, width, label="Clean Accuracy", color="#4CAF50")
    ax.bar(x + width / 2, adv_accs, width, label="Adversarial Accuracy", color="#F44336")

    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Robustness Score: {score} ({severity})")
    ax.set_xticks(x)
    ax.set_xticklabels(attacks, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 105)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    png_bytes = buf.read()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(save_path).write_bytes(png_bytes)

    return png_bytes


def _to_display(img: np.ndarray) -> np.ndarray:
    """Convert (C, H, W) → (H, W) for grayscale or (H, W, C) for colour."""
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    if img.ndim == 3 and img.shape[2] == 1:
        img = img.squeeze(axis=2)
    return np.clip(img, 0, 1)
