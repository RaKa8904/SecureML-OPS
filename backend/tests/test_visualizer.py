"""Unit tests for the perturbation visualizer."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from backend.utils.visualizer import (
    plot_perturbation_comparison,
    plot_score_breakdown,
)


class TestPerturbationComparison:
    def test_returns_png_bytes(self) -> None:
        orig = np.random.rand(1, 28, 28).astype(np.float32)
        adv = np.clip(orig + 0.1, 0, 1)
        result = plot_perturbation_comparison(orig, adv, "FGSM")
        assert isinstance(result, bytes)
        assert result[:4] == b"\x89PNG"

    def test_saves_to_file(self, tmp_path: Path) -> None:
        orig = np.random.rand(1, 28, 28).astype(np.float32)
        adv = np.clip(orig + 0.1, 0, 1)
        out = tmp_path / "test.png"
        plot_perturbation_comparison(orig, adv, "PGD", save_path=str(out))
        assert out.exists()
        assert out.stat().st_size > 0

    def test_handles_colour_images(self) -> None:
        orig = np.random.rand(3, 32, 32).astype(np.float32)
        adv = np.clip(orig + 0.05, 0, 1)
        result = plot_perturbation_comparison(orig, adv, "Square")
        assert result[:4] == b"\x89PNG"


class TestScoreBreakdown:
    def test_returns_png_bytes(self) -> None:
        breakdown = {
            "FGSM": {"clean_accuracy": 0.95, "adv_accuracy": 0.50},
            "PGD": {"clean_accuracy": 0.95, "adv_accuracy": 0.30},
        }
        result = plot_score_breakdown(breakdown, score=42.1, severity="HIGH")
        assert isinstance(result, bytes)
        assert result[:4] == b"\x89PNG"

    def test_saves_to_file(self, tmp_path: Path) -> None:
        breakdown = {
            "FGSM": {"clean_accuracy": 0.95, "adv_accuracy": 0.50},
        }
        out = tmp_path / "score.png"
        plot_score_breakdown(breakdown, 52.6, "MODERATE", save_path=str(out))
        assert out.exists()
