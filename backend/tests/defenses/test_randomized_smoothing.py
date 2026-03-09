"""Tests for randomized smoothing defense."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from backend.defenses.randomized_smoothing import (
    build_smoothed_classifier,
    certify_predictions,
)


class _TinyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.fc = nn.Linear(4 * 4 * 4, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x).view(x.size(0), -1))


@pytest.fixture()
def tiny_model() -> nn.Module:
    torch.manual_seed(42)
    model = _TinyCNN()
    model.eval()
    return model


@pytest.fixture()
def small_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    x = rng.random((8, 1, 28, 28), dtype=np.float32)
    y = rng.integers(0, 10, size=8)
    return x, y


class TestBuildSmoothedClassifier:
    def test_returns_smoothed_classifier(self, tiny_model):
        smoothed = build_smoothed_classifier(
            tiny_model, input_shape=(1, 28, 28), sample_size=10
        )
        assert hasattr(smoothed, "predict")
        assert hasattr(smoothed, "certify")

    def test_predictions_shape(self, tiny_model, small_data):
        smoothed = build_smoothed_classifier(
            tiny_model, input_shape=(1, 28, 28), sample_size=10
        )
        x, _ = small_data
        preds = smoothed.predict(x)
        assert preds.shape == (8, 10)


class TestCertifyPredictions:
    def test_returns_expected_keys(self, tiny_model, small_data):
        smoothed = build_smoothed_classifier(
            tiny_model, input_shape=(1, 28, 28), sample_size=10
        )
        x, y = small_data
        result = certify_predictions(smoothed, x, y, n_samples=10)
        assert "certified_accuracy" in result
        assert "avg_radius" in result
        assert "predictions" in result
        assert "radii" in result

    def test_certified_accuracy_range(self, tiny_model, small_data):
        smoothed = build_smoothed_classifier(
            tiny_model, input_shape=(1, 28, 28), sample_size=10
        )
        x, y = small_data
        result = certify_predictions(smoothed, x, y, n_samples=10)
        assert 0.0 <= result["certified_accuracy"] <= 1.0

    def test_predictions_array_shape(self, tiny_model, small_data):
        smoothed = build_smoothed_classifier(
            tiny_model, input_shape=(1, 28, 28), sample_size=10
        )
        x, y = small_data
        result = certify_predictions(smoothed, x, y, n_samples=10)
        assert result["predictions"].shape == (8,)
        assert result["radii"].shape == (8,)
