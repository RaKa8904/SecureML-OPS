"""Tests for preprocessing defenses."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from backend.defenses.preprocessing import (
    apply_feature_squeezing,
    apply_gaussian_augmentation,
    apply_jpeg_compression,
    wrap_model_with_preprocessor,
)


@pytest.fixture()
def sample_images() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.random((16, 1, 28, 28), dtype=np.float32)


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


class TestGaussianAugmentation:
    def test_output_shape(self, sample_images):
        result = apply_gaussian_augmentation(sample_images, sigma=0.1)
        assert result.shape == sample_images.shape

    def test_values_clipped(self, sample_images):
        result = apply_gaussian_augmentation(sample_images, sigma=0.5)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_output_differs(self, sample_images):
        result = apply_gaussian_augmentation(sample_images, sigma=0.1)
        assert not np.allclose(result, sample_images, atol=1e-6)


class TestJpegCompression:
    def test_output_shape(self, sample_images):
        result = apply_jpeg_compression(sample_images, quality=50)
        assert result.shape == sample_images.shape

    def test_different_from_input(self, sample_images):
        result = apply_jpeg_compression(sample_images, quality=30)
        assert not np.allclose(result, sample_images, atol=1e-6)


class TestFeatureSqueezing:
    def test_output_shape(self, sample_images):
        result = apply_feature_squeezing(sample_images, bit_depth=4)
        assert result.shape == sample_images.shape

    def test_reduced_precision(self, sample_images):
        result = apply_feature_squeezing(sample_images, bit_depth=1)
        unique_vals = np.unique(result)
        assert len(unique_vals) <= 4, "1-bit squeezing should have very few unique values"

    def test_values_in_range(self, sample_images):
        result = apply_feature_squeezing(sample_images, bit_depth=4)
        assert result.min() >= 0.0
        assert result.max() <= 1.0


class TestWrapModelWithPreprocessor:
    def test_gaussian_wrapper(self, sample_images):
        model = _TinyCNN()
        model.eval()
        classifier = wrap_model_with_preprocessor(
            model, input_shape=(1, 28, 28), defense_type="gaussian", sigma=0.1
        )
        preds = classifier.predict(sample_images)
        assert preds.shape == (16, 10)

    def test_feature_squeezing_wrapper(self, sample_images):
        model = _TinyCNN()
        model.eval()
        classifier = wrap_model_with_preprocessor(
            model, input_shape=(1, 28, 28), defense_type="feature_squeezing"
        )
        preds = classifier.predict(sample_images)
        assert preds.shape == (16, 10)

    def test_invalid_defense_type(self):
        model = _TinyCNN()
        with pytest.raises(ValueError, match="Unknown defense_type"):
            wrap_model_with_preprocessor(
                model, input_shape=(1, 28, 28), defense_type="invalid"
            )
