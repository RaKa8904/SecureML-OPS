"""Unit tests for the FGSM attack module (ART-based)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from torchvision import datasets, transforms

from backend.attacks.fgsm import FGSM


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #
class SimpleCNN(nn.Module):
    """Tiny CNN for MNIST — only used in tests."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(7),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 7 * 7, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))


@pytest.fixture(scope="module")
def trained_model() -> nn.Module:
    """Return a deterministically-initialised SimpleCNN in eval mode."""
    torch.manual_seed(0)
    model = SimpleCNN()
    model.eval()
    return model


@pytest.fixture(scope="module")
def mnist_batch() -> tuple[np.ndarray, np.ndarray]:
    """Load a small MNIST batch (32 images) as numpy arrays in [0, 1]."""
    ds = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False)
    images, labels = next(iter(loader))
    return images.numpy(), labels.numpy()


# ------------------------------------------------------------------ #
# Tests
# ------------------------------------------------------------------ #
class TestFGSM:
    EPSILON = 0.3

    def _run(self, model: nn.Module, data: tuple[np.ndarray, np.ndarray]) -> dict:
        attack = FGSM(epsilon=self.EPSILON)
        return attack.run(model, data)

    def test_returns_dict_with_required_keys(
        self, trained_model: nn.Module, mnist_batch: tuple[np.ndarray, np.ndarray]
    ) -> None:
        result = self._run(trained_model, mnist_batch)
        assert isinstance(result, dict)
        for key in ("attack", "type", "clean_accuracy", "adv_accuracy", "epsilon", "x_adv"):
            assert key in result

    def test_attack_name_and_type(
        self, trained_model: nn.Module, mnist_batch: tuple[np.ndarray, np.ndarray]
    ) -> None:
        result = self._run(trained_model, mnist_batch)
        assert result["attack"] == "FGSM"
        assert result["type"] == "white-box"

    def test_adversarial_examples_differ(
        self, trained_model: nn.Module, mnist_batch: tuple[np.ndarray, np.ndarray]
    ) -> None:
        x, _ = mnist_batch
        result = self._run(trained_model, mnist_batch)
        assert not np.allclose(result["x_adv"], x)

    def test_perturbation_within_epsilon(
        self, trained_model: nn.Module, mnist_batch: tuple[np.ndarray, np.ndarray]
    ) -> None:
        x, _ = mnist_batch
        result = self._run(trained_model, mnist_batch)
        max_diff = np.abs(result["x_adv"] - x).max()
        assert max_diff <= self.EPSILON + 1e-6

    def test_output_shape_matches_input(
        self, trained_model: nn.Module, mnist_batch: tuple[np.ndarray, np.ndarray]
    ) -> None:
        x, _ = mnist_batch
        result = self._run(trained_model, mnist_batch)
        assert result["x_adv"].shape == x.shape

    def test_output_clamped_to_valid_range(
        self, trained_model: nn.Module, mnist_batch: tuple[np.ndarray, np.ndarray]
    ) -> None:
        result = self._run(trained_model, mnist_batch)
        assert result["x_adv"].min() >= 0.0
        assert result["x_adv"].max() <= 1.0

    def test_accuracies_are_valid_floats(
        self, trained_model: nn.Module, mnist_batch: tuple[np.ndarray, np.ndarray]
    ) -> None:
        result = self._run(trained_model, mnist_batch)
        assert 0.0 <= result["clean_accuracy"] <= 1.0
        assert 0.0 <= result["adv_accuracy"] <= 1.0

    def test_adv_accuracy_lower_than_clean(
        self, trained_model: nn.Module, mnist_batch: tuple[np.ndarray, np.ndarray]
    ) -> None:
        result = self._run(trained_model, mnist_batch)
        assert result["adv_accuracy"] <= result["clean_accuracy"]

    def test_epsilon_matches(
        self, trained_model: nn.Module, mnist_batch: tuple[np.ndarray, np.ndarray]
    ) -> None:
        result = self._run(trained_model, mnist_batch)
        assert result["epsilon"] == self.EPSILON
