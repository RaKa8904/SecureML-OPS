"""Unit tests for the C&W attack module (ART-based)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from torchvision import datasets, transforms

from backend.attacks.cw import CW


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
    """Train a SimpleCNN briefly so it classifies some samples correctly."""
    torch.manual_seed(0)
    model = SimpleCNN()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ds = datasets.MNIST(root="data", train=True, download=True, transform=transforms.ToTensor())
    loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True)
    # 2 epochs on full training set — fast enough, gives ~90% accuracy
    for _ in range(2):
        for imgs, lbls in loader:
            optimizer.zero_grad()
            nn.functional.cross_entropy(model(imgs), lbls).backward()
            optimizer.step()
    model.eval()
    return model


@pytest.fixture(scope="module")
def mnist_batch() -> tuple[np.ndarray, np.ndarray]:
    """Smaller batch (16 samples) — C&W is slow."""
    ds = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False)
    images, labels = next(iter(loader))
    return images.numpy(), labels.numpy()


# ------------------------------------------------------------------ #
# Tests
# ------------------------------------------------------------------ #
class TestCW:
    EPSILON = 0.3

    def _run(self, model: nn.Module, data: tuple[np.ndarray, np.ndarray]) -> dict:
        attack = CW(epsilon=self.EPSILON, max_iter=20, batch_size=16)  # fast for CI
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
        assert result["attack"] == "C&W"
        assert result["type"] == "white-box"

    def test_adversarial_examples_differ(
        self, trained_model: nn.Module, mnist_batch: tuple[np.ndarray, np.ndarray]
    ) -> None:
        x, _ = mnist_batch
        result = self._run(trained_model, mnist_batch)
        # C&W only perturbs correctly-classified samples; check at least one differs
        assert np.any(np.abs(result["x_adv"] - x) > 1e-6)

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
        assert result["x_adv"].min() >= -1e-6
        assert result["x_adv"].max() <= 1.0 + 1e-6

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
