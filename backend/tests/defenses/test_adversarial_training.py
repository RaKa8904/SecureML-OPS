"""Tests for adversarial_training defense."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from backend.defenses.adversarial_training import adversarial_training


class _SimpleCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.fc = nn.Linear(8 * 4 * 4, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x).view(x.size(0), -1))


@pytest.fixture()
def simple_model() -> nn.Module:
    torch.manual_seed(42)
    model = _SimpleCNN()
    model.eval()
    return model


@pytest.fixture()
def small_train_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    x = rng.random((64, 1, 28, 28), dtype=np.float32)
    y = rng.integers(0, 10, size=64)
    return x, y


class TestAdversarialTraining:
    def test_returns_module(self, simple_model, small_train_data):
        result = adversarial_training(
            model=simple_model,
            train_data=small_train_data,
            input_shape=(1, 28, 28),
            nb_epochs=1,
            batch_size=32,
        )
        assert isinstance(result, nn.Module)

    def test_original_unchanged(self, simple_model, small_train_data):
        original_params = {
            k: v.clone() for k, v in simple_model.state_dict().items()
        }
        adversarial_training(
            model=simple_model,
            train_data=small_train_data,
            input_shape=(1, 28, 28),
            nb_epochs=1,
            batch_size=32,
        )
        for k, v in simple_model.state_dict().items():
            assert torch.equal(v, original_params[k]), (
                f"Original model parameter '{k}' was modified"
            )

    def test_hardened_model_differs(self, simple_model, small_train_data):
        hardened = adversarial_training(
            model=simple_model,
            train_data=small_train_data,
            input_shape=(1, 28, 28),
            nb_epochs=2,
            batch_size=32,
        )
        original_params = dict(simple_model.named_parameters())
        hardened_params = dict(hardened.named_parameters())
        any_different = any(
            not torch.equal(original_params[k].data, hardened_params[k].data)
            for k in original_params
        )
        assert any_different, "Hardened model should differ from original"

    def test_hardened_model_in_eval_mode(self, simple_model, small_train_data):
        hardened = adversarial_training(
            model=simple_model,
            train_data=small_train_data,
            input_shape=(1, 28, 28),
            nb_epochs=1,
            batch_size=32,
        )
        assert not hardened.training, "Returned model should be in eval mode"
