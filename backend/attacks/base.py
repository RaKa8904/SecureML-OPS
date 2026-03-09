"""Base class and shared helpers for all adversarial attack modules."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch.nn as nn
from art.estimators.classification import PyTorchClassifier


class BaseAttack(ABC):
    """Abstract base class that every attack module must inherit from.

    Parameters
    ----------
    epsilon : float
        Maximum perturbation budget.
    """

    def __init__(self, epsilon: float = 0.3) -> None:
        self.epsilon = epsilon

    @abstractmethod
    def run(
        self,
        model: nn.Module,
        data: tuple[np.ndarray, np.ndarray],
        **kwargs,
    ) -> dict:
        """Execute the attack on a batch of inputs.

        Parameters
        ----------
        model : nn.Module
            Target classifier in eval mode.
        data : tuple[ndarray, ndarray]
            ``(images, labels)`` — images shape ``(N, C, H, W)`` in [0, 1].

        Returns
        -------
        dict
            Keys: attack, type, clean_accuracy, adv_accuracy, epsilon, x_adv.
        """
        ...

    # ------------------------------------------------------------------ #
    # Shared helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _wrap_model(
        model: nn.Module,
        input_shape: tuple[int, ...],
        nb_classes: int = 10,
    ) -> PyTorchClassifier:
        """Wrap a PyTorch model in ART's PyTorchClassifier."""
        return PyTorchClassifier(
            model=model,
            loss=nn.CrossEntropyLoss(),
            input_shape=input_shape,
            nb_classes=nb_classes,
            clip_values=(0.0, 1.0),
        )

    @staticmethod
    def _accuracy(classifier: PyTorchClassifier, x: np.ndarray, y: np.ndarray) -> float:
        """Compute top-1 accuracy as a float in [0, 1]."""
        preds = np.argmax(classifier.predict(x), axis=1)
        return float(np.mean(preds == y))
