"""Adversarial Training defense — uses ART's AdversarialTrainer."""

from __future__ import annotations

import copy

import numpy as np
import torch.nn as nn
from art.attacks.evasion import ProjectedGradientDescent
from art.defences.trainer import AdversarialTrainer
from art.estimators.classification import PyTorchClassifier


def adversarial_training(
    model: nn.Module,
    train_data: tuple[np.ndarray, np.ndarray],
    input_shape: tuple[int, ...],
    nb_classes: int = 10,
    epsilon: float = 0.3,
    nb_epochs: int = 5,
    batch_size: int = 128,
    ratio: float = 0.5,
) -> nn.Module:
    """Apply adversarial training to a PyTorch model.

    Uses PGD-generated adversarial examples mixed with clean samples
    during training via ART's ``AdversarialTrainer``.

    Parameters
    ----------
    model : nn.Module
        The model to harden.  Will be deep-copied internally.
    train_data : tuple[ndarray, ndarray]
        ``(images, labels)`` — images shape ``(N, C, H, W)`` in [0, 1].
    input_shape : tuple[int, ...]
        Single sample shape, e.g. ``(1, 28, 28)``.
    nb_classes : int
        Number of output classes.
    epsilon : float
        PGD perturbation budget for generating adversarial examples.
    nb_epochs : int
        Number of adversarial training epochs.
    batch_size : int
        Training batch size.
    ratio : float
        Fraction of each batch that is adversarial (0–1).

    Returns
    -------
    nn.Module
        The adversarially-trained model (new copy; original unchanged).
    """
    hardened = copy.deepcopy(model)
    hardened.train()

    classifier = PyTorchClassifier(
        model=hardened,
        loss=nn.CrossEntropyLoss(),
        input_shape=input_shape,
        nb_classes=nb_classes,
        clip_values=(0.0, 1.0),
        optimizer=_make_optimizer(hardened),
    )

    pgd = ProjectedGradientDescent(
        estimator=classifier,
        eps=epsilon,
        eps_step=epsilon / 10,
        max_iter=10,
        batch_size=batch_size,
    )

    trainer = AdversarialTrainer(
        classifier=classifier,
        attacks=pgd,
        ratio=ratio,
    )

    x_train, y_train = train_data
    y_onehot = np.eye(nb_classes, dtype=np.float32)[y_train]

    trainer.fit(x_train, y_onehot, nb_epochs=nb_epochs, batch_size=batch_size)

    hardened.eval()
    return hardened


def _make_optimizer(model: nn.Module):
    """Create a default Adam optimizer for adversarial training."""
    import torch.optim as optim

    return optim.Adam(model.parameters(), lr=1e-3)
