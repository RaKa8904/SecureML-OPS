"""Transfer attack (grey-box) — surrogate CNN + PGD, no ART class."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from art.attacks.evasion import ProjectedGradientDescent

from backend.attacks.base import BaseAttack


class _SurrogateCNN(nn.Module):
    """Lightweight surrogate model for transfer attacks (MNIST-shaped)."""

    def __init__(self, input_channels: int = 1, nb_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, nb_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class Transfer(BaseAttack):
    """Grey-box transfer attack.

    Train a surrogate CNN on the same data, generate adversarial examples
    with PGD on the surrogate, then evaluate them against the target model.

    Parameters
    ----------
    epsilon : float
        L-inf perturbation budget.  Default ``0.3``.
    surrogate_epochs : int
        Number of epochs to train the surrogate.  Default ``3``.
    pgd_iter : int
        PGD iterations on the surrogate.  Default ``40``.
    nb_classes : int
        Number of output classes.  Default ``10``.
    """

    def __init__(
        self,
        epsilon: float = 0.3,
        surrogate_epochs: int = 3,
        pgd_iter: int = 40,
        nb_classes: int = 10,
    ) -> None:
        super().__init__(epsilon=epsilon)
        self.surrogate_epochs = surrogate_epochs
        self.pgd_iter = pgd_iter
        self.nb_classes = nb_classes

    def _train_surrogate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        input_channels: int,
    ) -> nn.Module:
        """Train a lightweight surrogate CNN on the provided data."""
        surrogate = _SurrogateCNN(
            input_channels=input_channels,
            nb_classes=self.nb_classes,
        )
        surrogate.train()
        optimizer = torch.optim.Adam(surrogate.parameters(), lr=1e-3)
        x_t = torch.from_numpy(x)
        y_t = torch.from_numpy(y).long()

        for _ in range(self.surrogate_epochs):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(surrogate(x_t), y_t)
            loss.backward()
            optimizer.step()

        surrogate.eval()
        return surrogate

    def run(
        self,
        model: nn.Module,
        data: tuple[np.ndarray, np.ndarray],
        **kwargs,
    ) -> dict:
        """Run transfer attack.

        Parameters
        ----------
        model : nn.Module
            Target classifier (expects input in [0, 1]).
        data : tuple[ndarray, ndarray]
            ``(images, labels)`` — images shape ``(N, C, H, W)`` in [0, 1].

        Returns
        -------
        dict
            Standard attack result dict.
        """
        x, y = data
        input_shape = x.shape[1:]
        input_channels = input_shape[0]

        # 1) Evaluate target model clean accuracy
        target_classifier = self._wrap_model(model, input_shape, self.nb_classes)
        clean_accuracy = self._accuracy(target_classifier, x, y)

        # 2) Train surrogate
        surrogate = self._train_surrogate(x, y, input_channels)

        # 3) Run PGD on surrogate via ART
        surrogate_classifier = self._wrap_model(surrogate, input_shape, self.nb_classes)
        pgd = ProjectedGradientDescent(
            estimator=surrogate_classifier,
            eps=self.epsilon,
            eps_step=self.epsilon / 10,
            max_iter=self.pgd_iter,
            verbose=False,
        )
        x_adv = pgd.generate(x=x)

        # 4) Evaluate adversarial examples on *target* model
        adv_accuracy = self._accuracy(target_classifier, x_adv, y)

        return {
            "attack": "Transfer",
            "type": "grey-box",
            "clean_accuracy": clean_accuracy,
            "adv_accuracy": adv_accuracy,
            "epsilon": self.epsilon,
            "x_adv": x_adv,
        }
