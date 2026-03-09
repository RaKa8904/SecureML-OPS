"""FGSM (Fast Gradient Sign Method) adversarial attack — ART-based."""

from __future__ import annotations

import numpy as np
import torch.nn as nn
from art.attacks.evasion import FastGradientMethod

from backend.attacks.base import BaseAttack


class FGSM(BaseAttack):
    """Single-step gradient-sign perturbation (Goodfellow et al., 2015).

    Wraps ART's ``FastGradientMethod`` with a PyTorch model.

    Parameters
    ----------
    epsilon : float
        L-inf perturbation budget.  Default ``0.3``.
    nb_classes : int
        Number of output classes.  Default ``10`` (MNIST / CIFAR-10).
    """

    def __init__(self, epsilon: float = 0.3, nb_classes: int = 10) -> None:
        super().__init__(epsilon=epsilon)
        self.nb_classes = nb_classes

    def run(
        self,
        model: nn.Module,
        data: tuple[np.ndarray, np.ndarray],
        **kwargs,
    ) -> dict:
        """Run FGSM on a full batch.

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
        input_shape = x.shape[1:]  # e.g. (1, 28, 28)

        classifier = self._wrap_model(model, input_shape, self.nb_classes)

        # Clean accuracy
        clean_accuracy = self._accuracy(classifier, x, y)

        # Generate adversarial examples via ART
        attack = FastGradientMethod(estimator=classifier, eps=self.epsilon)
        x_adv = attack.generate(x=x)

        # Adversarial accuracy
        adv_accuracy = self._accuracy(classifier, x_adv, y)

        return {
            "attack": "FGSM",
            "type": "white-box",
            "clean_accuracy": clean_accuracy,
            "adv_accuracy": adv_accuracy,
            "epsilon": self.epsilon,
            "x_adv": x_adv,
        }
