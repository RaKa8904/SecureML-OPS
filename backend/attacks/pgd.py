"""PGD (Projected Gradient Descent) adversarial attack — ART-based."""

from __future__ import annotations

import numpy as np
import torch.nn as nn
from art.attacks.evasion import ProjectedGradientDescent

from backend.attacks.base import BaseAttack


class PGD(BaseAttack):
    """Iterative projected gradient descent (Madry et al., 2018).

    Wraps ART's ``ProjectedGradientDescent`` with a PyTorch model.

    Parameters
    ----------
    epsilon : float
        L-inf perturbation budget.  Default ``0.3``.
    max_iter : int
        Number of PGD iterations.  Default ``40``.
    eps_step : float
        Step size per iteration.  Default ``epsilon / 10``.
    nb_classes : int
        Number of output classes.  Default ``10`` (MNIST / CIFAR-10).
    """

    def __init__(
        self,
        epsilon: float = 0.3,
        max_iter: int = 40,
        eps_step: float | None = None,
        nb_classes: int = 10,
    ) -> None:
        super().__init__(epsilon=epsilon)
        self.max_iter = max_iter
        self.eps_step = eps_step if eps_step is not None else epsilon / 10
        self.nb_classes = nb_classes

    def run(
        self,
        model: nn.Module,
        data: tuple[np.ndarray, np.ndarray],
        **kwargs,
    ) -> dict:
        """Run PGD on a full batch.

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

        classifier = self._wrap_model(model, input_shape, self.nb_classes)

        clean_accuracy = self._accuracy(classifier, x, y)

        attack = ProjectedGradientDescent(
            estimator=classifier,
            eps=self.epsilon,
            eps_step=self.eps_step,
            max_iter=self.max_iter,
            verbose=False,
        )
        x_adv = attack.generate(x=x)

        adv_accuracy = self._accuracy(classifier, x_adv, y)

        return {
            "attack": "PGD",
            "type": "white-box",
            "clean_accuracy": clean_accuracy,
            "adv_accuracy": adv_accuracy,
            "epsilon": self.epsilon,
            "x_adv": x_adv,
        }
