"""Square Attack (black-box) — ART-based."""

from __future__ import annotations

import numpy as np
import torch.nn as nn
from art.attacks.evasion import SquareAttack as SquareAttackART

from backend.attacks.base import BaseAttack


class Square(BaseAttack):
    """Score-based black-box attack with random square patches (Andriushchenko et al., 2020).

    Wraps ART's ``SquareAttack``.  Needs only confidence scores from the
    target model — no gradients or architecture knowledge.

    Parameters
    ----------
    epsilon : float
        L-inf perturbation budget.  Default ``0.3``.
    max_iter : int
        Maximum number of query iterations.  Default ``1000``.
    nb_classes : int
        Number of output classes.  Default ``10``.
    """

    def __init__(
        self,
        epsilon: float = 0.3,
        max_iter: int = 1000,
        nb_classes: int = 10,
    ) -> None:
        super().__init__(epsilon=epsilon)
        self.max_iter = max_iter
        self.nb_classes = nb_classes

    def run(
        self,
        model: nn.Module,
        data: tuple[np.ndarray, np.ndarray],
        **kwargs,
    ) -> dict:
        """Run Square Attack on a batch.

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

        attack = SquareAttackART(
            estimator=classifier,
            eps=self.epsilon,
            max_iter=self.max_iter,
            verbose=False,
        )
        x_adv = attack.generate(x=x, y=y)

        adv_accuracy = self._accuracy(classifier, x_adv, y)

        return {
            "attack": "Square",
            "type": "black-box",
            "clean_accuracy": clean_accuracy,
            "adv_accuracy": adv_accuracy,
            "epsilon": self.epsilon,
            "x_adv": x_adv,
        }
