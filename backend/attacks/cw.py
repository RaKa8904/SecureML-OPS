"""C&W (Carlini & Wagner L2) adversarial attack — ART-based."""

from __future__ import annotations

import numpy as np
import torch.nn as nn
from art.attacks.evasion import CarliniL2Method

from backend.attacks.base import BaseAttack


class CW(BaseAttack):
    """Optimization-based L2 attack (Carlini & Wagner, 2017).

    Wraps ART's ``CarliniL2Method``.  Slow — run on a small batch
    (≤ 100 samples) to keep execution time reasonable.

    Parameters
    ----------
    epsilon : float
        Recorded for the result dict; C&W minimises L2 directly
        rather than using a fixed epsilon budget.  Default ``0.3``.
    confidence : float
        Confidence parameter (kappa) for the attack.  Default ``0.0``.
    max_iter : int
        Maximum optimisation iterations.  Default ``100``.
    batch_size : int
        Batch size passed to ART's ``generate()``.  Default ``32``.
    nb_classes : int
        Number of output classes.  Default ``10``.
    """

    def __init__(
        self,
        epsilon: float = 0.3,
        confidence: float = 0.0,
        max_iter: int = 100,
        batch_size: int = 32,
        nb_classes: int = 10,
    ) -> None:
        super().__init__(epsilon=epsilon)
        self.confidence = confidence
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.nb_classes = nb_classes

    def run(
        self,
        model: nn.Module,
        data: tuple[np.ndarray, np.ndarray],
        **kwargs,
    ) -> dict:
        """Run C&W L2 on a batch.

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

        attack = CarliniL2Method(
            classifier=classifier,
            confidence=self.confidence,
            max_iter=self.max_iter,
            batch_size=self.batch_size,
            verbose=False,
        )
        x_adv = attack.generate(x=x)

        adv_accuracy = self._accuracy(classifier, x_adv, y)

        return {
            "attack": "C&W",
            "type": "white-box",
            "clean_accuracy": clean_accuracy,
            "adv_accuracy": adv_accuracy,
            "epsilon": self.epsilon,
            "x_adv": x_adv,
        }
