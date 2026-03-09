"""HopSkipJump adversarial attack (black-box) — ART-based."""

from __future__ import annotations

import numpy as np
import torch.nn as nn
from art.attacks.evasion import HopSkipJump as HopSkipJumpAttack

from backend.attacks.base import BaseAttack


class HopSkipJump(BaseAttack):
    """Decision-boundary-walking black-box attack (Chen et al., 2020).

    Wraps ART's ``HopSkipJump``.  Needs only output labels from the
    target model — no gradients or confidence scores.

    Parameters
    ----------
    epsilon : float
        Recorded for the result dict.  HopSkipJump does not use a fixed
        epsilon budget internally.  Default ``0.3``.
    max_iter : int
        Number of boundary-walk iterations.  Default ``50``.
    max_eval : int
        Maximum model evaluations per iteration.  Default ``1000``.
    init_eval : int
        Initial model evaluations for gradient estimation.  Default ``100``.
    nb_classes : int
        Number of output classes.  Default ``10``.
    """

    def __init__(
        self,
        epsilon: float = 0.3,
        max_iter: int = 50,
        max_eval: int = 1000,
        init_eval: int = 100,
        nb_classes: int = 10,
    ) -> None:
        super().__init__(epsilon=epsilon)
        self.max_iter = max_iter
        self.max_eval = max_eval
        self.init_eval = init_eval
        self.nb_classes = nb_classes

    def run(
        self,
        model: nn.Module,
        data: tuple[np.ndarray, np.ndarray],
        **kwargs,
    ) -> dict:
        """Run HopSkipJump on a batch.

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

        attack = HopSkipJumpAttack(
            classifier=classifier,
            max_iter=self.max_iter,
            max_eval=self.max_eval,
            init_eval=self.init_eval,
            verbose=False,
        )
        x_adv = attack.generate(x=x)

        adv_accuracy = self._accuracy(classifier, x_adv, y)

        return {
            "attack": "HopSkipJump",
            "type": "black-box",
            "clean_accuracy": clean_accuracy,
            "adv_accuracy": adv_accuracy,
            "epsilon": self.epsilon,
            "x_adv": x_adv,
        }
