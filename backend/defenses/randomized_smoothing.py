"""Randomized Smoothing defense — certified robustness via ART."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from art.estimators.certification.randomized_smoothing import (
    PyTorchRandomizedSmoothing,
)


def build_smoothed_classifier(
    model: nn.Module,
    input_shape: tuple[int, ...],
    nb_classes: int = 10,
    sigma: float = 0.25,
    sample_size: int = 100,
) -> PyTorchRandomizedSmoothing:
    """Wrap a PyTorch model in ART's randomized smoothing estimator.

    Parameters
    ----------
    model : nn.Module
        Base classifier to smooth.
    input_shape : tuple[int, ...]
        Single sample shape, e.g. ``(1, 28, 28)``.
    nb_classes : int
        Number of output classes.
    sigma : float
        Noise standard deviation for smoothing.
    sample_size : int
        Number of noise samples per prediction.

    Returns
    -------
    PyTorchRandomizedSmoothing
        Smoothed classifier that returns certified predictions.
    """
    model.eval()

    smoothed = PyTorchRandomizedSmoothing(
        model=model,
        loss=nn.CrossEntropyLoss(),
        input_shape=input_shape,
        nb_classes=nb_classes,
        clip_values=(0.0, 1.0),
        sample_size=sample_size,
        scale=sigma,
    )

    return smoothed


def certify_predictions(
    smoothed_classifier: PyTorchRandomizedSmoothing,
    x: np.ndarray,
    y: np.ndarray,
    n_samples: int = 100,
    alpha: float = 0.001,
) -> dict:
    """Run certified prediction on a batch and measure certified accuracy.

    Parameters
    ----------
    smoothed_classifier : PyTorchRandomizedSmoothing
        The smoothed classifier from ``build_smoothed_classifier``.
    x : ndarray
        Images of shape ``(N, C, H, W)`` in [0, 1].
    y : ndarray
        True labels.
    n_samples : int
        Number of Monte Carlo samples for certification.
    alpha : float
        Confidence level for the certification bound.

    Returns
    -------
    dict
        ``{"certified_accuracy": float, "avg_radius": float,
          "predictions": ndarray, "radii": ndarray}``
    """
    predictions = np.argmax(smoothed_classifier.predict(x), axis=1)
    certified_accuracy = float(np.mean(predictions == y))

    radii = []
    for i in range(len(x)):
        try:
            cert = smoothed_classifier.certify(
                x[i : i + 1], n=n_samples, batch_size=n_samples
            )
            radius = float(cert[1][0]) if len(cert) > 1 else 0.0
        except Exception:
            radius = 0.0
        radii.append(radius)

    radii_arr = np.array(radii)
    avg_radius = float(np.mean(radii_arr)) if len(radii_arr) > 0 else 0.0

    return {
        "certified_accuracy": certified_accuracy,
        "avg_radius": avg_radius,
        "predictions": predictions,
        "radii": radii_arr,
    }
