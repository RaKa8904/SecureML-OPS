"""Preprocessing-based defenses — Gaussian Augmentation, JPEG Compression, Feature Squeezing."""

from __future__ import annotations

import numpy as np
import torch.nn as nn
from art.defences.preprocessor import (
    FeatureSqueezing,
    GaussianAugmentation,
    JpegCompression,
)
from art.estimators.classification import PyTorchClassifier


def apply_gaussian_augmentation(
    x: np.ndarray,
    sigma: float = 0.1,
) -> np.ndarray:
    """Add Gaussian noise to inputs as a preprocessing defense.

    Parameters
    ----------
    x : ndarray
        Images of shape ``(N, C, H, W)`` in [0, 1].
    sigma : float
        Standard deviation of the Gaussian noise.

    Returns
    -------
    ndarray
        Augmented images clipped to [0, 1].
    """
    defense = GaussianAugmentation(sigma=sigma, augmentation=False)
    x_defended, _ = defense(x)
    return np.clip(x_defended, 0.0, 1.0)


def apply_jpeg_compression(
    x: np.ndarray,
    quality: int = 50,
) -> np.ndarray:
    """Apply JPEG compression to strip adversarial perturbations.

    Parameters
    ----------
    x : ndarray
        Images of shape ``(N, C, H, W)`` in [0, 1].
    quality : int
        JPEG quality factor (1–100). Lower = more smoothing.

    Returns
    -------
    ndarray
        JPEG-compressed images.
    """
    channels = x.shape[1]
    defense = JpegCompression(
        clip_values=(0.0, 1.0),
        quality=quality,
        channels_first=True,
    )
    x_defended, _ = defense(x)
    return x_defended


def apply_feature_squeezing(
    x: np.ndarray,
    bit_depth: int = 4,
) -> np.ndarray:
    """Reduce input precision to remove fine-grained perturbations.

    Parameters
    ----------
    x : ndarray
        Images of shape ``(N, C, H, W)`` in [0, 1].
    bit_depth : int
        Number of bits to keep per channel (1–8).

    Returns
    -------
    ndarray
        Squeezed images.
    """
    defense = FeatureSqueezing(clip_values=(0.0, 1.0), bit_depth=bit_depth)
    x_defended, _ = defense(x)
    return x_defended


def wrap_model_with_preprocessor(
    model: nn.Module,
    input_shape: tuple[int, ...],
    nb_classes: int = 10,
    defense_type: str = "feature_squeezing",
    **kwargs,
) -> PyTorchClassifier:
    """Wrap a PyTorch model with an ART preprocessor defense.

    Parameters
    ----------
    model : nn.Module
        Target classifier in eval mode.
    input_shape : tuple[int, ...]
        Single sample shape, e.g. ``(1, 28, 28)``.
    nb_classes : int
        Number of output classes.
    defense_type : str
        One of ``"gaussian"``, ``"jpeg"``, ``"feature_squeezing"``.
    **kwargs
        Passed to the chosen preprocessor constructor.

    Returns
    -------
    PyTorchClassifier
        ART classifier with the preprocessor attached.
    """
    preprocessors = {
        "gaussian": lambda: GaussianAugmentation(
            sigma=kwargs.get("sigma", 0.1), augmentation=False
        ),
        "jpeg": lambda: JpegCompression(
            clip_values=(0.0, 1.0),
            quality=kwargs.get("quality", 50),
            channels_first=True,
        ),
        "feature_squeezing": lambda: FeatureSqueezing(
            clip_values=(0.0, 1.0),
            bit_depth=kwargs.get("bit_depth", 4),
        ),
    }

    if defense_type not in preprocessors:
        raise ValueError(
            f"Unknown defense_type '{defense_type}'. "
            f"Choose from {list(preprocessors.keys())}."
        )

    defense = preprocessors[defense_type]()

    return PyTorchClassifier(
        model=model,
        loss=nn.CrossEntropyLoss(),
        input_shape=input_shape,
        nb_classes=nb_classes,
        clip_values=(0.0, 1.0),
        preprocessing_defences=[defense],
    )
