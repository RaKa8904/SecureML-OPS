"""Adversarial attack modules for SecureML Ops."""

from backend.attacks.base import BaseAttack
from backend.attacks.fgsm import FGSM

__all__ = ["BaseAttack", "FGSM"]
