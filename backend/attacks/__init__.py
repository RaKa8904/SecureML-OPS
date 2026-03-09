"""Adversarial attack modules for SecureML Ops."""

from backend.attacks.base import BaseAttack
from backend.attacks.fgsm import FGSM
from backend.attacks.pgd import PGD

__all__ = ["BaseAttack", "FGSM", "PGD"]
